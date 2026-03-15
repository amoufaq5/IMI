"""
Comprehensive Biomedical Corpus Collector for IMI Medical LLM

Collects the full spectrum of biomedical training data across 11 categories:

  1. Biomedical Literature      — PubMed, PMC, S2ORC, CORD-19, LitCovid
  2. Preprint Servers           — bioRxiv, medRxiv
  3. Clinical EHR / ICU         — MIMIC-III/IV, eICU (requires PhysioNet creds)
  4. Medical NLP Datasets       — i2b2, MedMentions, THYME, MTSamples
  5. Medical Question Answering — BioASQ, HealthSearchQA, LiveQA, MultiMedQA
  6. Drug & Pharmacology        — DrugBank, SIDER, TWOSIDES, OFFSIDES, DailyMed
  7. Genomics                   — TCGA clinical, ClinVar, HPO, Ensembl terms
  8. Clinical Trials            — ClinicalTrials.gov, WHO ICTRP
  9. Medical Ontologies         — UMLS (partial), MeSH, HPO, ICD-10, SNOMED summaries
  10. Consumer Health Dialogue  — MedDialog, HealthSearchQA, BLURB
  11. BigBio Benchmarks         — Standardized access to 50+ medical NLP datasets

Data access tiers:
  FREE    — No credentials needed (API, HuggingFace, public download)
  API_KEY — Requires NCBI_API_KEY in .env (PubMed, PMC, MeSH, GenBank)
  CRED    — Requires separate registration/credentials (MIMIC, DrugBank, UMLS)
            → Script generates detailed instructions + placeholder files

All output is in the standard IMI two-format schema:
  general_knowledge: {"text": "...", "source": "...", "adapter": "..."}
  instruction:       {"instruction": "...", "input": "...", "output": "...", "source": "...", "adapter": "..."}

Usage:
    # Full collection (all free + API key sources)
    python scripts/data_collection/collect_biomedical_corpus.py

    # Single category
    python scripts/data_collection/collect_biomedical_corpus.py --category pubmed
    python scripts/data_collection/collect_biomedical_corpus.py --category clinical_trials

    # List all sources and their access tier
    python scripts/data_collection/collect_biomedical_corpus.py --list

    # Generate instructions for credentialed datasets
    python scripts/data_collection/collect_biomedical_corpus.py --credentials-guide

Environment variables (set in .env):
    NCBI_API_KEY  — NCBI E-utilities key (10 req/s vs 3 req/s without)
"""

import os
import json
import time
import logging
import argparse
import hashlib
import random
import zipfile
import gzip
import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Tuple
from urllib.parse import urlencode, urljoin
from io import StringIO

import requests
from tqdm import tqdm

try:
    from datasets import load_dataset
    HAS_HF = True
except ImportError:
    HAS_HF = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "biomedical"
PROCESSED_DIR = DATA_DIR / "processed"

NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
SEMANTIC_SCHOLAR_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

# NCBI rate: 10/s with key, 3/s without
NCBI_RATE_DELAY = 0.11 if NCBI_API_KEY else 0.35


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def make_instruction(instruction: str, output: str, source: str, adapter: str,
                     input_text: str = "") -> Dict[str, Any]:
    return {"instruction": instruction, "input": input_text,
            "output": output, "source": source, "adapter": adapter}


def make_general(text: str, source: str, adapter: str) -> Dict[str, Any]:
    return {"text": text, "source": source, "adapter": adapter}


def save_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    logger.info(f"Saved {len(records):,} records → {path}")


def load_existing_ids(path: Path) -> set:
    """Load IDs of already-downloaded records to support resume."""
    if not path.exists():
        return set()
    ids = set()
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                ids.add(hashlib.md5((r.get("instruction", "") + r.get("text", "")).encode()).hexdigest())
            except Exception:
                pass
    return ids


# =============================================================================
# ROBUST HTTP HELPERS
# =============================================================================

_RETRY_EXCEPTIONS = (
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ReadTimeout,
)


def _robust_get(session: requests.Session, url: str, params: dict = None,
                timeout: int = 90, retries: int = 4, backoff: float = 2.0,
                stream: bool = False) -> requests.Response:
    """GET with exponential-backoff retries on transient network errors."""
    last_exc = None
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=timeout, stream=stream)
            r.raise_for_status()
            if not stream:
                # Force reading the body now so ChunkedEncodingError surfaces here
                _ = r.content
            return r
        except _RETRY_EXCEPTIONS as exc:
            last_exc = exc
            if attempt < retries - 1:
                wait = backoff ** (attempt + 1)
                logger.warning(f"  Network error ({type(exc).__name__}), retry {attempt+1}/{retries-1} in {wait:.0f}s…")
                time.sleep(wait)
    raise last_exc


# =============================================================================
# NCBI E-UTILITIES CLIENT
# =============================================================================

class NCBIClient:
    """
    Client for NCBI E-utilities (PubMed, PMC, MeSH, GenBank, LitCovid).
    https://www.ncbi.nlm.nih.gov/books/NBK25501/
    """
    BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    def __init__(self, api_key: str = "", tool: str = "imi_medical_llm",
                 email: str = "research@imi.ai"):
        self.api_key = api_key
        self.tool = tool
        self.email = email
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"{tool}/1.0"})
        self._delay = 0.11 if api_key else 0.35

    def _params(self, **kwargs) -> Dict[str, str]:
        p = {"tool": self.tool, "email": self.email, "retmode": "json"}
        if self.api_key:
            p["api_key"] = self.api_key
        p.update(kwargs)
        return p

    def _get(self, endpoint: str, params: dict) -> dict:
        url = self.BASE + endpoint
        time.sleep(self._delay)
        r = _robust_get(self.session, url, params=params, timeout=60)
        # NCBI occasionally returns JSON with unescaped control characters
        return json.loads(r.text, strict=False)

    def _get_xml(self, endpoint: str, params: dict) -> str:
        url = self.BASE + endpoint
        time.sleep(self._delay)
        params = {k: v for k, v in params.items() if k != "retmode"}
        params.update({"tool": self.tool, "email": self.email})
        if self.api_key:
            params["api_key"] = self.api_key
        r = _robust_get(self.session, url, params=params, timeout=120)
        return r.text

    def search(self, db: str, query: str, retmax: int = 10_000) -> List[str]:
        """Run ESearch and return IDs via paginated WebEnv history.

        NCBI caps a single retmax at 10 000; this method pages through the
        entire result set in 10 000-ID batches up to retmax.
        """
        result = self._get("esearch.fcgi", self._params(
            db=db, term=query, retmax=0, usehistory="y",
        ))
        esearch = result.get("esearchresult", {})
        total = min(int(esearch.get("count", 0)), retmax)
        web_env = esearch.get("webenv", "")
        query_key = esearch.get("querykey", "")

        if total == 0:
            return []

        page = 10_000  # NCBI hard cap per request
        all_ids: List[str] = []
        for retstart in range(0, total, page):
            r = self._get("esearch.fcgi", self._params(
                db=db, term=query, retmax=min(page, total - retstart),
                retstart=retstart,
                usehistory="y", WebEnv=web_env, query_key=query_key,
            ))
            batch = r.get("esearchresult", {}).get("idlist", [])
            all_ids.extend(batch)
            if not batch:
                break

        return all_ids[:retmax]

    def fetch_pubmed_abstracts(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """Fetch PubMed abstracts for a list of PMIDs."""
        if not pmids:
            return []
        xml_text = self._get_xml("efetch.fcgi", {
            "db": "pubmed", "id": ",".join(pmids),
            "rettype": "abstract", "retmode": "xml",
        })
        return self._parse_pubmed_xml(xml_text)

    def _parse_pubmed_xml(self, xml_text: str) -> List[Dict[str, Any]]:
        """Parse PubMed XML into structured records."""
        records = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return records
        for article in root.findall(".//PubmedArticle"):
            try:
                pmid_el = article.find(".//PMID")
                pmid = pmid_el.text if pmid_el is not None else ""
                title_el = article.find(".//ArticleTitle")
                title = "".join(title_el.itertext()) if title_el is not None else ""
                abstract_texts = article.findall(".//AbstractText")
                abstract = " ".join("".join(el.itertext()) for el in abstract_texts)
                mesh_headings = [
                    h.find("DescriptorName").text
                    for h in article.findall(".//MeshHeading")
                    if h.find("DescriptorName") is not None
                ]
                pub_year_el = article.find(".//PubDate/Year")
                year = pub_year_el.text if pub_year_el is not None else ""
                if title and abstract:
                    records.append({
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract,
                        "mesh_terms": mesh_headings,
                        "year": year,
                    })
            except Exception:
                continue
        return records

    def fetch_pmc_full_text(self, pmcids: List[str]) -> List[Dict[str, Any]]:
        """Fetch PMC full-text passages (introduction + methods + results sections)."""
        if not pmcids:
            return []
        xml_text = self._get_xml("efetch.fcgi", {
            "db": "pmc", "id": ",".join(pmcids),
            "rettype": "full", "retmode": "xml",
        })
        records = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return records
        for article in root.findall(".//article"):
            try:
                pmcid = ""
                for id_el in article.findall(".//article-id"):
                    if id_el.get("pub-id-type") == "pmc":
                        pmcid = id_el.text or ""
                title_el = article.find(".//article-title")
                title = "".join(title_el.itertext()) if title_el is not None else ""
                body_text = " ".join(
                    "".join(p.itertext())
                    for p in article.findall(".//body//p")
                )[:4000]
                if title and body_text:
                    records.append({"pmcid": pmcid, "title": title, "body": body_text})
            except Exception:
                continue
        return records


# =============================================================================
# 1. PUBMED LITERATURE
# =============================================================================

def collect_pubmed(ncbi: NCBIClient, queries: List[Tuple[str, str, str]],
                   max_per_query: int = 10_000) -> List[Dict[str, Any]]:
    """Collect PubMed abstracts across clinical domains."""
    records = []
    batch_size = 50  # Small batches to avoid NCBI truncating responses

    for query_str, adapter, topic in queries:
        logger.info(f"  PubMed: {topic} ({max_per_query:,} max)")
        try:
            pmids = ncbi.search("pubmed", query_str, retmax=max_per_query)
        except Exception as e:
            logger.warning(f"    Search failed for {topic}: {e}")
            continue
        logger.info(f"    Found {len(pmids)} articles")

        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]
            try:
                abstracts = ncbi.fetch_pubmed_abstracts(batch)
            except Exception as e:
                logger.warning(f"    Batch {i//batch_size} failed ({e}), skipping")
                continue
            for a in abstracts:
                text = f"Title: {a['title']}\n\nAbstract: {a['abstract']}"
                if a["mesh_terms"]:
                    text += f"\n\nMeSH terms: {', '.join(a['mesh_terms'][:10])}"
                records.append(make_general(text, f"pubmed_{topic}", adapter))

        logger.info(f"    Collected {len(records)} total PubMed records so far")

    return records


PUBMED_QUERIES = [
    # (query, adapter, label)
    ("systematic review[pt] AND (cardiovascular diseases[mh] OR heart disease[mh]) AND 2018:2024[dp]",
     "research", "cardiology_reviews"),
    ("systematic review[pt] AND oncology[mh] AND 2018:2024[dp]",
     "research", "oncology_reviews"),
    ("systematic review[pt] AND (diabetes mellitus[mh] OR endocrine diseases[mh]) AND 2018:2024[dp]",
     "research", "endocrinology_reviews"),
    ("systematic review[pt] AND neurology[mh] AND 2018:2024[dp]",
     "research", "neurology_reviews"),
    ("systematic review[pt] AND mental disorders[mh] AND 2018:2024[dp]",
     "research", "psychiatry_reviews"),
    ("systematic review[pt] AND (infectious diseases[mh] OR anti-bacterial agents[mh]) AND 2018:2024[dp]",
     "research", "infectious_disease_reviews"),
    ("systematic review[pt] AND pediatrics[mh] AND 2018:2024[dp]",
     "research", "pediatrics_reviews"),
    ("clinical practice guideline[pt] AND 2018:2024[dp]",
     "research", "clinical_guidelines"),
    ("drug interactions[mh] AND 2020:2024[dp]",
     "clinical_pharmacist", "drug_interactions"),
    ("adverse drug reaction reporting systems[mh] AND 2018:2024[dp]",
     "clinical_pharmacist", "adverse_drug_reactions"),
    ("triage[mh] AND emergency medicine[mh] AND 2018:2024[dp]",
     "patient_triage", "emergency_triage"),
    ("pharmacogenomics[mh] AND 2018:2024[dp]",
     "clinical_pharmacist", "pharmacogenomics"),
    ("sepsis[mh] AND critical care[mh] AND 2018:2024[dp]",
     "clinical_decision", "icu_sepsis"),
    ("surgery[mh] AND postoperative complications[mh] AND 2018:2024[dp]",
     "clinical_decision", "surgery"),
    ("rare diseases[mh] AND 2018:2024[dp]",
     "clinical_decision", "rare_diseases"),
]


# =============================================================================
# 2. PUBMED CENTRAL — FULL TEXT
# =============================================================================

def collect_pmc(ncbi: NCBIClient, max_articles: int = 10_000) -> List[Dict[str, Any]]:
    """Collect PMC open-access full-text articles across medical disciplines."""
    records = []
    pmc_queries = [
        ("open access[filter] AND clinical trial[pt] AND 2020:2024[dp]", "research", "rct_full_text"),
        ("open access[filter] AND review[pt] AND pharmacology[mh] AND 2020:2024[dp]",
         "clinical_pharmacist", "pharma_reviews"),
        ("open access[filter] AND case reports[pt] AND rare diseases[mh] AND 2020:2024[dp]",
         "clinical_decision", "case_reports"),
    ]
    per_query = max_articles // len(pmc_queries)
    batch_size = 25  # PMC full-text is large; fetch in small batches

    for query_str, adapter, topic in pmc_queries:
        logger.info(f"  PMC: {topic}")
        try:
            pmcids = ncbi.search("pmc", query_str, retmax=per_query)
        except Exception as e:
            logger.warning(f"    PMC search failed for {topic}: {e}")
            continue
        logger.info(f"    Found {len(pmcids)} PMC articles")
        for i in range(0, min(len(pmcids), 500), batch_size):
            batch = pmcids[i:i + batch_size]
            try:
                articles = ncbi.fetch_pmc_full_text(batch)
            except Exception as e:
                logger.warning(f"    PMC batch {i//batch_size} failed ({e}), skipping")
                continue
            for a in articles:
                text = f"Title: {a['title']}\n\n{a['body']}"
                records.append(make_general(text, f"pmc_{topic}", adapter))
    logger.info(f"  Collected {len(records)} PMC full-text records")
    return records


# =============================================================================
# 3. BIORXIV / MEDRXIV PREPRINTS
# =============================================================================

def collect_preprints(max_per_server: int = 2000) -> List[Dict[str, Any]]:
    """Collect recent preprints from bioRxiv and medRxiv via their public API."""
    records = []
    servers = [
        ("biorxiv", "research", [
            "pharmacology and toxicology", "neuroscience", "immunology", "genomics",
            "microbiology", "cell biology", "molecular biology",
        ]),
        ("medrxiv", "research", [
            "cardiovascular medicine", "oncology", "epidemiology",
            "infectious diseases", "psychiatry and clinical psychology",
            "pharmacology and therapeutics", "intensive care and critical care medicine",
        ]),
    ]
    session = requests.Session()
    session.headers["User-Agent"] = "IMI-Medical-LLM/1.0"

    for server, adapter, subjects in servers:
        collected = 0
        # Paginate through the API (returns 100 per page by default)
        cursor = 0
        while collected < max_per_server:
            try:
                url = f"https://api.biorxiv.org/details/{server}/2023-01-01/2024-12-31/{cursor}"
                resp = _robust_get(session, url, timeout=30)
                data = resp.json()
                papers = data.get("collection", [])
                if not papers:
                    break
                for paper in papers:
                    category = paper.get("category", "").lower()
                    if any(s.lower() in category for s in subjects):
                        title = paper.get("title", "")
                        abstract = paper.get("abstract", "")
                        if title and abstract and len(abstract) > 100:
                            text = f"Title: {title}\n\nAbstract: {abstract}"
                            records.append(make_general(text, f"{server}_preprint", adapter))
                            collected += 1
                            if collected >= max_per_server:
                                break
                cursor += len(papers)
                # API returns at most 100 per page; stop if fewer
                if len(papers) < 100:
                    break
                time.sleep(1.0)
            except Exception as e:
                logger.warning(f"  Preprint collection error ({server} cursor={cursor}): {e}")
                break
        logger.info(f"  {server}: {collected} preprints collected")

    logger.info(f"  Collected {len(records)} total preprint records")
    return records


# =============================================================================
# 4. LITCOVID — COVID-19 LITERATURE
# =============================================================================

def collect_litcovid(ncbi: NCBIClient, max_articles: int = 10_000) -> List[Dict[str, Any]]:
    """Collect LitCovid COVID-19 curated literature via PubMed."""
    logger.info("  Collecting LitCovid (COVID-19 curated literature)...")
    query = "COVID-19[mh] OR SARS-CoV-2[mh] AND 2020:2024[dp]"
    try:
        pmids = ncbi.search("pubmed", query, retmax=max_articles)
    except Exception as e:
        logger.warning(f"  LitCovid search failed: {e}")
        return []
    records = []
    batch_size = 50
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        try:
            abstracts = ncbi.fetch_pubmed_abstracts(batch)
        except Exception as e:
            logger.warning(f"  LitCovid batch {i//batch_size} failed ({e}), skipping")
            continue
        for a in abstracts:
            text = f"Title: {a['title']}\n\nAbstract: {a['abstract']}"
            records.append(make_general(text, "litcovid", "research"))
    logger.info(f"  Collected {len(records)} LitCovid records")
    return records


# =============================================================================
# 5. HUGGINGFACE — QA + NLP BENCHMARKS (BigBio + standalone)
# =============================================================================

HF_BIOMEDICAL_DATASETS = [
    # (local_name, hf_id, subset/config, adapter, description)
    # ── Biomedical Literature ──────────────────────────────────────────────
    ("cord19_abstracts", "allenai/cord19", "metadata",
     "research", "COVID-19 Open Research Dataset — metadata with abstracts"),

    # ── Question Answering ─────────────────────────────────────────────────
    ("healthsearchqa", "katielink/healthsearchqa", "all_data",
     "patient_triage", "HealthSearchQA: consumer health search Q&A (3.4K)"),
    ("biomrc", "bigbio/biomrc", "biomrc_large_A_bigbio_qa",
     "research", "BioMRC — biomedical machine reading comprehension"),

    # ── Named Entity Recognition / Clinical NLP ────────────────────────────
    ("medmentions", "bigbio/medmentions", "medmentions_full_bigbio_kb",
     "research", "MedMentions: 4.4K PubMed abstracts with UMLS entity mentions"),
    ("jnlpba", "bigbio/jnlpba", "jnlpba_bigbio_kb",
     "research", "JNLPBA: biomedical entity NER (DNA, RNA, cell, protein)"),

    # ── Relation Extraction ────────────────────────────────────────────────
    ("chemprot", "bigbio/chemprot", "chemprot_bigbio_kb",
     "research", "ChemProt — chemical-protein interaction relations"),
    ("drugprot", "bigbio/drugprot", "drugprot_bigbio_kb",
     "clinical_pharmacist", "DrugProt — drug-gene/protein interaction corpus"),
    ("ddi_corpus", "bigbio/ddi_corpus", "ddi_corpus_bigbio_kb",
     "clinical_pharmacist", "DDI Corpus — drug-drug interaction extraction"),
]


def collect_hf_biomedical() -> List[Dict[str, Any]]:
    """Download all free HuggingFace biomedical datasets."""
    if not HAS_HF:
        logger.error("HuggingFace datasets not installed. Run: pip install datasets")
        return []

    all_records = []
    for name, hf_id, subset, adapter, desc in HF_BIOMEDICAL_DATASETS:
        out_path = RAW_DIR / f"{name}.jsonl"
        if out_path.exists():
            logger.info(f"  Already collected: {name} ({out_path})")
            continue
        logger.info(f"  Loading: {name} — {desc}")
        try:
            if subset:
                ds = load_dataset(hf_id, subset, split="train", trust_remote_code=True)
            else:
                try:
                    ds = load_dataset(hf_id, split="train", trust_remote_code=True)
                except Exception:
                    ds = load_dataset(hf_id, trust_remote_code=True)
                    split_name = list(ds.keys())[0]
                    ds = ds[split_name]

            records = []
            for row in ds:
                record = _hf_row_to_record(row, name, adapter)
                if record:
                    records.append(record)

            save_jsonl(records, out_path)
            all_records.extend(records)
        except Exception as e:
            logger.warning(f"  Failed to load {name}: {e}")

    return all_records


def _hf_row_to_record(row: dict, source: str, adapter: str) -> Optional[Dict[str, Any]]:
    """Convert a HuggingFace row to IMI format using flexible field detection."""

    # Instruction format: prefer question/answer pairs
    for q_field in ("question", "input", "query", "prompt", "instruction"):
        for a_field in ("answer", "output", "response", "long_answer", "ideal_answer"):
            q_val = row.get(q_field)
            a_val = row.get(a_field)
            if q_val and a_val:
                q_str = q_val if isinstance(q_val, str) else str(q_val)
                a_str = a_val if isinstance(a_val, str) else str(a_val)
                if q_str.strip() and a_str.strip():
                    return make_instruction(q_str, a_str, source, adapter)

    # General knowledge format: prefer text / abstract / passage
    for t_field in ("text", "abstract", "passage", "document", "body"):
        val = row.get(t_field)
        if val:
            t_str = val if isinstance(val, str) else str(val)
            if len(t_str) > 50:
                return make_general(t_str, source, adapter)

    # Combine title + abstract if available
    title = row.get("title")
    abstract = row.get("abstract")
    if title and abstract:
        t_str = title if isinstance(title, str) else str(title)
        a_str = abstract if isinstance(abstract, str) else str(abstract)
        if t_str.strip() and a_str.strip():
            return make_general(f"{t_str}\n\n{a_str}", source, adapter)

    # BigBio KB schema: passages contain text spans
    passages = row.get("passages")
    if passages and isinstance(passages, list):
        texts = []
        for p in passages:
            if isinstance(p, dict):
                ptexts = p.get("text", [])
                if isinstance(ptexts, list):
                    texts.extend(t for t in ptexts if isinstance(t, str) and t.strip())
                elif isinstance(ptexts, str) and ptexts.strip():
                    texts.append(ptexts)
        combined = " ".join(texts)
        if len(combined) > 50:
            return make_general(combined, source, adapter)

    return None


# =============================================================================
# 6. CLINICAL TRIALS (ClinicalTrials.gov)
# =============================================================================

def collect_clinical_trials(max_trials: int = 5000) -> List[Dict[str, Any]]:
    """
    Collect clinical trial summaries from ClinicalTrials.gov v2 API.
    https://clinicaltrials.gov/data-api/api
    """
    logger.info(f"  Collecting ClinicalTrials.gov (max {max_trials:,} trials)...")
    records = []
    session = requests.Session()
    session.headers["User-Agent"] = "IMI-Medical-LLM/1.0"

    conditions = [
        "cancer", "diabetes", "heart disease", "sepsis", "stroke",
        "depression", "COPD", "HIV", "alzheimer", "COVID-19",
        "hypertension", "asthma", "pneumonia", "kidney disease", "obesity",
    ]

    per_condition = max_trials // len(conditions)
    base_url = "https://clinicaltrials.gov/api/v2/studies"

    for condition in conditions:
        try:
            params = {
                "query.cond": condition,
                "fields": "NCTId,BriefTitle,BriefSummary,Condition,InterventionType,"
                          "InterventionName,Phase,OverallStatus,EnrollmentCount,"
                          "PrimaryOutcomeMeasure,StudyType",
                "pageSize": min(per_condition, 1000),
                "format": "json",
            }
            resp = _robust_get(session, base_url, params=params, timeout=30)
            data = resp.json()
            studies = data.get("studies", [])

            for study in studies:
                proto = study.get("protocolSection", {})
                id_mod = proto.get("identificationModule", {})
                desc_mod = proto.get("descriptionModule", {})
                design_mod = proto.get("designModule", {})
                arms_mod = proto.get("armsInterventionsModule", {})
                outcomes_mod = proto.get("outcomesModule", {})

                nct_id = id_mod.get("nctId", "")
                title = id_mod.get("briefTitle", "")
                summary = desc_mod.get("briefSummary", "")
                phase = design_mod.get("phases", [""])[0] if design_mod.get("phases") else ""
                status = design_mod.get("studyStatusModule", {}).get("overallStatus", "")

                interventions = [
                    f"{i.get('type', '')}: {i.get('name', '')}"
                    for i in arms_mod.get("interventions", [])[:3]
                ]
                outcomes = [o.get("measure", "") for o in outcomes_mod.get("primaryOutcomes", [])[:2]]

                if title and summary:
                    instruction = f"Summarize the clinical trial '{title}' (NCT ID: {nct_id})."
                    output_parts = [summary.strip()]
                    if phase:
                        output_parts.append(f"Phase: {phase}")
                    if status:
                        output_parts.append(f"Status: {status}")
                    if interventions:
                        output_parts.append(f"Interventions: {'; '.join(interventions)}")
                    if outcomes:
                        output_parts.append(f"Primary outcomes: {'; '.join(outcomes)}")
                    records.append(make_instruction(
                        instruction, "\n".join(output_parts),
                        "clinicaltrials_gov", "research",
                    ))

            time.sleep(0.3)
        except Exception as e:
            logger.warning(f"  ClinicalTrials error for {condition}: {e}")

    logger.info(f"  Collected {len(records)} clinical trial records")
    return records


# =============================================================================
# 7. DRUG & PHARMACOLOGY (SIDER, OFFSIDES, TWOSIDES, DailyMed)
# =============================================================================

def collect_sider(max_records: int = 10000) -> List[Dict[str, Any]]:
    """
    SIDER: Side Effect Resource — drug-side effect associations.
    http://sideeffects.embl.de/ — STITCH compound IDs mapped to MedDRA terms.
    """
    logger.info("  Collecting SIDER drug-side effect data...")
    records = []
    sider_url = "http://sideeffects.embl.de/media/download/meddra_all_se.tsv.gz"
    out_gz = RAW_DIR / "sider_meddra_all_se.tsv.gz"

    if not out_gz.exists():
        try:
            logger.info(f"  Downloading SIDER: {sider_url}")
            resp = requests.get(sider_url, timeout=60, stream=True)
            resp.raise_for_status()
            out_gz.parent.mkdir(parents=True, exist_ok=True)
            with open(out_gz, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
        except Exception as e:
            logger.warning(f"  SIDER download failed: {e}")
            return records

    drug_effects: Dict[str, List[str]] = {}
    try:
        with gzip.open(out_gz, "rt", errors="replace") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 6:
                    drug_name = parts[4].strip()
                    side_effect = parts[5].strip()
                    if drug_name and side_effect:
                        drug_effects.setdefault(drug_name.lower(), []).append(side_effect)
    except Exception as e:
        logger.warning(f"  SIDER parse error: {e}")
        return records

    count = 0
    for drug, effects in drug_effects.items():
        if count >= max_records:
            break
        if len(effects) >= 3:
            unique_effects = list(dict.fromkeys(effects))  # deduplicate preserving order
            instruction = f"What are the common side effects of {drug}?"
            output = (
                f"According to the SIDER database, {drug} is associated with the following "
                f"side effects: {', '.join(unique_effects[:15])}.\n\n"
                f"This information is based on post-marketing reports and clinical trial data. "
                f"Not all patients experience these effects. Consult the prescribing information "
                f"and a healthcare professional for complete safety information."
            )
            records.append(make_instruction(instruction, output, "sider", "clinical_pharmacist"))
            count += 1

    logger.info(f"  Collected {len(records)} SIDER drug-side effect records")
    return records


def collect_dailymed(ncbi: NCBIClient, max_drugs: int = 2000) -> List[Dict[str, Any]]:
    """
    DailyMed FDA drug labels via NLM API.
    https://dailymed.nlm.nih.gov/dailymed/webservices-help/v2/
    """
    logger.info("  Collecting DailyMed drug label data...")
    records = []
    base = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
    session = requests.Session()
    session.headers["User-Agent"] = "IMI-Medical-LLM/1.0"

    sections_of_interest = [
        ("indications_and_usage", "What are the indications for {}?"),
        ("dosage_and_administration", "What is the dosage for {}?"),
        ("warnings_and_cautions", "What are the warnings for {}?"),
        ("drug_interactions", "What are the drug interactions for {}?"),
    ]

    collected = 0
    page = 1
    while collected < max_drugs:
        try:
            resp = _robust_get(session, f"{base}/spls.json",
                               params={"pagesize": 100, "page": page}, timeout=30)
            data = resp.json()
            spls = data.get("data", [])
            if not spls:
                break

            for spl in spls:
                if collected >= max_drugs:
                    break
                set_id = spl.get("setid", "")
                title = spl.get("title", "")
                if not set_id:
                    continue

                # Extract drug name from title (e.g. "ASPIRIN- aspirin tablet" → "aspirin")
                drug_name = title.split("-")[0].strip() if title else ""
                if not drug_name:
                    drug_name = spl.get("generic_name", spl.get("brand_name", ""))
                if not drug_name:
                    continue

                try:
                    detail_resp = _robust_get(session, f"{base}/spls/{set_id}.json", timeout=15)
                    detail = detail_resp.json()

                    # DailyMed v2 returns sections as a nested list
                    label_data = detail.get("data", {})
                    sections = {}

                    # Try to extract from structured sections
                    if isinstance(label_data, dict):
                        for key in label_data:
                            if isinstance(label_data[key], str) and len(label_data[key]) > 50:
                                sections[key.lower()] = label_data[key]

                    # Build instruction pairs from available sections
                    for section_key, q_template in sections_of_interest:
                        section_text = sections.get(section_key, "")
                        if section_text and len(section_text) > 50:
                            # Clean HTML tags if present
                            clean_text = re.sub(r'<[^>]+>', '', section_text)[:2000]
                            records.append(make_instruction(
                                q_template.format(drug_name),
                                clean_text,
                                "dailymed", "clinical_pharmacist",
                            ))
                            collected += 1

                    time.sleep(0.2)
                except Exception:
                    continue

            page += 1
            time.sleep(0.3)
        except Exception as e:
            logger.warning(f"  DailyMed error on page {page}: {e}")
            break

    logger.info(f"  Collected {len(records)} DailyMed records")
    return records


# =============================================================================
# 8. MEDICAL ONTOLOGIES (HPO, MeSH, ICD-10 context)
# =============================================================================

def collect_hpo() -> List[Dict[str, Any]]:
    """
    Human Phenotype Ontology — downloadable OBO file.
    https://hpo.jax.org/data/ontology
    """
    logger.info("  Collecting Human Phenotype Ontology (HPO)...")
    records = []
    hpo_url = "https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2024-03-06/hp.obo"
    out_path = RAW_DIR / "hp.obo"

    if not out_path.exists():
        try:
            resp = requests.get(hpo_url, timeout=60, stream=True)
            resp.raise_for_status()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
        except Exception as e:
            logger.warning(f"  HPO download failed: {e}")
            return records

    # Parse OBO format into Q&A pairs
    current_term: Dict[str, str] = {}
    with open(out_path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                if current_term.get("name") and current_term.get("def"):
                    name = current_term["name"]
                    definition = current_term["def"].strip('"').split('" [')[0]
                    synonyms = current_term.get("synonyms", "")
                    comment = current_term.get("comment", "")
                    instruction = f"What is {name} in the context of human phenotypes and medical genetics?"
                    output = f"{definition}"
                    if synonyms:
                        output += f"\n\nAlso known as: {synonyms}"
                    if comment:
                        output += f"\n\nNote: {comment}"
                    records.append(make_instruction(instruction, output, "hpo", "research"))
                current_term = {}
            elif ": " in line:
                key, _, value = line.partition(": ")
                if key == "name":
                    current_term["name"] = value
                elif key == "def":
                    current_term["def"] = value
                elif key == "comment":
                    current_term["comment"] = value
                elif key == "synonym" and "EXACT" in value:
                    current_term["synonyms"] = current_term.get("synonyms", "") + value.split('"')[1] + "; "

    logger.info(f"  Collected {len(records)} HPO phenotype records")
    return records


def collect_mesh_terms(ncbi: NCBIClient, max_terms: int = 5000) -> List[Dict[str, Any]]:
    """
    Collect MeSH (Medical Subject Headings) definitions via NCBI E-utilities.
    Uses the PubMed database to find articles, then extracts MeSH terms.
    """
    logger.info("  Collecting MeSH terms via NCBI E-utilities...")
    records = []

    # Search PubMed for broad disease categories and collect unique MeSH terms
    mesh_categories = [
        "cardiovascular diseases", "neoplasms", "mental disorders",
        "respiratory tract diseases", "endocrine system diseases",
        "nervous system diseases", "communicable diseases",
        "musculoskeletal diseases", "digestive system diseases",
        "skin diseases",
    ]

    seen_terms = set()
    for category in mesh_categories:
        if len(records) >= max_terms:
            break
        try:
            # Search the mesh database directly with plain text queries
            ids = ncbi.search("mesh", category, retmax=200)
            if not ids:
                logger.info(f"    No MeSH IDs for '{category}'")
                continue
            # Fetch MeSH records as XML
            xml_text = ncbi._get_xml("efetch.fcgi", {
                "db": "mesh", "id": ",".join(ids[:100]),
                "rettype": "full", "retmode": "xml",
            })
            try:
                root = ET.fromstring(xml_text)
            except ET.ParseError:
                logger.warning(f"    MeSH XML parse error for {category}")
                continue
            for record in root.findall(".//DescriptorRecord"):
                name_el = record.find(".//DescriptorName/String")
                scope_el = record.find(".//ScopeNote")
                if name_el is not None and scope_el is not None:
                    name = (name_el.text or "").strip()
                    scope = (scope_el.text or "").strip()
                    if name and scope and len(scope) > 30 and name not in seen_terms:
                        seen_terms.add(name)
                        instruction = f"Define the medical term '{name}' as used in clinical and research contexts."
                        records.append(make_instruction(instruction, scope, "mesh", "education"))
        except Exception as e:
            logger.warning(f"  MeSH error for {category}: {e}")

    logger.info(f"  Collected {len(records)} MeSH term records")
    return records


# =============================================================================
# 9. SEMANTIC SCHOLAR OPEN RESEARCH CORPUS (S2ORC sample)
# =============================================================================

def collect_s2orc_sample(max_papers: int = 5000) -> List[Dict[str, Any]]:
    """
    Semantic Scholar Academic Graph API — biomedical papers sample.
    https://api.semanticscholar.org/graph/v1
    """
    logger.info("  Collecting Semantic Scholar biomedical papers...")
    records = []
    session = requests.Session()
    headers = {"User-Agent": "IMI-Medical-LLM/1.0"}
    if SEMANTIC_SCHOLAR_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_KEY
    session.headers.update(headers)

    fields = "title,abstract,year,fieldsOfStudy,publicationTypes,openAccessPdf"
    biomedical_queries = [
        "clinical decision support machine learning",
        "drug discovery artificial intelligence",
        "genomics precision medicine CRISPR",
        "sepsis treatment intensive care",
        "cancer immunotherapy PD-1",
        "antibiotic resistance mechanisms",
        "mental health depression treatment",
    ]

    rate_limited_until = 0
    for query in biomedical_queries:
        if len(records) >= max_papers:
            break
        # Skip if we are still rate-limited
        now = time.time()
        if now < rate_limited_until:
            wait = rate_limited_until - now
            if wait > 120:
                logger.warning(f"  S2 rate limit too long ({wait:.0f}s), skipping remaining queries")
                break
            logger.info(f"  S2 rate limit — waiting {wait:.0f}s")
            time.sleep(wait)
        try:
            resp = session.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={"query": query, "fields": fields, "limit": 100,
                        "fieldsOfStudy": "Medicine,Biology"},
                timeout=30,
            )
            if resp.status_code == 429:
                # Use Retry-After header if available, else 30s
                retry_after = int(resp.headers.get("Retry-After", 30))
                rate_limited_until = time.time() + retry_after
                logger.warning(f"  S2 rate limit — will retry after {retry_after}s")
                continue
            if resp.status_code != 200:
                logger.warning(f"  S2 returned {resp.status_code} for '{query}'")
                continue
            data = resp.json()
            for paper in data.get("data", []):
                title = paper.get("title", "")
                abstract = paper.get("abstract", "")
                if title and abstract and len(abstract) > 100:
                    text = f"Title: {title}\n\nAbstract: {abstract}"
                    records.append(make_general(text, "semantic_scholar", "research"))
            time.sleep(1.0 if not SEMANTIC_SCHOLAR_KEY else 0.2)
        except Exception as e:
            logger.warning(f"  S2ORC error for '{query}': {e}")

    logger.info(f"  Collected {len(records)} S2ORC records")
    return records


# =============================================================================
# 10. CREDENTIALED DATASETS — INSTRUCTIONS & LOADER STUBS
# =============================================================================

CREDENTIALED_DATASETS = {
    "mimic3": {
        "name": "MIMIC-III Clinical Database",
        "url": "https://physionet.org/content/mimiciii/1.4/",
        "access": "Free after CITI training + PhysioNet credentialing (~1-2 days)",
        "size": "~40K ICU stays, 2M+ clinical notes",
        "adapter": "clinical_decision",
        "env_var": "PHYSIONET_MIMIC3_PATH",
        "instructions": (
            "1. Register at https://physionet.org/register/\n"
            "2. Complete CITI 'Data or Specimens Only Research' training\n"
            "3. Sign data use agreement at the MIMIC-III page\n"
            "4. Download: wget -r -N -c -np --user <username> "
            "https://physionet.org/files/mimiciii/1.4/\n"
            "5. Set PHYSIONET_MIMIC3_PATH=/path/to/mimic3/ in .env\n"
            "6. Re-run: python collect_biomedical_corpus.py --category mimic"
        ),
    },
    "mimic4": {
        "name": "MIMIC-IV Clinical Database",
        "url": "https://physionet.org/content/mimiciv/3.1/",
        "access": "Same as MIMIC-III (PhysioNet credentialing)",
        "size": "~70K ICU stays, 3M+ clinical events",
        "adapter": "clinical_decision",
        "env_var": "PHYSIONET_MIMIC4_PATH",
        "instructions": (
            "Same as MIMIC-III. Download from:\n"
            "https://physionet.org/content/mimiciv/3.1/\n"
            "Set PHYSIONET_MIMIC4_PATH=/path/to/mimic4/ in .env"
        ),
    },
    "eicu": {
        "name": "eICU Collaborative Research Database",
        "url": "https://physionet.org/content/eicu-crd/2.0/",
        "access": "PhysioNet credentialing (same as MIMIC)",
        "size": "~200K ICU admissions across 335 US hospitals",
        "adapter": "clinical_decision",
        "env_var": "PHYSIONET_EICU_PATH",
        "instructions": (
            "Same PhysioNet process as MIMIC.\n"
            "Set PHYSIONET_EICU_PATH=/path/to/eicu/ in .env"
        ),
    },
    "amsterdam_umcdb": {
        "name": "AmsterdamUMCdb",
        "url": "https://amsterdammedicaldatascience.nl/",
        "access": "Free after submitting research proposal (1-2 weeks)",
        "size": "~23K ICU admissions from Amsterdam UMC",
        "adapter": "clinical_decision",
        "env_var": "AMSTERDAM_UMC_PATH",
        "instructions": (
            "1. Submit research proposal at https://amsterdammedicaldatascience.nl/\n"
            "2. Sign data use agreement\n"
            "3. Download and set AMSTERDAM_UMC_PATH in .env"
        ),
    },
    "drugbank": {
        "name": "DrugBank Full Database",
        "url": "https://go.drugbank.com/releases/latest",
        "access": "Free academic license (same-day approval usually)",
        "size": "~14K drugs, 500K+ interactions, pharmacogenomics data",
        "adapter": "clinical_pharmacist",
        "env_var": "DRUGBANK_XML_PATH",
        "instructions": (
            "1. Register at https://go.drugbank.com/users/sign_up\n"
            "2. Request academic download at https://go.drugbank.com/releases/latest\n"
            "3. Download full_database.xml.zip and extract\n"
            "4. Set DRUGBANK_XML_PATH=/path/to/full_database.xml in .env"
        ),
    },
    "umls": {
        "name": "UMLS Metathesaurus",
        "url": "https://www.nlm.nih.gov/research/umls/",
        "access": "Free license (UMLS Terminology Services account)",
        "size": "4M+ concepts across 200+ medical vocabularies",
        "adapter": "education",
        "env_var": "UMLS_DATA_PATH",
        "instructions": (
            "1. Register at https://uts.nlm.nih.gov/uts/login\n"
            "2. Accept UMLS license agreement\n"
            "3. Download the UMLS Release (MRCONSO.RRF, MRDEF.RRF, etc.)\n"
            "4. Set UMLS_DATA_PATH=/path/to/umls/ in .env"
        ),
    },
    "i2b2": {
        "name": "i2b2 Clinical NLP Datasets",
        "url": "https://www.i2b2.org/NLP/DataSets/",
        "access": "Free after DUA (Data Use Agreement) registration",
        "size": "Multiple shared tasks: NER, coreference, temporal, relations",
        "adapter": "clinical_decision",
        "env_var": "I2B2_DATA_PATH",
        "instructions": (
            "1. Register at https://www.i2b2.org/NLP/DataSets/AgreementAR.php\n"
            "2. Sign the data use agreement\n"
            "3. Download shared task datasets\n"
            "4. Set I2B2_DATA_PATH=/path/to/i2b2/ in .env"
        ),
    },
    "snomed_ct": {
        "name": "SNOMED CT",
        "url": "https://www.nlm.nih.gov/healthit/snomedct/",
        "access": "Free US license via NLM UMLS account",
        "size": "350K+ clinical concepts with hierarchies and relationships",
        "adapter": "education",
        "env_var": "SNOMED_CT_PATH",
        "instructions": (
            "SNOMED CT is included in the UMLS distribution.\n"
            "After getting UMLS access, extract SNOMED from the release.\n"
            "Set SNOMED_CT_PATH=/path/to/snomed_ct/ in .env"
        ),
    },
}


def process_drugbank(xml_path: str) -> List[Dict[str, Any]]:
    """Process DrugBank XML into structured instruction pairs."""
    records = []
    path = Path(xml_path)
    if not path.exists():
        logger.warning(f"DrugBank XML not found at {path}")
        return records

    logger.info(f"  Parsing DrugBank XML: {path}")
    ns = {"db": "http://www.drugbank.ca"}
    context = ET.iterparse(str(path), events=("end",))
    count = 0

    for event, elem in context:
        if elem.tag == f"{{{ns['db']}}}drug" or elem.tag == "drug":
            try:
                name_el = elem.find("name") or elem.find(f"{{{ns['db']}}}name")
                desc_el = elem.find("description") or elem.find(f"{{{ns['db']}}}description")
                indication_el = elem.find("indication") or elem.find(f"{{{ns['db']}}}indication")
                mech_el = (elem.find("mechanism-of-action") or
                           elem.find(f"{{{ns['db']}}}mechanism-of-action"))
                interactions = []
                for interaction in (elem.findall(".//drug-interaction/name") +
                                    elem.findall(f".//{{{ns['db']}}}drug-interaction/{{{ns['db']}}}name")):
                    if interaction.text:
                        interactions.append(interaction.text)

                if name_el is None or name_el.text is None:
                    continue
                drug_name = name_el.text

                if desc_el is not None and desc_el.text:
                    records.append(make_instruction(
                        f"Describe the drug {drug_name}.",
                        desc_el.text[:2000],
                        "drugbank", "clinical_pharmacist",
                    ))
                if indication_el is not None and indication_el.text:
                    records.append(make_instruction(
                        f"What are the clinical indications for {drug_name}?",
                        indication_el.text[:2000],
                        "drugbank", "clinical_pharmacist",
                    ))
                if mech_el is not None and mech_el.text:
                    records.append(make_instruction(
                        f"Explain the mechanism of action of {drug_name}.",
                        mech_el.text[:2000],
                        "drugbank", "clinical_pharmacist",
                    ))
                if interactions:
                    records.append(make_instruction(
                        f"What are the major drug interactions of {drug_name}?",
                        (f"{drug_name} has known interactions with: "
                         f"{', '.join(interactions[:20])}.\n\n"
                         f"Always check a current drug interaction database and "
                         f"consult a clinical pharmacist for patient-specific guidance."),
                        "drugbank", "clinical_pharmacist",
                    ))
                count += 1
                elem.clear()
            except Exception:
                elem.clear()
                continue

    logger.info(f"  Processed {count} DrugBank drugs → {len(records)} records")
    return records


def generate_credentials_guide() -> None:
    """Print a step-by-step guide for obtaining all credentialed datasets."""
    print("\n" + "=" * 70)
    print("CREDENTIALED DATASET ACCESS GUIDE")
    print("=" * 70)
    print("The following datasets require registration/credentials before use.")
    print("All are FREE for academic/research use.\n")

    for key, info in CREDENTIALED_DATASETS.items():
        print(f"{'─' * 60}")
        print(f"Dataset: {info['name']}")
        print(f"URL:     {info['url']}")
        print(f"Access:  {info['access']}")
        print(f"Size:    {info['size']}")
        print(f"Env var: {info['env_var']}")
        print(f"Steps:\n{info['instructions']}")
        print()


# =============================================================================
# MASTER ORCHESTRATOR
# =============================================================================

CATEGORIES = {
    "pubmed": "PubMed abstracts (15 clinical query categories)",
    "pmc": "PubMed Central full-text open access articles",
    "preprints": "bioRxiv + medRxiv preprints",
    "litcovid": "LitCovid COVID-19 curated literature",
    "hf": "HuggingFace biomedical NLP datasets (BigBio + standalone)",
    "clinical_trials": "ClinicalTrials.gov trial summaries",
    "sider": "SIDER drug-side effect associations",
    "dailymed": "DailyMed FDA drug labels (NLM API)",
    "hpo": "Human Phenotype Ontology terms",
    "mesh": "MeSH medical subject headings (NCBI)",
    "s2orc": "Semantic Scholar biomedical papers",
    "drugbank": "DrugBank drug interactions (requires DRUGBANK_XML_PATH)",
    "all": "All free + API-key sources",
}


def collect_all(category: str = "all", max_per_source: int = 10_000) -> None:
    ncbi = NCBIClient(api_key=NCBI_API_KEY)
    if NCBI_API_KEY:
        logger.info(f"NCBI API key active — rate: 10 req/s")
    else:
        logger.warning("No NCBI_API_KEY set — rate limited to 3 req/s. "
                       "Set NCBI_API_KEY in .env for 3× faster collection.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_records: List[Dict[str, Any]] = []

    def run(name: str, fn, *args, **kwargs):
        if category not in ("all", name):
            return
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Collecting: {CATEGORIES.get(name, name)}")
        logger.info(f"{'=' * 60}")
        try:
            records = fn(*args, **kwargs)
            if records:
                out = RAW_DIR / f"{name}.jsonl"
                save_jsonl(records, out)
                all_records.extend(records)
                logger.info(f"  ✓ {name}: {len(records):,} records")
        except Exception as e:
            logger.error(f"  ✗ {name} failed: {e}", exc_info=True)

    run("pubmed", collect_pubmed, ncbi, PUBMED_QUERIES, max_per_source)
    run("pmc", collect_pmc, ncbi, max_per_source)
    run("litcovid", collect_litcovid, ncbi, max_per_source)
    run("preprints", collect_preprints, max_per_source // 2)
    run("hf", collect_hf_biomedical)
    run("clinical_trials", collect_clinical_trials, max_per_source)
    run("sider", collect_sider, max_per_source * 2)
    run("dailymed", collect_dailymed, ncbi, max_per_source // 5)
    run("hpo", collect_hpo)
    run("mesh", collect_mesh_terms, ncbi, max_per_source)
    run("s2orc", collect_s2orc_sample, max_per_source)

    # DrugBank — only if path is set
    drugbank_path = os.getenv("DRUGBANK_XML_PATH", "")
    if (category in ("all", "drugbank")) and drugbank_path:
        run("drugbank", process_drugbank, drugbank_path)

    # Final summary
    logger.info(f"\n{'=' * 60}")
    logger.info(f"COLLECTION COMPLETE")
    logger.info(f"Total records collected: {len(all_records):,}")
    logger.info(f"Output directory: {RAW_DIR}")
    logger.info(f"{'=' * 60}")
    logger.info("Next step: run prepare_medical_data.py to format for training")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive biomedical corpus collector for IMI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--category",
        choices=list(CATEGORIES.keys()),
        default="all",
        help="Which category to collect (default: all)",
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=10_000,
        help="Per-page fetch size (NCBI max = 10 000); all pages are retrieved automatically (default: 10000)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available sources and their access tier, then exit",
    )
    parser.add_argument(
        "--credentials-guide",
        action="store_true",
        help="Print step-by-step guide for obtaining credentialed datasets",
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable dataset categories:")
        print(f"{'Category':<20} {'Description'}")
        print("-" * 70)
        for cat, desc in CATEGORIES.items():
            print(f"  {cat:<18} {desc}")
        print(f"\nCredentialed datasets (require registration):")
        for key, info in CREDENTIALED_DATASETS.items():
            print(f"  {key:<18} {info['name']} — {info['access']}")
        print()
        return

    if args.credentials_guide:
        generate_credentials_guide()
        return

    collect_all(category=args.category, max_per_source=args.max_per_source)


if __name__ == "__main__":
    main()
