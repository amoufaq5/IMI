"""
UMI PubMed Data Ingestion Pipeline
Fetches and indexes medical literature from PubMed for RAG
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET

import httpx
from tqdm import tqdm


@dataclass
class PubMedArticle:
    """Represents a PubMed article."""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    pub_date: str
    mesh_terms: List[str]
    keywords: List[str]
    doi: Optional[str] = None


class PubMedClient:
    """
    Client for PubMed E-utilities API.
    https://www.ncbi.nlm.nih.gov/books/NBK25500/
    """
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    def __init__(self, api_key: Optional[str] = None, email: Optional[str] = None):
        """
        Initialize PubMed client.
        
        Args:
            api_key: NCBI API key (increases rate limit from 3 to 10 req/sec)
            email: Contact email (required by NCBI)
        """
        self.api_key = api_key or os.environ.get("NCBI_API_KEY")
        self.email = email or os.environ.get("NCBI_EMAIL", "umi@example.com")
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def search(
        self,
        query: str,
        max_results: int = 100,
        sort: str = "relevance",
    ) -> List[str]:
        """
        Search PubMed and return PMIDs.
        
        Args:
            query: Search query (PubMed syntax)
            max_results: Maximum number of results
            sort: Sort order (relevance, pub_date)
        
        Returns:
            List of PMIDs
        """
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "sort": sort,
            "retmode": "json",
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email
        
        response = await self.client.get(f"{self.BASE_URL}/esearch.fcgi", params=params)
        response.raise_for_status()
        
        data = response.json()
        return data.get("esearchresult", {}).get("idlist", [])
    
    async def fetch_articles(self, pmids: List[str]) -> List[PubMedArticle]:
        """
        Fetch full article details for given PMIDs.
        
        Args:
            pmids: List of PubMed IDs
        
        Returns:
            List of PubMedArticle objects
        """
        if not pmids:
            return []
        
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email
        
        response = await self.client.get(f"{self.BASE_URL}/efetch.fcgi", params=params)
        response.raise_for_status()
        
        return self._parse_xml(response.text)
    
    def _parse_xml(self, xml_text: str) -> List[PubMedArticle]:
        """Parse PubMed XML response."""
        articles = []
        
        try:
            root = ET.fromstring(xml_text)
            
            for article_elem in root.findall(".//PubmedArticle"):
                try:
                    # PMID
                    pmid_elem = article_elem.find(".//PMID")
                    pmid = pmid_elem.text if pmid_elem is not None else ""
                    
                    # Title
                    title_elem = article_elem.find(".//ArticleTitle")
                    title = title_elem.text if title_elem is not None else ""
                    
                    # Abstract
                    abstract_parts = []
                    for abstract_elem in article_elem.findall(".//AbstractText"):
                        label = abstract_elem.get("Label", "")
                        text = abstract_elem.text or ""
                        if label:
                            abstract_parts.append(f"{label}: {text}")
                        else:
                            abstract_parts.append(text)
                    abstract = " ".join(abstract_parts)
                    
                    # Authors
                    authors = []
                    for author_elem in article_elem.findall(".//Author"):
                        last_name = author_elem.findtext("LastName", "")
                        fore_name = author_elem.findtext("ForeName", "")
                        if last_name:
                            authors.append(f"{last_name} {fore_name}".strip())
                    
                    # Journal
                    journal_elem = article_elem.find(".//Journal/Title")
                    journal = journal_elem.text if journal_elem is not None else ""
                    
                    # Publication date
                    pub_date_elem = article_elem.find(".//PubDate")
                    if pub_date_elem is not None:
                        year = pub_date_elem.findtext("Year", "")
                        month = pub_date_elem.findtext("Month", "")
                        pub_date = f"{year} {month}".strip()
                    else:
                        pub_date = ""
                    
                    # MeSH terms
                    mesh_terms = []
                    for mesh_elem in article_elem.findall(".//MeshHeading/DescriptorName"):
                        if mesh_elem.text:
                            mesh_terms.append(mesh_elem.text)
                    
                    # Keywords
                    keywords = []
                    for kw_elem in article_elem.findall(".//Keyword"):
                        if kw_elem.text:
                            keywords.append(kw_elem.text)
                    
                    # DOI
                    doi = None
                    for id_elem in article_elem.findall(".//ArticleId"):
                        if id_elem.get("IdType") == "doi":
                            doi = id_elem.text
                            break
                    
                    if pmid and (title or abstract):
                        articles.append(PubMedArticle(
                            pmid=pmid,
                            title=title,
                            abstract=abstract,
                            authors=authors,
                            journal=journal,
                            pub_date=pub_date,
                            mesh_terms=mesh_terms,
                            keywords=keywords,
                            doi=doi,
                        ))
                
                except Exception as e:
                    print(f"Error parsing article: {e}")
                    continue
        
        except ET.ParseError as e:
            print(f"XML parse error: {e}")
        
        return articles
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class PubMedIngestionPipeline:
    """
    Pipeline for ingesting PubMed articles into UMI knowledge base.
    """
    
    # Comprehensive medical topics to index - expanded for maximum coverage
    MEDICAL_TOPICS = [
        # Common chronic conditions
        "diabetes mellitus treatment",
        "diabetes type 1 management",
        "diabetes type 2 guidelines",
        "diabetic complications",
        "hypertension management",
        "resistant hypertension",
        "asthma guidelines",
        "severe asthma treatment",
        "COPD treatment",
        "COPD exacerbation",
        "heart failure therapy",
        "heart failure preserved ejection fraction",
        "atrial fibrillation management",
        "coronary artery disease",
        "chronic kidney disease",
        "end stage renal disease",
        "liver cirrhosis management",
        "hepatitis B treatment",
        "hepatitis C treatment",
        "HIV antiretroviral therapy",
        "rheumatoid arthritis treatment",
        "osteoarthritis management",
        "osteoporosis treatment",
        "inflammatory bowel disease",
        "Crohn disease treatment",
        "ulcerative colitis",
        "multiple sclerosis treatment",
        "Parkinson disease management",
        "Alzheimer disease treatment",
        "epilepsy management",
        
        # Mental health
        "depression treatment",
        "treatment resistant depression",
        "anxiety disorders",
        "generalized anxiety disorder",
        "panic disorder treatment",
        "bipolar disorder management",
        "schizophrenia treatment",
        "PTSD treatment",
        "OCD treatment",
        "ADHD management",
        "eating disorders treatment",
        "substance use disorder",
        "alcohol use disorder",
        "opioid use disorder treatment",
        
        # Pain management
        "chronic pain management",
        "neuropathic pain treatment",
        "fibromyalgia management",
        "migraine treatment",
        "cluster headache",
        "low back pain guidelines",
        
        # Emergency medicine
        "acute myocardial infarction",
        "STEMI treatment",
        "NSTEMI management",
        "stroke treatment",
        "ischemic stroke thrombolysis",
        "hemorrhagic stroke",
        "sepsis management",
        "septic shock treatment",
        "anaphylaxis treatment",
        "status epilepticus",
        "diabetic ketoacidosis",
        "acute respiratory distress syndrome",
        "pulmonary embolism treatment",
        "acute pancreatitis",
        "gastrointestinal bleeding",
        "trauma resuscitation",
        "burn management",
        "poisoning treatment",
        "overdose management",
        
        # Infectious diseases
        "pneumonia treatment",
        "community acquired pneumonia",
        "hospital acquired pneumonia",
        "urinary tract infection",
        "skin soft tissue infection",
        "meningitis treatment",
        "endocarditis treatment",
        "osteomyelitis management",
        "tuberculosis treatment",
        "malaria treatment",
        "dengue fever management",
        "COVID-19 treatment",
        "influenza treatment",
        "fungal infection treatment",
        "parasitic infection",
        
        # Oncology
        "breast cancer treatment",
        "lung cancer treatment",
        "colorectal cancer",
        "prostate cancer management",
        "leukemia treatment",
        "lymphoma treatment",
        "melanoma treatment",
        "pancreatic cancer",
        "ovarian cancer treatment",
        "cervical cancer",
        "cancer immunotherapy",
        "chemotherapy side effects",
        "palliative care cancer",
        
        # Pharmacology
        "drug interactions",
        "adverse drug reactions",
        "antibiotic resistance",
        "antimicrobial stewardship",
        "opioid prescribing",
        "polypharmacy elderly",
        "drug dosing renal impairment",
        "drug dosing hepatic impairment",
        "pharmacogenomics",
        "therapeutic drug monitoring",
        
        # Diagnostics
        "differential diagnosis",
        "clinical decision making",
        "diagnostic accuracy",
        "point of care testing",
        "laboratory test interpretation",
        "imaging guidelines",
        "ECG interpretation",
        "chest X-ray interpretation",
        
        # Primary care & prevention
        "preventive medicine",
        "vaccination guidelines",
        "adult immunization",
        "childhood vaccination",
        "cancer screening",
        "cardiovascular risk assessment",
        "health promotion",
        "lifestyle modification",
        "smoking cessation",
        "obesity management",
        "metabolic syndrome",
        
        # Pediatrics
        "pediatric fever management",
        "pediatric asthma",
        "pediatric diabetes",
        "neonatal care",
        "pediatric infectious disease",
        "childhood obesity",
        "developmental disorders",
        "pediatric emergency",
        
        # Obstetrics & Gynecology
        "pregnancy complications",
        "gestational diabetes",
        "preeclampsia management",
        "postpartum hemorrhage",
        "contraception guidelines",
        "menopause management",
        "polycystic ovary syndrome",
        "endometriosis treatment",
        
        # Geriatrics
        "geriatric assessment",
        "falls prevention elderly",
        "delirium management",
        "dementia care",
        "frailty syndrome",
        "end of life care",
        
        # Surgery & procedures
        "preoperative assessment",
        "postoperative care",
        "surgical site infection",
        "venous thromboembolism prophylaxis",
        "wound care management",
        
        # Specialty topics
        "thyroid disorders",
        "adrenal insufficiency",
        "pituitary disorders",
        "anemia management",
        "coagulation disorders",
        "autoimmune diseases",
        "allergic rhinitis",
        "sleep disorders",
        "chronic fatigue syndrome",
    ]
    
    def __init__(
        self,
        output_dir: str = "data/knowledge_base/pubmed",
        api_key: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = PubMedClient(api_key=api_key)
        self.articles: List[PubMedArticle] = []
    
    async def fetch_topic(self, topic: str, max_results: int = 200) -> List[PubMedArticle]:
        """Fetch articles for a specific topic with multiple query strategies."""
        all_articles = []
        seen_pmids = set()
        
        # Strategy 1: High-quality reviews, guidelines, meta-analyses (most recent)
        query1 = f'({topic}) AND (systematic review[pt] OR guideline[pt] OR meta-analysis[pt] OR practice guideline[pt]) AND english[la] AND ("last 10 years"[dp])'
        pmids1 = await self.client.search(query1, max_results=max_results // 2, sort="pub_date")
        
        # Strategy 2: Clinical trials and research articles
        query2 = f'({topic}) AND (clinical trial[pt] OR randomized controlled trial[pt]) AND english[la] AND ("last 5 years"[dp])'
        pmids2 = await self.client.search(query2, max_results=max_results // 3, sort="relevance")
        
        # Strategy 3: Recent high-impact articles (any type)
        query3 = f'({topic}) AND english[la] AND ("last 2 years"[dp])'
        pmids3 = await self.client.search(query3, max_results=max_results // 4, sort="relevance")
        
        # Combine and deduplicate PMIDs
        all_pmids = []
        for pmid in pmids1 + pmids2 + pmids3:
            if pmid not in seen_pmids:
                seen_pmids.add(pmid)
                all_pmids.append(pmid)
        
        if not all_pmids:
            return []
        
        # Fetch in batches to respect rate limits
        batch_size = 50  # Increased batch size for efficiency
        
        for i in range(0, len(all_pmids), batch_size):
            batch = all_pmids[i:i + batch_size]
            batch_articles = await self.client.fetch_articles(batch)
            all_articles.extend(batch_articles)
            
            # Rate limiting (with API key: 10 req/sec, without: 3 req/sec)
            await asyncio.sleep(0.2 if self.client.api_key else 0.4)
        
        return all_articles
    
    async def run(self, max_per_topic: int = 50) -> None:
        """Run the full ingestion pipeline."""
        print("=" * 60)
        print("UMI PubMed Ingestion Pipeline")
        print("=" * 60)
        
        all_articles = []
        
        for topic in tqdm(self.MEDICAL_TOPICS, desc="Fetching topics"):
            try:
                articles = await self.fetch_topic(topic, max_results=max_per_topic)
                all_articles.extend(articles)
                print(f"  {topic}: {len(articles)} articles")
            except Exception as e:
                print(f"  Error fetching {topic}: {e}")
            
            # Rate limiting between topics
            await asyncio.sleep(1)
        
        # Deduplicate by PMID
        seen_pmids = set()
        unique_articles = []
        for article in all_articles:
            if article.pmid not in seen_pmids:
                seen_pmids.add(article.pmid)
                unique_articles.append(article)
        
        self.articles = unique_articles
        print(f"\nTotal unique articles: {len(self.articles)}")
        
        # Save articles
        await self.save()
        
        # Close client
        await self.client.close()
    
    async def save(self) -> None:
        """Save articles to disk."""
        # Save as JSONL for RAG indexing
        output_file = self.output_dir / "articles.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for article in self.articles:
                doc = {
                    "id": f"pubmed_{article.pmid}",
                    "title": article.title,
                    "content": f"{article.title}\n\n{article.abstract}",
                    "metadata": {
                        "pmid": article.pmid,
                        "authors": article.authors,
                        "journal": article.journal,
                        "pub_date": article.pub_date,
                        "mesh_terms": article.mesh_terms,
                        "keywords": article.keywords,
                        "doi": article.doi,
                        "source": "PubMed",
                    },
                }
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        print(f"Saved to: {output_file}")
        
        # Save statistics
        stats = {
            "total_articles": len(self.articles),
            "ingestion_date": datetime.now().isoformat(),
            "topics": self.MEDICAL_TOPICS,
        }
        
        stats_file = self.output_dir / "stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
    
    async def index_to_rag(self) -> None:
        """Index articles to RAG vector database."""
        # Import here to avoid circular imports
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        
        from src.ai.rag_service import RAGService
        
        rag = RAGService()
        await rag.initialize_collections()
        
        # Prepare documents
        documents = []
        for article in self.articles:
            documents.append({
                "id": f"pubmed_{article.pmid}",
                "title": article.title,
                "content": f"{article.title}\n\n{article.abstract}",
                "metadata": {
                    "source": "PubMed",
                    "pmid": article.pmid,
                    "journal": article.journal,
                },
            })
        
        # Index in batches
        indexed = await rag.index_batch("medical_literature", documents)
        print(f"Indexed {indexed} articles to RAG")


async def main():
    """Run the PubMed ingestion pipeline with maximum data collection."""
    pipeline = PubMedIngestionPipeline()
    # Increased to 200 articles per topic for comprehensive coverage
    await pipeline.run(max_per_topic=200)
    
    # Optionally index to RAG
    # await pipeline.index_to_rag()


if __name__ == "__main__":
    asyncio.run(main())
