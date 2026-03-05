"""
Clinical Guidelines & QA/QC Data Source Collector — Open Sources Only

Downloads open-access clinical practice guidelines, quality standards,
and regulatory data for the regulatory_qa adapter.

NO credentials, API keys, or registration required. All sources are publicly accessible.

Sources (all open/verified):
  - CDC (Centers for Disease Control) — open data portal
  - CMS (Centers for Medicare & Medicaid Services) — provider data
  - OpenFDA API — no API key required for basic queries
  - USPSTF (US Preventive Services Task Force) — public API
  - AHRQ (Agency for Healthcare Research and Quality) — public data
  - NICE (National Institute for Health and Care Excellence) — public API
  - NLM RxNorm — public REST API
  - ICD-10 codes — GitHub public dataset
  - Health.gov — public API

Usage:
    pip install requests
    python scripts/data_collection/collect_guidelines.py              # download all
    python scripts/data_collection/collect_guidelines.py --category clinical   # specific category
    python scripts/data_collection/collect_guidelines.py --list       # list all sources
"""
import io
import csv
import json
import logging
import argparse
import requests
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw" / "guidelines"
PROCESSED_DIR = DATA_DIR / "processed" / "regulatory_qa"

HEADERS = {
    "User-Agent": "IMI-Medical-AI-Research/1.0 (academic research)",
    "Accept": "application/json, text/csv, text/plain",
}


# ============================================================================
# GUIDELINE & QA/QC DATA SOURCES
#
# Categories:
#   clinical       — Clinical practice guidelines (CPGs)
#   quality        — QA/QC metrics, hospital quality measures
#   drug_safety    — Drug safety, adverse events, labeling
#   terminology    — Medical coding (ICD-10, LOINC, CPT, SNOMED)
#   preventive     — Preventive care, screening recommendations
#   infection      — Infection control, antimicrobial stewardship
#
# Format: (name, url, file_format, category, est_size, description)
# ============================================================================

GUIDELINE_SOURCES = [

    # ====================== CLINICAL PRACTICE GUIDELINES ======================

    ("uspstf_recommendations",
     "https://data.uspstf.gov/api/3/action/datastore_search?resource_id=uspstf-recommendations&limit=5000",
     "api_json", "clinical", "300+", "USPSTF preventive service recommendations — A/B/C/D/I grades"),

    ("ahrq_clinical_guidelines",
     "https://data.ahrq.gov/api/3/action/package_list",
     "api_json", "clinical", "500+", "AHRQ clinical guideline metadata index"),

    ("nice_guidelines_index",
     "https://api.nice.org.uk/services/guidance/published?pagesize=1000",
     "api_json", "clinical", "2K+", "NICE UK clinical guideline index (all published)"),

    ("who_guidelines_register",
     "https://app.magicapp.org/api/v1/guidelines?organization=who&limit=500",
     "api_json", "clinical", "200+", "WHO clinical guidelines register"),

    # ====================== QUALITY MEASURES & QA/QC ======================

    ("cms_quality_measures",
     "https://data.cms.gov/provider-data/sites/default/files/resources/37e3c1486af47e7575bbaa714e2d36c3/Timely_and_Effective_Care-Hospital.csv",
     "csv", "quality", "70K", "CMS Timely & Effective Care hospital quality measures"),

    ("cms_patient_experience",
     "https://data.cms.gov/provider-data/sites/default/files/resources/bc515b9549426e9a3a40096b89131e2f/HCAHPS-Hospital.csv",
     "csv", "quality", "5K", "CMS HCAHPS patient experience survey results"),

    ("cms_readmission_death",
     "https://data.cms.gov/provider-data/sites/default/files/resources/4dc1e1457c065a76b3ad05051cef3e05/Complications_and_Deaths-Hospital.csv",
     "csv", "quality", "20K", "CMS hospital complications, readmissions, and deaths"),

    ("cms_hospital_info",
     "https://data.cms.gov/provider-data/sites/default/files/resources/092fb35e1bb884a0584c47e0237aca67/Hospital_General_Information.csv",
     "csv", "quality", "5K", "CMS Hospital General Information — ratings and type"),

    ("cms_safety_measures",
     "https://data.cms.gov/provider-data/sites/default/files/resources/d0dcbbf5e0cd36f8dd9a2c12b0610024/Healthcare_Associated_Infections-Hospital.csv",
     "csv", "quality", "50K", "CMS Healthcare-Associated Infections hospital data"),

    ("cms_payment_value",
     "https://data.cms.gov/provider-data/sites/default/files/resources/d8d4c5ece25ed1e5c3db3f37b2e9e5ae/Payment_and_Value_of_Care-Hospital.csv",
     "csv", "quality", "15K", "CMS Payment and Value of Care hospital measures"),

    ("cms_outpatient_quality",
     "https://data.cms.gov/provider-data/sites/default/files/resources/0c0d0fa356eb99036c0a14d7c1e6c4b3/Outpatient_Imaging_Efficiency-Hospital.csv",
     "csv", "quality", "5K", "CMS Outpatient Imaging Efficiency measures"),

    # ====================== DRUG SAFETY & ADVERSE EVENTS ======================

    ("openfda_drug_events",
     "https://api.fda.gov/drug/event.json?limit=1000",
     "api_json", "drug_safety", "1K", "OpenFDA drug adverse event reports (sample batch)"),

    ("openfda_drug_recalls",
     "https://api.fda.gov/drug/enforcement.json?limit=1000",
     "api_json", "drug_safety", "1K", "OpenFDA drug recall enforcement reports"),

    ("openfda_drug_labeling",
     "https://api.fda.gov/drug/label.json?limit=1000",
     "api_json", "drug_safety", "1K", "OpenFDA drug labeling (indications, warnings, dosing)"),

    ("fda_medication_guides",
     "https://api.fda.gov/drug/label.json?search=openfda.product_type:\"HUMAN+PRESCRIPTION+DRUG\"&limit=1000",
     "api_json", "drug_safety", "1K", "FDA prescription drug medication guides"),

    ("fda_otc_monographs",
     "https://api.fda.gov/drug/label.json?search=openfda.product_type:\"HUMAN+OTC+DRUG\"&limit=1000",
     "api_json", "drug_safety", "1K", "FDA OTC drug monographs"),

    ("openfda_device_events",
     "https://api.fda.gov/device/event.json?limit=1000",
     "api_json", "drug_safety", "1K", "OpenFDA medical device adverse event reports"),

    ("openfda_device_recalls",
     "https://api.fda.gov/device/recall.json?limit=1000",
     "api_json", "drug_safety", "1K", "OpenFDA medical device recalls"),

    # ====================== MEDICAL TERMINOLOGY & CODING ======================

    ("icd10_codes_full",
     "https://raw.githubusercontent.com/kamillamagna/ICD-10-CSV/master/codes.csv",
     "csv", "terminology", "72K", "Complete ICD-10-CM diagnosis code descriptions"),

    ("icd10_categories",
     "https://raw.githubusercontent.com/kamillamagna/ICD-10-CSV/master/categories.csv",
     "csv", "terminology", "2K", "ICD-10 category groupings"),

    ("rxnorm_concepts",
     "https://rxnav.nlm.nih.gov/REST/allconcepts.json?tty=SBD",
     "api_json", "terminology", "50K", "RxNorm branded drug concepts via NLM API"),

    ("rxnorm_drug_classes",
     "https://rxnav.nlm.nih.gov/REST/rxclass/allClasses.json?classTypes=ATC1-4",
     "api_json", "terminology", "5K", "RxNorm drug classification (ATC codes)"),

    ("ndc_product_list",
     "https://download.open.fda.gov/drug/ndc/drug-ndc-0001-of-0001.json.zip",
     "zip_json", "terminology", "300K", "FDA National Drug Code product listing"),

    # ====================== PREVENTIVE CARE & SCREENING ======================

    ("cdc_immunization_schedule",
     "https://data.cdc.gov/api/views/fhky-rtsk/rows.csv?accessType=DOWNLOAD",
     "csv", "preventive", "500", "CDC recommended immunization schedule data"),

    ("cdc_chronic_disease_indicators",
     "https://data.cdc.gov/api/views/g4ie-h725/rows.csv?accessType=DOWNLOAD",
     "csv", "preventive", "400K", "CDC Chronic Disease Indicators by state"),

    ("cdc_brfss_prevalence",
     "https://data.cdc.gov/api/views/dttw-5yxu/rows.csv?accessType=DOWNLOAD",
     "csv", "preventive", "100K", "CDC BRFSS health risk behavior prevalence data"),

    ("healthgov_topics",
     "https://health.gov/myhealthfinder/api/v3/topicsearch.json?lang=en",
     "api_json", "preventive", "200", "Health.gov MyHealthfinder preventive health topics"),

    # ====================== INFECTION CONTROL & ANTIMICROBIAL ======================

    ("cdc_antibiotic_resistance",
     "https://data.cdc.gov/api/views/ffhk-4ku5/rows.csv?accessType=DOWNLOAD",
     "csv", "infection", "5K", "CDC Antibiotic Resistance Threats data"),

    ("cdc_hai_data",
     "https://data.cdc.gov/api/views/77hc-7uto/rows.csv?accessType=DOWNLOAD",
     "csv", "infection", "10K", "CDC Healthcare-Associated Infections data"),

    ("who_amr_surveillance",
     "https://extranet.who.int/gho/api/gho/GLASS_AMR.csv?$filter=TimeDim%20eq%202022",
     "csv", "infection", "5K", "WHO GLASS antimicrobial resistance surveillance 2022"),
]


# ============================================================================
# DOWNLOAD LOGIC
# ============================================================================

def download_source(name, url, fmt, dest):
    """Download a guideline data source"""
    if fmt == "skip":
        logger.info(f"  [SKIP] {name} — requires manual registration/download")
        return None

    try:
        resp = requests.get(url, headers=HEADERS, timeout=120)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.warning(f"  [WARN] {name} — HTTP error, trying without headers: {e}")
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
        except Exception as e2:
            raise RuntimeError(f"Cannot download {url}: {e2}")

    content = resp.content
    data = []

    if fmt == "csv":
        text = content.decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        data = [dict(row) for row in reader]

    elif fmt == "api_json":
        raw = json.loads(content)
        # Handle common API wrapper formats
        if isinstance(raw, list):
            data = raw
        elif isinstance(raw, dict):
            # OpenFDA format
            if "results" in raw:
                data = raw["results"]
            # CKAN/AHRQ format
            elif "result" in raw:
                result = raw["result"]
                if isinstance(result, list):
                    data = result
                elif isinstance(result, dict) and "records" in result:
                    data = result["records"]
                else:
                    data = [result]
            # Health.gov format
            elif "Result" in raw:
                result = raw["Result"]
                if "Resources" in result:
                    resources = result["Resources"]
                    if "Resource" in resources:
                        data = resources["Resource"]
                    else:
                        data = [resources]
                elif "Topics" in result:
                    data = result["Topics"]
                else:
                    data = [result]
            # NICE format
            elif "data" in raw:
                data = raw["data"]
            elif "guidelines" in raw:
                data = raw["guidelines"]
            else:
                data = [raw]

    elif fmt == "zip_json":
        import zipfile
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for name_in_zip in zf.namelist():
                if name_in_zip.endswith(".json"):
                    with zf.open(name_in_zip) as f:
                        raw = json.load(f)
                        if isinstance(raw, dict) and "results" in raw:
                            data = raw["results"]
                        elif isinstance(raw, list):
                            data = raw
                        else:
                            data = [raw]
                    break

    if data:
        with open(dest, "w") as f:
            json.dump(data, f, default=str)
    return data


def to_qa_format(item, source_name, category):
    """Convert guideline data to instruction/output format for training"""

    instruction = ""
    output = ""

    # CMS quality measures format
    if "Measure Name" in item and "Score" in item:
        measure = item.get("Measure Name", "")
        score = item.get("Score", "")
        hospital = item.get("Facility Name", item.get("Hospital Name", ""))
        state = item.get("State", "")
        instruction = f"What is the quality measure '{measure}' for {hospital}, {state}?"
        output = f"Score: {score}"
        footnote = item.get("Footnote", "")
        if footnote:
            output += f"\nNote: {footnote}"

    # OpenFDA drug labeling format
    elif "indications_and_usage" in item:
        brand = ""
        if "openfda" in item and isinstance(item["openfda"], dict):
            brands = item["openfda"].get("brand_name", [])
            brand = brands[0] if brands else ""
        indications = item.get("indications_and_usage", [""])[0] if isinstance(item.get("indications_and_usage"), list) else item.get("indications_and_usage", "")
        warnings = item.get("warnings", [""])[0] if isinstance(item.get("warnings"), list) else item.get("warnings", "")
        dosage = item.get("dosage_and_administration", [""])[0] if isinstance(item.get("dosage_and_administration"), list) else item.get("dosage_and_administration", "")

        if brand and indications:
            instruction = f"What are the indications, warnings, and dosing for {brand}?"
            parts = []
            if indications:
                parts.append(f"Indications: {indications}")
            if warnings:
                parts.append(f"Warnings: {warnings}")
            if dosage:
                parts.append(f"Dosage: {dosage}")
            output = "\n\n".join(parts)

    # OpenFDA adverse event format
    elif "patient" in item and "drug" in item:
        drugs = item.get("drug", [])
        reactions = item.get("patient", {}).get("reaction", [])
        if drugs and reactions:
            drug_names = [d.get("medicinalproduct", "") for d in drugs if d.get("medicinalproduct")]
            reaction_terms = [r.get("reactionmeddrapt", "") for r in reactions if r.get("reactionmeddrapt")]
            if drug_names and reaction_terms:
                instruction = f"What adverse events have been reported for {', '.join(drug_names[:3])}?"
                output = f"Reported reactions: {', '.join(reaction_terms)}"
                serious = item.get("serious", "")
                if serious == "1":
                    output += "\nClassification: Serious adverse event"

    # OpenFDA recall/enforcement format
    elif "reason_for_recall" in item:
        product = item.get("product_description", "")
        reason = item.get("reason_for_recall", "")
        classification = item.get("classification", "")
        instruction = f"What is the recall information for {product[:100]}?"
        output = f"Reason: {reason}\nClassification: {classification}"
        status = item.get("status", "")
        if status:
            output += f"\nStatus: {status}"

    # ICD-10 codes format
    elif "code" in item and "description" in item:
        instruction = f"What does ICD-10 code {item['code']} mean?"
        output = item["description"]

    # CDC data format
    elif "Topic" in item and "DataValue" in item:
        topic = item.get("Topic", "")
        question = item.get("Question", "")
        value = item.get("DataValue", "")
        location = item.get("LocationDesc", "")
        instruction = f"What is the prevalence of {topic} ({question}) in {location}?"
        output = f"Data value: {value}"
        unit = item.get("DataValueUnit", "")
        if unit:
            output += f" {unit}"

    # Health.gov preventive topics format
    elif "Title" in item and "Sections" in item:
        title = item.get("Title", "")
        sections = item.get("Sections", [])
        instruction = f"What is the preventive health recommendation for: {title}?"
        parts = []
        for sec in sections:
            if isinstance(sec, dict):
                sec_title = sec.get("Title", "")
                sec_content = sec.get("Content", sec.get("Description", ""))
                if sec_content:
                    parts.append(f"{sec_title}: {sec_content}" if sec_title else sec_content)
        output = "\n".join(parts) if parts else str(sections)

    # USPSTF recommendations format
    elif "recommendation" in item or "Recommendation" in item:
        rec = item.get("recommendation", item.get("Recommendation", ""))
        grade = item.get("grade", item.get("Grade", ""))
        instruction = f"What does the USPSTF recommend regarding: {rec}?"
        output = f"Grade: {grade}\nRecommendation: {rec}"

    # RxNorm format
    elif "rxcui" in item and "name" in item:
        instruction = f"What is RxNorm concept {item['rxcui']}?"
        output = f"Drug: {item['name']}"
        tty = item.get("tty", "")
        if tty:
            output += f"\nType: {tty}"

    # Generic fallback for any dict with text fields
    elif not instruction:
        text_fields = {k: v for k, v in item.items() if isinstance(v, str) and len(v) > 20}
        if len(text_fields) >= 2:
            keys = list(text_fields.keys())
            instruction = f"Explain the following {category} information: {text_fields[keys[0]][:200]}"
            output = text_fields[keys[1]][:500]

    if not instruction or not output:
        return None

    return {
        "instruction": instruction,
        "input": "",
        "output": output,
        "source": source_name,
        "adapter": "regulatory_qa",
        "category": category,
    }


def download_all(filter_category=None):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    sources = GUIDELINE_SOURCES
    if filter_category:
        sources = [s for s in GUIDELINE_SOURCES if s[3] == filter_category]
        logger.info(f"Filtering to category: {filter_category} ({len(sources)} sources)")

    stats = {"ok": 0, "skip": 0, "fail": 0, "total_examples": 0}

    for name, url, fmt, category, est_size, desc in sources:
        dest = RAW_DIR / f"{name}.json"

        if dest.exists():
            logger.info(f"[SKIP] {name} ({est_size}) — already exists")
            stats["skip"] += 1
            continue

        try:
            logger.info(f"[DL] {name} ({est_size}) <- {url[:80]}...")
            logger.info(f"     {desc}")

            data = download_source(name, url, fmt, dest)
            if data is None:
                stats["skip"] += 1
                continue

            logger.info(f"  -> {len(data):,} raw records")
            stats["ok"] += 1

            # Process to instruction format
            proc_path = PROCESSED_DIR / f"{name}.json"
            processed = []
            for item in data:
                if isinstance(item, dict):
                    result = to_qa_format(item, name, category)
                    if result:
                        processed.append(result)

            if processed:
                with open(proc_path, "w") as f:
                    json.dump(processed, f, default=str)
                logger.info(f"  -> {len(processed):,} processed -> regulatory_qa/{name}.json")
                stats["total_examples"] += len(processed)

        except Exception as e:
            logger.error(f"  X FAILED {name}: {e}")
            stats["fail"] += 1

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("GUIDELINES DOWNLOAD SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  Downloaded:  {stats['ok']}")
    logger.info(f"  Skipped:     {stats['skip']}")
    logger.info(f"  Failed:      {stats['fail']}")
    logger.info(f"  Total processed: {stats['total_examples']:,}")

    # Per-category breakdown
    logger.info("\nBy category:")
    categories = {}
    if PROCESSED_DIR.exists():
        for f in PROCESSED_DIR.glob("*.json"):
            with open(f) as fp:
                items = json.load(fp)
                if items and isinstance(items[0], dict):
                    cat = items[0].get("category", "unknown")
                    categories[cat] = categories.get(cat, 0) + len(items)
    for cat, count in sorted(categories.items()):
        logger.info(f"  {cat:25s} {count:>10,} examples")
    logger.info(f"{'='*60}")


def list_sources():
    """Print all available guideline sources"""
    print(f"\n{'='*70}")
    print("CLINICAL GUIDELINES & QA/QC DATA SOURCES")
    print(f"{'='*70}")
    print(f"{'#':>3}  {'Name':35s}  {'Category':15s}  {'Size':>6s}  Description")
    print("-" * 100)
    for i, (name, url, fmt, category, est_size, desc) in enumerate(GUIDELINE_SOURCES, 1):
        skip = " [manual]" if fmt == "skip" else ""
        print(f"{i:>3}  {name:35s}  {category:15s}  {est_size:>6s}  {desc}{skip}")
    print(f"\nTotal: {len(GUIDELINE_SOURCES)} sources")

    # Category summary
    cats = {}
    for _, _, _, cat, _, _ in GUIDELINE_SOURCES:
        cats[cat] = cats.get(cat, 0) + 1
    print("\nBy category:")
    for cat, count in sorted(cats.items()):
        print(f"  {cat:20s} {count} sources")


def main():
    parser = argparse.ArgumentParser(description="Download clinical guidelines & QA/QC data")
    parser.add_argument("--category", type=str, default=None,
                        choices=["clinical", "quality", "drug_safety", "terminology", "preventive", "infection"],
                        help="Filter by category")
    parser.add_argument("--list", action="store_true",
                        help="List all available sources without downloading")
    args = parser.parse_args()

    if args.list:
        list_sources()
    else:
        download_all(filter_category=args.category)


if __name__ == "__main__":
    main()
