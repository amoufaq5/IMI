"""
UMI Master Data Ingestion Pipeline
Runs all data ingestion pipelines to fetch MAXIMUM data from open sources
Fault-tolerant: continues even if individual scrapers fail
NO LIMITS on data fetching

Supported Data Sources (14 total):
- PubMed: Medical literature and research articles
- OpenFDA: Drug labels and information
- ClinicalTrials.gov: Clinical trial data
- RxNorm: Drug terminology and interactions
- WHO/ICD-10: Disease classification codes
- Kaggle: 80+ medical datasets with auto-download
- MedlinePlus: Consumer health information
- Open Targets: Drug-target-disease associations
- UMLS: Unified medical terminology (requires API key)
- SNOMED CT: Clinical terminology
- Orphanet: Rare disease information
- DisGeNET: Gene-disease associations
- ChEMBL: Bioactivity and drug data
- UniProt: Protein and gene data
"""

import asyncio
import argparse
import importlib
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add script directory to path
_script_dir = Path(__file__).parent.resolve()
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))


def safe_import(module_name: str, class_name: str):
    """Safely import a pipeline class, returning None if import fails."""
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except Exception as e:
        print(f"  WARNING: Could not import {class_name} from {module_name}: {e}")
        return None


# Import all pipelines with fallback
PubMedIngestionPipeline = safe_import("ingest_pubmed", "PubMedIngestionPipeline")
DrugIngestionPipeline = safe_import("ingest_drugbank", "DrugIngestionPipeline")
ClinicalTrialsIngestionPipeline = safe_import("ingest_clinicaltrials", "ClinicalTrialsIngestionPipeline")
RxNormIngestionPipeline = safe_import("ingest_rxnorm", "RxNormIngestionPipeline")
WHOIngestionPipeline = safe_import("ingest_who", "WHOIngestionPipeline")
KaggleIngestionPipeline = safe_import("ingest_kaggle", "KaggleIngestionPipeline")
MedlinePlusIngestionPipeline = safe_import("ingest_medlineplus", "MedlinePlusIngestionPipeline")
OpenTargetsIngestionPipeline = safe_import("ingest_opentargets", "OpenTargetsIngestionPipeline")
UMLSIngestionPipeline = safe_import("ingest_umls", "UMLSIngestionPipeline")
SNOMEDIngestionPipeline = safe_import("ingest_snomed", "SNOMEDIngestionPipeline")
OrphanetIngestionPipeline = safe_import("ingest_orphanet", "OrphanetIngestionPipeline")
DisGeNETIngestionPipeline = safe_import("ingest_disgenet", "DisGeNETIngestionPipeline")
ChEMBLIngestionPipeline = safe_import("ingest_chembl", "ChEMBLIngestionPipeline")
UniProtIngestionPipeline = safe_import("ingest_uniprot", "UniProtIngestionPipeline")


async def run_pipeline_safely(
    name: str,
    pipeline_class,
    output_dir: str,
    run_kwargs: Optional[Dict[str, Any]] = None,
    init_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a pipeline with comprehensive error handling."""
    if pipeline_class is None:
        print(f"  SKIPPED: {name} pipeline not available (import failed)")
        return {"status": "skipped", "reason": "import_failed"}
    
    try:
        init_args = {"output_dir": output_dir}
        if init_kwargs:
            init_args.update(init_kwargs)
        
        pipeline = pipeline_class(**init_args)
        
        if run_kwargs:
            await pipeline.run(**run_kwargs)
        else:
            await pipeline.run()
        
        # Get count from common attributes
        count = 0
        for attr in ["articles", "drugs", "trials", "codes", "topics", "diseases", 
                     "associations", "concepts", "molecules", "proteins", "datasets_processed",
                     "targets"]:
            if hasattr(pipeline, attr):
                val = getattr(pipeline, attr)
                count = len(val) if val else 0
                break
        
        return {"status": "success", "count": count}
        
    except Exception as e:
        print(f"  ERROR: {name} ingestion failed: {e}")
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}


async def run_all_pipelines(
    pubmed: bool = True,
    drugs: bool = True,
    trials: bool = True,
    rxnorm: bool = True,
    who: bool = True,
    kaggle: bool = True,
    medlineplus: bool = True,
    opentargets: bool = True,
    umls: bool = True,
    snomed: bool = True,
    orphanet: bool = True,
    disgenet: bool = True,
    chembl: bool = True,
    uniprot: bool = True,
    output_base: str = "data/knowledge_base",
    kaggle_api_key: Optional[str] = None,
    umls_api_key: Optional[str] = None,
    disgenet_api_key: Optional[str] = None,
):
    """
    Run all data ingestion pipelines with NO LIMITS.
    Fault-tolerant: continues even if individual scrapers fail.
    
    Args:
        pubmed: Run PubMed ingestion
        drugs: Run OpenFDA drug ingestion
        trials: Run ClinicalTrials.gov ingestion
        rxnorm: Run RxNorm ingestion
        who: Run WHO/ICD-10 ingestion
        kaggle: Run Kaggle datasets ingestion
        medlineplus: Run MedlinePlus ingestion
        opentargets: Run Open Targets ingestion
        umls: Run UMLS ingestion (requires API key)
        snomed: Run SNOMED CT ingestion
        orphanet: Run Orphanet ingestion
        disgenet: Run DisGeNET ingestion
        chembl: Run ChEMBL ingestion
        uniprot: Run UniProt ingestion
        output_base: Base output directory
        kaggle_api_key: Kaggle API key
        umls_api_key: UMLS API key
        disgenet_api_key: DisGeNET API key
    """
    print("=" * 70)
    print("UMI MASTER DATA INGESTION PIPELINE - NO LIMITS")
    print(f"Started at: {datetime.now().isoformat()}")
    print("Fault-tolerant mode: will continue if any scraper fails")
    print("=" * 70)
    
    results = {}
    
    # 1. PubMed - Medical Literature
    if pubmed:
        print("\n" + "=" * 70)
        print("1. PUBMED - Medical Literature (NO LIMIT)")
        print("=" * 70)
        results["pubmed"] = await run_pipeline_safely(
            "PubMed", PubMedIngestionPipeline, f"{output_base}/pubmed",
            run_kwargs={"max_per_topic": None}  # No limit
        )
    
    # 2. OpenFDA - Drug Information
    if drugs:
        print("\n" + "=" * 70)
        print("2. OPENFDA - Drug Information (NO LIMIT)")
        print("=" * 70)
        results["drugs"] = await run_pipeline_safely(
            "OpenFDA", DrugIngestionPipeline, f"{output_base}/drugs",
            run_kwargs={"max_per_category": None}  # No limit
        )
    
    # 3. ClinicalTrials.gov - Clinical Trials
    if trials:
        print("\n" + "=" * 70)
        print("3. CLINICALTRIALS.GOV - Clinical Trials (NO LIMIT)")
        print("=" * 70)
        results["trials"] = await run_pipeline_safely(
            "ClinicalTrials", ClinicalTrialsIngestionPipeline, f"{output_base}/clinical_trials",
            run_kwargs={"max_per_term": None}  # No limit
        )
    
    # 4. RxNorm - Drug Terminology
    if rxnorm:
        print("\n" + "=" * 70)
        print("4. RXNORM - Drug Terminology & Interactions")
        print("=" * 70)
        results["rxnorm"] = await run_pipeline_safely(
            "RxNorm", RxNormIngestionPipeline, f"{output_base}/rxnorm"
        )
    
    # 5. WHO/ICD-10 - Disease Classification
    if who:
        print("\n" + "=" * 70)
        print("5. WHO/ICD-10 - Disease Classification")
        print("=" * 70)
        results["who"] = await run_pipeline_safely(
            "WHO/ICD-10", WHOIngestionPipeline, f"{output_base}/who"
        )
    
    # 6. Kaggle - Medical Datasets
    if kaggle:
        print("\n" + "=" * 70)
        print("6. KAGGLE - Medical Datasets (80+ datasets)")
        print("=" * 70)
        results["kaggle"] = await run_pipeline_safely(
            "Kaggle", KaggleIngestionPipeline, f"{output_base}/kaggle",
            init_kwargs={"download_dir": "data/raw/kaggle", "api_key": kaggle_api_key},
            run_kwargs={"max_datasets": None}  # No limit
        )
    
    # 7. MedlinePlus - Consumer Health Information
    if medlineplus:
        print("\n" + "=" * 70)
        print("7. MEDLINEPLUS - Consumer Health Information")
        print("=" * 70)
        results["medlineplus"] = await run_pipeline_safely(
            "MedlinePlus", MedlinePlusIngestionPipeline, f"{output_base}/medlineplus"
        )
    
    # 8. Open Targets - Drug-Target-Disease Associations
    if opentargets:
        print("\n" + "=" * 70)
        print("8. OPEN TARGETS - Drug-Target-Disease Associations")
        print("=" * 70)
        results["opentargets"] = await run_pipeline_safely(
            "OpenTargets", OpenTargetsIngestionPipeline, f"{output_base}/opentargets"
        )
    
    # 9. UMLS - Unified Medical Language System
    if umls:
        print("\n" + "=" * 70)
        print("9. UMLS - Unified Medical Terminology")
        print("=" * 70)
        results["umls"] = await run_pipeline_safely(
            "UMLS", UMLSIngestionPipeline, f"{output_base}/umls",
            init_kwargs={"api_key": umls_api_key}
        )
    
    # 10. SNOMED CT - Clinical Terminology
    if snomed:
        print("\n" + "=" * 70)
        print("10. SNOMED CT - Clinical Terminology")
        print("=" * 70)
        results["snomed"] = await run_pipeline_safely(
            "SNOMED CT", SNOMEDIngestionPipeline, f"{output_base}/snomed"
        )
    
    # 11. Orphanet - Rare Diseases
    if orphanet:
        print("\n" + "=" * 70)
        print("11. ORPHANET - Rare Disease Information")
        print("=" * 70)
        results["orphanet"] = await run_pipeline_safely(
            "Orphanet", OrphanetIngestionPipeline, f"{output_base}/orphanet"
        )
    
    # 12. DisGeNET - Gene-Disease Associations
    if disgenet:
        print("\n" + "=" * 70)
        print("12. DISGENET - Gene-Disease Associations")
        print("=" * 70)
        results["disgenet"] = await run_pipeline_safely(
            "DisGeNET", DisGeNETIngestionPipeline, f"{output_base}/disgenet",
            init_kwargs={"api_key": disgenet_api_key}
        )
    
    # 13. ChEMBL - Bioactivity Data
    if chembl:
        print("\n" + "=" * 70)
        print("13. CHEMBL - Bioactivity & Drug Data")
        print("=" * 70)
        results["chembl"] = await run_pipeline_safely(
            "ChEMBL", ChEMBLIngestionPipeline, f"{output_base}/chembl"
        )
    
    # 14. UniProt - Protein Data
    if uniprot:
        print("\n" + "=" * 70)
        print("14. UNIPROT - Protein & Gene Data")
        print("=" * 70)
        results["uniprot"] = await run_pipeline_safely(
            "UniProt", UniProtIngestionPipeline, f"{output_base}/uniprot"
        )
    
    # Summary
    print("\n" + "=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)
    
    success_count = 0
    failed_count = 0
    skipped_count = 0
    total_items = 0
    
    for source, result in results.items():
        status = result.get("status", "unknown")
        if status == "success":
            count = result.get("count", 0)
            print(f"  {source.upper()}: ✓ Success - {count} items")
            success_count += 1
            total_items += count
        elif status == "skipped":
            print(f"  {source.upper()}: ⊘ Skipped - {result.get('reason', 'unknown')}")
            skipped_count += 1
        else:
            print(f"  {source.upper()}: ✗ Failed - {result.get('error', 'Unknown error')[:50]}")
            failed_count += 1
    
    print(f"\nTotal: {success_count} succeeded, {failed_count} failed, {skipped_count} skipped")
    print(f"Total items ingested: {total_items:,}")
    print(f"\nCompleted at: {datetime.now().isoformat()}")
    print("=" * 70)
    
    return results


def main():
    """Main entry point with CLI arguments."""
    parser = argparse.ArgumentParser(
        description="UMI Master Data Ingestion Pipeline - NO LIMITS"
    )
    
    # Data source toggles
    parser.add_argument("--no-pubmed", action="store_false", dest="pubmed", help="Skip PubMed")
    parser.add_argument("--no-drugs", action="store_false", dest="drugs", help="Skip OpenFDA")
    parser.add_argument("--no-trials", action="store_false", dest="trials", help="Skip ClinicalTrials")
    parser.add_argument("--no-rxnorm", action="store_false", dest="rxnorm", help="Skip RxNorm")
    parser.add_argument("--no-who", action="store_false", dest="who", help="Skip WHO/ICD-10")
    parser.add_argument("--no-kaggle", action="store_false", dest="kaggle", help="Skip Kaggle")
    parser.add_argument("--no-medlineplus", action="store_false", dest="medlineplus", help="Skip MedlinePlus")
    parser.add_argument("--no-opentargets", action="store_false", dest="opentargets", help="Skip OpenTargets")
    parser.add_argument("--no-umls", action="store_false", dest="umls", help="Skip UMLS")
    parser.add_argument("--no-snomed", action="store_false", dest="snomed", help="Skip SNOMED CT")
    parser.add_argument("--no-orphanet", action="store_false", dest="orphanet", help="Skip Orphanet")
    parser.add_argument("--no-disgenet", action="store_false", dest="disgenet", help="Skip DisGeNET")
    parser.add_argument("--no-chembl", action="store_false", dest="chembl", help="Skip ChEMBL")
    parser.add_argument("--no-uniprot", action="store_false", dest="uniprot", help="Skip UniProt")
    
    # Set defaults to True
    parser.set_defaults(
        pubmed=True, drugs=True, trials=True, rxnorm=True, who=True,
        kaggle=True, medlineplus=True, opentargets=True, umls=True,
        snomed=True, orphanet=True, disgenet=True, chembl=True, uniprot=True
    )
    
    # API keys
    parser.add_argument("--kaggle-key", type=str, default=os.environ.get("KAGGLE_KEY"), help="Kaggle API key")
    parser.add_argument("--umls-key", type=str, default=os.environ.get("UMLS_API_KEY"), help="UMLS API key")
    parser.add_argument("--disgenet-key", type=str, default=os.environ.get("DISGENET_API_KEY"), help="DisGeNET API key")
    
    # Output
    parser.add_argument("--output", type=str, default="data/knowledge_base", help="Base output directory")
    
    # Run only specific sources
    parser.add_argument("--only", type=str, nargs="+", help="Run only specified sources")
    
    args = parser.parse_args()
    
    # Handle --only flag
    if args.only:
        sources = [s.lower() for s in args.only]
        args.pubmed = "pubmed" in sources
        args.drugs = "drugs" in sources or "openfda" in sources
        args.trials = "trials" in sources or "clinicaltrials" in sources
        args.rxnorm = "rxnorm" in sources
        args.who = "who" in sources or "icd" in sources
        args.kaggle = "kaggle" in sources
        args.medlineplus = "medlineplus" in sources
        args.opentargets = "opentargets" in sources
        args.umls = "umls" in sources
        args.snomed = "snomed" in sources
        args.orphanet = "orphanet" in sources
        args.disgenet = "disgenet" in sources
        args.chembl = "chembl" in sources
        args.uniprot = "uniprot" in sources
    
    asyncio.run(run_all_pipelines(
        pubmed=args.pubmed,
        drugs=args.drugs,
        trials=args.trials,
        rxnorm=args.rxnorm,
        who=args.who,
        kaggle=args.kaggle,
        medlineplus=args.medlineplus,
        opentargets=args.opentargets,
        umls=args.umls,
        snomed=args.snomed,
        orphanet=args.orphanet,
        disgenet=args.disgenet,
        chembl=args.chembl,
        uniprot=args.uniprot,
        output_base=args.output,
        kaggle_api_key=args.kaggle_key,
        umls_api_key=args.umls_key,
        disgenet_api_key=args.disgenet_key,
    ))


if __name__ == "__main__":
    main()
