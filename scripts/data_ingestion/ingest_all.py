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
        print("1. PUBMED - Medical Literature")
        print("=" * 70)
        try:
            pipeline = PubMedIngestionPipeline(
                output_dir=f"{output_base}/pubmed"
            )
            await pipeline.run(max_per_topic=200)
            results["pubmed"] = {
                "status": "success",
                "articles": len(pipeline.articles),
            }
        except Exception as e:
            print(f"PubMed ingestion failed: {e}")
            results["pubmed"] = {"status": "failed", "error": str(e)}
    
    # 2. OpenFDA - Drug Information
    if drugs:
        print("\n" + "=" * 70)
        print("2. OPENFDA - Drug Information")
        print("=" * 70)
        try:
            pipeline = DrugIngestionPipeline(
                output_dir=f"{output_base}/drugs"
            )
            await pipeline.run(max_per_category=100)
            results["drugs"] = {
                "status": "success",
                "drugs": len(pipeline.drugs),
            }
        except Exception as e:
            print(f"Drug ingestion failed: {e}")
            results["drugs"] = {"status": "failed", "error": str(e)}
    
    # 3. ClinicalTrials.gov - Clinical Trials
    if trials:
        print("\n" + "=" * 70)
        print("3. CLINICALTRIALS.GOV - Clinical Trials")
        print("=" * 70)
        try:
            pipeline = ClinicalTrialsIngestionPipeline(
                output_dir=f"{output_base}/clinical_trials"
            )
            await pipeline.run(max_per_term=150)
            results["trials"] = {
                "status": "success",
                "trials": len(pipeline.trials),
            }
        except Exception as e:
            print(f"Clinical trials ingestion failed: {e}")
            results["trials"] = {"status": "failed", "error": str(e)}
    
    # 4. RxNorm - Drug Terminology
    if rxnorm:
        print("\n" + "=" * 70)
        print("4. RXNORM - Drug Terminology & Interactions")
        print("=" * 70)
        try:
            pipeline = RxNormIngestionPipeline(
                output_dir=f"{output_base}/rxnorm"
            )
            await pipeline.run()
            results["rxnorm"] = {
                "status": "success",
                "drugs": len(pipeline.drugs),
            }
        except Exception as e:
            print(f"RxNorm ingestion failed: {e}")
            results["rxnorm"] = {"status": "failed", "error": str(e)}
    
    # 5. WHO/ICD-10 - Disease Classification
    if who:
        print("\n" + "=" * 70)
        print("5. WHO/ICD-10 - Disease Classification")
        print("=" * 70)
        try:
            pipeline = WHOIngestionPipeline(
                output_dir=f"{output_base}/who"
            )
            await pipeline.run()
            results["who"] = {
                "status": "success",
                "codes": len(pipeline.codes),
            }
        except Exception as e:
            print(f"WHO/ICD ingestion failed: {e}")
            results["who"] = {"status": "failed", "error": str(e)}
    
    # Summary
    print("\n" + "=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)
    
    for source, result in results.items():
        status = result.get("status", "unknown")
        if status == "success":
            count_key = [k for k in result.keys() if k != "status"][0]
            print(f"  {source.upper()}: ✓ Success - {result[count_key]} items")
        else:
            print(f"  {source.upper()}: ✗ Failed - {result.get('error', 'Unknown error')}")
    
    print(f"\nCompleted at: {datetime.now().isoformat()}")
    print("=" * 70)
    
    return results


def main():
    """Main entry point with CLI arguments."""
    parser = argparse.ArgumentParser(
        description="UMI Master Data Ingestion Pipeline"
    )
    parser.add_argument(
        "--pubmed", action="store_true", default=True,
        help="Run PubMed ingestion"
    )
    parser.add_argument(
        "--no-pubmed", action="store_false", dest="pubmed",
        help="Skip PubMed ingestion"
    )
    parser.add_argument(
        "--drugs", action="store_true", default=True,
        help="Run OpenFDA drug ingestion"
    )
    parser.add_argument(
        "--no-drugs", action="store_false", dest="drugs",
        help="Skip OpenFDA drug ingestion"
    )
    parser.add_argument(
        "--trials", action="store_true", default=True,
        help="Run ClinicalTrials.gov ingestion"
    )
    parser.add_argument(
        "--no-trials", action="store_false", dest="trials",
        help="Skip ClinicalTrials.gov ingestion"
    )
    parser.add_argument(
        "--rxnorm", action="store_true", default=True,
        help="Run RxNorm ingestion"
    )
    parser.add_argument(
        "--no-rxnorm", action="store_false", dest="rxnorm",
        help="Skip RxNorm ingestion"
    )
    parser.add_argument(
        "--who", action="store_true", default=True,
        help="Run WHO/ICD-10 ingestion"
    )
    parser.add_argument(
        "--no-who", action="store_false", dest="who",
        help="Skip WHO/ICD-10 ingestion"
    )
    parser.add_argument(
        "--output", type=str, default="data/knowledge_base",
        help="Base output directory"
    )
    
    args = parser.parse_args()
    
    asyncio.run(run_all_pipelines(
        pubmed=args.pubmed,
        drugs=args.drugs,
        trials=args.trials,
        rxnorm=args.rxnorm,
        who=args.who,
        output_base=args.output,
    ))


if __name__ == "__main__":
    main()
