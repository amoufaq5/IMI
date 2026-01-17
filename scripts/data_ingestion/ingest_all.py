"""
UMI Master Data Ingestion Pipeline
<<<<<<< HEAD
Runs all data ingestion pipelines to fetch MAXIMUM data from open sources
Fault-tolerant: continues even if individual scrapers fail
NO LIMITS on data fetching
=======
Runs all data ingestion pipelines to fetch maximum data from open sources.
Designed to be fault-tolerant: if any scraper fails, others continue.
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)

Supported Data Sources (14 total):
- PubMed: Medical literature and research articles
- OpenFDA: Drug labels and information
- ClinicalTrials.gov: Clinical trial data
- RxNorm: Drug terminology and interactions
- WHO/ICD-10: Disease classification codes
<<<<<<< HEAD
- Kaggle: 80+ medical datasets with auto-download
=======
- Kaggle: Medical datasets with automatic downloads
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
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
<<<<<<< HEAD
import importlib
import os
=======
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
import sys
import traceback
from datetime import datetime
from pathlib import Path
<<<<<<< HEAD
from typing import Any, Dict, List, Optional

# Add script directory to path
=======
from typing import Any, Callable, Dict, Optional

# Add script directory to path for imports when run as standalone script
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
_script_dir = Path(__file__).parent.resolve()
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

<<<<<<< HEAD
=======
# Clear any cached imports
import importlib
if 'ingest_rxnorm' in sys.modules:
    del sys.modules['ingest_rxnorm']

>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)

def safe_import(module_name: str, class_name: str):
    """Safely import a pipeline class, returning None if import fails."""
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except Exception as e:
<<<<<<< HEAD
        print(f"  WARNING: Could not import {class_name} from {module_name}: {e}")
        return None


# Import all pipelines with fallback
=======
        print(f"WARNING: Could not import {class_name} from {module_name}: {e}")
        return None


# Import all pipelines with fallback for missing dependencies
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
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
<<<<<<< HEAD
    """Run a pipeline with comprehensive error handling."""
=======
    """
    Run a pipeline with comprehensive error handling.
    Returns result dict with status, count, and error info if failed.
    """
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
    if pipeline_class is None:
        print(f"  SKIPPED: {name} pipeline not available (import failed)")
        return {"status": "skipped", "reason": "import_failed"}
    
    try:
<<<<<<< HEAD
=======
        # Initialize pipeline
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
        init_args = {"output_dir": output_dir}
        if init_kwargs:
            init_args.update(init_kwargs)
        
        pipeline = pipeline_class(**init_args)
        
<<<<<<< HEAD
=======
        # Run pipeline
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
        if run_kwargs:
            await pipeline.run(**run_kwargs)
        else:
            await pipeline.run()
        
        # Get count from common attributes
        count = 0
        for attr in ["articles", "drugs", "trials", "codes", "topics", "diseases", 
<<<<<<< HEAD
                     "associations", "concepts", "molecules", "proteins", "datasets_processed",
                     "targets"]:
=======
                     "associations", "concepts", "molecules", "proteins", "datasets_processed"]:
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
            if hasattr(pipeline, attr):
                val = getattr(pipeline, attr)
                count = len(val) if val else 0
                break
        
        return {"status": "success", "count": count}
        
    except Exception as e:
<<<<<<< HEAD
        print(f"  ERROR: {name} ingestion failed: {e}")
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}
=======
        error_msg = str(e)
        print(f"  ERROR: {name} ingestion failed: {error_msg}")
        print(f"  Traceback: {traceback.format_exc()}")
        return {"status": "failed", "error": error_msg}
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)


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
<<<<<<< HEAD
    Run all data ingestion pipelines with NO LIMITS.
    Fault-tolerant: continues even if individual scrapers fail.
=======
    Run all data ingestion pipelines with NO CAPS on data fetching.
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
    
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
<<<<<<< HEAD
        orphanet: Run Orphanet ingestion
=======
        orphanet: Run Orphanet rare disease ingestion
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
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
<<<<<<< HEAD
        results["pubmed"] = await run_pipeline_safely(
            "PubMed", PubMedIngestionPipeline, f"{output_base}/pubmed",
            run_kwargs={"max_per_topic": None}  # No limit
        )
=======
        if PubMedIngestionPipeline is None:
            print("  SKIPPED: PubMed pipeline not available (import failed)")
            results["pubmed"] = {"status": "skipped", "reason": "import_failed"}
        else:
            try:
                pipeline = PubMedIngestionPipeline(
                    output_dir=f"{output_base}/pubmed"
                )
                await pipeline.run(max_per_topic=None)  # No limit
                results["pubmed"] = {
                    "status": "success",
                    "count": len(pipeline.articles),
                }
            except Exception as e:
                print(f"  ERROR: PubMed ingestion failed: {e}")
                print(f"  Continuing with next pipeline...")
                results["pubmed"] = {"status": "failed", "error": str(e)}
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
    
    # 2. OpenFDA - Drug Information
    if drugs:
        print("\n" + "=" * 70)
        print("2. OPENFDA - Drug Information (NO LIMIT)")
        print("=" * 70)
<<<<<<< HEAD
        results["drugs"] = await run_pipeline_safely(
            "OpenFDA", DrugIngestionPipeline, f"{output_base}/drugs",
            run_kwargs={"max_per_category": None}  # No limit
        )
=======
        if DrugIngestionPipeline is None:
            print("  SKIPPED: Drug pipeline not available (import failed)")
            results["drugs"] = {"status": "skipped", "reason": "import_failed"}
        else:
            try:
                pipeline = DrugIngestionPipeline(
                    output_dir=f"{output_base}/drugs"
                )
                await pipeline.run(max_per_category=None)  # No limit
                results["drugs"] = {
                    "status": "success",
                    "count": len(pipeline.drugs),
                }
            except Exception as e:
                print(f"  ERROR: Drug ingestion failed: {e}")
                print(f"  Continuing with next pipeline...")
                results["drugs"] = {"status": "failed", "error": str(e)}
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
    
    # 3. ClinicalTrials.gov - Clinical Trials
    if trials:
        print("\n" + "=" * 70)
        print("3. CLINICALTRIALS.GOV - Clinical Trials (NO LIMIT)")
        print("=" * 70)
<<<<<<< HEAD
        results["trials"] = await run_pipeline_safely(
            "ClinicalTrials", ClinicalTrialsIngestionPipeline, f"{output_base}/clinical_trials",
            run_kwargs={"max_per_term": None}  # No limit
        )
=======
        if ClinicalTrialsIngestionPipeline is None:
            print("  SKIPPED: ClinicalTrials pipeline not available")
            results["trials"] = {"status": "skipped", "reason": "import_failed"}
        else:
            try:
                pipeline = ClinicalTrialsIngestionPipeline(
                    output_dir=f"{output_base}/clinical_trials"
                )
                await pipeline.run(max_per_term=None)  # No limit
                results["trials"] = {
                    "status": "success",
                    "count": len(pipeline.trials),
                }
            except Exception as e:
                print(f"  ERROR: Clinical trials ingestion failed: {e}")
                print(f"  Continuing with next pipeline...")
                results["trials"] = {"status": "failed", "error": str(e)}
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
    
    # 4. RxNorm - Drug Terminology
    if rxnorm:
        print("\n" + "=" * 70)
        print("4. RXNORM - Drug Terminology & Interactions")
        print("=" * 70)
<<<<<<< HEAD
        results["rxnorm"] = await run_pipeline_safely(
            "RxNorm", RxNormIngestionPipeline, f"{output_base}/rxnorm"
        )
=======
        if RxNormIngestionPipeline is None:
            print("  SKIPPED: RxNorm pipeline not available")
            results["rxnorm"] = {"status": "skipped", "reason": "import_failed"}
        else:
            try:
                pipeline = RxNormIngestionPipeline(
                    output_dir=f"{output_base}/rxnorm"
                )
                await pipeline.run()
                results["rxnorm"] = {
                    "status": "success",
                    "count": len(pipeline.drugs),
                }
            except Exception as e:
                print(f"  ERROR: RxNorm ingestion failed: {e}")
                print(f"  Continuing with next pipeline...")
                results["rxnorm"] = {"status": "failed", "error": str(e)}
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
    
    # 5. WHO/ICD-10 - Disease Classification
    if who:
        print("\n" + "=" * 70)
        print("5. WHO/ICD-10 - Disease Classification")
        print("=" * 70)
<<<<<<< HEAD
        results["who"] = await run_pipeline_safely(
            "WHO/ICD-10", WHOIngestionPipeline, f"{output_base}/who"
        )
=======
        if WHOIngestionPipeline is None:
            print("  SKIPPED: WHO pipeline not available")
            results["who"] = {"status": "skipped", "reason": "import_failed"}
        else:
            try:
                pipeline = WHOIngestionPipeline(
                    output_dir=f"{output_base}/who"
                )
                await pipeline.run()
                results["who"] = {
                    "status": "success",
                    "count": len(pipeline.codes),
                }
            except Exception as e:
                print(f"  ERROR: WHO/ICD ingestion failed: {e}")
                print(f"  Continuing with next pipeline...")
                results["who"] = {"status": "failed", "error": str(e)}
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
    
    # 6. Kaggle - Medical Datasets
    if kaggle:
        print("\n" + "=" * 70)
<<<<<<< HEAD
        print("6. KAGGLE - Medical Datasets (80+ datasets)")
        print("=" * 70)
        results["kaggle"] = await run_pipeline_safely(
            "Kaggle", KaggleIngestionPipeline, f"{output_base}/kaggle",
            init_kwargs={"download_dir": "data/raw/kaggle", "api_key": kaggle_api_key},
            run_kwargs={"max_datasets": None}  # No limit
        )
=======
        print("6. KAGGLE - Medical Datasets (Auto-Download)")
        print("=" * 70)
        if KaggleIngestionPipeline is None:
            print("  SKIPPED: Kaggle pipeline not available")
            results["kaggle"] = {"status": "skipped", "reason": "import_failed"}
        else:
            try:
                pipeline = KaggleIngestionPipeline(
                    output_dir=f"{output_base}/kaggle",
                    download_dir="data/raw/kaggle"
                )
                await pipeline.run(max_datasets=None)  # No limit
                results["kaggle"] = {
                    "status": "success",
                    "count": len(pipeline.datasets_processed),
                }
            except Exception as e:
                print(f"  ERROR: Kaggle ingestion failed: {e}")
                print(f"  Continuing with next pipeline...")
                results["kaggle"] = {"status": "failed", "error": str(e)}
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
    
    # 7. MedlinePlus - Consumer Health Information
    if medlineplus:
        print("\n" + "=" * 70)
        print("7. MEDLINEPLUS - Consumer Health Information")
        print("=" * 70)
<<<<<<< HEAD
        results["medlineplus"] = await run_pipeline_safely(
            "MedlinePlus", MedlinePlusIngestionPipeline, f"{output_base}/medlineplus"
        )
=======
        if MedlinePlusIngestionPipeline is None:
            print("  SKIPPED: MedlinePlus pipeline not available")
            results["medlineplus"] = {"status": "skipped", "reason": "import_failed"}
        else:
            try:
                pipeline = MedlinePlusIngestionPipeline(
                    output_dir=f"{output_base}/medlineplus"
                )
                await pipeline.run()
                results["medlineplus"] = {
                    "status": "success",
                    "count": len(pipeline.topics),
                }
            except Exception as e:
                print(f"  ERROR: MedlinePlus ingestion failed: {e}")
                print(f"  Continuing with next pipeline...")
                results["medlineplus"] = {"status": "failed", "error": str(e)}
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
    
    # 8. Open Targets - Drug-Target-Disease Associations
    if opentargets:
        print("\n" + "=" * 70)
        print("8. OPEN TARGETS - Drug-Target-Disease Associations")
        print("=" * 70)
<<<<<<< HEAD
        results["opentargets"] = await run_pipeline_safely(
            "OpenTargets", OpenTargetsIngestionPipeline, f"{output_base}/opentargets"
        )
=======
        if OpenTargetsIngestionPipeline is None:
            print("  SKIPPED: Open Targets pipeline not available")
            results["opentargets"] = {"status": "skipped", "reason": "import_failed"}
        else:
            try:
                pipeline = OpenTargetsIngestionPipeline(
                    output_dir=f"{output_base}/opentargets"
                )
                await pipeline.run()
                results["opentargets"] = {
                    "status": "success",
                    "count": len(pipeline.diseases) + len(pipeline.drugs),
                }
            except Exception as e:
                print(f"  ERROR: Open Targets ingestion failed: {e}")
                print(f"  Continuing with next pipeline...")
                results["opentargets"] = {"status": "failed", "error": str(e)}
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
    
    # 9. UMLS - Unified Medical Language System
    if umls:
        print("\n" + "=" * 70)
        print("9. UMLS - Unified Medical Terminology")
        print("=" * 70)
<<<<<<< HEAD
        results["umls"] = await run_pipeline_safely(
            "UMLS", UMLSIngestionPipeline, f"{output_base}/umls",
            init_kwargs={"api_key": umls_api_key}
        )
=======
        if UMLSIngestionPipeline is None:
            print("  SKIPPED: UMLS pipeline not available")
            results["umls"] = {"status": "skipped", "reason": "import_failed"}
        else:
            try:
                pipeline = UMLSIngestionPipeline(
                    output_dir=f"{output_base}/umls"
                )
                await pipeline.run()
                results["umls"] = {
                    "status": "success",
                    "count": len(pipeline.concepts),
                }
            except Exception as e:
                print(f"  ERROR: UMLS ingestion failed: {e}")
                print(f"  Continuing with next pipeline...")
                results["umls"] = {"status": "failed", "error": str(e)}
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
    
    # 10. SNOMED CT - Clinical Terminology
    if snomed:
        print("\n" + "=" * 70)
        print("10. SNOMED CT - Clinical Terminology")
        print("=" * 70)
<<<<<<< HEAD
        results["snomed"] = await run_pipeline_safely(
            "SNOMED CT", SNOMEDIngestionPipeline, f"{output_base}/snomed"
        )
=======
        if SNOMEDIngestionPipeline is None:
            print("  SKIPPED: SNOMED CT pipeline not available")
            results["snomed"] = {"status": "skipped", "reason": "import_failed"}
        else:
            try:
                pipeline = SNOMEDIngestionPipeline(
                    output_dir=f"{output_base}/snomed"
                )
                await pipeline.run()
                results["snomed"] = {
                    "status": "success",
                    "count": len(pipeline.concepts),
                }
            except Exception as e:
                print(f"  ERROR: SNOMED CT ingestion failed: {e}")
                print(f"  Continuing with next pipeline...")
                results["snomed"] = {"status": "failed", "error": str(e)}
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
    
    # 11. Orphanet - Rare Diseases
    if orphanet:
        print("\n" + "=" * 70)
        print("11. ORPHANET - Rare Disease Information")
        print("=" * 70)
<<<<<<< HEAD
        results["orphanet"] = await run_pipeline_safely(
            "Orphanet", OrphanetIngestionPipeline, f"{output_base}/orphanet"
        )
=======
        if OrphanetIngestionPipeline is None:
            print("  SKIPPED: Orphanet pipeline not available")
            results["orphanet"] = {"status": "skipped", "reason": "import_failed"}
        else:
            try:
                pipeline = OrphanetIngestionPipeline(
                    output_dir=f"{output_base}/orphanet"
                )
                await pipeline.run()
                results["orphanet"] = {
                    "status": "success",
                    "count": len(pipeline.diseases),
                }
            except Exception as e:
                print(f"  ERROR: Orphanet ingestion failed: {e}")
                print(f"  Continuing with next pipeline...")
                results["orphanet"] = {"status": "failed", "error": str(e)}
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
    
    # 12. DisGeNET - Gene-Disease Associations
    if disgenet:
        print("\n" + "=" * 70)
        print("12. DISGENET - Gene-Disease Associations")
        print("=" * 70)
<<<<<<< HEAD
        results["disgenet"] = await run_pipeline_safely(
            "DisGeNET", DisGeNETIngestionPipeline, f"{output_base}/disgenet",
            init_kwargs={"api_key": disgenet_api_key}
        )
=======
        if DisGeNETIngestionPipeline is None:
            print("  SKIPPED: DisGeNET pipeline not available")
            results["disgenet"] = {"status": "skipped", "reason": "import_failed"}
        else:
            try:
                pipeline = DisGeNETIngestionPipeline(
                    output_dir=f"{output_base}/disgenet"
                )
                await pipeline.run()
                results["disgenet"] = {
                    "status": "success",
                    "count": len(pipeline.associations),
                }
            except Exception as e:
                print(f"  ERROR: DisGeNET ingestion failed: {e}")
                print(f"  Continuing with next pipeline...")
                results["disgenet"] = {"status": "failed", "error": str(e)}
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
    
    # 13. ChEMBL - Bioactivity Data
    if chembl:
        print("\n" + "=" * 70)
        print("13. CHEMBL - Bioactivity & Drug Data")
        print("=" * 70)
<<<<<<< HEAD
        results["chembl"] = await run_pipeline_safely(
            "ChEMBL", ChEMBLIngestionPipeline, f"{output_base}/chembl"
        )
=======
        if ChEMBLIngestionPipeline is None:
            print("  SKIPPED: ChEMBL pipeline not available")
            results["chembl"] = {"status": "skipped", "reason": "import_failed"}
        else:
            try:
                pipeline = ChEMBLIngestionPipeline(
                    output_dir=f"{output_base}/chembl"
                )
                await pipeline.run()
                results["chembl"] = {
                    "status": "success",
                    "count": len(pipeline.molecules) + len(pipeline.targets),
                }
            except Exception as e:
                print(f"  ERROR: ChEMBL ingestion failed: {e}")
                print(f"  Continuing with next pipeline...")
                results["chembl"] = {"status": "failed", "error": str(e)}
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
    
    # 14. UniProt - Protein Data
    if uniprot:
        print("\n" + "=" * 70)
        print("14. UNIPROT - Protein & Gene Data")
        print("=" * 70)
<<<<<<< HEAD
        results["uniprot"] = await run_pipeline_safely(
            "UniProt", UniProtIngestionPipeline, f"{output_base}/uniprot"
        )
=======
        if UniProtIngestionPipeline is None:
            print("  SKIPPED: UniProt pipeline not available")
            results["uniprot"] = {"status": "skipped", "reason": "import_failed"}
        else:
            try:
                pipeline = UniProtIngestionPipeline(
                    output_dir=f"{output_base}/uniprot"
                )
                await pipeline.run()
                results["uniprot"] = {
                    "status": "success",
                    "count": len(pipeline.proteins),
                }
            except Exception as e:
                print(f"  ERROR: UniProt ingestion failed: {e}")
                print(f"  Continuing with next pipeline...")
                results["uniprot"] = {"status": "failed", "error": str(e)}
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
    
    # Summary
    print("\n" + "=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)
    
    success_count = 0
    failed_count = 0
    skipped_count = 0
<<<<<<< HEAD
    total_items = 0
=======
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
    
    for source, result in results.items():
        status = result.get("status", "unknown")
        if status == "success":
            count = result.get("count", 0)
            print(f"  {source.upper()}: ✓ Success - {count} items")
            success_count += 1
<<<<<<< HEAD
            total_items += count
        elif status == "skipped":
            print(f"  {source.upper()}: ⊘ Skipped - {result.get('reason', 'unknown')}")
            skipped_count += 1
        else:
            print(f"  {source.upper()}: ✗ Failed - {result.get('error', 'Unknown error')[:50]}")
            failed_count += 1
=======
        elif status == "skipped":
            reason = result.get("reason", "unknown")
            print(f"  {source.upper()}: ⊘ Skipped - {reason}")
            skipped_count += 1
        else:
            print(f"  {source.upper()}: ✗ Failed - {result.get('error', 'Unknown error')}")
            failed_count += 1
    
    print(f"\nResults: {success_count} succeeded, {failed_count} failed, {skipped_count} skipped")
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
    
    print(f"\nTotal: {success_count} succeeded, {failed_count} failed, {skipped_count} skipped")
    print(f"Total items ingested: {total_items:,}")
    print(f"\nCompleted at: {datetime.now().isoformat()}")
    print("=" * 70)
    
    return results


def main():
    """Main entry point with CLI arguments."""
    parser = argparse.ArgumentParser(
<<<<<<< HEAD
        description="UMI Master Data Ingestion Pipeline - NO LIMITS"
=======
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
        "--kaggle", action="store_true", default=True,
        help="Run Kaggle datasets ingestion"
    )
    parser.add_argument(
        "--no-kaggle", action="store_false", dest="kaggle",
        help="Skip Kaggle datasets ingestion"
    )
    parser.add_argument(
        "--medlineplus", action="store_true", default=True,
        help="Run MedlinePlus ingestion"
    )
    parser.add_argument(
        "--no-medlineplus", action="store_false", dest="medlineplus",
        help="Skip MedlinePlus ingestion"
    )
    parser.add_argument(
        "--opentargets", action="store_true", default=True,
        help="Run Open Targets ingestion"
    )
    parser.add_argument(
        "--no-opentargets", action="store_false", dest="opentargets",
        help="Skip Open Targets ingestion"
    )
    parser.add_argument(
        "--umls", action="store_true", default=True,
        help="Run UMLS ingestion (requires API key)"
    )
    parser.add_argument(
        "--no-umls", action="store_false", dest="umls",
        help="Skip UMLS ingestion"
    )
    parser.add_argument(
        "--snomed", action="store_true", default=True,
        help="Run SNOMED CT ingestion"
    )
    parser.add_argument(
        "--no-snomed", action="store_false", dest="snomed",
        help="Skip SNOMED CT ingestion"
    )
    parser.add_argument(
        "--orphanet", action="store_true", default=True,
        help="Run Orphanet rare disease ingestion"
    )
    parser.add_argument(
        "--no-orphanet", action="store_false", dest="orphanet",
        help="Skip Orphanet ingestion"
    )
    parser.add_argument(
        "--disgenet", action="store_true", default=True,
        help="Run DisGeNET ingestion"
    )
    parser.add_argument(
        "--no-disgenet", action="store_false", dest="disgenet",
        help="Skip DisGeNET ingestion"
    )
    parser.add_argument(
        "--chembl", action="store_true", default=True,
        help="Run ChEMBL ingestion"
    )
    parser.add_argument(
        "--no-chembl", action="store_false", dest="chembl",
        help="Skip ChEMBL ingestion"
    )
    parser.add_argument(
        "--uniprot", action="store_true", default=True,
        help="Run UniProt ingestion"
    )
    parser.add_argument(
        "--no-uniprot", action="store_false", dest="uniprot",
        help="Skip UniProt ingestion"
    )
    parser.add_argument(
        "--output", type=str, default="data/knowledge_base",
        help="Base output directory"
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
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
