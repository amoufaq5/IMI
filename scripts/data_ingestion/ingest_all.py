"""
UMI Master Data Ingestion Pipeline
Runs all data ingestion pipelines to fetch maximum data from open sources
"""

import asyncio
import argparse
from datetime import datetime
from pathlib import Path

from ingest_pubmed import PubMedIngestionPipeline
from ingest_drugbank import DrugIngestionPipeline
from ingest_clinicaltrials import ClinicalTrialsIngestionPipeline
from ingest_rxnorm import RxNormIngestionPipeline
from ingest_who import WHOIngestionPipeline


async def run_all_pipelines(
    pubmed: bool = True,
    drugs: bool = True,
    trials: bool = True,
    rxnorm: bool = True,
    who: bool = True,
    output_base: str = "data/knowledge_base",
):
    """
    Run all data ingestion pipelines.
    
    Args:
        pubmed: Run PubMed ingestion
        drugs: Run OpenFDA drug ingestion
        trials: Run ClinicalTrials.gov ingestion
        rxnorm: Run RxNorm ingestion
        who: Run WHO/ICD-10 ingestion
        output_base: Base output directory
    """
    print("=" * 70)
    print("UMI MASTER DATA INGESTION PIPELINE")
    print(f"Started at: {datetime.now().isoformat()}")
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
