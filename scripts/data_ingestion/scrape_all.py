"""
IMI Master Data Ingestion - Run all scrapers and consolidate data
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add script directory to path
sys.path.insert(0, str(Path(__file__).parent))

from base_scraper import logger
from scrape_pubmed import PubMedScraper
from scrape_medical_datasets import MedicalDatasetScraper


async def run_all_scrapers(
    output_base: str = "data/raw",
    pubmed: bool = True,
    datasets: bool = True,
    pubmed_articles_per_topic: int = 200,
    max_samples_per_dataset: Optional[int] = None,
) -> Dict[str, any]:
    """
    Run all data scrapers and consolidate results.
    
    Args:
        output_base: Base output directory
        pubmed: Run PubMed scraper
        datasets: Run HuggingFace medical datasets scraper
        pubmed_articles_per_topic: Articles per topic for PubMed
        max_samples_per_dataset: Max samples per HF dataset
    
    Returns:
        Summary of all scraping results
    """
    results = {}
    start_time = datetime.now()
    
    print("=" * 70)
    print("IMI DATA INGESTION PIPELINE")
    print(f"Started: {start_time.isoformat()}")
    print("=" * 70)
    
    # 1. PubMed
    if pubmed:
        print("\n[1/2] PUBMED - Medical Literature")
        print("-" * 50)
        try:
            scraper = PubMedScraper(output_dir=f"{output_base}/pubmed")
            results["pubmed"] = await scraper.run(
                articles_per_topic=pubmed_articles_per_topic
            )
            print(f"  ✓ PubMed: {results['pubmed']['items_scraped']} articles")
        except Exception as e:
            logger.error(f"PubMed scraper failed: {e}")
            results["pubmed"] = {"error": str(e)}
    
    # 2. HuggingFace Medical Datasets
    if datasets:
        print("\n[2/2] HUGGINGFACE - Medical Datasets")
        print("-" * 50)
        try:
            scraper = MedicalDatasetScraper(output_dir=f"{output_base}/medical_datasets")
            results["datasets"] = await scraper.run(
                max_samples_per_dataset=max_samples_per_dataset
            )
            scraper.save_by_dataset()
            print(f"  ✓ Datasets: {results['datasets']['items_scraped']} samples")
        except Exception as e:
            logger.error(f"Dataset scraper failed: {e}")
            results["datasets"] = {"error": str(e)}
    
    # Summary
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70)
    print(f"Duration: {elapsed:.1f} seconds")
    
    total_items = sum(
        r.get("items_scraped", 0) for r in results.values() 
        if isinstance(r, dict) and "items_scraped" in r
    )
    print(f"Total items: {total_items:,}")
    
    # Save summary
    summary = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "elapsed_seconds": elapsed,
        "total_items": total_items,
        "results": results,
    }
    
    summary_path = Path(output_base) / "ingestion_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nSummary saved to: {summary_path}")
    
    return summary


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="IMI Master Data Ingestion")
    parser.add_argument("--output", default="data/raw", help="Output base directory")
    parser.add_argument("--no-pubmed", action="store_true", help="Skip PubMed")
    parser.add_argument("--no-datasets", action="store_true", help="Skip HF datasets")
    parser.add_argument("--pubmed-articles", type=int, default=200, help="PubMed articles per topic")
    parser.add_argument("--max-samples", type=int, help="Max samples per HF dataset")
    args = parser.parse_args()
    
    await run_all_scrapers(
        output_base=args.output,
        pubmed=not args.no_pubmed,
        datasets=not args.no_datasets,
        pubmed_articles_per_topic=args.pubmed_articles,
        max_samples_per_dataset=args.max_samples,
    )


if __name__ == "__main__":
    asyncio.run(main())
