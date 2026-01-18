"""
IMI Medical Dataset Scraper - Download open medical datasets from HuggingFace
"""

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from base_scraper import BaseScraper, ScrapedItem, logger


class MedicalDatasetScraper(BaseScraper[Dict]):
    """
    Scraper for open medical datasets from HuggingFace.
    Downloads and processes datasets for training.
    """
    
    SOURCE_NAME = "medical_datasets"
    
    # High-quality medical datasets on HuggingFace
    DATASETS = {
        # Medical QA datasets
        "pubmedqa": {
            "name": "pubmed_qa",
            "subset": "pqa_labeled",
            "split": "train",
            "description": "PubMed Question Answering dataset",
        },
        "medmcqa": {
            "name": "openlifescienceai/medmcqa",
            "subset": None,
            "split": "train",
            "description": "Medical entrance exam questions (AIIMS/NEET)",
        },
        "medqa": {
            "name": "bigbio/med_qa",
            "subset": "med_qa_en_source",
            "split": "train",
            "description": "USMLE-style medical questions",
        },
        # Medical instruction datasets
        "medical_meadow_flashcards": {
            "name": "medalpaca/medical_meadow_medical_flashcards",
            "subset": None,
            "split": "train",
            "description": "Medical flashcards for learning",
        },
        "medical_meadow_wikidoc": {
            "name": "medalpaca/medical_meadow_wikidoc",
            "subset": None,
            "split": "train",
            "description": "WikiDoc medical articles",
        },
        "medical_meadow_wikidoc_patient": {
            "name": "medalpaca/medical_meadow_wikidoc_patient_information",
            "subset": None,
            "split": "train",
            "description": "Patient-friendly medical information",
        },
        # Clinical datasets
        "chatdoctor": {
            "name": "lavita/ChatDoctor-HealthCareMagic-100k",
            "subset": None,
            "split": "train",
            "description": "Doctor-patient conversations",
        },
        # Drug information
        "drug_info": {
            "name": "truehealth/drugbank",
            "subset": None,
            "split": "train",
            "description": "Drug information database",
        },
    }
    
    def __init__(self, output_dir: str):
        super().__init__(output_dir)
        self.datasets_loaded = {}
    
    async def scrape(
        self,
        dataset_names: Optional[List[str]] = None,
        max_samples_per_dataset: Optional[int] = None,
        **kwargs
    ) -> List[Dict]:
        """
        Download and process medical datasets.
        
        Args:
            dataset_names: Specific datasets to download (None = all)
            max_samples_per_dataset: Limit samples per dataset
        """
        from datasets import load_dataset
        
        datasets_to_load = dataset_names or list(self.DATASETS.keys())
        all_items = []
        
        for ds_key in datasets_to_load:
            if ds_key not in self.DATASETS:
                logger.warning(f"Unknown dataset: {ds_key}")
                continue
            
            ds_config = self.DATASETS[ds_key]
            logger.info(f"Loading {ds_key}: {ds_config['description']}")
            
            try:
                # Load dataset
                if ds_config["subset"]:
                    dataset = load_dataset(
                        ds_config["name"],
                        ds_config["subset"],
                        split=ds_config["split"],
                        trust_remote_code=True
                    )
                else:
                    dataset = load_dataset(
                        ds_config["name"],
                        split=ds_config["split"],
                        trust_remote_code=True
                    )
                
                # Limit samples if specified
                if max_samples_per_dataset and len(dataset) > max_samples_per_dataset:
                    dataset = dataset.shuffle(seed=42).select(range(max_samples_per_dataset))
                
                # Process items
                for idx, item in enumerate(dataset):
                    processed = {
                        "dataset": ds_key,
                        "dataset_name": ds_config["name"],
                        "index": idx,
                        "data": item,
                    }
                    all_items.append(processed)
                
                logger.info(f"  Loaded {len(dataset)} samples from {ds_key}")
                self.datasets_loaded[ds_key] = len(dataset)
                
            except Exception as e:
                logger.error(f"Failed to load {ds_key}: {e}")
                self.errors.append({"dataset": ds_key, "error": str(e)})
        
        logger.info(f"Total samples loaded: {len(all_items)}")
        return all_items
    
    def item_to_scraped(self, item: Dict) -> ScrapedItem:
        """Convert dataset item to ScrapedItem."""
        data = item["data"]
        ds_key = item["dataset"]
        
        # Extract question/answer based on dataset format
        if ds_key == "pubmedqa":
            title = data.get("question", "")
            content = json.dumps({
                "question": data.get("question", ""),
                "context": data.get("context", {}).get("contexts", []),
                "long_answer": data.get("long_answer", ""),
                "final_decision": data.get("final_decision", ""),
            }, ensure_ascii=False)
        
        elif ds_key in ["medmcqa", "medqa"]:
            title = data.get("question", "")[:200]
            content = json.dumps({
                "question": data.get("question", ""),
                "options": data.get("options", data.get("answer_idx", {})),
                "answer": data.get("answer", data.get("answer_idx", "")),
                "explanation": data.get("exp", data.get("explanation", "")),
            }, ensure_ascii=False)
        
        elif "meadow" in ds_key:
            title = data.get("input", data.get("instruction", ""))[:200]
            content = json.dumps({
                "instruction": data.get("instruction", ""),
                "input": data.get("input", ""),
                "output": data.get("output", ""),
            }, ensure_ascii=False)
        
        elif ds_key == "chatdoctor":
            title = data.get("instruction", data.get("input", ""))[:200]
            content = json.dumps({
                "instruction": data.get("instruction", ""),
                "input": data.get("input", ""),
                "output": data.get("output", ""),
            }, ensure_ascii=False)
        
        else:
            title = str(data)[:200]
            content = json.dumps(data, ensure_ascii=False, default=str)
        
        return ScrapedItem(
            id=f"{ds_key}_{item['index']}",
            title=title,
            content=content,
            source=ds_key,
            metadata={
                "dataset_name": item["dataset_name"],
                "index": item["index"],
            }
        )
    
    def save_by_dataset(self):
        """Save items grouped by dataset."""
        from collections import defaultdict
        
        by_dataset = defaultdict(list)
        for item in self.items:
            by_dataset[item["dataset"]].append(item)
        
        for ds_key, items in by_dataset.items():
            output_path = self.output_dir / f"{ds_key}.jsonl"
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in items:
                    scraped = self.item_to_scraped(item)
                    f.write(scraped.to_json() + '\n')
            logger.info(f"Saved {len(items)} items to {output_path}")


async def main():
    """Run medical dataset scraper."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download medical datasets from HuggingFace")
    parser.add_argument("--output", default="data/raw/medical_datasets", help="Output directory")
    parser.add_argument("--datasets", nargs="+", help="Specific datasets to download")
    parser.add_argument("--max-samples", type=int, help="Max samples per dataset")
    args = parser.parse_args()
    
    scraper = MedicalDatasetScraper(output_dir=args.output)
    result = await scraper.run(
        dataset_names=args.datasets,
        max_samples_per_dataset=args.max_samples
    )
    
    # Also save by dataset
    scraper.save_by_dataset()
    
    print(f"\nResults: {result}")
    print(f"Datasets loaded: {scraper.datasets_loaded}")


if __name__ == "__main__":
    asyncio.run(main())
