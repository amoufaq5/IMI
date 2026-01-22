"""
Data Preparation Script

Combines all data sources and prepares them for training:
- Merges collected datasets
- Merges synthetic data
- Merges PDF-extracted data
- Creates train/val splits
- Formats for instruction tuning
"""
import json
import logging
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_json_files(directory: Path, pattern: str = "*.json") -> List[Dict[str, Any]]:
    """Load all JSON files from a directory"""
    all_data = []
    
    if not directory.exists():
        return all_data
    
    for json_file in directory.glob(pattern):
        try:
            with open(json_file) as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    all_data.append(data)
            logger.info(f"  Loaded {json_file.name}")
        except Exception as e:
            logger.warning(f"  Failed to load {json_file}: {e}")
    
    return all_data


def prepare_adapter_data(adapter_name: str) -> Dict[str, List[Dict]]:
    """Prepare all data for a specific adapter"""
    logger.info(f"\nPreparing data for: {adapter_name}")
    
    all_examples = []
    
    # 1. Collected datasets (processed)
    processed_dir = DATA_DIR / "processed" / adapter_name
    if processed_dir.exists():
        examples = load_json_files(processed_dir)
        all_examples.extend(examples)
        logger.info(f"  Processed data: {len(examples)} examples")
    
    # 2. Synthetic data
    synthetic_file = DATA_DIR / "synthetic" / f"{adapter_name}_synthetic.json"
    if synthetic_file.exists():
        with open(synthetic_file) as f:
            examples = json.load(f)
            all_examples.extend(examples)
            logger.info(f"  Synthetic data: {len(examples)} examples")
    
    # 3. PDF-extracted data (for regulatory_qa)
    if adapter_name == "regulatory_qa":
        pdf_file = DATA_DIR / "processed" / "regulatory_qa" / "regulatory_pdfs.json"
        if pdf_file.exists():
            with open(pdf_file) as f:
                examples = json.load(f)
                all_examples.extend(examples)
                logger.info(f"  PDF data: {len(examples)} examples")
    
    # 4. Train directory (merged data from collect_datasets.py)
    train_file = DATA_DIR / "train" / f"{adapter_name}_train.json"
    if train_file.exists():
        with open(train_file) as f:
            examples = json.load(f)
            all_examples.extend(examples)
            logger.info(f"  Train data: {len(examples)} examples")
    
    if not all_examples:
        logger.warning(f"  No data found for {adapter_name}")
        return {"train": [], "val": []}
    
    # Deduplicate by instruction hash
    seen = set()
    unique_examples = []
    for ex in all_examples:
        key = hash(ex.get("instruction", "") + ex.get("input", ""))
        if key not in seen:
            seen.add(key)
            unique_examples.append(ex)
    
    logger.info(f"  Total unique: {len(unique_examples)} examples")
    
    # Shuffle
    random.shuffle(unique_examples)
    
    # Split 90/10
    split_idx = int(len(unique_examples) * 0.9)
    
    return {
        "train": unique_examples[:split_idx],
        "val": unique_examples[split_idx:],
    }


def prepare_all_data(adapters: List[str] = None):
    """Prepare data for all adapters"""
    logger.info("=" * 60)
    logger.info("Data Preparation for Training")
    logger.info("=" * 60)
    
    if adapters is None:
        adapters = [
            "patient_triage",
            "clinical_pharmacist",
            "clinical_decision",
            "education",
            "regulatory_qa",
            "research",
        ]
    
    # Output directory
    output_dir = DATA_DIR / "final"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {}
    
    for adapter in adapters:
        data = prepare_adapter_data(adapter)
        
        # Save
        if data["train"]:
            train_path = output_dir / f"{adapter}_train.json"
            val_path = output_dir / f"{adapter}_val.json"
            
            with open(train_path, 'w') as f:
                json.dump(data["train"], f, indent=2)
            with open(val_path, 'w') as f:
                json.dump(data["val"], f, indent=2)
            
            stats[adapter] = {
                "train": len(data["train"]),
                "val": len(data["val"]),
            }
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Preparation Summary")
    logger.info("=" * 60)
    
    total_train = 0
    total_val = 0
    
    for adapter, counts in stats.items():
        logger.info(f"  {adapter}: {counts['train']} train, {counts['val']} val")
        total_train += counts["train"]
        total_val += counts["val"]
    
    logger.info(f"\n  Total: {total_train} train, {total_val} val")
    logger.info(f"  Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--adapters", nargs="+", help="Specific adapters to prepare")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    prepare_all_data(args.adapters)


if __name__ == "__main__":
    main()
