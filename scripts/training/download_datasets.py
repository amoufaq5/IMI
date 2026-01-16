"""
UMI Medical Dataset Downloader
Downloads open-source medical datasets for training
"""

import os
import subprocess
import sys
from pathlib import Path


def create_directories():
    """Create data directories."""
    dirs = [
        "data/raw/pubmedqa",
        "data/raw/medqa",
        "data/raw/medmcqa",
        "data/raw/drugbank",
        "data/training",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"Created: {d}")


def download_pubmedqa():
    """Download PubMedQA dataset."""
    print("\n" + "=" * 50)
    print("Downloading PubMedQA...")
    print("=" * 50)
    
    url = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
    output = "data/raw/pubmedqa/ori_pqal.json"
    
    try:
        import requests
        response = requests.get(url)
        response.raise_for_status()
        
        with open(output, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"Downloaded: {output}")
    except Exception as e:
        print(f"Failed to download PubMedQA: {e}")
        print("Manual download: https://pubmedqa.github.io/")


def download_medqa():
    """Download MedQA dataset."""
    print("\n" + "=" * 50)
    print("Downloading MedQA...")
    print("=" * 50)
    
    print("MedQA requires manual download due to size.")
    print("Instructions:")
    print("1. Go to: https://github.com/jind11/MedQA")
    print("2. Download the dataset")
    print("3. Place train.jsonl in data/raw/medqa/")


def download_medmcqa():
    """Download MedMCQA dataset."""
    print("\n" + "=" * 50)
    print("Downloading MedMCQA...")
    print("=" * 50)
    
    try:
        from datasets import load_dataset
        
        dataset = load_dataset("medmcqa", split="train")
        
        output = "data/raw/medmcqa/train.json"
        dataset.to_json(output)
        
        print(f"Downloaded: {output}")
        print(f"Total samples: {len(dataset)}")
    except Exception as e:
        print(f"Failed to download MedMCQA: {e}")
        print("Manual download: https://medmcqa.github.io/")


def download_medical_meadow():
    """Download Medical Meadow dataset (curated medical QA)."""
    print("\n" + "=" * 50)
    print("Downloading Medical Meadow...")
    print("=" * 50)
    
    try:
        from datasets import load_dataset
        
        # Medical Meadow is a curated collection
        dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")
        
        output = "data/raw/medical_meadow/flashcards.json"
        Path("data/raw/medical_meadow").mkdir(parents=True, exist_ok=True)
        dataset.to_json(output)
        
        print(f"Downloaded: {output}")
        print(f"Total samples: {len(dataset)}")
    except Exception as e:
        print(f"Failed to download Medical Meadow: {e}")


def download_chatdoctor():
    """Download ChatDoctor dataset."""
    print("\n" + "=" * 50)
    print("Downloading ChatDoctor Dataset...")
    print("=" * 50)
    
    try:
        from datasets import load_dataset
        
        dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")
        
        output = "data/raw/chatdoctor/healthcare_magic.json"
        Path("data/raw/chatdoctor").mkdir(parents=True, exist_ok=True)
        dataset.to_json(output)
        
        print(f"Downloaded: {output}")
        print(f"Total samples: {len(dataset)}")
    except Exception as e:
        print(f"Failed to download ChatDoctor: {e}")


def print_drugbank_instructions():
    """Print instructions for DrugBank."""
    print("\n" + "=" * 50)
    print("DrugBank Setup Instructions")
    print("=" * 50)
    print("""
DrugBank requires registration for download:

1. Go to: https://go.drugbank.com/releases/latest
2. Create a free academic account (or purchase commercial license)
3. Download "All Drug Vocabulary" (CSV)
4. Place the file in: data/raw/drugbank/drugbank_vocabulary.csv

Alternative: Use open drug databases:
- RxNorm: https://www.nlm.nih.gov/research/umls/rxnorm/
- OpenFDA: https://open.fda.gov/apis/drug/
""")


def main():
    """Download all available datasets."""
    print("=" * 60)
    print("UMI Medical Dataset Downloader")
    print("=" * 60)
    
    # Create directories
    create_directories()
    
    # Download datasets
    download_pubmedqa()
    download_medmcqa()
    download_medical_meadow()
    download_chatdoctor()
    
    # Print manual download instructions
    download_medqa()
    print_drugbank_instructions()
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Complete any manual downloads listed above")
    print("2. Run: python scripts/training/prepare_data.py")
    print("3. Run: python scripts/training/fine_tune.py")


if __name__ == "__main__":
    main()
