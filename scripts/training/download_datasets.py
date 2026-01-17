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


def download_bioasq():
    """Download BioASQ dataset for biomedical QA."""
    print("\n" + "=" * 50)
    print("Downloading BioASQ...")
    print("=" * 50)
    
    try:
        from datasets import load_dataset
        
        dataset = load_dataset("rag-datasets/rag-mini-bioasq", "question-answer-passages", split="test")
        
        output = "data/raw/bioasq/bioasq_qa.json"
        Path("data/raw/bioasq").mkdir(parents=True, exist_ok=True)
        dataset.to_json(output)
        
        print(f"Downloaded: {output}")
        print(f"Total samples: {len(dataset)}")
    except Exception as e:
        print(f"Failed to download BioASQ: {e}")
        print("Manual download: http://bioasq.org/")


def download_icd10_codes():
    """Download ICD-10 codes for diagnosis coding."""
    print("\n" + "=" * 50)
    print("Downloading ICD-10 Codes...")
    print("=" * 50)
    
    url = "https://raw.githubusercontent.com/kamillamagna/ICD-10-CSV/master/codes.csv"
    output = "data/raw/icd10/icd10_codes.csv"
    
    try:
        import requests
        Path("data/raw/icd10").mkdir(parents=True, exist_ok=True)
        
        response = requests.get(url)
        response.raise_for_status()
        
        with open(output, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"Downloaded: {output}")
    except Exception as e:
        print(f"Failed to download ICD-10: {e}")
        print("Manual download: https://www.cms.gov/medicare/coding/icd10")


def download_loinc_common():
    """Download common LOINC codes for lab interpretation."""
    print("\n" + "=" * 50)
    print("Downloading Common LOINC Codes...")
    print("=" * 50)
    
    print("LOINC requires registration for full download.")
    print("Instructions:")
    print("1. Go to: https://loinc.org/downloads/")
    print("2. Create a free account")
    print("3. Download 'LOINC Table Core'")
    print("4. Place in: data/raw/loinc/")
    
    # Create sample common lab codes
    common_labs = [
        {"code": "2345-7", "name": "Glucose", "unit": "mg/dL", "low": 70, "high": 100},
        {"code": "2160-0", "name": "Creatinine", "unit": "mg/dL", "low": 0.7, "high": 1.3},
        {"code": "3094-0", "name": "BUN", "unit": "mg/dL", "low": 7, "high": 20},
        {"code": "2951-2", "name": "Sodium", "unit": "mEq/L", "low": 136, "high": 145},
        {"code": "2823-3", "name": "Potassium", "unit": "mEq/L", "low": 3.5, "high": 5.0},
        {"code": "2075-0", "name": "Chloride", "unit": "mEq/L", "low": 98, "high": 106},
        {"code": "1963-8", "name": "Bicarbonate", "unit": "mEq/L", "low": 22, "high": 29},
        {"code": "17861-6", "name": "Calcium", "unit": "mg/dL", "low": 8.5, "high": 10.5},
        {"code": "718-7", "name": "Hemoglobin", "unit": "g/dL", "low": 12.0, "high": 17.5},
        {"code": "4544-3", "name": "Hematocrit", "unit": "%", "low": 36, "high": 50},
        {"code": "6690-2", "name": "WBC", "unit": "K/uL", "low": 4.5, "high": 11.0},
        {"code": "777-3", "name": "Platelets", "unit": "K/uL", "low": 150, "high": 400},
        {"code": "1742-6", "name": "ALT", "unit": "U/L", "low": 7, "high": 56},
        {"code": "1920-8", "name": "AST", "unit": "U/L", "low": 10, "high": 40},
        {"code": "1975-2", "name": "Bilirubin Total", "unit": "mg/dL", "low": 0.1, "high": 1.2},
        {"code": "2885-2", "name": "Total Protein", "unit": "g/dL", "low": 6.0, "high": 8.3},
        {"code": "1751-7", "name": "Albumin", "unit": "g/dL", "low": 3.5, "high": 5.0},
        {"code": "2532-0", "name": "LDH", "unit": "U/L", "low": 140, "high": 280},
        {"code": "2571-8", "name": "Triglycerides", "unit": "mg/dL", "low": 0, "high": 150},
        {"code": "2093-3", "name": "Total Cholesterol", "unit": "mg/dL", "low": 0, "high": 200},
        {"code": "2085-9", "name": "HDL Cholesterol", "unit": "mg/dL", "low": 40, "high": 999},
        {"code": "13457-7", "name": "LDL Cholesterol", "unit": "mg/dL", "low": 0, "high": 100},
        {"code": "4548-4", "name": "HbA1c", "unit": "%", "low": 4.0, "high": 5.6},
        {"code": "3016-3", "name": "TSH", "unit": "mIU/L", "low": 0.4, "high": 4.0},
        {"code": "14749-6", "name": "Free T4", "unit": "ng/dL", "low": 0.8, "high": 1.8},
    ]
    
    import json
    output = "data/raw/loinc/common_labs.json"
    Path("data/raw/loinc").mkdir(parents=True, exist_ok=True)
    
    with open(output, 'w') as f:
        json.dump(common_labs, f, indent=2)
    
    print(f"Created common lab reference: {output}")


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
    download_bioasq()
    download_icd10_codes()
    download_loinc_common()
    
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
