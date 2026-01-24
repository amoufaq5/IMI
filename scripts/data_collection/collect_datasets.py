"""
Medical Dataset Collection Script

Downloads and prepares open medical datasets that don't require credentials/licenses:
- MedQA (USMLE questions)
- MedMCQA (Medical MCQs)
- PubMedQA (PubMed Q&A)
- HealthCareMagic-100k (Doctor-patient conversations)
- MTSamples (Medical transcriptions)
- Drug interactions from open sources
"""
import os
import json
import asyncio
import logging
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm
import zipfile
import tarfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base data directory
DATA_DIR = Path(__file__).parent.parent.parent / "data"


@dataclass
class DatasetConfig:
    """Configuration for a dataset"""
    name: str
    url: str
    description: str
    format: str  # json, csv, zip, tar.gz
    adapter_type: str  # Which adapter this data is for
    processing_fn: Optional[str] = None


# Open datasets that don't require credentials
DATASETS = [
    # ==================== EDUCATION / USMLE ====================
    DatasetConfig(
        name="medqa",
        url="https://huggingface.co/datasets/bigbio/med_qa/resolve/main/data/US/train.jsonl",
        description="USMLE-style medical questions (~10k)",
        format="jsonl",
        adapter_type="education",
    ),
    DatasetConfig(
        name="medmcqa",
        url="https://huggingface.co/datasets/medmcqa/resolve/main/data/train.json",
        description="Medical MCQ dataset from Indian medical exams (~180k)",
        format="json",
        adapter_type="education",
    ),
    DatasetConfig(
        name="medical_meadow_wikidoc",
        url="https://huggingface.co/datasets/medalpaca/medical_meadow_wikidoc/resolve/main/medical_meadow_wikidoc.json",
        description="WikiDoc medical articles Q&A (~10k)",
        format="json",
        adapter_type="education",
    ),
    DatasetConfig(
        name="medical_meadow_wikidoc_patient",
        url="https://huggingface.co/datasets/medalpaca/medical_meadow_wikidoc_patient_information/resolve/main/medical_meadow_wikidoc_patient_information.json",
        description="WikiDoc patient information Q&A (~5k)",
        format="json",
        adapter_type="education",
    ),
    DatasetConfig(
        name="medical_meadow_flashcards",
        url="https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards/resolve/main/medical_meadow_medical_flashcards.json",
        description="Medical flashcards for studying (~33k)",
        format="json",
        adapter_type="education",
    ),
    DatasetConfig(
        name="medquad",
        url="https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset/resolve/main/train.json",
        description="Medical Q&A from NIH websites (~16k)",
        format="json",
        adapter_type="education",
    ),
    
    # ==================== PATIENT TRIAGE / CONVERSATIONS ====================
    DatasetConfig(
        name="healthcaremagic",
        url="https://huggingface.co/datasets/wangrongsheng/HealthCareMagic-100k-en/resolve/main/HealthCareMagic-100k.json",
        description="Doctor-patient conversation dataset (~100k)",
        format="json",
        adapter_type="patient_triage",
    ),
    DatasetConfig(
        name="medical_meadow_mediqa",
        url="https://huggingface.co/datasets/medalpaca/medical_meadow_mediqa/resolve/main/medical_meadow_mediqa.json",
        description="Medical Q&A from consumer health questions (~2k)",
        format="json",
        adapter_type="patient_triage",
    ),
    DatasetConfig(
        name="chatdoctor",
        url="https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k/resolve/main/chatdoctor-healthcaremagic.json",
        description="ChatDoctor training conversations (~100k)",
        format="json",
        adapter_type="patient_triage",
    ),
    DatasetConfig(
        name="icliniq",
        url="https://huggingface.co/datasets/lavita/ChatDoctor-iCliniq/resolve/main/chatdoctor-icliniq.json",
        description="iCliniq doctor-patient conversations (~10k)",
        format="json",
        adapter_type="patient_triage",
    ),
    DatasetConfig(
        name="medical_meadow_health_advice",
        url="https://huggingface.co/datasets/medalpaca/medical_meadow_health_advice/resolve/main/medical_meadow_health_advice.json",
        description="Health advice conversations (~8k)",
        format="json",
        adapter_type="patient_triage",
    ),
    DatasetConfig(
        name="meddialog_en",
        url="https://huggingface.co/datasets/UCSD-AI4H/Medical-Dialogue-System/resolve/main/english/train.json",
        description="Medical dialogue system conversations (~200k+)",
        format="json",
        adapter_type="patient_triage",
    ),
    
    # ==================== RESEARCH / LITERATURE ====================
    DatasetConfig(
        name="pubmedqa",
        url="https://huggingface.co/datasets/pubmed_qa/resolve/main/pqa_labeled/train.json",
        description="PubMed question answering dataset (~1k labeled)",
        format="json",
        adapter_type="research",
    ),
    DatasetConfig(
        name="medical_meadow_pubmed_causal",
        url="https://huggingface.co/datasets/medalpaca/medical_meadow_pubmed_causal/resolve/main/medical_meadow_pubmed_causal.json",
        description="PubMed causal language modeling (~2.5M)",
        format="json",
        adapter_type="research",
    ),
    DatasetConfig(
        name="sciq",
        url="https://huggingface.co/datasets/allenai/sciq/resolve/main/data/train.json",
        description="Science questions including biology/medicine (~12k)",
        format="json",
        adapter_type="research",
    ),
    
    # ==================== CLINICAL / DOCTOR ====================
    DatasetConfig(
        name="medical_meadow_cord19",
        url="https://huggingface.co/datasets/medalpaca/medical_meadow_cord19/resolve/main/medical_meadow_cord19.json",
        description="COVID-19 research papers Q&A (~1k)",
        format="json",
        adapter_type="clinical_decision",
    ),
    DatasetConfig(
        name="medal_ner",
        url="https://huggingface.co/datasets/bigbio/medal/resolve/main/data/train.jsonl",
        description="Medical entity recognition abbreviations (~14M)",
        format="jsonl",
        adapter_type="clinical_decision",
    ),
    
    # ==================== PHARMACIST / DRUG INFO ====================
    DatasetConfig(
        name="drug_combo_extraction",
        url="https://huggingface.co/datasets/allenai/drug-combo-extraction/resolve/main/data/train.jsonl",
        description="Drug combination extraction from literature",
        format="jsonl",
        adapter_type="clinical_pharmacist",
    ),
    DatasetConfig(
        name="medical_meadow_openassistant",
        url="https://huggingface.co/datasets/medalpaca/medical_meadow_medqa/resolve/main/medical_meadow_medqa.json",
        description="Medical OpenAssistant conversations (~10k)",
        format="json",
        adapter_type="clinical_pharmacist",
    ),
    
    # ==================== MENTAL HEALTH ====================
    DatasetConfig(
        name="mental_health_counseling",
        url="https://huggingface.co/datasets/Amod/mental_health_counseling_conversations/resolve/main/data/train.json",
        description="Mental health counseling conversations (~3k)",
        format="json",
        adapter_type="patient_triage",
    ),
    DatasetConfig(
        name="counsel_chat",
        url="https://huggingface.co/datasets/nbertagnolli/counsel-chat/resolve/main/data/train.json",
        description="Counseling chat conversations (~2k)",
        format="json",
        adapter_type="patient_triage",
    ),
    
    # ==================== MULTILINGUAL (English subset) ====================
    DatasetConfig(
        name="medical_meadow_mmmlu",
        url="https://huggingface.co/datasets/medalpaca/medical_meadow_mmmlu/resolve/main/medical_meadow_mmmlu.json",
        description="Multilingual medical MMLU (~3k)",
        format="json",
        adapter_type="education",
    ),
]


class DatasetCollector:
    """Collects and processes medical datasets"""
    
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.raw_dir = data_dir / "raw"
        self.processed_dir = data_dir / "processed"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url: str, dest_path: Path, desc: str = "Downloading") -> bool:
        """Download a file with progress bar"""
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dest_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            return True
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return False
    
    def download_dataset(self, config: DatasetConfig) -> Optional[Path]:
        """Download a single dataset"""
        logger.info(f"Downloading {config.name}: {config.description}")
        
        # Determine file extension
        ext = config.format
        if ext == "jsonl":
            ext = "jsonl"
        elif ext == "json":
            ext = "json"
        
        dest_path = self.raw_dir / f"{config.name}.{ext}"
        
        if dest_path.exists():
            logger.info(f"  Already exists: {dest_path}")
            return dest_path
        
        success = self.download_file(config.url, dest_path, f"  {config.name}")
        
        if success:
            logger.info(f"  Saved to: {dest_path}")
            return dest_path
        return None
    
    def process_medqa(self, raw_path: Path) -> List[Dict[str, Any]]:
        """Process MedQA dataset into instruction format"""
        processed = []
        
        with open(raw_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                
                # Build question with options
                question = item.get('question', '')
                options = item.get('options', {})
                answer_idx = item.get('answer_idx', '')
                
                options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
                
                processed.append({
                    "instruction": f"Answer this USMLE-style medical question:\n\n{question}\n\n{options_text}",
                    "input": "",
                    "output": f"The correct answer is {answer_idx}. {options.get(answer_idx, '')}",
                    "source": "medqa",
                    "adapter": "education",
                })
        
        return processed
    
    def process_medmcqa(self, raw_path: Path) -> List[Dict[str, Any]]:
        """Process MedMCQA dataset"""
        processed = []
        
        with open(raw_path, 'r') as f:
            data = json.load(f)
        
        for item in data:
            question = item.get('question', '')
            options = [
                item.get('opa', ''),
                item.get('opb', ''),
                item.get('opc', ''),
                item.get('opd', ''),
            ]
            answer_idx = item.get('cop', 0)  # 1-indexed
            explanation = item.get('exp', '')
            
            options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
            answer_letter = chr(64 + answer_idx) if answer_idx else 'A'
            
            output = f"The correct answer is {answer_letter}."
            if explanation:
                output += f"\n\nExplanation: {explanation}"
            
            processed.append({
                "instruction": f"Answer this medical question:\n\n{question}\n\n{options_text}",
                "input": "",
                "output": output,
                "source": "medmcqa",
                "adapter": "education",
            })
        
        return processed
    
    def process_healthcaremagic(self, raw_path: Path) -> List[Dict[str, Any]]:
        """Process HealthCareMagic conversations"""
        processed = []
        
        with open(raw_path, 'r') as f:
            data = json.load(f)
        
        for item in data:
            instruction = item.get('instruction', item.get('input', ''))
            output = item.get('output', item.get('response', ''))
            
            if instruction and output:
                processed.append({
                    "instruction": "You are a helpful medical assistant. Answer the patient's question.",
                    "input": instruction,
                    "output": output,
                    "source": "healthcaremagic",
                    "adapter": "patient_triage",
                })
        
        return processed
    
    def process_pubmedqa(self, raw_path: Path) -> List[Dict[str, Any]]:
        """Process PubMedQA dataset"""
        processed = []
        
        with open(raw_path, 'r') as f:
            data = json.load(f)
        
        for key, item in data.items():
            question = item.get('QUESTION', '')
            context = " ".join(item.get('CONTEXTS', []))
            answer = item.get('final_decision', '')
            long_answer = item.get('LONG_ANSWER', '')
            
            output = f"Answer: {answer}"
            if long_answer:
                output += f"\n\nExplanation: {long_answer}"
            
            processed.append({
                "instruction": f"Based on the following research context, answer the question.\n\nContext: {context[:1000]}...",
                "input": question,
                "output": output,
                "source": "pubmedqa",
                "adapter": "research",
            })
        
        return processed
    
    def process_generic(self, raw_path: Path, adapter: str) -> List[Dict[str, Any]]:
        """Generic processor for instruction-format datasets"""
        processed = []
        
        with open(raw_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            for item in data:
                processed.append({
                    "instruction": item.get('instruction', ''),
                    "input": item.get('input', ''),
                    "output": item.get('output', ''),
                    "source": raw_path.stem,
                    "adapter": adapter,
                })
        
        return processed
    
    def process_dataset(self, config: DatasetConfig, raw_path: Path) -> List[Dict[str, Any]]:
        """Process a dataset based on its type"""
        logger.info(f"Processing {config.name}...")
        
        processors = {
            "medqa": self.process_medqa,
            "medmcqa": self.process_medmcqa,
            "healthcaremagic": self.process_healthcaremagic,
            "pubmedqa": self.process_pubmedqa,
        }
        
        processor = processors.get(config.name)
        if processor:
            return processor(raw_path)
        else:
            return self.process_generic(raw_path, config.adapter_type)
    
    def save_processed(self, data: List[Dict[str, Any]], name: str, adapter: str):
        """Save processed data"""
        # Save by adapter type
        adapter_dir = self.processed_dir / adapter
        adapter_dir.mkdir(exist_ok=True)
        
        output_path = adapter_dir / f"{name}.json"
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"  Saved {len(data)} examples to {output_path}")
    
    def collect_all(self):
        """Download and process all datasets"""
        logger.info("=" * 60)
        logger.info("IMI Medical Dataset Collection")
        logger.info("=" * 60)
        
        stats = {"downloaded": 0, "processed": 0, "failed": 0}
        
        for config in DATASETS:
            logger.info(f"\n--- {config.name} ---")
            
            # Download
            raw_path = self.download_dataset(config)
            if not raw_path:
                stats["failed"] += 1
                continue
            stats["downloaded"] += 1
            
            # Process
            try:
                processed = self.process_dataset(config, raw_path)
                if processed:
                    self.save_processed(processed, config.name, config.adapter_type)
                    stats["processed"] += 1
            except Exception as e:
                logger.error(f"  Failed to process: {e}")
                stats["failed"] += 1
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("Collection Summary")
        logger.info("=" * 60)
        logger.info(f"Downloaded: {stats['downloaded']}")
        logger.info(f"Processed: {stats['processed']}")
        logger.info(f"Failed: {stats['failed']}")
        
        # Show data by adapter
        logger.info("\nData by adapter type:")
        for adapter_dir in self.processed_dir.iterdir():
            if adapter_dir.is_dir():
                files = list(adapter_dir.glob("*.json"))
                total = 0
                for f in files:
                    with open(f) as fp:
                        total += len(json.load(fp))
                logger.info(f"  {adapter_dir.name}: {total} examples")
    
    def merge_by_adapter(self):
        """Merge all datasets by adapter type for training"""
        logger.info("\nMerging datasets by adapter type...")
        
        train_dir = self.data_dir / "train"
        train_dir.mkdir(exist_ok=True)
        
        for adapter_dir in self.processed_dir.iterdir():
            if adapter_dir.is_dir():
                all_data = []
                for f in adapter_dir.glob("*.json"):
                    with open(f) as fp:
                        all_data.extend(json.load(fp))
                
                if all_data:
                    # Shuffle
                    import random
                    random.shuffle(all_data)
                    
                    # Split train/val
                    split_idx = int(len(all_data) * 0.9)
                    train_data = all_data[:split_idx]
                    val_data = all_data[split_idx:]
                    
                    # Save
                    with open(train_dir / f"{adapter_dir.name}_train.json", 'w') as f:
                        json.dump(train_data, f)
                    with open(train_dir / f"{adapter_dir.name}_val.json", 'w') as f:
                        json.dump(val_data, f)
                    
                    logger.info(f"  {adapter_dir.name}: {len(train_data)} train, {len(val_data)} val")


def main():
    collector = DatasetCollector()
    collector.collect_all()
    collector.merge_by_adapter()


if __name__ == "__main__":
    main()
