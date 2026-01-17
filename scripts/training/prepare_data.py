"""
UMI Medical Data Preparation Pipeline
Prepares medical datasets for fine-tuning Mistral on medical domain
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib

import pandas as pd
from tqdm import tqdm


@dataclass
class MedicalQAPair:
    """A single medical question-answer pair for training."""
    question: str
    answer: str
    category: str  # diagnosis, drug, procedure, general
    source: str
    difficulty: str = "medium"  # easy, medium, hard
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_instruction_format(self) -> Dict[str, str]:
        """Convert to instruction-following format for fine-tuning."""
        return {
            "instruction": self.question,
            "input": "",
            "output": self.answer,
            "category": self.category,
        }
    
    def to_chat_format(self) -> Dict[str, Any]:
        """Convert to chat format for Mistral fine-tuning."""
        return {
            "messages": [
                {"role": "user", "content": self.question},
                {"role": "assistant", "content": self.answer},
            ],
            "category": self.category,
        }


class MedicalDataSources:
    """
    Handlers for various open medical data sources.
    """
    
    @staticmethod
    def load_pubmedqa(data_path: str) -> List[MedicalQAPair]:
        """
        Load PubMedQA dataset.
        Source: https://pubmedqa.github.io/
        """
        pairs = []
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for pmid, item in tqdm(data.items(), desc="Loading PubMedQA"):
                question = item.get("QUESTION", "")
                context = " ".join(item.get("CONTEXTS", []))
                long_answer = item.get("LONG_ANSWER", "")
                
                if question and long_answer:
                    # Create contextual answer
                    answer = f"Based on medical literature:\n\n{long_answer}"
                    
                    pairs.append(MedicalQAPair(
                        question=question,
                        answer=answer,
                        category="research",
                        source="PubMedQA",
                        metadata={"pmid": pmid, "context": context[:500]},
                    ))
        except FileNotFoundError:
            print(f"PubMedQA file not found: {data_path}")
        
        return pairs
    
    @staticmethod
    def load_medqa(data_path: str) -> List[MedicalQAPair]:
        """
        Load MedQA dataset (USMLE-style questions).
        Source: https://github.com/jind11/MedQA
        """
        pairs = []
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Loading MedQA"):
                    item = json.loads(line)
                    
                    question = item.get("question", "")
                    options = item.get("options", {})
                    answer_idx = item.get("answer_idx", "")
                    
                    if question and answer_idx:
                        # Format as clinical reasoning
                        options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
                        correct_answer = options.get(answer_idx, "")
                        
                        formatted_q = f"{question}\n\nOptions:\n{options_text}"
                        formatted_a = f"The correct answer is {answer_idx}: {correct_answer}\n\nReasoning: This is based on clinical guidelines and medical evidence."
                        
                        pairs.append(MedicalQAPair(
                            question=formatted_q,
                            answer=formatted_a,
                            category="diagnosis",
                            source="MedQA",
                            difficulty="hard",
                        ))
        except FileNotFoundError:
            print(f"MedQA file not found: {data_path}")
        
        return pairs
    
    @staticmethod
    def load_medmcqa(data_path: str) -> List[MedicalQAPair]:
        """
        Load MedMCQA dataset (Indian medical entrance exams).
        Source: https://medmcqa.github.io/
        """
        pairs = []
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Loading MedMCQA"):
                    item = json.loads(line)
                    
                    question = item.get("question", "")
                    options = [item.get(f"op{c}", "") for c in "abcd"]
                    correct = item.get("cop", 0)  # 1-indexed
                    explanation = item.get("exp", "")
                    subject = item.get("subject_name", "general")
                    
                    if question and correct:
                        options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
                        correct_letter = chr(64 + correct)
                        correct_text = options[correct - 1] if correct <= len(options) else ""
                        
                        formatted_q = f"{question}\n\n{options_text}"
                        formatted_a = f"The answer is {correct_letter}: {correct_text}"
                        if explanation:
                            formatted_a += f"\n\nExplanation: {explanation}"
                        
                        pairs.append(MedicalQAPair(
                            question=formatted_q,
                            answer=formatted_a,
                            category=subject.lower(),
                            source="MedMCQA",
                            difficulty="medium",
                        ))
        except FileNotFoundError:
            print(f"MedMCQA file not found: {data_path}")
        
        return pairs
    
    @staticmethod
    def load_drugbank_qa(data_path: str) -> List[MedicalQAPair]:
        """
        Generate QA pairs from DrugBank data.
        Source: https://go.drugbank.com/releases/latest
        """
        pairs = []
        
        try:
            # Assuming CSV format with drug information
            df = pd.read_csv(data_path)
            
            for _, row in tqdm(df.iterrows(), desc="Processing DrugBank", total=len(df)):
                drug_name = row.get("name", "")
                description = row.get("description", "")
                indication = row.get("indication", "")
                mechanism = row.get("mechanism-of-action", "")
                
                if drug_name and description:
                    # Generate multiple QA pairs per drug
                    
                    # What is X?
                    pairs.append(MedicalQAPair(
                        question=f"What is {drug_name}?",
                        answer=description[:1000],
                        category="drug",
                        source="DrugBank",
                    ))
                    
                    # What is X used for?
                    if indication:
                        pairs.append(MedicalQAPair(
                            question=f"What is {drug_name} used for?",
                            answer=f"{drug_name} is indicated for: {indication[:800]}",
                            category="drug",
                            source="DrugBank",
                        ))
                    
                    # How does X work?
                    if mechanism:
                        pairs.append(MedicalQAPair(
                            question=f"How does {drug_name} work?",
                            answer=f"Mechanism of action: {mechanism[:800]}",
                            category="drug",
                            source="DrugBank",
                        ))
        except FileNotFoundError:
            print(f"DrugBank file not found: {data_path}")
        
        return pairs
    
    @staticmethod
    def generate_asmethod_training_data() -> List[MedicalQAPair]:
        """
        Generate synthetic ASMETHOD consultation training data.
        """
        pairs = []
        
        # Template-based generation for ASMETHOD protocol
        scenarios = [
            {
                "symptom": "headache",
                "age": 35,
                "duration": "2 days",
                "severity": "moderate",
                "other_symptoms": "sensitivity to light",
                "medications": "none",
                "history": "occasional migraines",
                "diagnosis": "Tension headache with possible migraine features",
                "recommendation": "OTC pain relief (paracetamol or ibuprofen), rest in dark room, stay hydrated",
                "urgency": "routine",
            },
            {
                "symptom": "chest pain",
                "age": 55,
                "duration": "30 minutes",
                "severity": "severe",
                "other_symptoms": "shortness of breath, sweating",
                "medications": "blood pressure medication",
                "history": "hypertension",
                "diagnosis": "Possible cardiac event - EMERGENCY",
                "recommendation": "Call 999 immediately. Chew aspirin if available. Do not drive yourself.",
                "urgency": "emergency",
            },
            {
                "symptom": "sore throat",
                "age": 28,
                "duration": "3 days",
                "severity": "mild",
                "other_symptoms": "runny nose, mild cough",
                "medications": "none",
                "history": "none significant",
                "diagnosis": "Viral upper respiratory infection (common cold)",
                "recommendation": "Rest, fluids, throat lozenges, paracetamol for discomfort. See GP if symptoms worsen or persist beyond 7 days.",
                "urgency": "routine",
            },
            {
                "symptom": "abdominal pain",
                "age": 42,
                "duration": "6 hours",
                "severity": "severe",
                "other_symptoms": "nausea, pain in right lower abdomen",
                "medications": "none",
                "history": "none",
                "diagnosis": "Possible appendicitis - requires urgent evaluation",
                "recommendation": "Go to A&E for evaluation. Do not eat or drink. Surgical assessment may be needed.",
                "urgency": "urgent",
            },
            {
                "symptom": "skin rash",
                "age": 8,
                "duration": "1 day",
                "severity": "mild",
                "other_symptoms": "itching, no fever",
                "medications": "none",
                "history": "eczema",
                "diagnosis": "Likely eczema flare or contact dermatitis",
                "recommendation": "Apply emollient cream, avoid scratching, antihistamine for itching. See GP if spreading or not improving in 3 days.",
                "urgency": "routine",
            },
        ]
        
        for scenario in scenarios:
            # Full consultation format
            question = f"""Patient presents with the following:
- Age: {scenario['age']} years old
- Main symptom: {scenario['symptom']}
- Duration: {scenario['duration']}
- Severity: {scenario['severity']}
- Other symptoms: {scenario['other_symptoms']}
- Current medications: {scenario['medications']}
- Medical history: {scenario['history']}

Using the ASMETHOD protocol, provide an assessment and recommendation."""

            answer = f"""**ASMETHOD Assessment**

**A - Age**: {scenario['age']} years old
**S - Self/Other**: Patient presenting for themselves
**M - Medications**: {scenario['medications']}
**E - Exact Symptoms**: {scenario['symptom'].capitalize()} - {scenario['severity']} severity
**T - Time/Duration**: {scenario['duration']}
**H - History**: {scenario['history']}
**O - Other Symptoms**: {scenario['other_symptoms']}
**D - Danger Signs**: {'RED FLAG - Seek immediate medical attention' if scenario['urgency'] == 'emergency' else 'No immediate danger signs identified'}

**Assessment**: {scenario['diagnosis']}

**Recommendation**: {scenario['recommendation']}

**Urgency Level**: {scenario['urgency'].upper()}

{'⚠️ IMPORTANT: This assessment indicates a potential emergency. Please seek immediate medical attention.' if scenario['urgency'] == 'emergency' else ''}

*Disclaimer: This is an AI-assisted assessment and does not replace professional medical advice.*"""

            pairs.append(MedicalQAPair(
                question=question,
                answer=answer,
                category="consultation",
                source="ASMETHOD_Synthetic",
                difficulty="medium",
                metadata={"urgency": scenario["urgency"]},
            ))
        
        return pairs


class DataPreparationPipeline:
    """
    Main pipeline for preparing medical training data.
    """
    
    def __init__(self, output_dir: str = "data/training"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.all_pairs: List[MedicalQAPair] = []
    
    def load_all_sources(self, data_dir: str = "data/raw", include_ingested: bool = True) -> None:
        """Load data from all available sources."""
        data_path = Path(data_dir)
        
        # Load each source if available
        sources = MedicalDataSources()
        
        # PubMedQA
        pubmedqa_path = data_path / "pubmedqa" / "ori_pqal.json"
        if pubmedqa_path.exists():
            self.all_pairs.extend(sources.load_pubmedqa(str(pubmedqa_path)))
        
        # MedQA
        medqa_path = data_path / "medqa" / "train.jsonl"
        if medqa_path.exists():
            self.all_pairs.extend(sources.load_medqa(str(medqa_path)))
        
        # MedMCQA
        medmcqa_path = data_path / "medmcqa" / "train.json"
        if medmcqa_path.exists():
            self.all_pairs.extend(sources.load_medmcqa(str(medmcqa_path)))
        
        # DrugBank
        drugbank_path = data_path / "drugbank" / "drugbank_vocabulary.csv"
        if drugbank_path.exists():
            self.all_pairs.extend(sources.load_drugbank_qa(str(drugbank_path)))
        
        # Always add ASMETHOD synthetic data
        self.all_pairs.extend(sources.generate_asmethod_training_data())
        
        # Load from ingested data (knowledge_base)
        if include_ingested:
            self._load_ingested_data()
        
        print(f"Total QA pairs loaded: {len(self.all_pairs)}")
    
    def _load_ingested_data(self, kb_dir: str = "data/knowledge_base") -> None:
        """Load QA pairs from ingested knowledge base data."""
        from convert_ingested_data import IngestedDataConverter
        
        kb_path = Path(kb_dir)
        if not kb_path.exists():
            print("Knowledge base directory not found, skipping ingested data")
            return
        
        try:
            converter = IngestedDataConverter(
                knowledge_base_dir=kb_dir,
                output_dir="data/training/from_ingestion",
            )
            converter.convert_all()
            
            # Convert to MedicalQAPair format
            for pair in converter.all_pairs:
                self.all_pairs.append(MedicalQAPair(
                    question=pair.question,
                    answer=pair.answer,
                    category=pair.category,
                    source=pair.source,
                    metadata=pair.metadata,
                ))
            
            print(f"Loaded {len(converter.all_pairs)} pairs from ingested data")
        except Exception as e:
            print(f"Error loading ingested data: {e}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep medical symbols
        text = re.sub(r'[^\w\s\-\.,;:?!°%/()\'\"μ±≤≥<>]', '', text)
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        return text.strip()
    
    def filter_quality(self, min_q_len: int = 20, min_a_len: int = 50) -> None:
        """Filter out low-quality pairs."""
        original_count = len(self.all_pairs)
        
        filtered = []
        for pair in self.all_pairs:
            # Length checks
            if len(pair.question) < min_q_len or len(pair.answer) < min_a_len:
                continue
            
            # Clean text
            pair.question = self.clean_text(pair.question)
            pair.answer = self.clean_text(pair.answer)
            
            # Skip if cleaning made it too short
            if len(pair.question) < min_q_len or len(pair.answer) < min_a_len:
                continue
            
            filtered.append(pair)
        
        self.all_pairs = filtered
        print(f"Filtered: {original_count} -> {len(self.all_pairs)} pairs")
    
    def deduplicate(self) -> None:
        """Remove duplicate entries."""
        seen = set()
        unique = []
        
        for pair in self.all_pairs:
            # Create hash of question
            q_hash = hashlib.md5(pair.question.lower().encode()).hexdigest()
            
            if q_hash not in seen:
                seen.add(q_hash)
                unique.append(pair)
        
        original = len(self.all_pairs)
        self.all_pairs = unique
        print(f"Deduplicated: {original} -> {len(self.all_pairs)} pairs")
    
    def split_data(
        self,
        train_ratio: float = 0.9,
        val_ratio: float = 0.05,
        test_ratio: float = 0.05,
    ) -> Tuple[List[MedicalQAPair], List[MedicalQAPair], List[MedicalQAPair]]:
        """Split data into train/val/test sets."""
        import random
        random.shuffle(self.all_pairs)
        
        n = len(self.all_pairs)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train = self.all_pairs[:train_end]
        val = self.all_pairs[train_end:val_end]
        test = self.all_pairs[val_end:]
        
        print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
        return train, val, test
    
    def save_for_training(
        self,
        format: str = "chat",  # "chat" or "instruction"
    ) -> Dict[str, str]:
        """Save processed data in format ready for fine-tuning."""
        train, val, test = self.split_data()
        
        output_files = {}
        
        for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
            output_path = self.output_dir / f"{split_name}.jsonl"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for pair in split_data:
                    if format == "chat":
                        data = pair.to_chat_format()
                    else:
                        data = pair.to_instruction_format()
                    
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
            
            output_files[split_name] = str(output_path)
            print(f"Saved {split_name}: {output_path}")
        
        # Save statistics
        stats = {
            "total_pairs": len(self.all_pairs),
            "train_size": len(train),
            "val_size": len(val),
            "test_size": len(test),
            "categories": {},
            "sources": {},
        }
        
        for pair in self.all_pairs:
            stats["categories"][pair.category] = stats["categories"].get(pair.category, 0) + 1
            stats["sources"][pair.source] = stats["sources"].get(pair.source, 0) + 1
        
        stats_path = self.output_dir / "stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return output_files


def main():
    """Run the data preparation pipeline."""
    print("=" * 60)
    print("UMI Medical Data Preparation Pipeline")
    print("=" * 60)
    
    pipeline = DataPreparationPipeline(output_dir="data/training")
    
    # Load all available data sources
    pipeline.load_all_sources(data_dir="data/raw")
    
    # Clean and filter
    pipeline.filter_quality()
    
    # Remove duplicates
    pipeline.deduplicate()
    
    # Save for training
    output_files = pipeline.save_for_training(format="chat")
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print(f"Output files: {output_files}")
    print("=" * 60)


if __name__ == "__main__":
    main()
