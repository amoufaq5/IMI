"""
IMI Data Processor - Convert scraped data to training format
Supports multiple output formats for SFT, DPO, and reasoning training
"""

import json
import random
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from tqdm import tqdm


@dataclass
class TrainingExample:
    """A single training example."""
    id: str
    messages: List[Dict[str, str]]  # [{"role": "user/assistant", "content": "..."}]
    category: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_chat_format(self) -> Dict[str, Any]:
        """Standard chat format for SFT."""
        return {
            "id": self.id,
            "messages": self.messages,
            "category": self.category,
            "source": self.source,
        }
    
    def to_sharegpt_format(self) -> Dict[str, Any]:
        """ShareGPT format (used by many fine-tuning frameworks)."""
        conversations = []
        for msg in self.messages:
            role = "human" if msg["role"] == "user" else "gpt"
            conversations.append({"from": role, "value": msg["content"]})
        
        return {
            "id": self.id,
            "conversations": conversations,
        }
    
    def to_alpaca_format(self) -> Dict[str, str]:
        """Alpaca instruction format."""
        instruction = ""
        input_text = ""
        output = ""
        
        for msg in self.messages:
            if msg["role"] == "user":
                if not instruction:
                    instruction = msg["content"]
                else:
                    input_text = msg["content"]
            elif msg["role"] == "assistant":
                output = msg["content"]
        
        return {
            "instruction": instruction,
            "input": input_text,
            "output": output,
        }


@dataclass 
class DPOExample:
    """A DPO (Direct Preference Optimization) training example."""
    id: str
    prompt: str
    chosen: str      # Preferred response
    rejected: str    # Non-preferred response
    category: str
    source: str
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
        }


class DataProcessor:
    """
    Process scraped medical data into training formats.
    Supports SFT, DPO, and Chain-of-Thought reasoning formats.
    """
    
    def __init__(
        self,
        raw_data_dir: str = "data/raw",
        output_dir: str = "data/processed",
    ):
        self.raw_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.examples: List[TrainingExample] = []
        self.dpo_examples: List[DPOExample] = []
    
    def load_jsonl(self, filepath: Path) -> List[Dict]:
        """Load JSONL file."""
        items = []
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return items
    
    def process_pubmed(self, max_samples: Optional[int] = None):
        """Process PubMed articles into QA pairs."""
        pubmed_file = self.raw_dir / "pubmed" / "data.jsonl"
        items = self.load_jsonl(pubmed_file)
        
        if max_samples:
            items = items[:max_samples]
        
        for item in tqdm(items, desc="Processing PubMed"):
            metadata = item.get("metadata", {})
            title = item.get("title", "")
            content = item.get("content", "")
            
            if not title or not content or len(content) < 100:
                continue
            
            # Generate QA pairs from article
            questions = [
                f"What does the research say about {title.lower().rstrip('.')}?",
                f"Summarize the findings from the study: {title}",
                f"What are the key conclusions about {title.lower().rstrip('.')}?",
            ]
            
            for i, question in enumerate(questions):
                example_id = f"pubmed_{item.get('id', '')}_{i}"
                
                # Create answer with reasoning structure
                answer = self._create_medical_answer(
                    question=question,
                    content=content,
                    source_type="research",
                    metadata=metadata,
                )
                
                self.examples.append(TrainingExample(
                    id=example_id,
                    messages=[
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ],
                    category="medical_research",
                    source="pubmed",
                    metadata=metadata,
                ))
    
    def process_medical_qa(self, max_samples: Optional[int] = None):
        """Process medical QA datasets (MedQA, MedMCQA, PubMedQA)."""
        datasets_dir = self.raw_dir / "medical_datasets"
        
        # Process each QA dataset
        for ds_name in ["medqa", "medmcqa", "pubmedqa"]:
            ds_file = datasets_dir / f"{ds_name}.jsonl"
            items = self.load_jsonl(ds_file)
            
            if max_samples:
                items = items[:max_samples]
            
            for item in tqdm(items, desc=f"Processing {ds_name}"):
                try:
                    content = json.loads(item.get("content", "{}"))
                except:
                    content = {}
                
                question = content.get("question", "")
                if not question:
                    continue
                
                # Build answer based on dataset type
                if ds_name in ["medqa", "medmcqa"]:
                    answer = self._process_mcq(content)
                else:  # pubmedqa
                    answer = self._process_pubmedqa(content)
                
                if not answer:
                    continue
                
                example_id = f"{ds_name}_{item.get('id', '')}"
                
                self.examples.append(TrainingExample(
                    id=example_id,
                    messages=[
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ],
                    category="medical_qa",
                    source=ds_name,
                ))
    
    def process_medical_instructions(self, max_samples: Optional[int] = None):
        """Process instruction-following medical datasets."""
        datasets_dir = self.raw_dir / "medical_datasets"
        
        instruction_datasets = [
            "medical_meadow_flashcards",
            "medical_meadow_wikidoc",
            "medical_meadow_wikidoc_patient",
            "chatdoctor",
        ]
        
        for ds_name in instruction_datasets:
            ds_file = datasets_dir / f"{ds_name}.jsonl"
            items = self.load_jsonl(ds_file)
            
            if max_samples:
                items = items[:max_samples]
            
            for item in tqdm(items, desc=f"Processing {ds_name}"):
                try:
                    content = json.loads(item.get("content", "{}"))
                except:
                    content = {}
                
                instruction = content.get("instruction", "")
                input_text = content.get("input", "")
                output = content.get("output", "")
                
                if not output or len(output) < 20:
                    continue
                
                # Combine instruction and input
                if input_text:
                    user_content = f"{instruction}\n\n{input_text}"
                else:
                    user_content = instruction
                
                if not user_content:
                    continue
                
                example_id = f"{ds_name}_{item.get('id', '')}"
                
                self.examples.append(TrainingExample(
                    id=example_id,
                    messages=[
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": output},
                    ],
                    category="medical_instruction",
                    source=ds_name,
                ))
    
    def _process_mcq(self, content: Dict) -> str:
        """Process multiple choice question into reasoning answer."""
        question = content.get("question", "")
        options = content.get("options", {})
        answer_key = content.get("answer", "")
        explanation = content.get("explanation", "")
        
        if not options or not answer_key:
            return ""
        
        # Build structured answer with reasoning
        answer_parts = []
        
        # Step 1: Analyze the question
        answer_parts.append("Let me analyze this medical question step by step.\n")
        
        # Step 2: Consider options
        answer_parts.append("**Analysis of options:**")
        if isinstance(options, dict):
            for key, value in options.items():
                answer_parts.append(f"- Option {key}: {value}")
        elif isinstance(options, list):
            for i, opt in enumerate(options):
                answer_parts.append(f"- Option {chr(65+i)}: {opt}")
        
        # Step 3: Reasoning
        if explanation:
            answer_parts.append(f"\n**Reasoning:**\n{explanation}")
        
        # Step 4: Conclusion
        if isinstance(answer_key, int) and isinstance(options, list):
            correct = options[answer_key] if answer_key < len(options) else str(answer_key)
        elif isinstance(options, dict):
            correct = options.get(str(answer_key), str(answer_key))
        else:
            correct = str(answer_key)
        
        answer_parts.append(f"\n**Answer:** {correct}")
        
        return "\n".join(answer_parts)
    
    def _process_pubmedqa(self, content: Dict) -> str:
        """Process PubMedQA into reasoning answer."""
        question = content.get("question", "")
        contexts = content.get("context", [])
        long_answer = content.get("long_answer", "")
        decision = content.get("final_decision", "")
        
        if not long_answer:
            return ""
        
        answer_parts = []
        
        # Include context summary if available
        if contexts and isinstance(contexts, list):
            answer_parts.append("Based on the medical literature:\n")
        
        # Main answer
        answer_parts.append(long_answer)
        
        # Final decision if available
        if decision:
            answer_parts.append(f"\n**Conclusion:** {decision}")
        
        return "\n".join(answer_parts)
    
    def _create_medical_answer(
        self,
        question: str,
        content: str,
        source_type: str,
        metadata: Dict,
    ) -> str:
        """Create a well-structured medical answer with reasoning."""
        # Extract abstract if present
        parts = content.split("\n\n", 1)
        title = parts[0] if parts else ""
        abstract = parts[1] if len(parts) > 1 else content
        
        # Truncate if too long
        if len(abstract) > 2000:
            abstract = abstract[:2000] + "..."
        
        answer_parts = []
        
        # Add source context
        journal = metadata.get("journal", "")
        if journal:
            answer_parts.append(f"According to research published in {journal}:\n")
        else:
            answer_parts.append("Based on medical research:\n")
        
        # Main content
        answer_parts.append(abstract)
        
        # Add MeSH terms as key topics if available
        mesh_terms = metadata.get("mesh_terms", [])
        if mesh_terms and len(mesh_terms) > 0:
            terms = ", ".join(mesh_terms[:5])
            answer_parts.append(f"\n**Key topics:** {terms}")
        
        return "\n".join(answer_parts)
    
    def create_dpo_examples(self, num_examples: int = 1000):
        """
        Create DPO examples from existing training data.
        Uses heuristics to create chosen/rejected pairs.
        """
        if len(self.examples) < 100:
            print("Not enough examples to create DPO pairs")
            return
        
        # Group by category
        by_category = defaultdict(list)
        for ex in self.examples:
            by_category[ex.category].append(ex)
        
        created = 0
        for category, examples in by_category.items():
            if len(examples) < 10:
                continue
            
            # Create DPO pairs within category
            for i in range(min(num_examples // len(by_category), len(examples) - 1)):
                ex1 = examples[i]
                
                # Get the user message and assistant response
                user_msg = ex1.messages[0]["content"] if ex1.messages else ""
                chosen = ex1.messages[1]["content"] if len(ex1.messages) > 1 else ""
                
                if not user_msg or not chosen or len(chosen) < 50:
                    continue
                
                # Create a "rejected" response (shorter, less detailed)
                rejected = self._create_rejected_response(chosen)
                
                self.dpo_examples.append(DPOExample(
                    id=f"dpo_{ex1.id}",
                    prompt=user_msg,
                    chosen=chosen,
                    rejected=rejected,
                    category=category,
                    source=ex1.source,
                ))
                created += 1
        
        print(f"Created {created} DPO examples")
    
    def _create_rejected_response(self, chosen: str) -> str:
        """Create a rejected (lower quality) response from chosen."""
        # Strategy: Create a shorter, less detailed version
        sentences = chosen.split(". ")
        
        if len(sentences) <= 2:
            # Just truncate
            return chosen[:len(chosen)//2] + "..."
        
        # Take only first 1-2 sentences
        rejected = ". ".join(sentences[:2]) + "."
        
        # Remove any structured elements
        rejected = rejected.replace("**", "")
        rejected = rejected.replace("*", "")
        
        return rejected
    
    def deduplicate(self):
        """Remove duplicate examples based on content hash."""
        seen = set()
        unique = []
        
        for ex in self.examples:
            # Hash based on user message
            content = ex.messages[0]["content"] if ex.messages else ""
            h = hashlib.md5(content.encode()).hexdigest()
            
            if h not in seen:
                seen.add(h)
                unique.append(ex)
        
        removed = len(self.examples) - len(unique)
        self.examples = unique
        print(f"Deduplicated: removed {removed} duplicates, {len(unique)} remaining")
    
    def split_data(
        self,
        train_ratio: float = 0.9,
        val_ratio: float = 0.05,
        test_ratio: float = 0.05,
        seed: int = 42,
    ) -> Tuple[List[TrainingExample], List[TrainingExample], List[TrainingExample]]:
        """Split data into train/val/test sets."""
        random.seed(seed)
        shuffled = self.examples.copy()
        random.shuffle(shuffled)
        
        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train = shuffled[:train_end]
        val = shuffled[train_end:val_end]
        test = shuffled[val_end:]
        
        return train, val, test
    
    def save(
        self,
        format: str = "chat",  # chat, sharegpt, alpaca
        include_dpo: bool = True,
    ):
        """Save processed data to files."""
        # Split data
        train, val, test = self.split_data()
        
        # Save SFT data
        for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
            output_path = self.output_dir / f"sft_{split_name}.jsonl"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for ex in split_data:
                    if format == "chat":
                        data = ex.to_chat_format()
                    elif format == "sharegpt":
                        data = ex.to_sharegpt_format()
                    else:
                        data = ex.to_alpaca_format()
                    
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
            
            print(f"Saved {len(split_data)} examples to {output_path}")
        
        # Save DPO data
        if include_dpo and self.dpo_examples:
            dpo_path = self.output_dir / "dpo_train.jsonl"
            with open(dpo_path, 'w', encoding='utf-8') as f:
                for ex in self.dpo_examples:
                    f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + '\n')
            print(f"Saved {len(self.dpo_examples)} DPO examples to {dpo_path}")
        
        # Save statistics
        stats = {
            "total_examples": len(self.examples),
            "train_examples": len(train),
            "val_examples": len(val),
            "test_examples": len(test),
            "dpo_examples": len(self.dpo_examples),
            "categories": dict(defaultdict(int, {
                ex.category: sum(1 for e in self.examples if e.category == ex.category)
                for ex in self.examples
            })),
            "sources": dict(defaultdict(int, {
                ex.source: sum(1 for e in self.examples if e.source == ex.source)
                for ex in self.examples
            })),
        }
        
        stats_path = self.output_dir / "stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved statistics to {stats_path}")
    
    def process_all(self, max_samples_per_source: Optional[int] = None):
        """Process all data sources."""
        print("=" * 60)
        print("IMI DATA PROCESSOR")
        print("=" * 60)
        
        # Process each source
        print("\n[1/3] Processing PubMed articles...")
        self.process_pubmed(max_samples=max_samples_per_source)
        
        print("\n[2/3] Processing Medical QA datasets...")
        self.process_medical_qa(max_samples=max_samples_per_source)
        
        print("\n[3/3] Processing Medical instruction datasets...")
        self.process_medical_instructions(max_samples=max_samples_per_source)
        
        # Deduplicate
        print("\nDeduplicating...")
        self.deduplicate()
        
        # Create DPO examples
        print("\nCreating DPO examples...")
        self.create_dpo_examples()
        
        print(f"\nTotal training examples: {len(self.examples)}")
        print(f"Total DPO examples: {len(self.dpo_examples)}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Process medical data for training")
    parser.add_argument("--raw-dir", default="data/raw", help="Raw data directory")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument("--format", choices=["chat", "sharegpt", "alpaca"], default="chat")
    parser.add_argument("--max-samples", type=int, help="Max samples per source")
    args = parser.parse_args()
    
    processor = DataProcessor(
        raw_data_dir=args.raw_dir,
        output_dir=args.output_dir,
    )
    
    processor.process_all(max_samples_per_source=args.max_samples)
    processor.save(format=args.format)
    
    print("\nData processing complete!")


if __name__ == "__main__":
    main()
