"""
UMI Ingested Data to Training Data Converter
Converts data from knowledge_base (scraped data) to training QA pairs
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm


@dataclass
class TrainingQAPair:
    """A single QA pair for training."""
    question: str
    answer: str
    category: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_chat_format(self) -> Dict[str, Any]:
        """Convert to chat format for fine-tuning."""
        return {
            "messages": [
                {"role": "user", "content": self.question},
                {"role": "assistant", "content": self.answer},
            ],
            "category": self.category,
            "source": self.source,
        }
    
    def to_instruction_format(self) -> Dict[str, str]:
        """Convert to instruction format."""
        return {
            "instruction": self.question,
            "input": "",
            "output": self.answer,
            "category": self.category,
        }


class PubMedConverter:
    """Convert PubMed articles to QA pairs."""
    
    QUESTION_TEMPLATES = [
        "What does research say about {topic}?",
        "Summarize the findings on {topic}.",
        "What are the key points about {topic} according to medical literature?",
        "Explain what studies show about {topic}.",
        "What is the current medical understanding of {topic}?",
    ]
    
    SPECIFIC_TEMPLATES = [
        ("What are the main findings of the study '{title}'?", "abstract"),
        ("Summarize the research paper: {title}", "abstract"),
        ("What does the article '{title}' conclude?", "abstract"),
    ]
    
    @classmethod
    def convert(cls, input_file: Path) -> List[TrainingQAPair]:
        """Convert PubMed JSONL to QA pairs."""
        pairs = []
        
        if not input_file.exists():
            print(f"PubMed file not found: {input_file}")
            return pairs
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Converting PubMed"):
                try:
                    article = json.loads(line)
                    pairs.extend(cls._article_to_qa(article))
                except json.JSONDecodeError:
                    continue
        
        return pairs
    
    @classmethod
    def _article_to_qa(cls, article: Dict) -> List[TrainingQAPair]:
        """Convert a single article to QA pairs."""
        pairs = []
        
        title = article.get("title", "")
        content = article.get("content", "")
        metadata = article.get("metadata", {})
        
        if not title or not content or len(content) < 100:
            return pairs
        
        # Extract topic from title
        topic = title.lower().replace(":", "").strip()
        
        # Generate general topic question
        template = random.choice(cls.QUESTION_TEMPLATES)
        question = template.format(topic=topic)
        
        # Create structured answer
        answer = f"Based on medical research:\n\n{content[:1500]}"
        
        if metadata.get("mesh_terms"):
            mesh = ", ".join(metadata["mesh_terms"][:5])
            answer += f"\n\nRelated topics: {mesh}"
        
        pairs.append(TrainingQAPair(
            question=question,
            answer=answer,
            category="research",
            source="PubMed",
            metadata={"pmid": metadata.get("pmid", "")},
        ))
        
        # Generate title-specific question
        specific_template, _ = random.choice(cls.SPECIFIC_TEMPLATES)
        specific_q = specific_template.format(title=title[:100])
        
        pairs.append(TrainingQAPair(
            question=specific_q,
            answer=content[:1500],
            category="research",
            source="PubMed",
            metadata={"pmid": metadata.get("pmid", "")},
        ))
        
        return pairs


class DrugConverter:
    """Convert drug data to QA pairs."""
    
    @classmethod
    def convert(cls, input_file: Path) -> List[TrainingQAPair]:
        """Convert drug JSONL to QA pairs."""
        pairs = []
        
        if not input_file.exists():
            print(f"Drug file not found: {input_file}")
            return pairs
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Converting Drugs"):
                try:
                    drug = json.loads(line)
                    pairs.extend(cls._drug_to_qa(drug))
                except json.JSONDecodeError:
                    continue
        
        return pairs
    
    @classmethod
    def _drug_to_qa(cls, drug: Dict) -> List[TrainingQAPair]:
        """Convert a single drug entry to QA pairs."""
        pairs = []
        
        name = drug.get("title", "")
        content = drug.get("content", "")
        metadata = drug.get("metadata", {})
        
        if not name or not content:
            return pairs
        
        # What is this drug?
        pairs.append(TrainingQAPair(
            question=f"What is {name}?",
            answer=content[:1200],
            category="drug",
            source="OpenFDA",
        ))
        
        # Indications
        indications = metadata.get("indications", [])
        if indications:
            ind_text = "\n".join([f"- {ind}" for ind in indications[:10]])
            pairs.append(TrainingQAPair(
                question=f"What is {name} used for? What are its indications?",
                answer=f"{name} is indicated for:\n\n{ind_text}",
                category="drug",
                source="OpenFDA",
            ))
        
        # Warnings
        warnings = metadata.get("warnings", [])
        if warnings:
            warn_text = "\n".join([f"- {w}" for w in warnings[:10]])
            pairs.append(TrainingQAPair(
                question=f"What are the warnings for {name}?",
                answer=f"Important warnings for {name}:\n\n{warn_text}\n\nAlways consult a healthcare provider before use.",
                category="drug",
                source="OpenFDA",
            ))
        
        # Contraindications
        contraindications = metadata.get("contraindications", [])
        if contraindications:
            contra_text = "\n".join([f"- {c}" for c in contraindications[:10]])
            pairs.append(TrainingQAPair(
                question=f"Who should not take {name}? What are the contraindications?",
                answer=f"{name} is contraindicated in:\n\n{contra_text}",
                category="drug",
                source="OpenFDA",
            ))
        
        # Side effects
        side_effects = metadata.get("side_effects", [])
        if side_effects:
            se_text = "\n".join([f"- {se}" for se in side_effects[:15]])
            pairs.append(TrainingQAPair(
                question=f"What are the side effects of {name}?",
                answer=f"Possible side effects of {name} include:\n\n{se_text}\n\nContact your doctor if you experience severe side effects.",
                category="drug",
                source="OpenFDA",
            ))
        
        return pairs


class ClinicalTrialsConverter:
    """Convert clinical trials data to QA pairs."""
    
    @classmethod
    def convert(cls, input_file: Path) -> List[TrainingQAPair]:
        """Convert clinical trials JSONL to QA pairs."""
        pairs = []
        
        if not input_file.exists():
            print(f"Clinical trials file not found: {input_file}")
            return pairs
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Converting Clinical Trials"):
                try:
                    trial = json.loads(line)
                    pairs.extend(cls._trial_to_qa(trial))
                except json.JSONDecodeError:
                    continue
        
        return pairs
    
    @classmethod
    def _trial_to_qa(cls, trial: Dict) -> List[TrainingQAPair]:
        """Convert a single trial to QA pairs."""
        pairs = []
        
        title = trial.get("title", "")
        content = trial.get("content", "")
        metadata = trial.get("metadata", {})
        
        if not title or not content:
            return pairs
        
        nct_id = metadata.get("nct_id", "")
        conditions = metadata.get("conditions", [])
        phase = metadata.get("phase", "")
        status = metadata.get("status", "")
        
        # General trial question
        if conditions:
            condition = conditions[0]
            pairs.append(TrainingQAPair(
                question=f"Are there any clinical trials for {condition}?",
                answer=f"Yes, there are clinical trials investigating treatments for {condition}.\n\nExample trial: {title}\n\n{content[:1000]}\n\nFor more information, search ClinicalTrials.gov with ID: {nct_id}",
                category="clinical_trial",
                source="ClinicalTrials.gov",
                metadata={"nct_id": nct_id},
            ))
        
        # Specific trial question
        pairs.append(TrainingQAPair(
            question=f"What is clinical trial {nct_id} about?",
            answer=content[:1200],
            category="clinical_trial",
            source="ClinicalTrials.gov",
            metadata={"nct_id": nct_id},
        ))
        
        return pairs


class RxNormConverter:
    """Convert RxNorm drug data to QA pairs."""
    
    @classmethod
    def convert(cls, input_file: Path) -> List[TrainingQAPair]:
        """Convert RxNorm JSONL to QA pairs."""
        pairs = []
        
        if not input_file.exists():
            print(f"RxNorm file not found: {input_file}")
            return pairs
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Converting RxNorm"):
                try:
                    drug = json.loads(line)
                    pairs.extend(cls._drug_to_qa(drug))
                except json.JSONDecodeError:
                    continue
        
        return pairs
    
    @classmethod
    def _drug_to_qa(cls, drug: Dict) -> List[TrainingQAPair]:
        """Convert RxNorm drug to QA pairs."""
        pairs = []
        
        name = drug.get("title", "")
        content = drug.get("content", "")
        metadata = drug.get("metadata", {})
        
        if not name or not content:
            return pairs
        
        # Drug information
        pairs.append(TrainingQAPair(
            question=f"Tell me about the medication {name}.",
            answer=content[:1200],
            category="drug",
            source="RxNorm",
        ))
        
        # Drug class
        drug_classes = metadata.get("drug_classes", [])
        if drug_classes:
            classes_text = ", ".join(drug_classes[:5])
            pairs.append(TrainingQAPair(
                question=f"What drug class does {name} belong to?",
                answer=f"{name} belongs to the following drug class(es): {classes_text}",
                category="drug",
                source="RxNorm",
            ))
        
        # Brand names
        brand_names = metadata.get("brand_names", [])
        if brand_names:
            brands = ", ".join(brand_names[:10])
            pairs.append(TrainingQAPair(
                question=f"What are the brand names for {name}?",
                answer=f"{name} is available under these brand names: {brands}",
                category="drug",
                source="RxNorm",
            ))
        
        return pairs


class ICDConverter:
    """Convert ICD-10 codes to QA pairs."""
    
    @classmethod
    def convert(cls, input_file: Path) -> List[TrainingQAPair]:
        """Convert ICD JSONL to QA pairs."""
        pairs = []
        
        if not input_file.exists():
            print(f"ICD file not found: {input_file}")
            return pairs
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Converting ICD-10"):
                try:
                    code = json.loads(line)
                    pairs.extend(cls._code_to_qa(code))
                except json.JSONDecodeError:
                    continue
        
        return pairs
    
    @classmethod
    def _code_to_qa(cls, code_entry: Dict) -> List[TrainingQAPair]:
        """Convert ICD code to QA pairs."""
        pairs = []
        
        title = code_entry.get("title", "")
        content = code_entry.get("content", "")
        metadata = code_entry.get("metadata", {})
        
        if not title or not content:
            return pairs
        
        icd_code = metadata.get("code", "")
        category = metadata.get("category", "")
        
        # What is this code?
        pairs.append(TrainingQAPair(
            question=f"What is ICD-10 code {icd_code}?",
            answer=content,
            category="diagnosis",
            source="ICD-10",
        ))
        
        # Code lookup by description
        description = metadata.get("description", title)
        pairs.append(TrainingQAPair(
            question=f"What is the ICD-10 code for {description.lower()}?",
            answer=f"The ICD-10 code for {description} is {icd_code}.\n\n{content}",
            category="diagnosis",
            source="ICD-10",
        ))
        
        return pairs


<<<<<<< HEAD
=======
class GenericJSONLConverter:
    """Generic converter for any JSONL data source with standard format."""
    
    @classmethod
    def convert(cls, input_file: Path, source_name: str, category: str) -> List[TrainingQAPair]:
        """Convert any JSONL file with id/title/content/metadata format."""
        pairs = []
        
        if not input_file.exists():
            return pairs
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    pairs.extend(cls._item_to_qa(item, source_name, category))
                except json.JSONDecodeError:
                    continue
        
        return pairs
    
    @classmethod
    def _item_to_qa(cls, item: Dict, source_name: str, category: str) -> List[TrainingQAPair]:
        """Convert a single item to QA pairs."""
        pairs = []
        
        title = item.get("title", "")
        content = item.get("content", "")
        
        if not title or not content or len(content) < 50:
            return pairs
        
        # Generate a question based on the title
        pairs.append(TrainingQAPair(
            question=f"What can you tell me about {title}?",
            answer=content[:1500],
            category=category,
            source=source_name,
        ))
        
        return pairs


>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
class IngestedDataConverter:
    """
    Main converter that processes all ingested data sources
    and creates training-ready QA pairs.
<<<<<<< HEAD
=======
    Handles errors gracefully - skips sources that fail.
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
    """
    
    def __init__(
        self,
        knowledge_base_dir: str = "data/knowledge_base",
        output_dir: str = "data/training/from_ingestion",
    ):
        self.kb_dir = Path(knowledge_base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.all_pairs: List[TrainingQAPair] = []
    
<<<<<<< HEAD
    def convert_all(self) -> None:
        """Convert all ingested data sources."""
=======
    def _safe_convert(self, converter_func, *args, source_name: str = "Unknown") -> List[TrainingQAPair]:
        """Safely run a converter, catching any errors."""
        try:
            return converter_func(*args)
        except Exception as e:
            print(f"  WARNING: Error converting {source_name}: {e}")
            return []
    
    def convert_all(self) -> None:
        """Convert all ingested data sources with error handling."""
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
        print("=" * 60)
        print("Converting Ingested Data to Training Format")
        print("=" * 60)
        
        # PubMed
        pubmed_file = self.kb_dir / "pubmed" / "articles.jsonl"
        if pubmed_file.exists():
<<<<<<< HEAD
            pairs = PubMedConverter.convert(pubmed_file)
=======
            pairs = self._safe_convert(PubMedConverter.convert, pubmed_file, source_name="PubMed")
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
            self.all_pairs.extend(pairs)
            print(f"PubMed: {len(pairs)} QA pairs")
        
        # Drugs (OpenFDA)
        drugs_file = self.kb_dir / "drugs" / "drugs.jsonl"
        if drugs_file.exists():
<<<<<<< HEAD
            pairs = DrugConverter.convert(drugs_file)
=======
            pairs = self._safe_convert(DrugConverter.convert, drugs_file, source_name="Drugs")
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
            self.all_pairs.extend(pairs)
            print(f"Drugs: {len(pairs)} QA pairs")
        
        # Clinical Trials
        trials_file = self.kb_dir / "clinical_trials" / "trials.jsonl"
        if trials_file.exists():
<<<<<<< HEAD
            pairs = ClinicalTrialsConverter.convert(trials_file)
=======
            pairs = self._safe_convert(ClinicalTrialsConverter.convert, trials_file, source_name="ClinicalTrials")
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
            self.all_pairs.extend(pairs)
            print(f"Clinical Trials: {len(pairs)} QA pairs")
        
        # RxNorm
        rxnorm_file = self.kb_dir / "rxnorm" / "drugs.jsonl"
        if rxnorm_file.exists():
<<<<<<< HEAD
            pairs = RxNormConverter.convert(rxnorm_file)
=======
            pairs = self._safe_convert(RxNormConverter.convert, rxnorm_file, source_name="RxNorm")
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
            self.all_pairs.extend(pairs)
            print(f"RxNorm: {len(pairs)} QA pairs")
        
        # ICD-10
        icd_file = self.kb_dir / "who" / "icd_codes.jsonl"
        if icd_file.exists():
<<<<<<< HEAD
            pairs = ICDConverter.convert(icd_file)
            self.all_pairs.extend(pairs)
            print(f"ICD-10: {len(pairs)} QA pairs")
        
=======
            pairs = self._safe_convert(ICDConverter.convert, icd_file, source_name="ICD-10")
            self.all_pairs.extend(pairs)
            print(f"ICD-10: {len(pairs)} QA pairs")
        
        # NEW SOURCES - Use generic converter for standard JSONL format
        
        # Kaggle datasets
        kaggle_file = self.kb_dir / "kaggle" / "datasets.jsonl"
        if kaggle_file.exists():
            pairs = self._safe_convert(
                GenericJSONLConverter.convert, kaggle_file, "Kaggle", "dataset",
                source_name="Kaggle"
            )
            self.all_pairs.extend(pairs)
            print(f"Kaggle: {len(pairs)} QA pairs")
        
        # MedlinePlus
        medlineplus_file = self.kb_dir / "medlineplus" / "topics.jsonl"
        if medlineplus_file.exists():
            pairs = self._safe_convert(
                GenericJSONLConverter.convert, medlineplus_file, "MedlinePlus", "health_info",
                source_name="MedlinePlus"
            )
            self.all_pairs.extend(pairs)
            print(f"MedlinePlus: {len(pairs)} QA pairs")
        
        # Open Targets - diseases
        opentargets_diseases = self.kb_dir / "opentargets" / "diseases.jsonl"
        if opentargets_diseases.exists():
            pairs = self._safe_convert(
                GenericJSONLConverter.convert, opentargets_diseases, "OpenTargets", "disease",
                source_name="OpenTargets Diseases"
            )
            self.all_pairs.extend(pairs)
            print(f"Open Targets Diseases: {len(pairs)} QA pairs")
        
        # Open Targets - drugs
        opentargets_drugs = self.kb_dir / "opentargets" / "drugs.jsonl"
        if opentargets_drugs.exists():
            pairs = self._safe_convert(
                GenericJSONLConverter.convert, opentargets_drugs, "OpenTargets", "drug",
                source_name="OpenTargets Drugs"
            )
            self.all_pairs.extend(pairs)
            print(f"Open Targets Drugs: {len(pairs)} QA pairs")
        
        # UMLS
        umls_file = self.kb_dir / "umls" / "concepts.jsonl"
        if umls_file.exists():
            pairs = self._safe_convert(
                GenericJSONLConverter.convert, umls_file, "UMLS", "terminology",
                source_name="UMLS"
            )
            self.all_pairs.extend(pairs)
            print(f"UMLS: {len(pairs)} QA pairs")
        
        # SNOMED CT
        snomed_file = self.kb_dir / "snomed" / "concepts.jsonl"
        if snomed_file.exists():
            pairs = self._safe_convert(
                GenericJSONLConverter.convert, snomed_file, "SNOMED CT", "terminology",
                source_name="SNOMED CT"
            )
            self.all_pairs.extend(pairs)
            print(f"SNOMED CT: {len(pairs)} QA pairs")
        
        # Orphanet
        orphanet_file = self.kb_dir / "orphanet" / "rare_diseases.jsonl"
        if orphanet_file.exists():
            pairs = self._safe_convert(
                GenericJSONLConverter.convert, orphanet_file, "Orphanet", "rare_disease",
                source_name="Orphanet"
            )
            self.all_pairs.extend(pairs)
            print(f"Orphanet: {len(pairs)} QA pairs")
        
        # DisGeNET
        disgenet_file = self.kb_dir / "disgenet" / "gene_disease_associations.jsonl"
        if disgenet_file.exists():
            pairs = self._safe_convert(
                GenericJSONLConverter.convert, disgenet_file, "DisGeNET", "gene_disease",
                source_name="DisGeNET"
            )
            self.all_pairs.extend(pairs)
            print(f"DisGeNET: {len(pairs)} QA pairs")
        
        # ChEMBL molecules
        chembl_molecules = self.kb_dir / "chembl" / "molecules.jsonl"
        if chembl_molecules.exists():
            pairs = self._safe_convert(
                GenericJSONLConverter.convert, chembl_molecules, "ChEMBL", "drug",
                source_name="ChEMBL Molecules"
            )
            self.all_pairs.extend(pairs)
            print(f"ChEMBL Molecules: {len(pairs)} QA pairs")
        
        # ChEMBL targets
        chembl_targets = self.kb_dir / "chembl" / "targets.jsonl"
        if chembl_targets.exists():
            pairs = self._safe_convert(
                GenericJSONLConverter.convert, chembl_targets, "ChEMBL", "target",
                source_name="ChEMBL Targets"
            )
            self.all_pairs.extend(pairs)
            print(f"ChEMBL Targets: {len(pairs)} QA pairs")
        
        # UniProt
        uniprot_file = self.kb_dir / "uniprot" / "proteins.jsonl"
        if uniprot_file.exists():
            pairs = self._safe_convert(
                GenericJSONLConverter.convert, uniprot_file, "UniProt", "protein",
                source_name="UniProt"
            )
            self.all_pairs.extend(pairs)
            print(f"UniProt: {len(pairs)} QA pairs")
        
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
        print(f"\nTotal QA pairs: {len(self.all_pairs)}")
    
    def deduplicate(self) -> None:
        """Remove duplicate questions."""
        import hashlib
        
        seen = set()
        unique = []
        
        for pair in self.all_pairs:
            q_hash = hashlib.md5(pair.question.lower().encode()).hexdigest()
            if q_hash not in seen:
                seen.add(q_hash)
                unique.append(pair)
        
        original = len(self.all_pairs)
        self.all_pairs = unique
        print(f"Deduplicated: {original} -> {len(self.all_pairs)} pairs")
    
    def save(self, format: str = "chat") -> Dict[str, str]:
        """Save converted data for training."""
        # Shuffle
        random.shuffle(self.all_pairs)
        
        # Split 90/5/5
        n = len(self.all_pairs)
        train_end = int(n * 0.9)
        val_end = train_end + int(n * 0.05)
        
        train = self.all_pairs[:train_end]
        val = self.all_pairs[train_end:val_end]
        test = self.all_pairs[val_end:]
        
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
            print(f"Saved {split_name}: {len(split_data)} pairs -> {output_path}")
        
        # Save stats
        stats = {
            "total_pairs": len(self.all_pairs),
            "train_size": len(train),
            "val_size": len(val),
            "test_size": len(test),
            "sources": {},
            "categories": {},
        }
        
        for pair in self.all_pairs:
            stats["sources"][pair.source] = stats["sources"].get(pair.source, 0) + 1
            stats["categories"][pair.category] = stats["categories"].get(pair.category, 0) + 1
        
        stats_path = self.output_dir / "stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return output_files


def main():
    """Run the conversion pipeline."""
    converter = IngestedDataConverter(
        knowledge_base_dir="data/knowledge_base",
        output_dir="data/training/from_ingestion",
    )
    
    # Convert all sources
    converter.convert_all()
    
    # Deduplicate
    converter.deduplicate()
    
    # Save
    output_files = converter.save(format="chat")
    
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)
    print(f"\nOutput files: {output_files}")
    print("\nTo use with fine-tuning, merge with other training data or use directly.")


if __name__ == "__main__":
    main()
