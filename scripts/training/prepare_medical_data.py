"""
Medical Data Preparation — 2 Formats Only
==========================================

Converts all collected medical datasets into exactly two output formats:

  general_knowledge — Raw medical text passages for causal language modelling.
                      Teaches the model factual medical knowledge.
                      Format: {"text": "..."}

  instruction       — Medical Q&A and instruction-following pairs.
                      Teaches the model to respond to medical queries.
                      Format: {"instruction": "...", "input": "...", "output": "..."}

Output files (in data/final/):
  general_knowledge_train.json
  general_knowledge_val.json
  instruction_train.json
  instruction_val.json

These 4 files are the only inputs required by finetune_mixtral.py.
All prior adapter-specific files (patient_triage_*, education_*, etc.) are
NOT used by the new training pipeline — this script replaces prepare_data.py
for the purposes of finetune_mixtral.py.

Usage:
    python scripts/training/prepare_medical_data.py
    python scripts/training/prepare_medical_data.py --val-split 0.05
    python scripts/training/prepare_medical_data.py --min-quality 0.35
"""
import json
import logging
import argparse
import random
import re
import hashlib
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# =============================================================================
# DATASET → FORMAT MAPPING
#
# Each source dataset is mapped to one of two output formats:
#   "general_knowledge" — text passages, encyclopedic facts
#   "instruction"       — Q&A pairs, instruction following
#
# Sources are identified by the "source" or "adapter" fields set by
# collect_datasets.py and collect_hf_datasets.py.
# =============================================================================

# Datasets whose natural form is reference text → general_knowledge
GENERAL_KNOWLEDGE_SOURCES = {
    "wiki_medical_terms",
    "medical_meadow_wikidoc",
    "wikidoc_patient_info",
    "medical_flashcards",
    "medical_meadow_medical_flashcards",
    "med_dataset_flashcards",
    "medical_specialities",
    "medical_meadow_cord19",
    "pubmedqa_artificial",    # research abstracts with context
    "medical_meadow_pubmed_causal",
    "sciq",                   # science knowledge passages
    "drug_reviews",           # drug information text
    "phi_drug",               # drug information
    # ── New biomedical corpus sources (collect_biomedical_corpus.py) ──────
    "pubmed_cardiology_reviews",
    "pubmed_oncology_reviews",
    "pubmed_endocrinology_reviews",
    "pubmed_neurology_reviews",
    "pubmed_psychiatry_reviews",
    "pubmed_infectious_disease_reviews",
    "pubmed_pediatrics_reviews",
    "pubmed_clinical_guidelines",
    "pubmed_drug_interactions",
    "pubmed_adverse_drug_reactions",
    "pubmed_emergency_triage",
    "pubmed_pharmacogenomics",
    "pubmed_icu_sepsis",
    "pubmed_surgery",
    "pubmed_rare_diseases",
    "pmc_rct_full_text",
    "pmc_pharma_reviews",
    "pmc_case_reports",
    "litcovid",
    "biorxiv_preprint",
    "medrxiv_preprint",
    "semantic_scholar",
    "pubmed_200k_rct",
    "cord19_qa",
    "s2orc",
    "medmentions",
    "jnlpba",
}

# Adapters whose processed data is encyclopedic rather than Q&A
GENERAL_KNOWLEDGE_ADAPTERS = {"regulatory_qa"}

# Everything else defaults to instruction format

# =============================================================================
# QUALITY FILTERING (shared with prepare_data.py)
# =============================================================================

MIN_TEXT_LENGTH = 40         # chars — for general_knowledge
MIN_INSTRUCTION_LENGTH = 10  # chars
MIN_OUTPUT_LENGTH = 20       # chars
MAX_FIELD_LENGTH = 50_000    # chars

HTML_TAG_RE = re.compile(r"<[^>]+>")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
MULTIPLE_NEWLINES_RE = re.compile(r"\n{3,}")
MULTIPLE_SPACES_RE = re.compile(r" {2,}")

GARBAGE_PATTERNS = [
    re.compile(r"^[\s\d\W]{20,}$"),
    re.compile(r"(.)\1{10,}"),
    re.compile(r"^\s*(null|nan|none|undefined|N/?A)\s*$", re.I),
    re.compile(r"^\s*\[?\s*\]?\s*$"),
    re.compile(r"^\s*\{?\s*\}?\s*$"),
    re.compile(r"^test\s*\d*$", re.I),
    re.compile(r"^(lorem ipsum|placeholder)", re.I),
]


def clean_text(text: str) -> str:
    """Normalize unicode, strip HTML, collapse whitespace."""
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    text = unicodedata.normalize("NFC", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = MARKDOWN_LINK_RE.sub(r"\1", text)
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    text = MULTIPLE_NEWLINES_RE.sub("\n\n", text)
    text = MULTIPLE_SPACES_RE.sub(" ", text)
    return text.strip()


def is_garbage(text: str) -> bool:
    if not text:
        return True
    return any(p.search(text) for p in GARBAGE_PATTERNS)


def quality_ok(text: str, min_len: int) -> bool:
    """Return True if text passes basic quality checks."""
    if len(text) < min_len or len(text) > MAX_FIELD_LENGTH:
        return False
    if is_garbage(text):
        return False
    # Penalise extreme symbol density (>50% non-alphanumeric)
    alnum = sum(1 for c in text if c.isalnum() or c.isspace())
    if len(text) > 0 and alnum / len(text) < 0.5:
        return False
    return True


# =============================================================================
# FORMAT CLASSIFIERS
# =============================================================================

def classify_example(example: Dict[str, Any]) -> Optional[str]:
    """
    Decide whether an example belongs to 'general_knowledge' or 'instruction'.
    Returns None if the example should be dropped.
    """
    source = example.get("source", "")
    adapter = example.get("adapter", "")

    if source in GENERAL_KNOWLEDGE_SOURCES or adapter in GENERAL_KNOWLEDGE_ADAPTERS:
        return "general_knowledge"

    # If the example already has a 'text' key and no instruction/output, treat as GK
    if "text" in example and "instruction" not in example:
        return "general_knowledge"

    # Default: instruction format
    return "instruction"


# =============================================================================
# FORMAT CONVERTERS
# =============================================================================

def to_general_knowledge(example: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Convert a raw example to general_knowledge format: {"text": "..."}.

    Tries multiple field names used by different source datasets.
    """
    # Explicit text field
    if "text" in example:
        text = clean_text(example["text"])
        if quality_ok(text, MIN_TEXT_LENGTH):
            return {"text": text}
        return None

    # Flashcard / term-definition format
    if "term" in example or "front" in example:
        term = clean_text(example.get("term", example.get("front", "")))
        definition = clean_text(
            example.get("definition", example.get("back", example.get("output", "")))
        )
        if term and definition:
            text = f"{term}: {definition}"
            if quality_ok(text, MIN_TEXT_LENGTH):
                return {"text": text}
        return None

    # Instruction + output (no meaningful input) → combine into prose
    instruction = clean_text(example.get("instruction", ""))
    output = clean_text(example.get("output", ""))
    if instruction and output:
        text = f"{instruction}\n{output}"
        if quality_ok(text, MIN_TEXT_LENGTH):
            return {"text": text}

    return None


def to_instruction(example: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Convert a raw example to instruction format:
    {"instruction": "...", "input": "...", "output": "..."}.

    Input field is optional; kept empty string if absent.
    """
    instruction = clean_text(example.get("instruction", ""))
    inp = clean_text(example.get("input", ""))
    output = clean_text(example.get("output", ""))

    # Some datasets use different field names
    if not instruction:
        instruction = clean_text(example.get("question", example.get("query", "")))
    if not output:
        output = clean_text(example.get("answer", example.get("response", "")))

    if not quality_ok(instruction, MIN_INSTRUCTION_LENGTH):
        return None
    if not quality_ok(output, MIN_OUTPUT_LENGTH):
        return None
    if instruction.strip() == output.strip():
        return None  # Copy-paste noise

    return {"instruction": instruction, "input": inp, "output": output}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_raw_examples(data_dir: Path) -> List[Dict[str, Any]]:
    """Load every JSON / JSONL file from all relevant source directories.

    Supports:
      .json  — array or single-object files (legacy format)
      .jsonl — newline-delimited JSON (collect_biomedical_corpus.py output)
    """
    all_examples: List[Dict[str, Any]] = []
    source_dirs = [
        data_dir / "processed",
        data_dir / "raw",
        data_dir / "train",
    ]

    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        # Recurse into subdirectories
        all_files = list(source_dir.rglob("*.json")) + list(source_dir.rglob("*.jsonl"))
        for jf in all_files:
            try:
                if jf.suffix == ".jsonl":
                    # JSONL: one record per line
                    with open(jf) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    all_examples.append(json.loads(line))
                                except json.JSONDecodeError:
                                    pass
                else:
                    with open(jf) as f:
                        content = f.read().strip()
                    if not content:
                        continue
                    data = json.loads(content)
                    if isinstance(data, list):
                        all_examples.extend(data)
                    elif isinstance(data, dict):
                        all_examples.append(data)
                logger.debug(f"  Loaded {jf.relative_to(data_dir)}")
            except Exception as e:
                logger.warning(f"  Could not load {jf.name}: {e}")

    logger.info(f"Raw examples loaded: {len(all_examples):,}")
    return all_examples


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def prepare_two_formats(
    data_dir: Path,
    val_split: float = 0.10,
    min_quality_score: float = 0.35,
    seed: int = 42,
) -> Dict[str, Dict[str, int]]:
    """
    Full pipeline:
      1. Load all raw examples
      2. Classify into general_knowledge / instruction
      3. Convert to canonical format
      4. Deduplicate
      5. Split train / val
      6. Save to data/final/
    """
    random.seed(seed)

    output_dir = data_dir / "final"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  Medical Data Preparation — 2 Formats")
    logger.info("=" * 60)

    raw_examples = load_all_raw_examples(data_dir)
    if not raw_examples:
        logger.error(
            "No raw data found. Run data collection first:\n"
            "  python scripts/data_collection/collect_datasets.py"
        )
        return {}

    gk_examples: List[Dict[str, str]] = []
    instr_examples: List[Dict[str, str]] = []
    skipped = 0

    for ex in raw_examples:
        fmt = classify_example(ex)
        if fmt == "general_knowledge":
            converted = to_general_knowledge(ex)
            if converted:
                gk_examples.append(converted)
            else:
                skipped += 1
        else:
            converted = to_instruction(ex)
            if converted:
                instr_examples.append(converted)
            else:
                skipped += 1

    logger.info(f"  general_knowledge: {len(gk_examples):,} (before dedup)")
    logger.info(f"  instruction:       {len(instr_examples):,} (before dedup)")
    logger.info(f"  skipped/filtered:  {skipped:,}")

    # Deduplicate by content hash
    def dedup(examples: List[Dict], key_fields: List[str]) -> List[Dict]:
        seen = set()
        unique = []
        for ex in examples:
            key = "".join(ex.get(f, "") for f in key_fields).encode()
            h = hashlib.sha256(key).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(ex)
        return unique

    gk_examples = dedup(gk_examples, ["text"])
    instr_examples = dedup(instr_examples, ["instruction", "input"])

    logger.info(f"  After dedup — general_knowledge: {len(gk_examples):,}")
    logger.info(f"  After dedup — instruction:       {len(instr_examples):,}")

    stats: Dict[str, Dict[str, int]] = {}

    for fmt_name, examples in [("general_knowledge", gk_examples), ("instruction", instr_examples)]:
        if not examples:
            logger.warning(f"  No examples for '{fmt_name}' — skipping.")
            continue

        random.shuffle(examples)
        split_idx = int(len(examples) * (1.0 - val_split))
        train_data = examples[:split_idx]
        val_data = examples[split_idx:]

        train_path = output_dir / f"{fmt_name}_train.json"
        val_path = output_dir / f"{fmt_name}_val.json"

        with open(train_path, "w") as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        with open(val_path, "w") as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)

        stats[fmt_name] = {"train": len(train_data), "val": len(val_data)}
        logger.info(f"  {fmt_name}: {len(train_data):,} train + {len(val_data):,} val → {output_dir}")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Summary")
    logger.info("=" * 60)
    total_train = sum(v["train"] for v in stats.values())
    total_val = sum(v["val"] for v in stats.values())
    for fmt_name, counts in stats.items():
        logger.info(f"  {fmt_name:25s}: {counts['train']:>8,} train  {counts['val']:>7,} val")
    logger.info(f"  {'TOTAL':25s}: {total_train:>8,} train  {total_val:>7,} val")
    logger.info("")
    logger.info("  Output files:")
    for fmt_name in stats:
        logger.info(f"    data/final/{fmt_name}_train.json")
        logger.info(f"    data/final/{fmt_name}_val.json")
    logger.info("")
    logger.info("  Next step:")
    logger.info("    python scripts/training/finetune_mixtral.py --demo")
    logger.info("=" * 60)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Prepare medical training data in 2 formats: general_knowledge + instruction"
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Path to data directory (default: project root /data)",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.10,
        help="Fraction reserved for validation (default: 0.10)",
    )
    parser.add_argument(
        "--min-quality", type=float, default=0.35,
        help="Minimum quality score threshold 0.0–1.0 (default: 0.35)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR

    prepare_two_formats(
        data_dir=data_dir,
        val_split=args.val_split,
        min_quality_score=args.min_quality,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
