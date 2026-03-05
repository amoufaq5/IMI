"""
Data Preparation Script

Combines all open/verified data sources and prepares them for training:
- Merges collected datasets (HuggingFace, GitHub, CDC/CMS open data)
- Merges synthetic data
- Merges PDF-extracted data
- Validates and cleans data (removes garbage entries)
- Creates train/val splits
- Formats for instruction tuning

Data Quality Pipeline:
  1. Load raw data from all sources
  2. Validate structure (required fields present)
  3. Clean text (remove HTML, fix encoding, normalize whitespace)
  4. Filter garbage (too short, too long, non-medical, corrupt)
  5. Deduplicate (by instruction+input hash)
  6. Quality score and filter low-quality entries
  7. Split train/val
"""
import json
import logging
import argparse
import random
import re
import hashlib
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# ============================================================================
# DATA QUALITY CONSTANTS
# ============================================================================

# Minimum/maximum character lengths for valid fields
MIN_INSTRUCTION_LENGTH = 10
MIN_OUTPUT_LENGTH = 15
MAX_FIELD_LENGTH = 50_000  # 50K chars max per field

# Patterns that indicate garbage/corrupt data
GARBAGE_PATTERNS = [
    re.compile(r"^[\s\d\W]{20,}$"),                   # Only whitespace/numbers/symbols
    re.compile(r"(.)\1{10,}"),                          # Same character repeated 10+ times
    re.compile(r"^\s*null\s*$", re.I),                 # Literal "null"
    re.compile(r"^\s*nan\s*$", re.I),                  # Literal "nan"
    re.compile(r"^\s*none\s*$", re.I),                 # Literal "none"
    re.compile(r"^\s*undefined\s*$", re.I),            # Literal "undefined"
    re.compile(r"^\s*N/?A\s*$", re.I),                 # Literal "N/A"
    re.compile(r"^\s*\[?\s*\]?\s*$"),                  # Empty brackets
    re.compile(r"^\s*\{?\s*\}?\s*$"),                  # Empty braces
    re.compile(r"^test\s*\d*$", re.I),                 # Test entries
    re.compile(r"^(lorem ipsum|placeholder)", re.I),   # Placeholder text
]

# HTML/markup patterns to strip
HTML_TAG_RE = re.compile(r"<[^>]+>")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
MULTIPLE_NEWLINES_RE = re.compile(r"\n{3,}")
MULTIPLE_SPACES_RE = re.compile(r" {2,}")

# Characters that indicate encoding corruption
ENCODING_CORRUPTION_PATTERNS = [
    re.compile(r"[\ufffd]{3,}"),          # Unicode replacement chars
    re.compile(r"\\x[0-9a-f]{2}", re.I), # Escaped hex bytes
    re.compile(r"\u00c3[\u0080-\u00bf]{2,}"),  # Mojibake (UTF-8 read as Latin-1)
]


# ============================================================================
# TEXT CLEANING FUNCTIONS
# ============================================================================

def clean_text(text: str) -> str:
    """Clean a text field: fix encoding, strip HTML, normalize whitespace."""
    if not isinstance(text, str):
        return str(text) if text is not None else ""

    # Normalize Unicode (NFC form)
    text = unicodedata.normalize("NFC", text)

    # Strip HTML tags
    text = HTML_TAG_RE.sub(" ", text)

    # Convert markdown links to just the text
    text = MARKDOWN_LINK_RE.sub(r"\1", text)

    # Fix common encoding issues
    text = text.replace("\x00", "")       # Null bytes
    text = text.replace("\r\n", "\n")     # Normalize line endings
    text = text.replace("\r", "\n")
    text = text.replace("\t", " ")        # Tabs to spaces

    # Collapse excessive whitespace
    text = MULTIPLE_NEWLINES_RE.sub("\n\n", text)
    text = MULTIPLE_SPACES_RE.sub(" ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def is_garbage(text: str) -> bool:
    """Check if text matches garbage/corrupt patterns."""
    if not text:
        return True
    for pattern in GARBAGE_PATTERNS:
        if pattern.search(text):
            return True
    return False


def has_encoding_corruption(text: str) -> bool:
    """Check if text has encoding corruption artifacts."""
    for pattern in ENCODING_CORRUPTION_PATTERNS:
        if pattern.search(text):
            return True
    return False


def compute_quality_score(example: Dict[str, Any]) -> float:
    """
    Compute a quality score (0.0 - 1.0) for a training example.

    Factors:
    - Instruction clarity (length, not too short/long)
    - Output quality (length, not too short/long, informative)
    - No encoding corruption
    - Has proper sentence structure
    """
    score = 1.0

    instruction = example.get("instruction", "")
    output = example.get("output", "")
    inp = example.get("input", "")

    # Instruction quality
    if len(instruction) < 20:
        score -= 0.15
    elif len(instruction) > 10_000:
        score -= 0.10

    # Output quality
    if len(output) < 30:
        score -= 0.20
    elif len(output) > 30_000:
        score -= 0.10

    # Check for sentence structure (at least one period or question mark)
    if not re.search(r"[.?!]", output):
        score -= 0.10

    # Penalize encoding issues
    combined = instruction + inp + output
    if has_encoding_corruption(combined):
        score -= 0.30

    # Penalize excessive special characters (>30% non-alphanumeric)
    alnum_count = sum(1 for c in combined if c.isalnum() or c.isspace())
    if len(combined) > 0 and alnum_count / len(combined) < 0.5:
        score -= 0.25

    # Penalize if instruction == output (lazy copy-paste data)
    if instruction and output and instruction.strip() == output.strip():
        score -= 0.40

    return max(0.0, min(1.0, score))


# ============================================================================
# VALIDATION & FILTERING
# ============================================================================

def validate_example(example: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate a single training example.

    Returns:
        (is_valid, rejection_reason)
    """
    if not isinstance(example, dict):
        return False, "not_dict"

    instruction = example.get("instruction", "")
    output = example.get("output", "")

    # Required fields must exist
    if not instruction:
        return False, "missing_instruction"
    if not output:
        return False, "missing_output"

    # Type check
    if not isinstance(instruction, str) or not isinstance(output, str):
        return False, "non_string_field"

    # Length checks
    if len(instruction) < MIN_INSTRUCTION_LENGTH:
        return False, "instruction_too_short"
    if len(output) < MIN_OUTPUT_LENGTH:
        return False, "output_too_short"
    if len(instruction) > MAX_FIELD_LENGTH:
        return False, "instruction_too_long"
    if len(output) > MAX_FIELD_LENGTH:
        return False, "output_too_long"

    # Garbage detection
    if is_garbage(instruction):
        return False, "garbage_instruction"
    if is_garbage(output):
        return False, "garbage_output"

    return True, None


def clean_and_validate_examples(
    examples: List[Dict[str, Any]],
    min_quality_score: float = 0.4,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Clean, validate, and filter a list of training examples.

    Returns:
        (clean_examples, rejection_stats)
    """
    clean = []
    rejection_stats: Dict[str, int] = {}

    for ex in examples:
        # Step 1: Validate structure
        is_valid, reason = validate_example(ex)
        if not is_valid:
            rejection_stats[reason] = rejection_stats.get(reason, 0) + 1
            continue

        # Step 2: Clean text fields
        cleaned = {
            "instruction": clean_text(ex.get("instruction", "")),
            "input": clean_text(ex.get("input", "")),
            "output": clean_text(ex.get("output", "")),
            "source": ex.get("source", "unknown"),
            "adapter": ex.get("adapter", ""),
        }

        # Step 3: Re-validate after cleaning
        is_valid, reason = validate_example(cleaned)
        if not is_valid:
            rejection_stats[f"post_clean_{reason}"] = rejection_stats.get(f"post_clean_{reason}", 0) + 1
            continue

        # Step 4: Quality score filter
        score = compute_quality_score(cleaned)
        if score < min_quality_score:
            rejection_stats["low_quality_score"] = rejection_stats.get("low_quality_score", 0) + 1
            continue

        clean.append(cleaned)

    return clean, rejection_stats


# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================

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


def prepare_adapter_data(adapter_name: str, min_quality_score: float = 0.4) -> Dict[str, List[Dict]]:
    """Prepare all data for a specific adapter with preprocessing and validation"""
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

    raw_count = len(all_examples)

    # ── PREPROCESSING PIPELINE ──────────────────────────────────
    logger.info(f"  Running preprocessing pipeline on {raw_count} examples...")

    # Step 1: Clean, validate, and quality-filter
    clean_examples, rejection_stats = clean_and_validate_examples(
        all_examples, min_quality_score=min_quality_score
    )

    rejected_total = raw_count - len(clean_examples)
    if rejection_stats:
        logger.info(f"  Rejected {rejected_total} examples ({100*rejected_total/raw_count:.1f}%):")
        for reason, count in sorted(rejection_stats.items(), key=lambda x: -x[1]):
            logger.info(f"    {reason}: {count}")

    # Step 2: Deduplicate by instruction+input content hash
    seen = set()
    unique_examples = []
    for ex in clean_examples:
        content = (ex.get("instruction", "") + ex.get("input", "")).encode("utf-8")
        key = hashlib.sha256(content).hexdigest()
        if key not in seen:
            seen.add(key)
            unique_examples.append(ex)

    dedup_removed = len(clean_examples) - len(unique_examples)
    logger.info(f"  Deduplication removed {dedup_removed} duplicates")
    logger.info(f"  Final: {raw_count} raw → {len(unique_examples)} clean unique examples")

    # Shuffle
    random.shuffle(unique_examples)

    # Split 90/10
    split_idx = int(len(unique_examples) * 0.9)

    return {
        "train": unique_examples[:split_idx],
        "val": unique_examples[split_idx:],
    }


def prepare_all_data(adapters: List[str] = None, min_quality_score: float = 0.4):
    """Prepare data for all adapters with full preprocessing pipeline"""
    logger.info("=" * 60)
    logger.info("Data Preparation for Training (with preprocessing)")
    logger.info("=" * 60)
    logger.info(f"Min quality score: {min_quality_score}")

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
        data = prepare_adapter_data(adapter, min_quality_score=min_quality_score)

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
    parser = argparse.ArgumentParser(description="Prepare training data with preprocessing")
    parser.add_argument("--adapters", nargs="+", help="Specific adapters to prepare")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--min-quality-score", type=float, default=0.4,
        help="Minimum quality score (0.0-1.0) to keep an example (default: 0.4)",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    prepare_all_data(args.adapters, min_quality_score=args.min_quality_score)


if __name__ == "__main__":
    main()
