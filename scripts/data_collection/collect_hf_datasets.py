"""
Quick HuggingFace Dataset Downloader for RunPod

Downloads medical datasets using the `datasets` library (reliable, no 404s).
Run this instead of collect_datasets.py if you're having URL issues.

Usage:
    pip install datasets
    python scripts/data_collection/collect_hf_datasets.py
"""
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from datasets import load_dataset
except ImportError:
    raise RuntimeError("Install datasets first: pip install datasets")

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


# ============================================================================
# Dataset catalogue: (name, hf_id, hf_subset, adapter_type)
# ============================================================================
HF_DATASETS = [
    # SHARED FOUNDATION / EDUCATION
    ("medqa", "bigbio/med_qa", "med_qa_en_source", "education"),
    ("medmcqa", "openlifescienceai/medmcqa", None, "education"),
    ("medalpaca_medqa", "medalpaca/medical_meadow_medqa", None, "education"),
    ("medical_flashcards", "medalpaca/medical_meadow_medical_flashcards", None, "education"),
    ("medical_meadow_mmmlu", "medalpaca/medical_meadow_mmmlu", None, "education"),
    ("medquad", "keivalya/MedQuad-MedicalQnADataset", None, "education"),
    ("headqa_en", "dvilares/head_qa", "en", "education"),

    # CLINICAL / DOCTOR
    ("medical_meadow_wikidoc", "medalpaca/medical_meadow_wikidoc", None, "clinical_decision"),
    ("chatdoctor_icliniq", "lavita/ChatDoctor-iCliniq", None, "clinical_decision"),
    ("healthcaremagic_100k", "lavita/ChatDoctor-HealthCareMagic-100k", None, "clinical_decision"),

    # PATIENT
    ("wikidoc_patient_info", "medalpaca/medical_meadow_wikidoc_patient_information", None, "patient_triage"),
    ("medical_meadow_health_advice", "medalpaca/medical_meadow_health_advice", None, "patient_triage"),
    ("medical_meadow_mediqa", "medalpaca/medical_meadow_mediqa", None, "patient_triage"),
    ("empathetic_dialogues", "facebook/empathetic_dialogues", None, "patient_triage"),
    ("mental_health_counseling", "Amod/mental_health_counseling_conversations", None, "patient_triage"),
    ("counsel_chat", "nbertagnolli/counsel-chat", None, "patient_triage"),

    # RESEARCH
    ("pubmedqa_labeled", "qiaojin/PubMedQA", "pqa_labeled", "research"),
    ("pubmedqa_artificial", "qiaojin/PubMedQA", "pqa_artificial", "research"),
    ("medical_meadow_cord19", "medalpaca/medical_meadow_cord19", None, "research"),
    ("sciq", "allenai/sciq", None, "research"),
]


def download_all():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    stats = {"ok": 0, "skip": 0, "fail": 0}

    for name, hf_id, subset, adapter in HF_DATASETS:
        dest = RAW_DIR / f"{name}.json"

        if dest.exists():
            logger.info(f"[SKIP] {name} — already exists")
            stats["skip"] += 1
            continue

        try:
            logger.info(f"[DL] {name} from {hf_id} (subset={subset})")
            if subset:
                ds = load_dataset(hf_id, subset, split="train", trust_remote_code=True)
            else:
                ds = load_dataset(hf_id, split="train", trust_remote_code=True)

            data = [dict(row) for row in ds]
            with open(dest, "w") as f:
                json.dump(data, f)

            logger.info(f"  ✓ {len(data)} examples → {dest}")
            stats["ok"] += 1

            # Also save to processed dir by adapter
            adapter_dir = PROCESSED_DIR / adapter
            adapter_dir.mkdir(parents=True, exist_ok=True)
            proc_path = adapter_dir / f"{name}.json"

            # Convert to instruction format
            processed = []
            for item in data:
                processed.append({
                    "instruction": item.get("instruction", item.get("question", item.get("QUESTION", ""))),
                    "input": item.get("input", item.get("context", "")),
                    "output": item.get("output", item.get("answer", item.get("response", item.get("LONG_ANSWER", "")))),
                    "source": name,
                    "adapter": adapter,
                })

            # Filter out empty entries
            processed = [p for p in processed if p["instruction"] and p["output"]]

            if processed:
                with open(proc_path, "w") as f:
                    json.dump(processed, f)
                logger.info(f"  ✓ {len(processed)} processed → {proc_path}")

        except Exception as e:
            logger.error(f"  ✗ FAILED {name}: {e}")
            stats["fail"] += 1

    logger.info("\n" + "=" * 50)
    logger.info(f"Done: {stats['ok']} downloaded, {stats['skip']} skipped, {stats['fail']} failed")
    logger.info("=" * 50)

    # Summary by adapter
    logger.info("\nData by adapter:")
    for d in sorted(PROCESSED_DIR.iterdir()):
        if d.is_dir():
            total = 0
            for f in d.glob("*.json"):
                with open(f) as fp:
                    total += len(json.load(fp))
            logger.info(f"  {d.name}: {total:,} examples")


if __name__ == "__main__":
    download_all()
