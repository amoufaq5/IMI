"""
Comprehensive Medical Dataset Downloader — Open & Verified Sources Only

Downloads open-source medical datasets from verified, credential-free sources:
  - HuggingFace Hub (via `datasets` library) — NO login required
  - GitHub repositories (direct URL) — public repos only
  - US Government open data (CDC, CMS) — NO API keys required

NO credentials, API keys, or accounts needed. All sources are publicly accessible.

Usage:
    pip install datasets requests
    python scripts/data_collection/collect_hf_datasets.py            # download all
    python scripts/data_collection/collect_hf_datasets.py --adapter patient_triage
    python scripts/data_collection/collect_hf_datasets.py --source hf       # HuggingFace only
    python scripts/data_collection/collect_hf_datasets.py --source url      # direct URL only
    python scripts/data_collection/collect_hf_datasets.py --list

Total estimated examples: ~8M+
"""
import os
import io
import csv
import json
import logging
import zipfile
import argparse
import requests
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    logging.warning("'datasets' library not installed — HuggingFace downloads disabled. pip install datasets")

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


# ============================================================================
# COMPREHENSIVE DATASET CATALOGUE — 100+ open-source medical datasets
#
# HF_DATASETS: Downloaded via HuggingFace `datasets` library
# URL_DATASETS: Downloaded via direct URL (GitHub, NIH, FDA, academic)
#
# Adapter types:
#   education         — USMLE, MCQ, flashcards, medical knowledge
#   clinical_decision — Clinical notes, doctor reasoning, diagnosis
#   patient_triage    — Patient Q&A, symptom assessment, health advice
#   research          — PubMed, biomedical literature, evidence-based QA
#   clinical_pharmacist — Drug info, interactions, pharmacology
#   regulatory_qa     — Guidelines, protocols, medical standards
# ============================================================================

# Format: (name, hf_id, hf_subset, adapter_type, est_size, description)
HF_DATASETS = [

    # ====================== EDUCATION / USMLE / MCQ ======================
    ("medqa", "bigbio/med_qa", "med_qa_en_source", "education", "12K", "USMLE-style medical questions"),
    ("medmcqa", "openlifescienceai/medmcqa", None, "education", "194K", "Medical MCQs, 21 subjects (AIIMS/NEET)"),
    ("medalpaca_medqa", "medalpaca/medical_meadow_medqa", None, "education", "10K", "USMLE + chain-of-thought reasoning"),
    ("medical_flashcards", "medalpaca/medical_meadow_medical_flashcards", None, "education", "34K", "High-yield medical flashcards"),
    ("medical_meadow_mmmlu", "medalpaca/medical_meadow_mmmlu", None, "education", "3K", "Multilingual medical MMLU"),
    ("medquad", "keivalya/MedQuad-MedicalQnADataset", None, "education", "16K", "Medical Q&A from NIH websites"),
    ("headqa_en", "dvilares/head_qa", "en", "education", "7K", "Spanish medical exams translated to English"),
    ("sciq", "allenai/sciq", None, "education", "12K", "Science questions including biology/medicine"),
    ("medical_instruction_100k", "Mohammed-Altaf/medical-instruction-100k", None, "education", "100K", "Broad medical instruction following"),
    # med_dataset_flashcards only has a "test" split — handled via split fallback in loader
    ("med_dataset_flashcards", "Med-dataset/Med_Dataset", None, "education", "10K", "Medical concept flashcards: anatomy, pharma, pathology"),
    ("medical_reasoning", "mamachang/medical-reasoning", None, "education", "4K", "Medical reasoning chains"),
    ("medical_meadow_alpaca", "monology/medical_meadow_alpaca", None, "education", "6K", "Medical Alpaca instruction data"),
    ("wiki_medical_terms", "gamino/wiki_medical_terms", None, "education", "7K", "Wikipedia medical terminology definitions"),
    # medical_specialities requires one of 35 specialty configs — loaded via MULTI_CONFIG_DATASETS
    ("medical_specialities", "HPAI-BSC/medical-specialities", None, "education", "14K", "Questions across medical specialties"),
    ("medqa_alpaca_format", "maximegmd/medqa_alpaca_format", None, "education", "12K", "MedQA in Alpaca instruction format"),
    ("alpacare_medinstruct", "lavita/AlpaCare-MedInstruct-52k", None, "education", "52K", "AlpaCare medical instruction tuning"),
    ("medical_qa_eswardivi", "eswardivi/medical_qa", None, "education", "6K", "General medical Q&A pairs"),
    ("medical_sciences_stackexchange", "ymoslem/MedicalSciences-StackExchange", None, "education", "5K", "Medical StackExchange Q&A"),

    # ====================== CLINICAL / DOCTOR ======================
    ("medical_meadow_wikidoc", "medalpaca/medical_meadow_wikidoc", None, "clinical_decision", "10K", "Clinical reference articles encyclopedic"),
    ("chatdoctor_icliniq", "lavita/ChatDoctor-iCliniq", None, "clinical_decision", "11K", "Real doctor-patient clinical dialogues"),
    ("healthcaremagic_100k", "lavita/ChatDoctor-HealthCareMagic-100k", None, "clinical_decision", "100K", "Doctor answers to patient queries"),
    ("ai_medical_chatbot", "ruslanmv/ai-medical-chatbot", None, "clinical_decision", "257K", "Doctor-patient dialogues (250K+)"),
    ("healthcaremagic_ruslan", "ruslanmv/HealthCareMagic-100k", None, "clinical_decision", "112K", "HealthCareMagic cleaned dialogues"),
    ("icliniq_7k", "ruslanmv/icliniq-7k", None, "clinical_decision", "7K", "iCliniq real doctor consultations"),
    ("know_medical_dialogue_v2", "knowrohit07/know_medical_dialogue_v2", None, "clinical_decision", "6K", "Medical dialogue dataset v2"),
    ("soap_summary", "omi-health/medical-dialogue-to-soap-summary", None, "clinical_decision", "10K", "Medical dialogue to SOAP note summaries"),
    ("medical_ai_cleaned_alpaca", "abhikrnigam/medical_ai_cleaned_alpaca", None, "clinical_decision", "257K", "Cleaned medical Alpaca data"),
    ("medical_meadow_pubmed_causal", "medalpaca/medical_meadow_pubmed_causal", None, "clinical_decision", "2K", "PubMed causal language modeling"),

    # ====================== PATIENT / TRIAGE ======================
    ("wikidoc_patient_info", "medalpaca/medical_meadow_wikidoc_patient_information", None, "patient_triage", "6K", "Patient-facing medical explanations"),
    ("medical_meadow_health_advice", "medalpaca/medical_meadow_health_advice", None, "patient_triage", "10K", "Patient health advice questions"),
    ("medical_meadow_mediqa", "medalpaca/medical_meadow_mediqa", None, "patient_triage", "2K", "Consumer health Q&A (MEDIQA)"),
    ("empathetic_dialogues", "facebook/empathetic_dialogues", None, "patient_triage", "25K", "Empathy in conversation tone training"),
    # mental_health_counseling: LICENSE.txt included in data scan — use data_files override (see SPECIAL_DATA_FILES)
    ("mental_health_counseling", "Amod/mental_health_counseling_conversations", None, "patient_triage", "3K", "Mental health counseling conversations"),
    ("counsel_chat", "nbertagnolli/counsel-chat", None, "patient_triage", "2K", "Counseling chat conversations"),
    # medical_qa_datasets and medical_question_answering require a config — use "all-processed"
    ("medical_qa_datasets", "lavita/medical-qa-datasets", "all-processed", "patient_triage", "1.7M", "Aggregated medical Q&A (ChatDoctor+iCliniq+MedQuAD)"),
    ("medical_question_answering", "Malikeh1375/medical-question-answering-datasets", "all-processed", "patient_triage", "1.3M", "Curated medical question-answering pairs"),
    ("complete_medical_symptoms", "mohammad2928git/complete_medical_symptom_dataset", None, "patient_triage", "1.3M", "Complete medical symptom-disease mapping"),
    ("ai_medical_dataset", "ruslanmv/ai-medical-dataset", None, "patient_triage", "50K", "General medical chatbot dataset"),

    # ====================== RESEARCH / LITERATURE ======================
    ("pubmedqa_labeled", "qiaojin/PubMedQA", "pqa_labeled", "research", "1K", "PubMedQA evidence-based Q&A (gold)"),
    ("pubmedqa_artificial", "qiaojin/PubMedQA", "pqa_artificial", "research", "211K", "PubMedQA artificial large volume pretraining"),
    ("medical_meadow_cord19", "medalpaca/medical_meadow_cord19", None, "research", "1K", "COVID-19 research papers Q&A"),

    # ====================== PHARMACIST / DRUG INFO ======================
    ("drug_reviews", "lewtun/drug-reviews", None, "clinical_pharmacist", "215K", "Drug reviews with conditions and ratings"),
    ("phi_drug", "Shishir1807/Phi_Drug", None, "clinical_pharmacist", "53K", "Drug information instruction data"),
    ("medicine_review", "Shivani-3112/medicine-review", None, "clinical_pharmacist", "161K", "Medicine reviews with side effects and ratings"),
]


# Datasets that require loading multiple configs and concatenating them
MULTI_CONFIG_DATASETS = {
    "medical_specialities": [
        "Cardiology", "Hematology", "Oncology", "Endocrinology", "Respiratory",
        "Allergy", "Dermatology", "Nephrology", "Gastroenterology", "Rheumatology",
        "Otorhinolaryngology", "Anesthesiology", "Biochemistry", "Pharmacology",
        "Psychiatry", "Microbiology", "Physiology", "Pathology", "Obstetrics",
        "Gynecology", "Surgery", "Emergency", "Orthopedics", "Neurology", "Urology",
        "Anatomy", "Genetics", "Radiology", "Ophthalmology", "Odontology",
        "Pediatrics", "Geriatrics", "Nursing", "Chemistry", "Psychology",
    ],
}

# Datasets where data_files must be restricted (e.g. LICENSE.txt picked up as data)
SPECIAL_DATA_FILES = {
    # The repo includes a LICENSE-RAIL-D.txt that confuses the auto-detect;
    # restrict to CSV/JSON data files only.
    "mental_health_counseling": {"train": ["*.csv", "*.json", "*.jsonl"]},
}


# ============================================================================
# NON-HUGGINGFACE DATASETS — Direct URL downloads
# Format: (name, url, file_format, adapter_type, est_size, description)
#   file_format: "json", "csv", "jsonl", "zip_csv", "zip_json", "tsv"
#                "zip_csv:<member>" to pick a specific file inside a zip
# ============================================================================

URL_DATASETS = [
    # ====================== GITHUB — Medical NLP (public repos) ===============
    # mtsamples.csv is in the data/ subdirectory (not repo root)
    ("mtsamples_transcriptions",
     "https://raw.githubusercontent.com/salgadev/medical-nlp/master/data/mtsamples.csv",
     "csv", "clinical_decision", "5K", "MTSamples medical transcriptions — 40 specialties"),

    ("chatdoctor_200k",
     "https://raw.githubusercontent.com/Kent0n-Li/ChatDoctor/main/chatdoctor5k.json",
     "json", "clinical_decision", "5K", "ChatDoctor patient-doctor conversations"),

    # MedCalc-Bench: files live in datasets/ (plural) and train is train_data.csv (not train_set.csv)
    ("medcalc_bench_train",
     "https://raw.githubusercontent.com/ncbi-nlp/MedCalc-Bench/main/datasets/train_data.csv",
     "csv", "education", "10K", "MedCalc-Bench medical calculator QA — train (NeurIPS 2024)"),

    ("medcalc_bench_test",
     "https://raw.githubusercontent.com/ncbi-nlm/MedCalc-Bench/main/datasets/test_set.csv",
     "csv", "education", "1K", "MedCalc-Bench medical calculator QA — test set"),

    # ====================== GITHUB — Symptom/Disease Datasets =================
    # Files live in Data/ and MasterData/ subdirectories (not repo root)
    ("symptom_disease_dataset",
     "https://raw.githubusercontent.com/itachi9604/healthcare-chatbot/master/Data/Training.csv",
     "csv", "patient_triage", "5K", "Symptom to disease prediction dataset"),

    ("symptom_severity",
     "https://raw.githubusercontent.com/itachi9604/healthcare-chatbot/master/MasterData/Symptom_severity.csv",
     "csv", "patient_triage", "130", "Symptom severity weights"),

    ("symptom_description",
     "https://raw.githubusercontent.com/itachi9604/healthcare-chatbot/master/MasterData/symptom_Description.csv",
     "csv", "patient_triage", "40", "Symptom descriptions for patient education"),

    ("symptom_precaution",
     "https://raw.githubusercontent.com/itachi9604/healthcare-chatbot/master/MasterData/symptom_precaution.csv",
     "csv", "patient_triage", "40", "Symptom precaution advice"),

    # ====================== GITHUB — Medical Dialogues ========================
    ("mts_dialog_clinical_notes",
     "https://raw.githubusercontent.com/abachaa/MTS-Dialog/main/Main-Dataset/MTS-Dialog-TrainingSet.csv",
     "csv", "clinical_decision", "1.2K", "Medical dialogue to clinical note generation"),

    ("mts_dialog_test",
     "https://raw.githubusercontent.com/abachaa/MTS-Dialog/main/Main-Dataset/MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv",
     "csv", "clinical_decision", "200", "MTS-Dialog test set (MEDIQA-Chat 2023)"),

    # ====================== GITHUB — Open Medical Resources ===================
    ("icd10_codes",
     "https://raw.githubusercontent.com/kamillamagna/ICD-10-CSV/master/codes.csv",
     "csv", "regulatory_qa", "72K", "Complete ICD-10 diagnosis code descriptions"),

    # ====================== GITHUB — Synthea (synthetic patient data) =========
    # Synthea CSV data is packaged as a zip in downloads/; extract the target member
    ("synthea_sample_patients",
     "https://github.com/synthetichealth/synthea-sample-data/raw/master/downloads/synthea_sample_data_csv_nov2021.zip",
     "zip_csv:patients.csv", "clinical_decision", "1K", "Synthea synthetic patient demographics"),

    ("synthea_sample_conditions",
     "https://github.com/synthetichealth/synthea-sample-data/raw/master/downloads/synthea_sample_data_csv_nov2021.zip",
     "zip_csv:conditions.csv", "clinical_decision", "10K", "Synthea synthetic patient conditions"),

    ("synthea_sample_medications",
     "https://github.com/synthetichealth/synthea-sample-data/raw/master/downloads/synthea_sample_data_csv_nov2021.zip",
     "zip_csv:medications.csv", "clinical_pharmacist", "10K", "Synthea synthetic patient medications"),

    ("synthea_sample_allergies",
     "https://github.com/synthetichealth/synthea-sample-data/raw/master/downloads/synthea_sample_data_csv_nov2021.zip",
     "zip_csv:allergies.csv", "patient_triage", "1K", "Synthea synthetic patient allergies"),

    ("synthea_sample_procedures",
     "https://github.com/synthetichealth/synthea-sample-data/raw/master/downloads/synthea_sample_data_csv_nov2021.zip",
     "zip_csv:procedures.csv", "clinical_decision", "10K", "Synthea synthetic patient procedures"),

    ("synthea_sample_observations",
     "https://github.com/synthetichealth/synthea-sample-data/raw/master/downloads/synthea_sample_data_csv_nov2021.zip",
     "zip_csv:observations.csv", "clinical_decision", "100K", "Synthea synthetic patient lab observations"),

    # ====================== US GOV — CDC Open Data (no API key) ==============
    ("cdc_wonder_mortality_2022",
     "https://data.cdc.gov/api/views/bi63-dtpu/rows.csv?accessType=DOWNLOAD",
     "csv", "research", "1M", "CDC WONDER cause of death detailed mortality 2022"),

    # ====================== US GOV — CMS Open Data ===========================
    # Stable CMS SODA API endpoint (no hash in URL)
    ("cms_hospital_compare",
     "https://data.cms.gov/provider-data/api/1/datastore/query/xubh-q36u/0?results_format=csv",
     "csv", "regulatory_qa", "5K", "CMS Hospital General Information"),
]


# Cache for zip downloads — avoid re-downloading the same zip for each member
_ZIP_CACHE: dict = {}  # url -> bytes


# ============================================================================
# DOWNLOAD & PROCESSING
# ============================================================================

def to_instruction_format(item, source_name, adapter):
    """Convert any dataset row to instruction/input/output format"""
    instruction_fields = ["instruction", "question", "QUESTION", "input_text", "query", "prompt", "Description",
                          "medical_specialty", "Disease", "Drug", "drug_name"]
    input_fields = ["input", "context", "Context", "CONTEXTS", "patient", "dialogue", "transcription", "description",
                    "Symptom", "prognosis", "SUBJECT"]
    output_fields = ["output", "answer", "response", "Response", "LONG_ANSWER", "answer_text", "completion", "Doctor",
                     "keywords", "sample_name", "Precaution", "review", "rating"]

    instruction = ""
    for f in instruction_fields:
        val = item.get(f)
        if val and isinstance(val, str) and len(val.strip()) > 3:
            instruction = val.strip()
            break
        elif val and isinstance(val, list):
            instruction = " ".join(str(v) for v in val).strip()
            break

    inp = ""
    for f in input_fields:
        val = item.get(f)
        if val and isinstance(val, str):
            inp = val.strip()
            break
        elif val and isinstance(val, list):
            inp = " ".join(str(v) for v in val).strip()
            break

    output = ""
    for f in output_fields:
        val = item.get(f)
        if val and isinstance(val, str) and len(val.strip()) > 3:
            output = val.strip()
            break
        elif val and isinstance(val, list):
            output = " ".join(str(v) for v in val).strip()
            break

    # Handle MCQ datasets (medmcqa, headqa, etc.)
    if not output and "cop" in item:
        idx = item.get("cop", 0)
        options = [item.get(f"op{c}", "") for c in "abcd"]
        if 1 <= idx <= 4 and options[idx-1]:
            output = f"The correct answer is {chr(64+idx)}. {options[idx-1]}"
            exp = item.get("exp", "")
            if exp:
                output += f"\n\nExplanation: {exp}"

    # Handle PubMedQA format
    if not output and "final_decision" in item:
        output = f"Answer: {item['final_decision']}"
        long = item.get("LONG_ANSWER", "")
        if long:
            output += f"\n\n{long}"

    # Handle ChatDoctor / dialogue format
    if not instruction and "Patient" in item:
        instruction = item["Patient"]
    if not output and "Doctor" in item:
        output = item["Doctor"]

    # Handle mental-health counseling format (Context / Response)
    if not instruction and "Context" in item:
        instruction = item["Context"]
    if not output and "Response" in item:
        output = item["Response"]

    # Handle MTSamples format
    if not instruction and "medical_specialty" in item:
        instruction = f"Generate a clinical note for {item.get('medical_specialty', 'general')} specialty."
        inp = item.get("description", "")
        output = item.get("transcription", "")

    # Handle disease-symptom datasets
    if not instruction and "Disease" in item:
        instruction = f"What are the symptoms and treatment for {item.get('Disease', '')}?"
        symptoms = [item.get(f"Symptom_{i}", "") for i in range(1, 18) if item.get(f"Symptom_{i}")]
        if symptoms:
            output = f"Symptoms: {', '.join(symptoms)}"

    # Handle drug review datasets
    if not instruction and "drugName" in item:
        cond = item.get("condition", "a condition")
        instruction = f"What is the patient experience with {item['drugName']} for {cond}?"
        output = item.get("review", "")

    # Handle ICD codes
    if not instruction and "code" in item and "description" in item:
        instruction = f"What is ICD-10 code {item['code']}?"
        output = item.get("description", "")

    if not instruction or not output:
        return None

    return {
        "instruction": instruction,
        "input": inp,
        "output": output,
        "source": source_name,
        "adapter": adapter,
    }


def _load_zip_member(content: bytes, member_name: str) -> list:
    """Extract a named CSV member from a zip archive."""
    data = []
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        # Find exact match first, then case-insensitive search
        names = zf.namelist()
        target = None
        for n in names:
            if n.endswith(f"/{member_name}") or n == member_name:
                target = n
                break
        if target is None:
            # Fuzzy: any entry whose basename matches
            for n in names:
                if Path(n).name == member_name:
                    target = n
                    break
        if target is None:
            raise FileNotFoundError(f"{member_name} not found in zip. Members: {names[:10]}")
        with zf.open(target) as f:
            text = f.read().decode("utf-8", errors="replace")
            reader = csv.DictReader(io.StringIO(text))
            data = [dict(row) for row in reader]
    return data


def download_url_dataset(name, url, file_format, dest):
    """Download a dataset from a direct URL and save as JSON."""
    if file_format == "skip":
        logger.info(f"  [SKIP] {name} — requires manual download")
        return None

    # zip_csv:<member> — download zip once (cached), extract named CSV
    if file_format.startswith("zip_csv:"):
        member = file_format.split(":", 1)[1]
        if url not in _ZIP_CACHE:
            logger.info(f"    Downloading zip archive...")
            resp = requests.get(url, timeout=120, stream=True)
            resp.raise_for_status()
            _ZIP_CACHE[url] = resp.content
        data = _load_zip_member(_ZIP_CACHE[url], member)
        if data:
            with open(dest, "w") as f:
                json.dump(data, f)
        return data

    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()
    content = resp.content

    data = []
    if file_format == "json":
        data = json.loads(content)
        if isinstance(data, dict):
            for key in ["data", "rows", "items", "records"]:
                if key in data:
                    data = data[key]
                    break
            else:
                data = [data]

    elif file_format == "jsonl":
        for line in content.decode("utf-8", errors="replace").strip().split("\n"):
            if line.strip():
                data.append(json.loads(line))

    elif file_format == "csv":
        text = content.decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        data = [dict(row) for row in reader]

    elif file_format == "tsv":
        text = content.decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text), delimiter="\t")
        data = [dict(row) for row in reader]

    elif file_format == "tsv_gz":
        import gzip
        text = gzip.decompress(content).decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text), delimiter="\t")
        data = [dict(row) for row in reader]

    elif file_format == "zip_csv":
        # Extract the first CSV found in the archive
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for name_in_zip in zf.namelist():
                if name_in_zip.endswith(".csv"):
                    with zf.open(name_in_zip) as f:
                        text = f.read().decode("utf-8", errors="replace")
                        reader = csv.DictReader(io.StringIO(text))
                        data.extend(dict(row) for row in reader)
                    break

    elif file_format == "zip_json":
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for name_in_zip in zf.namelist():
                if name_in_zip.endswith(".json"):
                    with zf.open(name_in_zip) as f:
                        data = json.load(f)
                    break

    if data:
        with open(dest, "w") as f:
            json.dump(data, f)
    return data


def _load_hf_dataset(hf_id, subset, name):
    """
    Robust HuggingFace dataset loader with automatic split + config fallbacks.

    Priority:
      1. Exact subset + split="train"
      2. split="test" (if "train" doesn't exist)
      3. First available split
      4. Multi-config concatenation (MULTI_CONFIG_DATASETS)
      5. data_files override (SPECIAL_DATA_FILES)
    """
    kwargs = dict(trust_remote_code=True)

    # Apply data_files override if configured
    if name in SPECIAL_DATA_FILES:
        kwargs["data_files"] = SPECIAL_DATA_FILES[name]

    # Multi-config datasets: concatenate all configs
    if name in MULTI_CONFIG_DATASETS:
        all_rows = []
        configs = MULTI_CONFIG_DATASETS[name]
        for cfg in configs:
            try:
                ds = load_dataset(hf_id, cfg, split="train", **kwargs)
                all_rows.extend([dict(row) for row in ds])
            except Exception as e:
                logger.debug(f"    Config {cfg} failed: {e}")
        return all_rows

    # Standard single-config loading with split fallback
    load_args = [hf_id] + ([subset] if subset else [])

    for split in ("train", "test", "validation"):
        try:
            ds = load_dataset(*load_args, split=split, **kwargs)
            return [dict(row) for row in ds]
        except ValueError as e:
            if "Unknown split" in str(e) or "splits" in str(e).lower():
                continue
            # Missing config error — re-raise to caller
            raise
        except Exception as e:
            err = str(e)
            if "Config name is missing" in err or "pick one among" in err:
                raise
            if "Unknown split" in err:
                continue
            raise

    # Last resort: load as DatasetDict and take first split
    try:
        ds_dict = load_dataset(*load_args, **kwargs)
        first_split = list(ds_dict.keys())[0]
        logger.info(f"    Using split '{first_split}' (no 'train' split found)")
        return [dict(row) for row in ds_dict[first_split]]
    except Exception:
        raise


def download_all(filter_adapter=None, source_filter=None):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    stats = {"ok": 0, "skip": 0, "fail": 0, "total_examples": 0}

    # ---- HuggingFace datasets ----
    if source_filter in (None, "hf"):
        if not HAS_DATASETS:
            logger.warning("Skipping HuggingFace datasets — 'datasets' library not installed")
        else:
            hf_list = HF_DATASETS
            if filter_adapter:
                hf_list = [d for d in HF_DATASETS if d[3] == filter_adapter]

            logger.info(f"\n{'='*60}")
            logger.info(f"HUGGINGFACE DATASETS ({len(hf_list)} datasets)")
            logger.info(f"{'='*60}")

            for name, hf_id, subset, adapter, est_size, desc in hf_list:
                dest = RAW_DIR / f"{name}.json"
                if dest.exists():
                    logger.info(f"[SKIP] {name} ({est_size})")
                    stats["skip"] += 1
                    continue
                try:
                    logger.info(f"[HF] {name} ({est_size}) <- {hf_id}")
                    data = _load_hf_dataset(hf_id, subset, name)
                    with open(dest, "w") as f:
                        json.dump(data, f)
                    logger.info(f"  -> {len(data):,} raw examples")
                    stats["ok"] += 1
                    _process_to_adapter(data, name, adapter, stats)
                except Exception as e:
                    logger.error(f"  X FAILED {name}: {e}")
                    stats["fail"] += 1

    # ---- URL-based datasets ----
    if source_filter in (None, "url"):
        url_list = URL_DATASETS
        if filter_adapter:
            url_list = [d for d in URL_DATASETS if d[3] == filter_adapter]

        logger.info(f"\n{'='*60}")
        logger.info(f"DIRECT URL DATASETS ({len(url_list)} datasets)")
        logger.info(f"{'='*60}")

        for name, url, fmt, adapter, est_size, desc in url_list:
            dest = RAW_DIR / f"{name}.json"
            if dest.exists():
                logger.info(f"[SKIP] {name} ({est_size})")
                stats["skip"] += 1
                continue
            try:
                logger.info(f"[URL] {name} ({est_size}) <- {url[:80]}...")
                data = download_url_dataset(name, url, fmt, dest)
                if data is None:
                    stats["skip"] += 1
                    continue
                logger.info(f"  -> {len(data):,} raw examples")
                stats["ok"] += 1
                _process_to_adapter(data, name, adapter, stats)
            except Exception as e:
                logger.error(f"  X FAILED {name}: {e}")
                stats["fail"] += 1

    _print_summary(stats)


def _process_to_adapter(data, name, adapter, stats):
    """Process raw data to instruction format and save to adapter dir"""
    adapter_dir = PROCESSED_DIR / adapter
    adapter_dir.mkdir(parents=True, exist_ok=True)
    proc_path = adapter_dir / f"{name}.json"
    processed = []
    for item in data:
        result = to_instruction_format(item, name, adapter)
        if result:
            processed.append(result)
    if processed:
        with open(proc_path, "w") as f:
            json.dump(processed, f)
        logger.info(f"  -> {len(processed):,} processed -> {adapter}/{name}.json")
        stats["total_examples"] += len(processed)


def _print_summary(stats):
    """Print download summary"""
    logger.info(f"\n{'='*60}")
    logger.info("DOWNLOAD SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  Downloaded:  {stats['ok']}")
    logger.info(f"  Skipped:     {stats['skip']}")
    logger.info(f"  Failed:      {stats['fail']}")
    logger.info(f"  Total new examples: {stats['total_examples']:,}")
    logger.info("\nExamples by adapter type:")
    grand_total = 0
    if PROCESSED_DIR.exists():
        for d in sorted(PROCESSED_DIR.iterdir()):
            if d.is_dir():
                total = 0
                files = list(d.glob("*.json"))
                for f in files:
                    with open(f) as fp:
                        total += len(json.load(fp))
                grand_total += total
                logger.info(f"  {d.name:25s} {total:>10,} examples  ({len(files)} datasets)")
    logger.info(f"  {'GRAND TOTAL':25s} {grand_total:>10,} examples")
    logger.info(f"{'='*60}")


def list_datasets():
    """Print all available datasets"""
    print(f"\n{'='*60}")
    print(f"HUGGINGFACE DATASETS ({len(HF_DATASETS)})")
    print(f"{'='*60}")
    print(f"{'#':>3}  {'Name':35s}  {'Source':50s}  {'Adapter':20s}  {'Size':>6s}")
    print("-" * 120)
    for i, (name, hf_id, subset, adapter, est_size, desc) in enumerate(HF_DATASETS, 1):
        hf_str = f"{hf_id}" + (f" [{subset}]" if subset else "")
        hf_str += " [multi-config]" if name in MULTI_CONFIG_DATASETS else ""
        print(f"{i:>3}  {name:35s}  {hf_str:50s}  {adapter:20s}  {est_size:>6s}")

    print(f"\n{'='*60}")
    print(f"DIRECT URL DATASETS ({len(URL_DATASETS)})")
    print(f"{'='*60}")
    for i, (name, url, fmt, adapter, est_size, desc) in enumerate(URL_DATASETS, 1):
        print(f"{len(HF_DATASETS)+i:>3}  {name:35s}  {url[:50]:50s}  {adapter:20s}  {est_size:>6s}")

    total = len(HF_DATASETS) + len(URL_DATASETS)
    print(f"\nTotal: {total} datasets ({len(HF_DATASETS)} HF + {len(URL_DATASETS)} URL)")


def main():
    parser = argparse.ArgumentParser(description="Download open-source medical datasets")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Filter by adapter type")
    parser.add_argument("--source", type=str, default=None, choices=["hf", "url"],
                        help="Download only HF or only URL datasets")
    parser.add_argument("--list", action="store_true",
                        help="List all available datasets without downloading")
    args = parser.parse_args()

    if args.list:
        list_datasets()
    else:
        download_all(filter_adapter=args.adapter, source_filter=args.source)


if __name__ == "__main__":
    main()
