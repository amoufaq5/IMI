# IMI Fine-Tuning Guide: End-to-End

## Overview

This guide covers the complete fine-tuning pipeline from raw data to trained adapters.

```
1. Collect Data → 2. Ingest & Process → 3. Prepare Splits → 4. Train → 5. Evaluate & Save
```

---

## Step 1: Collect Data

### A. Open Medical Datasets (automated)

```bash
python scripts/data_collection/collect_datasets.py
```

Downloads MedQA, MedMCQA, PubMedQA, HealthCareMagic, MTSamples, drug interaction data.

### B. Google Drive — Pharma QA Files

```bash
# Install gdown first
pip install gdown

# Download from shared folder
python scripts/data_collection/download_gdrive.py \
    --folder-url "https://drive.google.com/drive/folders/YOUR_FOLDER_ID" \
    --adapter regulatory_qa

# Or use folder ID directly
python scripts/data_collection/download_gdrive.py \
    --folder-id "1aBcDeFgHiJkLmNoPqRsTuVwXyZ" \
    --adapter regulatory_qa
```

**Supported file formats:** `.json`, `.jsonl`, `.txt`, `.pdf`, `.csv`

**JSON format expected** (any of these work):
```json
{"instruction": "What is the SOP for...", "input": "", "output": "The SOP requires..."}
{"question": "What are GMP requirements?", "answer": "GMP requires..."}
{"text": "Raw regulatory text content..."}
```

### C. Local WHO & FDA Guidelines (PDFs from your desktop)

```bash
# 1. Copy your PDFs into the data/pdfs/ directory
cp ~/Desktop/WHO_guidelines/*.pdf data/pdfs/
cp ~/Desktop/FDA_regulations/*.pdf data/pdfs/

# 2. Run the PDF ingester — auto-detects WHO/FDA/EMA/ICH/GMP sources
python scripts/data_collection/ingest_pdfs.py
```

The ingester:
- Extracts text from PDFs (PyMuPDF or pdfminer)
- Auto-detects source org (WHO, FDA, etc.) from filename/content
- Splits into sections by headings
- Generates Q&A pairs (summary, compliance, key requirements)
- Saves to `data/processed/regulatory_qa/regulatory_pdfs.json`

### D. Synthetic Data Generation

```bash
python scripts/data_collection/synthetic_generator.py
```

---

## Step 2: Prepare Training Data

```bash
# Merges all sources (open datasets + Google Drive + PDFs + synthetic)
# Creates train/val splits per adapter
python scripts/training/prepare_data.py --adapter all
```

Output structure:
```
data/final/
├── patient_triage_train.json
├── patient_triage_val.json
├── clinical_pharmacist_train.json
├── clinical_pharmacist_val.json
├── regulatory_qa_train.json      ← includes your GDrive + PDF data
├── regulatory_qa_val.json
└── ...
```

---

## Step 3: Train Adapters

### Hardware: 6×A100 (your setup)

With 7B model + 4-bit QLoRA, each A100 uses only ~6 GB VRAM. You have massive headroom.

```bash
# Train ALL 6 adapters in parallel — each on its own GPU
python scripts/training/train_lora.py --adapter all --parallel
```

### What happens during training

1. **Parent process** detects 6 GPUs, assigns 1 adapter per GPU
2. Launches 6 **subprocesses** with `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5`
3. Each subprocess:
   - Loads Meditron-7B in 4-bit QLoRA (~4 GB VRAM)
   - Attaches LoRA weights (r=16, alpha=32) to `q_proj`, `v_proj`
   - Trains on adapter-specific data
   - Saves adapter weights to `adapters/<adapter_name>/`
4. Parent monitors all processes, reports completion

### Training parameters (per adapter)

| Parameter | Value |
|-----------|-------|
| Base model | Meditron-7B (4-bit QLoRA) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, v_proj |
| Batch size | 4 |
| Gradient accumulation | 4 (effective batch = 16) |
| Max sequence length | 2048 |
| Learning rate | 2e-4 (most adapters) |

### Expected training time on 6×A100

| Metric | Value |
|--------|-------|
| Wall time | **~2.5 hours** (slowest adapter) |
| GPU-hours | ~12 total across all 6 |
| VRAM per GPU | ~6 GB (out of 40-80 GB) |

### Single adapter or debugging

```bash
# Train one adapter on GPU 0
python scripts/training/train_lora.py --adapter patient_triage --gpu 0

# Sequential (single GPU fallback)
python scripts/training/train_lora.py --adapter all
```

---

## Step 4: Evaluate

```bash
python scripts/training/evaluate_adapter.py --adapter patient_triage
python scripts/training/evaluate_adapter.py --adapter regulatory_qa
```

Metrics: perplexity, USMLE MCQ accuracy, triage F1, safety audit, ROUGE scores.

---

## Step 5: Save Trained Adapters

### What gets saved

After training, each adapter saves to:
```
adapters/
├── patient_triage/
│   ├── adapter_model.safetensors   (~30-60 MB per adapter)
│   ├── adapter_config.json
│   └── training_metadata.json
├── clinical_pharmacist/
├── clinical_decision/
├── education/
├── regulatory_qa/
└── research/
```

**Total size: ~200-400 MB** for all 6 adapters (NOT the full 7B model — just the LoRA deltas).

### DO NOT push adapter weights to GitHub

GitHub has a 100 MB file limit and repos should stay lean. The `.gitignore` already excludes:
- `models/` (base model weights)
- `adapters/` (trained LoRA weights)
- `*.safetensors`, `*.bin`, `*.pt`
- `data/` (training data)

### Where to store adapters

#### Option A: HuggingFace Hub (Recommended)

Free, versioned, easy to load in code. Private repos available.

```bash
# Install huggingface_hub
pip install huggingface_hub

# Login (one-time)
huggingface-cli login

# Push each adapter to HF Hub
python -c "
from huggingface_hub import HfApi
api = HfApi()

adapters = [
    'patient_triage', 'clinical_pharmacist', 'clinical_decision',
    'education', 'regulatory_qa', 'research',
]

for adapter in adapters:
    api.upload_folder(
        folder_path=f'adapters/{adapter}',
        repo_id=f'your-org/imi-{adapter}',
        repo_type='model',
        private=True,  # Keep private for now
    )
    print(f'Pushed {adapter}')
"
```

Then in production, load with:
```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "your-org/imi-patient_triage")
```

#### Option B: Cloud Storage (S3 / GCS)

```bash
# AWS S3
aws s3 sync adapters/ s3://your-bucket/imi-adapters/ --exclude "*.pyc"

# Google Cloud Storage
gsutil -m rsync -r adapters/ gs://your-bucket/imi-adapters/
```

#### Option C: RunPod Persistent Volume

If training on RunPod, adapters are already on the persistent volume at `/workspace/imi/adapters/`. They survive pod restarts. To download locally:

```bash
# From your local machine
scp -r root@your-pod-ip:/workspace/imi/adapters/ ./adapters/
```

#### Option D: Git LFS (if you really want them in git)

Not recommended, but possible for small teams:

```bash
git lfs install
git lfs track "adapters/**/*.safetensors"
git add .gitattributes
git add adapters/
git commit -m "Add trained adapters via LFS"
git push
```

---

## What to Push to GitHub

| Push to GitHub | Do NOT push |
|---------------|-------------|
| All source code (`src/`, `scripts/`) | `models/` (base model) |
| Config files (`.env.example`, `requirements.txt`) | `adapters/` (trained weights) |
| Documentation (`docs/`) | `data/` (training data) |
| `.gitignore` | `.env` (secrets) |
| Tests (`tests/`) | `__pycache__/` |

---

## Complete Pipeline (Copy-Paste)

```bash
# === ON RUNPOD (6×A100-80GB) ===

# 1. Setup
git clone https://github.com/your-org/imi.git
cd imi
pip install -r requirements.txt

# 2. Collect 40+ open datasets
python scripts/data_collection/collect_datasets.py

# 3. Download your Google Drive pharma QA files
python scripts/data_collection/download_gdrive.py \
    --folder-url "https://drive.google.com/drive/folders/YOUR_FOLDER_ID" \
    --adapter regulatory_qa

# 4. Copy WHO/FDA PDFs (upload to RunPod first via scp or web UI)
# scp ~/Desktop/WHO_guidelines/*.pdf root@pod-ip:/workspace/imi/data/pdfs/
python scripts/data_collection/ingest_pdfs.py

# 5. Generate synthetic data
python scripts/data_collection/synthetic_generator.py --num-examples 5000

# 6. Prepare final training splits
python scripts/training/prepare_data.py --adapter all

# 7. STAGE 1: Foundation training (~12hrs on 1×A100)
python scripts/training/train_foundation.py

# 8. STAGE 2: DPO safety alignment (~2hrs on 1×A100)
python scripts/training/train_dpo.py export   # export seed safety pairs
python scripts/training/train_dpo.py train --foundation-path models/foundation

# 9. STAGE 3: Train all adapters in parallel (~8hrs on 6×A100)
python scripts/training/train_lora.py \
    --adapter all --parallel \
    --base-model models/dpo_aligned

# 10. Evaluate (must pass all thresholds before deployment)
python scripts/training/evaluate_adapter.py --adapter patient_triage
python scripts/training/evaluate_adapter.py --adapter clinical_decision
python scripts/training/evaluate_adapter.py --adapter regulatory_qa

# 11. Start vLLM inference server
bash scripts/start_vllm.sh          # all adapters
bash scripts/start_vllm.sh --mvp    # doctor + patient only

# 12. Push adapters to HuggingFace Hub
huggingface-cli login
# (run the upload script from Step 5 above)
```
