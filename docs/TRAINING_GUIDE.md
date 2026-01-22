# IMI Training Guide

Complete guide for training domain-specific LoRA adapters for the IMI Medical LLM Platform.

---

## Overview

IMI uses **LoRA (Low-Rank Adaptation)** to fine-tune the Meditron base model for specific medical domains. This approach:
- Reduces training compute by 90%+
- Allows multiple specialized adapters
- Preserves base model knowledge
- Enables quick adapter switching

---

## Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA COLLECTION                              │
├─────────────────────────────────────────────────────────────────┤
│  1. collect_datasets.py  → Download open medical datasets       │
│  2. ingest_pdfs.py       → Extract WHO/FDA regulations          │
│  3. synthetic_generator.py → Generate synthetic cases           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DATA PREPARATION                             │
├─────────────────────────────────────────────────────────────────┤
│  prepare_data.py → Merge, deduplicate, split train/val         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING                                     │
├─────────────────────────────────────────────────────────────────┤
│  train_lora.py → Train LoRA adapters with PEFT                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     EVALUATION                                   │
├─────────────────────────────────────────────────────────────────┤
│  evaluate_adapter.py → Measure perplexity, generate samples     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Data Collection

### 1.1 Download Open Datasets

```bash
# Download MedQA, MedMCQA, PubMedQA, HealthCareMagic, etc.
python scripts/data_collection/collect_datasets.py
```

**Datasets collected (no credentials required):**

| Dataset | Size | Use Case |
|---------|------|----------|
| MedQA | ~10k | USMLE questions |
| MedMCQA | ~180k | Medical MCQs |
| PubMedQA | ~1k | Research Q&A |
| HealthCareMagic | ~100k | Patient conversations |
| ChatDoctor | ~100k | Medical dialogues |

### 1.2 Ingest Your WHO/FDA PDFs

```bash
# Place PDFs in data/pdfs/ directory
cp /path/to/your/WHO_*.pdf data/pdfs/
cp /path/to/your/FDA_*.pdf data/pdfs/

# Run ingestion
python scripts/data_collection/ingest_pdfs.py
```

**Supported PDF types:**
- WHO guidelines
- FDA regulations (21 CFR)
- ICH guidelines
- GMP documents
- EMA guidelines

### 1.3 Generate Synthetic Data

```bash
# Generate 5000 synthetic examples
python scripts/data_collection/synthetic_generator.py --num-examples 5000
```

**Synthetic data types:**
- Patient triage cases with symptoms/severity/recommendations
- Drug interaction scenarios
- Clinical decision support cases
- USMLE-style questions
- Regulatory compliance scenarios

---

## Step 2: Data Preparation

```bash
# Merge all data sources and create train/val splits
python scripts/training/prepare_data.py
```

This will:
1. Merge collected datasets
2. Merge synthetic data
3. Merge PDF-extracted data
4. Deduplicate by instruction hash
5. Create 90/10 train/val splits
6. Save to `data/final/`

**Output structure:**
```
data/final/
├── patient_triage_train.json
├── patient_triage_val.json
├── clinical_pharmacist_train.json
├── clinical_pharmacist_val.json
├── education_train.json
├── education_val.json
├── regulatory_qa_train.json
├── regulatory_qa_val.json
└── ...
```

---

## Step 3: Training

### Hardware Requirements

| Configuration | GPU VRAM | Training Time |
|---------------|----------|---------------|
| 4-bit (recommended) | 16GB | 4-8 hours |
| 8-bit | 24GB | 3-6 hours |
| Full precision | 48GB+ | 2-4 hours |

### Train Single Adapter

```bash
# Train patient triage adapter
python scripts/training/train_lora.py --adapter patient_triage

# Train with custom settings
python scripts/training/train_lora.py \
    --adapter education \
    --epochs 5 \
    --batch-size 2 \
    --learning-rate 1e-4
```

### Train All Adapters

```bash
python scripts/training/train_lora.py --adapter all
```

### Training Configuration

Each adapter has optimized defaults:

| Adapter | LoRA r | Alpha | LR | Epochs |
|---------|--------|-------|-----|--------|
| patient_triage | 16 | 32 | 2e-4 | 3 |
| clinical_pharmacist | 16 | 32 | 2e-4 | 3 |
| clinical_decision | 16 | 32 | 1e-4 | 4 |
| education | 16 | 32 | 2e-4 | 3 |
| regulatory_qa | 16 | 32 | 2e-4 | 3 |
| research | 16 | 32 | 1e-4 | 4 |

### Resume Training

```bash
python scripts/training/train_lora.py \
    --adapter patient_triage \
    --resume-from adapters/patient_triage/checkpoint-500
```

---

## Step 4: Evaluation

```bash
# Evaluate trained adapter
python scripts/training/evaluate_adapter.py --adapter patient_triage
```

**Metrics:**
- Perplexity on validation set
- Sample generations for manual review
- Results saved to `adapters/{name}/evaluation_results.json`

---

## Data Format

All training data uses the instruction format:

```json
{
  "instruction": "You are a medical triage assistant. Assess the patient.",
  "input": "45yo male with chest pain, shortness of breath",
  "output": "Triage Level: URGENT. Recommend immediate evaluation...",
  "source": "synthetic_triage",
  "adapter": "patient_triage"
}
```

---

## Directory Structure After Training

```
imi/
├── data/
│   ├── raw/              # Downloaded datasets
│   ├── processed/        # Processed by adapter type
│   ├── synthetic/        # Generated synthetic data
│   ├── pdfs/             # Your WHO/FDA PDFs
│   └── final/            # Ready for training
├── adapters/
│   ├── patient_triage/
│   │   ├── adapter_model.bin
│   │   ├── adapter_config.json
│   │   └── evaluation_results.json
│   ├── clinical_pharmacist/
│   ├── education/
│   └── ...
└── scripts/
    ├── data_collection/
    │   ├── collect_datasets.py
    │   ├── ingest_pdfs.py
    │   └── synthetic_generator.py
    └── training/
        ├── prepare_data.py
        ├── train_lora.py
        └── evaluate_adapter.py
```

---

## Quick Start

```bash
# 1. Collect data
python scripts/data_collection/collect_datasets.py
python scripts/data_collection/synthetic_generator.py --num-examples 2000

# 2. Add your PDFs (optional)
cp ~/Documents/WHO_GMP_*.pdf data/pdfs/
python scripts/data_collection/ingest_pdfs.py

# 3. Prepare data
python scripts/training/prepare_data.py

# 4. Train (start with one adapter)
python scripts/training/train_lora.py --adapter patient_triage

# 5. Evaluate
python scripts/training/evaluate_adapter.py --adapter patient_triage
```

---

## Cloud Training (RunPod)

For faster training, use RunPod with A100 GPUs:

```bash
# On RunPod instance
git clone <your-repo>
cd imi
pip install -r requirements.txt

# Train with 8-bit for A100
python scripts/training/train_lora.py \
    --adapter all \
    --use-8bit \
    --batch-size 8
```

See `docs/RUNPOD_DEPLOYMENT_GUIDE.md` for detailed cloud setup.

---

## Troubleshooting

### Out of Memory
- Reduce batch size: `--batch-size 1`
- Use 4-bit quantization (default)
- Reduce max sequence length in config

### Slow Training
- Increase gradient accumulation steps
- Use mixed precision (enabled by default)
- Use cloud GPU (A100 recommended)

### Poor Results
- Increase training data (more synthetic examples)
- Train for more epochs
- Adjust learning rate (try 1e-4 or 3e-4)
- Check data quality
