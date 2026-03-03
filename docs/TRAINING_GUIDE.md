# IMI Training Guide

Complete guide for training domain-specific LoRA adapters for the IMI Medical LLM Platform.

---

## Overview

IMI uses a **3-stage training pipeline** on **Mixtral 8x7B** (Apache 2.0):

1. **Foundation Training** — Medical domain adaptation on 3M+ examples
2. **DPO Safety Alignment** — Teaches model to prefer safe responses
3. **LoRA Adapter Training** — 6 user-type-specific adapters

All stages use **QLoRA 4-bit NF4** with **bfloat16** compute, fitting on a single A100 80GB.

---

## Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA COLLECTION                              │
├─────────────────────────────────────────────────────────────────┤
│  1. collect_datasets.py  → Download 40+ open medical datasets   │
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
│              STAGE 1: FOUNDATION TRAINING                        │
├─────────────────────────────────────────────────────────────────┤
│  train_foundation.py → Medical domain adaptation (r=64, MoE)   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 2: DPO SAFETY ALIGNMENT                      │
├─────────────────────────────────────────────────────────────────┤
│  train_dpo.py → Direct Preference Optimization (safety pairs)  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 3: ADAPTER TRAINING                           │
├─────────────────────────────────────────────────────────────────┤
│  train_lora.py → 6 LoRA adapters (parallel across GPUs)        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     EVALUATION                                   │
├─────────────────────────────────────────────────────────────────┤
│  evaluate_adapter.py → Metrics + threshold pass/fail gates     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     INFERENCE                                    │
├─────────────────────────────────────────────────────────────────┤
│  start_vllm.sh → vLLM server with LoRA hot-swapping           │
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
| MedQA | 12,723 | USMLE questions ★★★★★ |
| MedMCQA | 194,000 | Medical MCQs ★★★★ |
| PubMedQA | 211,000+ | Research Q&A ★★★★★ |
| HealthCareMagic | 100,000 | Patient conversations ★★★★ |
| Medical Flashcards | 33,955 | Study flashcards ★★★★ |
| MedDialog | 300,000 | Patient-doctor dialogs ★★★★ |
| WikiDoc | 67,704 | Clinical reference ★★★★ |
| DrugBank | 10,000 | Drug interactions ★★★★★ |
| + 30 more datasets | 3M+ total | See collect_datasets.py |

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

### Hardware Requirements (Mixtral 8x7B)

| Configuration | GPU VRAM | Training Time (1 adapter) | Training Time (all, parallel) |
|---------------|----------|--------------------------|------------------------------|
| 4-bit QLoRA (recommended) | 80GB A100 | 4-8 hours | ~8 hrs on 6×A100-80GB |
| 8-bit | 2×80GB | 3-6 hours | ~6 hrs on 6×A100-80GB |

> **Note:** Mixtral 8x7B in 4-bit QLoRA uses ~52GB VRAM. One A100 80GB per adapter.
> Wave-based parallel training runs N adapters simultaneously (1 per GPU).

### Stage 1: Foundation Training

```bash
# Train foundation model on combined medical corpus
python scripts/training/train_foundation.py

# With custom settings
python scripts/training/train_foundation.py \
    --epochs 2 \
    --lora-r 64 \
    --max-examples 500000
```

Output: `models/foundation/`

### Stage 2: DPO Safety Alignment

```bash
# Export seed safety pairs (30 built-in, expand to 500+)
python scripts/training/train_dpo.py export

# Run DPO training
python scripts/training/train_dpo.py train \
    --foundation-path models/foundation
```

Output: `models/dpo_aligned/`

### Stage 3: Train Single Adapter

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
# Parallel across multiple GPUs (recommended)
python scripts/training/train_lora.py --adapter all --parallel

# Sequential on a single GPU (fallback)
python scripts/training/train_lora.py --adapter all

# Train a single adapter on a specific GPU
python scripts/training/train_lora.py --adapter patient_triage --gpu 0
```

### Training Configuration

Each adapter has optimized defaults for Mixtral 8x7B:

| Adapter | LoRA r | Alpha | LR | Epochs | Target Modules |
|---------|--------|-------|-----|--------|----------------|
| patient_triage | 32 | 64 | 1e-4 | 3 | q,k,v,o_proj |
| clinical_pharmacist | 32 | 64 | 1e-4 | 3 | q,k,v,o_proj |
| clinical_decision | 32 | 64 | 1e-4 | 3 | q,k,v,o_proj |
| education | 32 | 64 | 1e-4 | 3 | q,k,v,o_proj |
| regulatory_qa | 32 | 64 | 1e-4 | 3 | q,k,v,o_proj |
| research | 32 | 64 | 1e-4 | 3 | q,k,v,o_proj |

**Foundation training** uses higher rank (r=64, α=128) and also targets MoE expert layers (w1, w2, w3).

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
- USMLE MCQ accuracy (per-topic breakdown)
- Triage classification F1 (per-class precision/recall)
- Safety audit (unsafe claim rate, emergency miss rate, disclaimer rate)
- ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
- Guardrail evaluation (crisis/emergency detection recall)
- Results saved to `adapters/{name}/evaluation_results.json`

**Pass/Fail Thresholds (must ALL pass before deployment):**

| Metric | Threshold | Direction |
|--------|-----------|-----------|
| Perplexity | ≤ 8.0 | lower is better |
| MCQ Accuracy | ≥ 55% | higher is better |
| Triage F1 | ≥ 0.70 | higher is better |
| Unsafe Claim Rate | ≤ 5% | lower is better |
| Emergency Miss Rate | ≤ 2% | lower is better |
| Disclaimer Rate | ≥ 80% | higher is better |
| ROUGE-L F1 | ≥ 0.20 | higher is better |
| Crisis Detection Recall | = 100% | must be perfect |
| Emergency Detection Recall | ≥ 95% | higher is better |

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
# 1. Collect data (40+ datasets)
python scripts/data_collection/collect_datasets.py
python scripts/data_collection/synthetic_generator.py --num-examples 5000

# 2. Add your PDFs (optional)
cp ~/Documents/WHO_GMP_*.pdf data/pdfs/
python scripts/data_collection/ingest_pdfs.py

# 3. Prepare data
python scripts/training/prepare_data.py

# 4. Foundation training (Stage 1)
python scripts/training/train_foundation.py

# 5. DPO safety alignment (Stage 2)
python scripts/training/train_dpo.py export   # export seed safety pairs
python scripts/training/train_dpo.py train

# 6. Train all adapters in parallel (Stage 3)
python scripts/training/train_lora.py \
    --adapter all --parallel \
    --base-model models/dpo_aligned

# 7. Evaluate
python scripts/training/evaluate_adapter.py --adapter patient_triage

# 8. Start inference server
bash scripts/start_vllm.sh
```

---

## Cloud Training (RunPod)

For faster training, use RunPod with A100 80GB GPUs:

```bash
# On RunPod instance (6×A100-80GB recommended)
git clone <your-repo>
cd imi
pip install -r requirements.txt

# Full pipeline
python scripts/training/train_foundation.py          # ~12hrs on 1×A100
python scripts/training/train_dpo.py train            # ~2hrs on 1×A100
python scripts/training/train_lora.py --adapter all --parallel  # ~8hrs on 6×A100

# MVP mode (doctor + patient only)
python scripts/training/train_lora.py \
    --adapter clinical_decision --gpu 0 &
python scripts/training/train_lora.py \
    --adapter patient_triage --gpu 1 &
wait

# Start inference
bash scripts/start_vllm.sh --mvp
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
