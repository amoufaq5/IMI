# IMI Training Guide

Complete guide for training the IMI Medical LLM on Mixtral 8x7B.

---

## Overview

Two training paths depending on your goal:

| Path | When to use | GPU | Cost estimate |
|------|-------------|-----|---------------|
| **Adapter only** (QLoRA) | MVP, proof of concept, fast iteration | 1× A100 80GB | ~$20–80 |
| **Foundation → Adapter** | Production quality, deep domain shift | 8× A100 80GB + 1× A100 80GB | ~$300–600 |

Both paths use the same data collection and preparation steps.

---

## Pipeline Overview

```
┌──────────────────────────────────────────────────────────┐
│  1. DATA COLLECTION                                       │
│     collect_datasets.py  → 40+ open medical datasets     │
│     ingest_pdfs.py        → WHO/FDA regulation PDFs       │
│     synthetic_generator.py → synthetic training cases    │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  2. DATA PREPARATION                                      │
│     prepare_medical_data.py → 2 formats:                 │
│       general_knowledge_{train,val}.json                  │
│       instruction_{train,val}.json                        │
└──────────────────────────────────────────────────────────┘
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
┌────────────────────┐   ┌────────────────────────────────┐
│  PATH A            │   │  PATH B (production)           │
│  ADAPTER ONLY      │   │                                │
│                    │   │  3. FOUNDATION TRAINING        │
│  finetune_         │   │     train_foundation.py        │
│  mixtral.py        │   │     8× A100 80GB               │
│  1× A100 80GB      │   │     torchrun --nproc_per_node=8│
│                    │   │            │                   │
│                    │   │            ▼                   │
│                    │   │  4. ADAPTER TRAINING           │
│                    │   │     finetune_mixtral.py        │
│                    │   │     --base-model models/found. │
│                    │   │     1× A100 80GB               │
└────────────────────┘   └────────────────────────────────┘
             │                       │
             └───────────┬───────────┘
                         ▼
┌──────────────────────────────────────────────────────────┐
│  5. EVALUATE                                             │
│     evaluate_adapter.py                                  │
└──────────────────────────────────────────────────────────┘
```

---

## Step 1: Environment Setup on RunPod

### 1.1 Pull the repository

```bash
# On RunPod terminal — first time
git clone https://github.com/amoufaq5/IMI.git
cd IMI

# If you already cloned and need the latest changes
git fetch origin
git checkout claude/cleanup-mixtral-medical-finetune-vsaQb
git pull origin claude/cleanup-mixtral-medical-finetune-vsaQb
```

> **Note:** RunPod pods are ephemeral. Use a **Network Volume** (persistent disk)
> and clone the repo there. Mount it at `/workspace` so it survives pod restarts.

```bash
# Recommended RunPod setup — clone to persistent volume
cd /workspace
git clone https://github.com/amoufaq5/IMI.git
cd /workspace/IMI
```

### 1.2 Install dependencies

**For adapter training only (single A100 80GB):**
```bash
bash scripts/install_training.sh
```

**For foundation training (8× A100 80GB):**
```bash
bash scripts/install_training.sh --foundation
```

The script installs packages in the correct order. Do **not** run
`pip install -r requirements-training.txt` directly — order matters.

Verify the install:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
# Expected: True  8    (or True  1 for single GPU)
```

---

## Step 2: Data Collection

```bash
# Download 40+ open medical datasets (no API keys required)
python scripts/data_collection/collect_datasets.py

# Optional: extract WHO/FDA regulation PDFs
# Place PDFs in data/pdfs/ first
python scripts/data_collection/ingest_pdfs.py

# Optional: generate synthetic training cases
python scripts/data_collection/synthetic_generator.py --num-examples 5000
```

This populates `data/raw/` and `data/processed/`. No credentials required —
all datasets are open-access (MIT, Apache 2.0, CC0 licenses).

---

## Step 3: Data Preparation

```bash
python scripts/training/prepare_medical_data.py
```

This produces exactly **2 output formats** in `data/final/`:

| File | Format | Content |
|------|--------|---------|
| `general_knowledge_train.json` | `{"text": "..."}` | Medical text passages, drug info, encyclopedic facts |
| `general_knowledge_val.json` | `{"text": "..."}` | Validation split (10%) |
| `instruction_train.json` | `{"instruction":"...","input":"...","output":"..."}` | Q&A pairs, instruction-following |
| `instruction_val.json` | `{"instruction":"...","input":"...","output":"..."}` | Validation split (10%) |

Options:
```bash
# Custom validation split
python scripts/training/prepare_medical_data.py --val-split 0.05

# Custom data directory
python scripts/training/prepare_medical_data.py --data-dir /workspace/data
```

---

## Step 4A: Adapter Training Only (Path A — single A100 80GB)

Skip this if doing Path B (foundation → adapter).

### Demo test first (always do this before full training)

```bash
python scripts/training/finetune_mixtral.py --demo
```

This runs 10 steps on 100 examples. Takes ~5 minutes. Confirms:
- Model loads correctly
- Data files are found
- GPU has enough memory
- Training loop runs without error

### Full adapter training

```bash
# Recommended: A100 80GB
python scripts/training/finetune_mixtral.py --gpu-tier A100_80GB

# If you have A100 40GB
python scripts/training/finetune_mixtral.py --gpu-tier A100_40GB

# Train on specific data format only
python scripts/training/finetune_mixtral.py \
    --gpu-tier A100_80GB \
    --data-format instruction    # or "general_knowledge" or "both" (default)

# Resume from checkpoint
python scripts/training/finetune_mixtral.py \
    --gpu-tier A100_80GB \
    --resume-from models/mixtral-medical-qlora/checkpoint-500

# With experiment tracking
python scripts/training/finetune_mixtral.py \
    --gpu-tier A100_80GB \
    --report-to wandb
```

**GPU tier settings:**

| Tier | Batch | Seq len | Grad accum | LoRA r | Effective batch |
|------|-------|---------|------------|--------|-----------------|
| `A100_40GB` | 1 | 1024 | 16 | 16 | 16 |
| `A100_80GB` | 4 | 2048 | 4 | 32 | 16 |
| `H100_80GB` | 8 | 2048 | 2 | 64 | 16 |

Output saved to `models/mixtral-medical-qlora/`.

---

## Step 4B: Foundation Training (Path B — 8× A100 80GB)

This trains all 46.7B parameters — no LoRA, no quantization.
Requires `--foundation` flag in install step and DeepSpeed ZeRO Stage 3.

### Memory layout with ZeRO-3 across 8× A100 80GB

```
Per GPU (80 GB budget):
  Model shard (ZeRO-3):     46.7B × 2B / 8 GPUs  ≈ 11.7 GB
  Gradient shard:                                  ≈ 11.7 GB
  Optimizer shard (Adam):                          ≈ 23.4 GB
  Activations (batch=4, grad_ckpt):               ≈ 15–20 GB
  NCCL buffers:                                    ≈  5 GB
  Total:                                           ≈ 67–72 GB  ✓
```

### Run foundation training

```bash
torchrun --nproc_per_node=8 \
    scripts/training/train_foundation.py \
    --deepspeed configs/deepspeed_zero3.json
```

**With custom options:**
```bash
torchrun --nproc_per_node=8 \
    scripts/training/train_foundation.py \
    --deepspeed configs/deepspeed_zero3.json \
    --epochs 1 \
    --output-dir models/foundation \
    --max-examples 1000000
```

**Resume from checkpoint:**
```bash
torchrun --nproc_per_node=8 \
    scripts/training/train_foundation.py \
    --deepspeed configs/deepspeed_zero3.json \
    --resume-from models/foundation/checkpoint-1000
```

**Expected training time (8× A100 80GB, 1M examples, seq=2048):**
- ~10–18 hours for 1 epoch

Output saved to `models/foundation/`.

### Verify DeepSpeed is working before starting

```bash
# Check all 8 GPUs are visible
nvidia-smi --list-gpus

# Check DeepSpeed can see them
python -c "import torch; print(f'{torch.cuda.device_count()} GPUs')"

# Check ds_report for compiled ops
ds_report
```

### After foundation training — run adapter training

```bash
# Use the foundation model as the base (single A100 80GB)
python scripts/training/finetune_mixtral.py \
    --base-model models/foundation \
    --gpu-tier A100_80GB
```

---

## Step 5: Evaluate

```bash
python scripts/training/evaluate_adapter.py --adapter patient_triage
```

**Pass/fail thresholds:**

| Metric | Threshold |
|--------|-----------|
| Perplexity | ≤ 8.0 |
| MCQ Accuracy | ≥ 55% |
| Triage F1 | ≥ 0.70 |
| Unsafe Claim Rate | ≤ 5% |
| Crisis Detection Recall | 100% |

---

## RunPod — Practical Tips

### Pulling latest code during a session

```bash
# Basic pull (same branch)
cd /workspace/IMI
git pull origin claude/cleanup-mixtral-medical-finetune-vsaQb

# If you get "local changes would be overwritten"
git stash
git pull origin claude/cleanup-mixtral-medical-finetune-vsaQb
git stash pop

# Switch to a different branch
git fetch origin
git checkout <branch-name>
```

### Save data to network volume before pod stops

```bash
# Copy trained model to persistent volume
cp -r models/foundation /workspace/models/foundation

# Copy processed data
cp -r data/final /workspace/data/final
```

### Reconnect to a running training job

```bash
# If tmux was used before starting training
tmux attach -t training

# Start training inside tmux so it survives disconnections
tmux new -s training
torchrun --nproc_per_node=8 scripts/training/train_foundation.py \
    --deepspeed configs/deepspeed_zero3.json
# Detach: Ctrl+B then D
```

### Monitor GPU usage during training

```bash
# In a separate terminal or tmux pane
watch -n 2 nvidia-smi
```

---

## Troubleshooting

### OOM on foundation training
- Reduce batch size: `--batch-size 2` (currently 4)
- Increase gradient accumulation: the effective batch stays the same
- Enable CPU offload in `configs/deepspeed_zero3.json`:
  ```json
  "offload_optimizer": { "device": "cpu" },
  "offload_param": { "device": "cpu" }
  ```
  Warning: CPU offload reduces speed by ~50%

### OOM on adapter training
- Switch to a smaller tier: `--gpu-tier A100_40GB`
- Reduce seq length: override in the tier config

### NCCL errors on multi-GPU
```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1   # disable InfiniBand if not available
torchrun --nproc_per_node=8 ...
```

### DeepSpeed not found
```bash
bash scripts/install_training.sh --foundation
ds_report   # should show available ops
```

### `torch.cuda.is_available()` returns False
```bash
# Check driver
nvidia-smi
# Reinstall torch with correct CUDA tag
bash scripts/install_training.sh --cuda 12.1
```

### Training loss is NaN from step 1
- Lower the learning rate: `--lr 1e-5`
- Check data: run `prepare_medical_data.py` again to re-filter
- Verify bf16 is supported: A100 and H100 support it; older GPUs do not

---

## Directory Structure After Training

```
IMI/
├── configs/
│   └── deepspeed_zero3.json       # ZeRO Stage 3 config
├── data/
│   ├── raw/                       # Downloaded datasets
│   ├── processed/                 # Intermediate processed data
│   └── final/
│       ├── general_knowledge_train.json
│       ├── general_knowledge_val.json
│       ├── instruction_train.json
│       └── instruction_val.json
├── models/
│   ├── foundation/                # Full fine-tuned base (Path B)
│   │   ├── config.json
│   │   ├── model.safetensors.*
│   │   ├── tokenizer.json
│   │   └── foundation_metadata.json
│   └── mixtral-medical-qlora/     # QLoRA adapter (Path A or B step 2)
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── training_metadata.json
├── requirements-training.txt
└── scripts/
    ├── install_training.sh
    ├── data_collection/
    │   ├── collect_datasets.py
    │   ├── ingest_pdfs.py
    │   └── synthetic_generator.py
    └── training/
        ├── prepare_medical_data.py   # produces 2-format data
        ├── train_foundation.py       # full fine-tuning, 8× A100
        ├── finetune_mixtral.py       # QLoRA adapter, 1× A100
        └── evaluate_adapter.py
```
