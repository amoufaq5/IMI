# UMI Training on RunPod - Complete Guide

## Overview
This guide covers deploying and running the UMI training pipeline on RunPod with 4x NVIDIA H100 GPUs.

---

## 1. Create RunPod Instance

### Recommended Configuration
| Setting | Value |
|---------|-------|
| **GPU Type** | NVIDIA H100 80GB |
| **GPU Count** | 4 |
| **Container Image** | `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04` |
| **Volume Size** | 500GB+ (for data and checkpoints) |
| **Volume Mount** | `/workspace` |

### Alternative (Budget)
| Setting | Value |
|---------|-------|
| **GPU Type** | NVIDIA A100 80GB |
| **GPU Count** | 4 |

---

## 2. Initial Setup Commands

```bash
# Connect to your RunPod instance via SSH or Web Terminal

# Update system
apt-get update && apt-get install -y git wget curl vim htop nvtop

# Verify GPUs
nvidia-smi

# Clone the repository
cd /workspace
git clone https://github.com/amoufaq5/IMI.git umi
cd umi

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools
```

---

## 3. Install Dependencies

```bash
# Install PyTorch with CUDA 12.1 (optimized for H100)
pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Flash Attention 2 (required for 8192 seq length)
pip install flash-attn --no-build-isolation

# Install DeepSpeed
pip install deepspeed

# Install all project dependencies
pip install -r requirements.txt

# Verify installations
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
python -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')"
```

---

## 4. Configure Environment Variables

```bash
# Create .env file
cat > .env << 'EOF'
# API Keys for Data Ingestion
KAGGLE_KEY=KGAT_873bea3468d7cc753df8ed8e4ecb3399
UMLS_API_KEY=your-umls-key-here
DISGENET_API_KEY=your-disgenet-key-here

# Hugging Face (for model download)
HF_TOKEN=your-huggingface-token

# Training settings
CUDA_VISIBLE_DEVICES=0,1,2,3
NCCL_DEBUG=INFO
NCCL_IB_DISABLE=0
NCCL_P2P_DISABLE=0

# Optimize for H100
TORCH_CUDA_ARCH_LIST="9.0"
EOF

# Load environment
source .env
export $(grep -v '^#' .env | xargs)

# Setup Kaggle credentials
mkdir -p ~/.kaggle
echo '{"username":"your-kaggle-username","key":"KGAT_873bea3468d7cc753df8ed8e4ecb3399"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

---

## 5. Run Data Ingestion (Optional but Recommended)

```bash
# Create data directories
mkdir -p data/knowledge_base data/training data/raw

# Run all scrapers (NO LIMITS - this will take several hours)
python scripts/data_ingestion/ingest_all.py \
    --output data/knowledge_base \
    --kaggle-key "$KAGGLE_KEY"

# Or run specific sources only (faster)
python scripts/data_ingestion/ingest_all.py \
    --only pubmed drugs trials chembl opentargets \
    --output data/knowledge_base

# Convert to training format
python scripts/training/convert_ingested_data.py \
    --input data/knowledge_base \
    --output data/training
```

---

## 6. Run Training

### Option A: Full Pipeline (Recommended)
```bash
# Run complete pipeline: ingest -> convert -> train -> evaluate
python scripts/training/train_pipeline.py \
    --num-gpus 4 \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --max-seq-length 8192 \
    --epochs 3 \
    --batch-size 1 \
    --lr 2e-5

# For continuous training (runs indefinitely)
python scripts/training/train_pipeline.py \
    --continuous \
    --num-gpus 4
```

### Option B: Direct DeepSpeed Training
```bash
# Make launch script executable
chmod +x scripts/training/launch_distributed.sh

# Run with DeepSpeed ZeRO-2
NUM_GPUS=4 MODE=full ./scripts/training/launch_distributed.sh

# Or manually with deepspeed
deepspeed --num_gpus=4 scripts/training/fine_tune_distributed.py \
    --deepspeed \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --output_dir outputs/model \
    --max_seq_length 8192 \
    --num_epochs 3 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5
```

### Option C: FSDP Training (Alternative to DeepSpeed)
```bash
torchrun --nproc_per_node=4 scripts/training/fine_tune_distributed.py \
    --fsdp \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --output_dir outputs/model \
    --max_seq_length 8192 \
    --num_epochs 3
```

---

## 7. Monitor Training

### Terminal 1: Training
```bash
# Your training command runs here
```

### Terminal 2: GPU Monitoring
```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Or use nvtop for better visualization
nvtop
```

### Terminal 3: Logs
```bash
# Watch training logs
tail -f outputs/model/training.log

# Check TensorBoard
tensorboard --logdir outputs/model --port 6006 --bind_all
```

---

## 8. Resume from Checkpoint (After Interruption)

```bash
# The pipeline automatically resumes from last checkpoint
python scripts/training/train_pipeline.py --num-gpus 4

# Or manually specify checkpoint
deepspeed --num_gpus=4 scripts/training/fine_tune_distributed.py \
    --deepspeed \
    --resume_from_checkpoint outputs/model/checkpoint-latest \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --output_dir outputs/model
```

---

## 9. Export and Save Model

```bash
# After training completes, the model is saved at:
ls -la outputs/model/

# Copy to persistent storage (if using ephemeral instance)
cp -r outputs/model /workspace/saved_models/

# Or upload to Hugging Face Hub
pip install huggingface_hub
huggingface-cli login
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='outputs/model',
    repo_id='your-username/umi-mistral-7b',
    repo_type='model'
)
"
```

---

## 10. Quick Start Script (Copy-Paste Ready)

```bash
#!/bin/bash
# Save as: setup_and_train.sh

set -e

echo "=== UMI Training Setup on RunPod ==="

# 1. System setup
apt-get update && apt-get install -y git wget curl htop nvtop

# 2. Clone repo
cd /workspace
git clone https://github.com/amoufaq5/IMI.git umi || true
cd umi

# 3. Python environment
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools

# 4. Install dependencies
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn --no-build-isolation
pip install deepspeed
pip install -r requirements.txt

# 5. Set environment
export KAGGLE_KEY="KGAT_873bea3468d7cc753df8ed8e4ecb3399"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 6. Create directories
mkdir -p data/knowledge_base data/training outputs

# 7. Run training pipeline
python scripts/training/train_pipeline.py \
    --num-gpus 4 \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --max-seq-length 8192 \
    --epochs 3

echo "=== Training Complete ==="
```

**Run with:**
```bash
chmod +x setup_and_train.sh
./setup_and_train.sh
```

---

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size or use ZeRO-3
deepspeed --num_gpus=4 scripts/training/fine_tune_distributed.py \
    --deepspeed configs/deepspeed_zero3.json \
    --batch_size 1 \
    --gradient_accumulation_steps 32
```

### NCCL Errors
```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
```

### Flash Attention Build Fails
```bash
# Install pre-built wheel
pip install flash-attn --no-build-isolation

# Or disable flash attention
python scripts/training/fine_tune_distributed.py --no-flash-attention
```

### Checkpoint Corruption
```bash
# Delete corrupted checkpoint and restart
rm -rf outputs/model/checkpoint-*
python scripts/training/train_pipeline.py --num-gpus 4
```

---

## Estimated Training Time (4x H100)

| Data Size | Epochs | Estimated Time |
|-----------|--------|----------------|
| 100K samples | 3 | ~2 hours |
| 500K samples | 3 | ~8 hours |
| 1M samples | 3 | ~16 hours |
| Full dataset | 3 | ~24-48 hours |

---

## Cost Estimate (RunPod)

| GPU | Price/hr | 4 GPUs/hr | 24hr Training |
|-----|----------|-----------|---------------|
| H100 80GB | ~$3.50 | ~$14.00 | ~$336 |
| A100 80GB | ~$2.00 | ~$8.00 | ~$192 |
| A100 40GB | ~$1.50 | ~$6.00 | ~$144 |

---

## Support

- **Repository**: https://github.com/amoufaq5/IMI
- **Issues**: Open a GitHub issue for bugs
- **RunPod Docs**: https://docs.runpod.io
