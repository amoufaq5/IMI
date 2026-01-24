# RunPod Deployment Guide

Complete guide for deploying and training IMI on RunPod GPU cloud.

---

## Step 1: Create RunPod Account

1. Go to [runpod.io](https://runpod.io)
2. Sign up and add credits ($10-50 recommended for testing)

---

## Step 2: Launch a GPU Pod

### Recommended Configurations

| Use Case | GPU | VRAM | Cost/hr | Pod Type |
|----------|-----|------|---------|----------|
| **Development** | RTX 3090 | 24GB | ~$0.30 | Community |
| **Training (7B)** | RTX A5000 | 24GB | ~$0.40 | Secure |
| **Training (fast)** | A100 40GB | 40GB | ~$1.50 | Secure |
| **Training (70B)** | A100 80GB | 80GB | ~$2.00 | Secure |

### Launch Steps

1. Click **"Deploy"** → **"GPU Pods"**
2. Select template: **"RunPod Pytorch 2.1"** or **"RunPod Transformers"**
3. Choose GPU (A100 40GB recommended for training)
4. Set volume size: **100GB** (for model + data)
5. Click **"Deploy"**

---

## Step 3: Connect to Your Pod

### Option A: Web Terminal
Click **"Connect"** → **"Start Web Terminal"**

### Option B: SSH (Recommended)
```bash
# Get SSH command from RunPod dashboard
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_rsa
```

---

## Step 4: Clone and Setup IMI

```bash
# Navigate to workspace (persistent storage)
cd /workspace

# Clone your repository
git clone https://github.com/YOUR_USERNAME/imi.git
cd imi

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install additional training dependencies
pip install bitsandbytes accelerate peft trl
```

---

## Step 5: Download Meditron Model

```bash
# Option 1: Let it download automatically during training
# (Downloads to ~/.cache/huggingface/)

# Option 2: Pre-download to workspace (persists across restarts)
cd /workspace
huggingface-cli download epfl-llm/meditron-7b --local-dir ./models/meditron-7b

# For 70B (requires 150GB+ storage)
# huggingface-cli download epfl-llm/meditron-70b --local-dir ./models/meditron-70b
```

---

## Step 6: Prepare Training Data

```bash
cd /workspace/imi

# Collect open datasets
python scripts/data_collection/collect_datasets.py

# Generate synthetic data
python scripts/data_collection/synthetic_generator.py --num-examples 5000

# If you have PDFs, upload them first (see Step 7)
# python scripts/data_collection/ingest_pdfs.py

# Prepare final training data
python scripts/training/prepare_data.py
```

---

## Step 7: Upload Your Data (Optional)

### Upload PDFs via SCP
```bash
# From your local machine
scp -P <port> ~/Documents/WHO_*.pdf root@<pod-ip>:/workspace/imi/data/pdfs/
scp -P <port> ~/Documents/FDA_*.pdf root@<pod-ip>:/workspace/imi/data/pdfs/
```

### Upload via RunPod File Browser
1. Click **"Connect"** → **"HTTP Service [8888]"**
2. Use JupyterLab file browser to upload

---

## Step 8: Train Adapters

```bash
cd /workspace/imi
source venv/bin/activate

# Train single adapter (4-6 hours on A100)
python scripts/training/train_lora.py --adapter patient_triage

# Train with local model path
python scripts/training/train_lora.py \
    --adapter patient_triage \
    --base-model /workspace/models/meditron-7b

# Train all adapters
python scripts/training/train_lora.py --adapter all

# Train with 8-bit (faster on A100)
python scripts/training/train_lora.py \
    --adapter all \
    --use-8bit \
    --batch-size 8
```

### Training in Background (Recommended)
```bash
# Use screen to keep training running after disconnect
screen -S training

# Run training
python scripts/training/train_lora.py --adapter all

# Detach: Ctrl+A, then D
# Reattach: screen -r training
```

---

## Step 9: Download Trained Adapters

```bash
# From your local machine
scp -r -P <port> root@<pod-ip>:/workspace/imi/adapters ./adapters/
```

Or use RunPod's file browser to download.

---

## Step 10: Run the API Server

```bash
cd /workspace/imi
source venv/bin/activate

# Start server
python scripts/run_server.py --host 0.0.0.0 --port 8000

# Or with uvicorn directly
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Access via RunPod's **"Connect"** → **"HTTP Service [8000]"**

---

## Complete Setup Script

Save this as `setup_runpod.sh`:

```bash
#!/bin/bash
set -e

echo "=== IMI RunPod Setup ==="

# Navigate to workspace
cd /workspace

# Clone repo (replace with your repo URL)
if [ ! -d "imi" ]; then
    git clone https://github.com/YOUR_USERNAME/imi.git
fi
cd imi

# Setup Python environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install bitsandbytes accelerate peft trl

# Create data directories
mkdir -p data/pdfs data/raw data/processed data/synthetic data/final
mkdir -p adapters models

# Collect data
echo "=== Collecting Datasets ==="
python scripts/data_collection/collect_datasets.py

echo "=== Generating Synthetic Data ==="
python scripts/data_collection/synthetic_generator.py --num-examples 3000

echo "=== Preparing Training Data ==="
python scripts/training/prepare_data.py

echo "=== Setup Complete ==="
echo "To train: python scripts/training/train_lora.py --adapter patient_triage"
```

Run with:
```bash
chmod +x setup_runpod.sh
./setup_runpod.sh
```

---

## Cost Estimates

| Task | GPU | Time | Cost |
|------|-----|------|------|
| Data collection | Any | 30 min | ~$0.20 |
| Train 1 adapter (7B) | A100 40GB | 4-6 hrs | ~$8 |
| Train all adapters (7B) | A100 40GB | 24-30 hrs | ~$45 |
| Train 1 adapter (70B) | A100 80GB | 12-18 hrs | ~$36 |

**Tip:** Use spot instances for 50-70% savings (but may be interrupted).

---

## Troubleshooting

### Out of Memory
```bash
# Use 4-bit quantization (default)
python scripts/training/train_lora.py --adapter patient_triage --use-4bit

# Reduce batch size
python scripts/training/train_lora.py --adapter patient_triage --batch-size 1
```

### Pod Disconnected During Training
```bash
# If using screen, reattach
screen -r training

# Check if still running
ps aux | grep python
```

### Model Download Failed
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/
huggingface-cli download epfl-llm/meditron-7b --local-dir /workspace/models/meditron-7b
```

### Storage Full
```bash
# Check usage
df -h

# Clear HuggingFace cache
rm -rf ~/.cache/huggingface/

# Remove old checkpoints
rm -rf adapters/*/checkpoint-*
```

---

## Quick Reference

```bash
# SSH to pod
ssh root@<ip> -p <port>

# Activate environment
cd /workspace/imi && source venv/bin/activate

# Train
python scripts/training/train_lora.py --adapter patient_triage

# Run server
python scripts/run_server.py --port 8000

# Download adapters (from local)
scp -r -P <port> root@<ip>:/workspace/imi/adapters ./
```
