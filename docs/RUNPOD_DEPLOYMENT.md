# UMI - RunPod Deployment Guide

Complete guide for deploying UMI Medical LLM on RunPod with GPU-accelerated AI inference.

**Last Updated**: January 2026

---

## Quick Reference

| Component | Recommendation |
|-----------|----------------|
| **GPU** | RTX 4090 (24GB) for MVP |
| **Volume** | 50GB minimum |
| **Time to Deploy** | ~30 minutes |
| **Monthly Cost** | ~$500 (24/7) |

---

## Why RunPod?

- **GPU Access**: A100, H100, RTX 4090 on-demand
- **Cost-Effective**: Pay per hour, no long-term commitment
- **Fast Deployment**: Pre-configured ML environments
- **Scalable**: Spin up multiple pods as needed
- **Persistent Storage**: Volumes survive pod restarts

## New Features (January 2026)

- **Dosage Calculator**: Weight/age-based dosing with renal/hepatic adjustments
- **Lab Interpreter**: Automated lab result analysis with clinical guidance
- **ICD-10 Coding**: AI-assisted diagnosis code suggestions
- **Enhanced Training Data**: MedMCQA, BioASQ, ICD-10, LOINC datasets

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        RunPod                                │
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │   API Pod       │    │      GPU Pod (vLLM)             │ │
│  │   (CPU)         │◄──►│   Mistral-7B / Llama-2          │ │
│  │   FastAPI       │    │   Medical Embedding Model       │ │
│  └────────┬────────┘    └─────────────────────────────────┘ │
│           │                                                  │
└───────────┼──────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────┐
│              External Services (Cloud)                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Supabase │  │ Upstash  │  │ Qdrant   │  │ MinIO/S3 │    │
│  │ Postgres │  │ Redis    │  │ Cloud    │  │          │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## Step 1: Prepare External Services

Before deploying to RunPod, set up managed services:

### PostgreSQL (Supabase - Free Tier)
1. Go to https://supabase.com
2. Create new project
3. Copy connection string from Settings > Database

### Redis (Upstash - Free Tier)
1. Go to https://upstash.com
2. Create Redis database
3. Copy REST URL and token

### Vector Database (Qdrant Cloud - Free Tier)
1. Go to https://cloud.qdrant.io
2. Create cluster
3. Copy URL and API key

---

## Step 2: Create RunPod Template

### Option A: GPU Pod (Full AI Capabilities)

Create a new template on RunPod:

**Template Settings:**
- **Name**: UMI-GPU
- **Container Image**: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
- **Docker Command**: Leave empty (we'll use start script)
- **Volume Mount**: `/workspace` (50GB minimum)
- **Expose Ports**: `8000/http`

**Environment Variables:**
```
APP_ENV=production
DEBUG=false
HOST=0.0.0.0
PORT=8000

# External Database (Supabase)
DATABASE_URL=postgresql+asyncpg://postgres:[PASSWORD]@db.[PROJECT].supabase.co:5432/postgres

# External Redis (Upstash)
REDIS_URL=rediss://default:[PASSWORD]@[ENDPOINT].upstash.io:6379

# External Vector DB (Qdrant Cloud)
QDRANT_URL=https://[CLUSTER-ID].aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your-qdrant-api-key

# Security
SECRET_KEY=your-production-secret-key
JWT_SECRET_KEY=your-jwt-secret-key
ENCRYPTION_KEY=your-fernet-key

# AI Models (Local on GPU)
USE_LOCAL_LLM=true
LLM_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HF_TOKEN=your-huggingface-token

# Features
FEATURE_CONSULTATION_ENABLED=true
FEATURE_PHARMA_ENABLED=true
FEATURE_IMAGING_ENABLED=true
```

### Option B: CPU Pod (API Only, External LLM)

For cost savings, run API on CPU and use OpenAI:

**Template Settings:**
- **Name**: UMI-CPU
- **Container Image**: `python:3.11-slim`
- **Volume Mount**: `/workspace` (20GB)
- **Expose Ports**: `8000/http`

**Environment Variables:**
```
# Use OpenAI instead of local models
USE_LOCAL_LLM=false
OPENAI_API_KEY=sk-your-openai-key

# Same external services as above...
```

---

## Step 3: Deployment Script

Create this script in your repo as `scripts/runpod_start.sh`:

```bash
#!/bin/bash
set -e

echo "=== UMI RunPod Startup Script ==="

# Navigate to workspace
cd /workspace

# Clone or update repository
if [ -d "IMI" ]; then
    echo "Updating existing repository..."
    cd IMI
    git pull origin main
else
    echo "Cloning repository..."
    git clone https://github.com/amoufaq5/IMI.git
    cd IMI
fi

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install GPU-specific packages if CUDA available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected, installing CUDA packages..."
    pip install torch==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118
    pip install vllm==0.3.2
    pip install flash-attn --no-build-isolation
fi

# Download models (if using local LLM)
if [ "$USE_LOCAL_LLM" = "true" ]; then
    echo "Pre-downloading AI models..."
    python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_name = os.environ.get('LLM_MODEL_NAME', 'mistralai/Mistral-7B-Instruct-v0.2')
print(f'Downloading {model_name}...')
AutoTokenizer.from_pretrained(model_name)
AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype='auto')
print('Model downloaded successfully!')
"
fi

# Run database migrations
echo "Running database migrations..."
alembic upgrade head

# Start the application
echo "Starting UMI API server..."
exec uvicorn src.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --timeout-keep-alive 30
```

---

## Step 4: Deploy on RunPod

### Via RunPod Console

1. Go to https://runpod.io/console/pods
2. Click **+ Deploy**
3. Select GPU (recommended: RTX 4090 or A100 40GB)
4. Choose your template (UMI-GPU)
5. Set volume size: 50GB
6. Click **Deploy**

### Via RunPod CLI

```bash
# Install RunPod CLI
pip install runpod

# Configure API key
runpod config --api-key YOUR_RUNPOD_API_KEY

# Deploy pod
runpod pod create \
    --name umi-production \
    --gpu-type "NVIDIA RTX 4090" \
    --image runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04 \
    --volume-size 50 \
    --ports 8000/http \
    --env-file .env.runpod
```

---

## Step 5: Post-Deployment Setup

### Connect to Pod

```bash
# Via SSH (get command from RunPod console)
ssh root@[POD_IP] -p [SSH_PORT] -i ~/.ssh/id_rsa

# Or use RunPod web terminal
```

### Run Startup Script

```bash
# First time setup
cd /workspace
git clone https://github.com/your-org/umi.git
cd umi
chmod +x scripts/runpod_start.sh
./scripts/runpod_start.sh
```

### Verify Deployment

```bash
# Check health endpoint
curl http://localhost:8000/health

# Check GPU utilization
nvidia-smi

# View logs
tail -f /workspace/umi/logs/umi.log
```

---

## Step 6: Configure Domain & SSL

### Using RunPod Proxy (Easiest)

RunPod provides automatic HTTPS via their proxy:
- Your endpoint: `https://[POD_ID]-8000.proxy.runpod.net`

### Using Cloudflare Tunnel (Custom Domain)

```bash
# Install cloudflared
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared-linux-amd64
mv cloudflared-linux-amd64 /usr/local/bin/cloudflared

# Authenticate
cloudflared tunnel login

# Create tunnel
cloudflared tunnel create umi-api

# Configure tunnel
cat > ~/.cloudflared/config.yml << EOF
tunnel: [TUNNEL_ID]
credentials-file: /root/.cloudflared/[TUNNEL_ID].json

ingress:
  - hostname: api.umi-health.com
    service: http://localhost:8000
  - service: http_status:404
EOF

# Run tunnel
cloudflared tunnel run umi-api
```

---

## Cost Optimization

### GPU Selection Guide

| GPU | VRAM | Cost/hr | Best For |
|-----|------|---------|----------|
| RTX 3090 | 24GB | ~$0.44 | Development, testing |
| RTX 4090 | 24GB | ~$0.69 | Production (7B models) |
| A100 40GB | 40GB | ~$1.89 | Production (13B+ models) |
| A100 80GB | 80GB | ~$2.49 | Large models, fine-tuning |

### Cost-Saving Tips

1. **Use Spot Instances**: 50-70% cheaper, good for dev
2. **Scale Down Off-Hours**: Stop pods when not needed
3. **Use CPU for API**: Run API on cheap CPU, only GPU for inference
4. **Batch Requests**: Queue requests to maximize GPU utilization

### Estimated Monthly Costs

| Configuration | Hours/Month | Cost |
|--------------|-------------|------|
| RTX 4090 (24/7) | 720 | ~$500 |
| RTX 4090 (12hr/day) | 360 | ~$250 |
| A100 40GB (24/7) | 720 | ~$1,360 |
| CPU + OpenAI | 720 | ~$50 + API costs |

---

## Monitoring & Logging

### View Logs

```bash
# Application logs
tail -f /workspace/umi/logs/umi.log

# System logs
journalctl -f
```

### GPU Monitoring

```bash
# Real-time GPU stats
watch -n 1 nvidia-smi

# GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### Health Checks

```bash
# Create monitoring script
cat > /workspace/health_check.sh << 'EOF'
#!/bin/bash
while true; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
    if [ "$STATUS" != "200" ]; then
        echo "$(date): Health check failed with status $STATUS"
        # Restart application
        pkill -f uvicorn
        cd /workspace/umi && ./scripts/runpod_start.sh &
    fi
    sleep 60
done
EOF
chmod +x /workspace/health_check.sh
```

---

## Troubleshooting

### Pod Won't Start
```bash
# Check container logs in RunPod console
# Common issues:
# - Out of disk space: Increase volume size
# - OOM: Choose larger GPU
# - Port conflict: Check exposed ports
```

### Model Loading Fails
```bash
# Check available VRAM
nvidia-smi

# Use smaller model
export LLM_MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Or use quantized model
export LLM_MODEL_NAME=TheBloke/Mistral-7B-Instruct-v0.2-GPTQ
```

### Database Connection Issues
```bash
# Test connection
python -c "
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
import os

async def test():
    engine = create_async_engine(os.environ['DATABASE_URL'])
    async with engine.connect() as conn:
        result = await conn.execute('SELECT 1')
        print('Database connected!')

asyncio.run(test())
"
```

### Slow Inference
```bash
# Enable Flash Attention
pip install flash-attn --no-build-isolation

# Use vLLM for faster inference
# Already included in requirements.txt
```

---

## Quick Reference

| Resource | URL |
|----------|-----|
| RunPod Console | https://runpod.io/console |
| RunPod Docs | https://docs.runpod.io |
| Supabase | https://supabase.com |
| Upstash Redis | https://upstash.com |
| Qdrant Cloud | https://cloud.qdrant.io |

### Useful Commands

```bash
# Start UMI
cd /workspace/IMI && source venv/bin/activate && uvicorn src.main:app --host 0.0.0.0 --port 8000

# Stop UMI
pkill -f uvicorn

# Update code
cd /workspace/IMI && git pull && pip install -r requirements.txt

# Run training pipeline (optional)
cd /workspace/IMI && python scripts/training/download_datasets.py
python scripts/training/prepare_data.py
python scripts/training/fine_tune.py --epochs 3

# Ingest medical knowledge
python scripts/data_ingestion/ingest_pubmed.py
python scripts/data_ingestion/ingest_drugbank.py

# View GPU usage
nvidia-smi -l 1

# Check disk space
df -h /workspace
```
