# UMI RunPod Complete Deployment Guide

**Complete step-by-step guide for deploying UMI Medical AI Platform on RunPod**

**Version**: 1.0  
**Last Updated**: January 2026  
**Repository**: https://github.com/amoufaq5/IMI

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [GPU Selection Guide](#gpu-selection-guide)
4. [External Services Setup](#external-services-setup)
5. [RunPod Account Setup](#runpod-account-setup)
6. [Pod Template Creation](#pod-template-creation)
7. [Deployment Steps](#deployment-steps)
8. [Application Configuration](#application-configuration)
9. [AI Model Setup](#ai-model-setup)
10. [Database Initialization](#database-initialization)
11. [Starting the Application](#starting-the-application)
12. [Domain & SSL Configuration](#domain--ssl-configuration)
13. [Testing & Verification](#testing--verification)
14. [Monitoring & Logging](#monitoring--logging)
15. [Troubleshooting](#troubleshooting)
16. [Cost Optimization](#cost-optimization)
17. [Quick Reference](#quick-reference)

---

## Overview

UMI (Universal Medical Intelligence) is an AI-powered medical consultation and pharmaceutical QA/QC platform. This guide covers deploying UMI on RunPod with GPU-accelerated inference.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         RunPod GPU Pod                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    UMI Application                         │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │  │
│  │  │  FastAPI    │  │  vLLM       │  │  Embedding      │   │  │
│  │  │  Backend    │  │  Inference  │  │  Service        │   │  │
│  │  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘   │  │
│  │         │                │                   │            │  │
│  │         └────────────────┴───────────────────┘            │  │
│  │                          │                                 │  │
│  └──────────────────────────┼─────────────────────────────────┘  │
│                             │                                    │
│                    GPU: RTX 4090 / A100                         │
└─────────────────────────────┼────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    External Cloud Services                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Supabase   │  │   Upstash    │  │   Qdrant     │          │
│  │  PostgreSQL  │  │    Redis     │  │  Vector DB   │          │
│  │   Database   │  │    Cache     │  │   Search     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### Key Features

- **AI Medical Consultation**: ASMETHOD-based symptom assessment
- **Drug Information**: Search, interactions, OTC recommendations
- **Dosage Calculator**: Weight/age-based dosing, renal/hepatic adjustments
- **Lab Interpreter**: Lab result analysis with clinical guidance
- **ICD-10 Coding**: Diagnosis code lookup and AI-assisted suggestions
- **Pharma QA/QC**: Document generation, facility compliance
- **Medical Imaging**: X-ray, dermoscopy analysis (optional)
- **RAG Pipeline**: Medical knowledge retrieval

---

## Prerequisites

Before starting, ensure you have:

- [ ] GitHub account with access to the UMI repository
- [ ] RunPod account with payment method
- [ ] Hugging Face account with API token
- [ ] Basic command line knowledge
- [ ] ~$50-100 initial credits for RunPod

### Accounts to Create

| Service | URL | Purpose |
|---------|-----|---------|
| RunPod | https://runpod.io | GPU hosting |
| Supabase | https://supabase.com | PostgreSQL database |
| Upstash | https://upstash.com | Redis cache |
| Qdrant Cloud | https://cloud.qdrant.io | Vector database |
| Hugging Face | https://huggingface.co | AI model access |

---

## GPU Selection Guide

### Recommended GPU for MVP

| Phase | GPU | VRAM | Cost/hr | Recommendation |
|-------|-----|------|---------|----------------|
| **Development** | RTX 3090 | 24GB | ~$0.44 | Testing only |
| **MVP Launch** | **RTX 4090** | 24GB | ~$0.69 | **Best value** |
| **Scale** | A100 40GB | 40GB | ~$1.89 | High traffic |
| **Enterprise** | H100 | 80GB | ~$3.50 | Maximum performance |

### Performance Comparison (Mistral-7B)

| GPU | Tokens/sec | Batch Size | Monthly Cost (24/7) |
|-----|------------|------------|---------------------|
| RTX 3090 | ~30 | 1-2 | ~$320 |
| **RTX 4090** | **~50** | **2-4** | **~$500** |
| A100 40GB | ~80 | 4-8 | ~$1,360 |
| H100 | ~150 | 16+ | ~$2,500 |

### Capacity Calculation

```
RTX 4090 Throughput:
├── Tokens per second: ~50
├── Average response: 500 tokens
├── Time per response: 10 seconds
├── Requests per minute: 6
├── Requests per hour: 360
└── Daily capacity: 8,640 consultations

MVP Target: 100-500 users/day = 1 GPU handles easily
```

### When to Scale

Add more GPUs when:
- Response latency > 15 seconds consistently
- Queue depth > 10 requests
- Daily users > 5,000

---

## External Services Setup

### Step 1: PostgreSQL (Supabase)

1. Go to https://supabase.com
2. Click **"Start your project"** → Sign up/Login
3. Click **"New Project"**
4. Configure:
   - **Organization**: Create or select
   - **Project name**: `umi-production`
   - **Database password**: Generate a strong password (save this!)
   - **Region**: Choose closest to your users
5. Wait for project creation (~2 minutes)
6. Go to **Settings** → **Database**
7. Copy the **Connection string (URI)**

**Format your connection string:**
```
Original:  postgresql://postgres:[PASSWORD]@db.xxxx.supabase.co:5432/postgres
For UMI:   postgresql+asyncpg://postgres:[PASSWORD]@db.xxxx.supabase.co:5432/postgres
```

### Step 2: Redis (Upstash)

1. Go to https://upstash.com
2. Sign up/Login
3. Click **"Create Database"**
4. Configure:
   - **Name**: `umi-redis`
   - **Type**: Regional
   - **Region**: `eu-west-1` (or closest to your users)
   - **TLS**: Enabled (default)
5. After creation, copy:
   - **UPSTASH_REDIS_REST_URL**
   - **UPSTASH_REDIS_REST_TOKEN**

**Connection string format:**
```
rediss://default:[TOKEN]@[ENDPOINT].upstash.io:6379
```

### Step 3: Vector Database (Qdrant Cloud)

1. Go to https://cloud.qdrant.io
2. Sign up/Login
3. Click **"Create Cluster"**
4. Configure:
   - **Name**: `umi-vectors`
   - **Cloud Provider**: AWS
   - **Region**: `eu-west-1`
   - **Configuration**: Free tier (1GB) for MVP
5. After creation, copy:
   - **Cluster URL**: `https://xxx.aws.cloud.qdrant.io:6333`
   - **API Key**: From the dashboard

### Step 4: Hugging Face Token

1. Go to https://huggingface.co
2. Sign up/Login
3. Go to **Settings** → **Access Tokens**
4. Click **"New token"**
5. Configure:
   - **Name**: `umi-runpod`
   - **Type**: Read
6. Copy the token

---

## RunPod Account Setup

### Create Account

1. Go to https://runpod.io
2. Click **"Sign Up"**
3. Verify email
4. Go to **Settings** → **Billing**
5. Add payment method
6. Add **$50-100** credits to start

### Generate API Key (Optional - for CLI)

1. Go to **Settings** → **API Keys**
2. Click **"Create API Key"**
3. Save the key securely

---

## Pod Template Creation

### Create Template

1. Go to RunPod Console → **Templates**
2. Click **"New Template"**
3. Fill in the following:

**Basic Settings:**
```
Template Name: UMI-Medical-MVP

Container Image: 
runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

Docker Command: (leave empty for now)

Volume Mount Path: /workspace

Expose HTTP Ports: 8000
```

**Environment Variables:**

Copy and paste these, replacing placeholders with your actual values:

```bash
# Application
APP_NAME=UMI
APP_ENV=production
DEBUG=false
HOST=0.0.0.0
PORT=8000

# Database (from Supabase)
DATABASE_URL=postgresql+asyncpg://postgres:YOUR_PASSWORD@db.YOUR_PROJECT.supabase.co:5432/postgres

# Redis (from Upstash)
REDIS_URL=rediss://default:YOUR_TOKEN@YOUR_ENDPOINT.upstash.io:6379

# Vector DB (from Qdrant)
QDRANT_URL=https://YOUR_CLUSTER.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your-qdrant-api-key

# Security Keys (generate these - see below)
SECRET_KEY=your-secret-key-min-32-chars
JWT_SECRET_KEY=your-jwt-secret-key-min-32-chars
ENCRYPTION_KEY=your-fernet-encryption-key

# AI Configuration
USE_LOCAL_LLM=true
LLM_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HF_TOKEN=your-huggingface-token

# Features
FEATURE_CONSULTATION_ENABLED=true
FEATURE_PHARMA_ENABLED=true
FEATURE_IMAGING_ENABLED=false
```

### Generate Security Keys

Run this Python script locally to generate secure keys:

```python
import secrets
from cryptography.fernet import Fernet

print("=== UMI Security Keys ===")
print(f"SECRET_KEY={secrets.token_urlsafe(32)}")
print(f"JWT_SECRET_KEY={secrets.token_urlsafe(32)}")
print(f"ENCRYPTION_KEY={Fernet.generate_key().decode()}")
```

**Example output:**
```
SECRET_KEY=Abc123XyzSecureRandomString...
JWT_SECRET_KEY=Def456AnotherSecureString...
ENCRYPTION_KEY=base64EncodedFernetKey...
```

4. Click **"Save Template"**

---

## Deployment Steps

### Step 1: Deploy Pod

1. Go to RunPod Console → **Pods**
2. Click **"+ Deploy"**
3. Select GPU:
   ```
   ✓ NVIDIA RTX 4090
     24GB VRAM | ~$0.69/hr
   ```
4. Select your template: **UMI-Medical-MVP**
5. Configure:
   - **Volume Size**: 50 GB
   - **Volume Mount**: /workspace
6. Choose deployment type:
   - **On-Demand**: Guaranteed availability (~$0.69/hr)
   - **Spot**: 50% cheaper but can be interrupted (~$0.35/hr)
7. Click **"Deploy"**
8. Wait for pod to start (~1-2 minutes)

### Step 2: Connect to Pod

**Option A: Web Terminal (Easiest)**
1. Click on your running pod
2. Click **"Connect"** → **"Web Terminal"**

**Option B: SSH**
1. Click **"Connect"** → **"SSH"**
2. Copy the SSH command:
   ```bash
   ssh root@[POD_IP] -p [SSH_PORT] -i ~/.ssh/id_rsa
   ```

---

## Application Configuration

### Initial System Setup

```bash
# Update system packages
apt-get update && apt-get install -y git curl wget

# Navigate to workspace
cd /workspace

# Clone the UMI repository
git clone https://github.com/amoufaq5/IMI.git
cd IMI

# Create Python virtual environment
python -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt
```

### Install GPU-Optimized Packages

```bash
# Install PyTorch with CUDA support
pip install torch==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install vLLM for fast inference
pip install vllm==0.3.2

# Install Flash Attention (optional, improves speed)
pip install flash-attn --no-build-isolation
```

---

## AI Model Setup

### Download Mistral-7B Model

This takes approximately 10-15 minutes:

```bash
python << 'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Get model name from environment or use default
model_name = os.environ.get('LLM_MODEL_NAME', 'mistralai/Mistral-7B-Instruct-v0.2')

print(f"Downloading {model_name}...")
print("This may take 10-15 minutes...")

# Download tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("✓ Tokenizer downloaded")

# Download model with GPU optimization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("✓ Model downloaded")

# Verify GPU memory usage
gpu_memory = torch.cuda.memory_allocated() / 1e9
print(f"✓ GPU Memory Used: {gpu_memory:.2f} GB")
print("✓ Model ready for inference!")
EOF
```

### Download Embedding Model

```bash
python << 'EOF'
from sentence_transformers import SentenceTransformer

model_name = "sentence-transformers/all-MiniLM-L6-v2"
print(f"Downloading {model_name}...")

model = SentenceTransformer(model_name)
print("✓ Embedding model downloaded")

# Test embedding
test_embedding = model.encode("Test medical query")
print(f"✓ Embedding dimension: {len(test_embedding)}")
EOF
```

### Verify GPU Status

```bash
# Check GPU is recognized
nvidia-smi

# Expected output shows:
# - GPU: NVIDIA RTX 4090
# - Memory: ~14GB used (after model loading)
# - Utilization: Ready for inference
```

---

## Database Initialization

### Run Database Migrations

```bash
# Ensure you're in the project directory with venv activated
cd /workspace/IMI
source venv/bin/activate

# Run Alembic migrations
alembic upgrade head
```

### Verify Database Connection

```bash
python << 'EOF'
import asyncio
import os

async def test_database():
    from sqlalchemy.ext.asyncio import create_async_engine
    
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        print("✗ DATABASE_URL not set")
        return
    
    try:
        engine = create_async_engine(database_url)
        async with engine.connect() as conn:
            result = await conn.execute("SELECT 1")
            print("✓ Database connection successful!")
    except Exception as e:
        print(f"✗ Database connection failed: {e}")

asyncio.run(test_database())
EOF
```

### Initialize RAG Collections (Optional)

```bash
python << 'EOF'
import asyncio
from src.core.database import init_db
from src.ai.rag_service import RAGService

async def init_rag():
    print("Initializing RAG collections...")
    await init_db()
    
    rag = RAGService()
    await rag.initialize()
    print("✓ RAG collections initialized")

asyncio.run(init_rag())
EOF
```

---

## Starting the Application

### Option 1: Foreground (Testing)

```bash
cd /workspace/IMI
source venv/bin/activate

# Start with live output
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### Option 2: Background (Production)

```bash
cd /workspace/IMI
source venv/bin/activate

# Start in background with logging
nohup uvicorn src.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 2 \
    --timeout-keep-alive 30 \
    > /workspace/umi.log 2>&1 &

# Verify it's running
sleep 3
curl http://localhost:8000/health
```

### Option 3: Create Startup Script (Recommended)

```bash
cat > /workspace/start_umi.sh << 'EOF'
#!/bin/bash
set -e

echo "=== Starting UMI Application ==="

# Navigate to project
cd /workspace/IMI

# Activate virtual environment
source venv/bin/activate

# Pull latest code
echo "Pulling latest code..."
git pull origin main

# Install any new dependencies
pip install -r requirements.txt --quiet

# Run database migrations
echo "Running migrations..."
alembic upgrade head

# Start application
echo "Starting UMI server..."
exec uvicorn src.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 2 \
    --timeout-keep-alive 30
EOF

chmod +x /workspace/start_umi.sh
```

Run the startup script:
```bash
/workspace/start_umi.sh
```

### Configure Auto-Start on Pod Restart

Update your RunPod template Docker Command to:
```
bash -c "sleep 10 && cd /workspace && ./start_umi.sh"
```

---

## Domain & SSL Configuration

### Option 1: RunPod Proxy (Easiest)

RunPod automatically provides HTTPS:

```
Your API URL: https://[POD_ID]-8000.proxy.runpod.net

Example: https://abc123xyz-8000.proxy.runpod.net
```

Find your POD_ID in the RunPod console.

### Option 2: Cloudflare Tunnel (Custom Domain)

For a custom domain like `api.yourdomain.com`:

```bash
# Install cloudflared
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared-linux-amd64
mv cloudflared-linux-amd64 /usr/local/bin/cloudflared

# Login to Cloudflare (opens browser)
cloudflared tunnel login

# Create tunnel
cloudflared tunnel create umi-api

# Note the tunnel ID from output

# Configure tunnel
cat > ~/.cloudflared/config.yml << EOF
tunnel: YOUR_TUNNEL_ID
credentials-file: /root/.cloudflared/YOUR_TUNNEL_ID.json

ingress:
  - hostname: api.yourdomain.com
    service: http://localhost:8000
  - service: http_status:404
EOF

# Add DNS record in Cloudflare dashboard:
# Type: CNAME
# Name: api
# Target: YOUR_TUNNEL_ID.cfargotunnel.com

# Run tunnel in background
nohup cloudflared tunnel run umi-api > /workspace/cloudflare.log 2>&1 &
```

---

## Testing & Verification

### Health Check

```bash
# Local test
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","service":"umi"}

# External test (replace with your POD_ID)
curl https://[POD_ID]-8000.proxy.runpod.net/health
```

### Full API Test Suite

```bash
# Set your base URL
BASE_URL="https://[POD_ID]-8000.proxy.runpod.net"

# 1. Health check
echo "=== Health Check ==="
curl -s $BASE_URL/health | jq

# 2. Register a test user
echo "=== Register User ==="
curl -s -X POST $BASE_URL/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "SecurePass123!",
    "confirm_password": "SecurePass123!"
  }' | jq

# 3. Login and get token
echo "=== Login ==="
TOKEN=$(curl -s -X POST $BASE_URL/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "SecurePass123!"
  }' | jq -r '.access_token')
echo "Token: ${TOKEN:0:20}..."

# 4. Start a consultation
echo "=== Start Consultation ==="
curl -s -X POST $BASE_URL/api/v1/consultations \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "initial_message": "I have been experiencing headaches for the past 3 days"
  }' | jq

# 5. Search for drugs
echo "=== Drug Search ==="
curl -s "$BASE_URL/api/v1/drugs/search?q=paracetamol" \
  -H "Authorization: Bearer $TOKEN" | jq

# 6. Check drug interactions
echo "=== Drug Interactions ==="
curl -s -X POST $BASE_URL/api/v1/drugs/interactions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "drug_names": ["aspirin", "ibuprofen"]
  }' | jq

# 7. Calculate medication dosage
echo "=== Dosage Calculator ==="
curl -s -X POST $BASE_URL/api/v1/clinical/dosage/calculate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "drug_name": "amoxicillin",
    "patient": {
      "age_years": 35,
      "weight_kg": 70,
      "sex": "male",
      "serum_creatinine": 1.0
    }
  }' | jq

# 8. Interpret lab results
echo "=== Lab Interpreter ==="
curl -s -X POST $BASE_URL/api/v1/clinical/labs/interpret \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "results": [
      {"test_name": "glucose", "value": 126},
      {"test_name": "hemoglobin", "value": 14.5},
      {"test_name": "potassium", "value": 4.2}
    ],
    "sex": "male"
  }' | jq

# 9. Get ICD-10 code suggestions
echo "=== ICD-10 Coding ==="
curl -s -X POST $BASE_URL/api/v1/clinical/icd10/suggest \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": "patient presents with chest pain and shortness of breath",
    "diagnoses": ["hypertension"]
  }' | jq
```

### OpenAPI Documentation

Access interactive API docs:
- **Swagger UI**: `https://[POD_ID]-8000.proxy.runpod.net/docs`
- **ReDoc**: `https://[POD_ID]-8000.proxy.runpod.net/redoc`

---

## Monitoring & Logging

### View Application Logs

```bash
# Real-time logs
tail -f /workspace/umi.log

# Last 100 lines
tail -n 100 /workspace/umi.log

# Search for errors
grep -i error /workspace/umi.log
```

### GPU Monitoring

```bash
# Real-time GPU stats (updates every second)
watch -n 1 nvidia-smi

# One-time GPU status
nvidia-smi

# GPU memory and utilization only
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### System Resources

```bash
# Disk usage
df -h /workspace

# Memory usage
free -h

# Running processes
htop
```

### Create Health Check Script

```bash
cat > /workspace/health_check.sh << 'EOF'
#!/bin/bash

while true; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    if [ "$STATUS" = "200" ]; then
        echo "[$TIMESTAMP] ✓ Health check passed"
    else
        echo "[$TIMESTAMP] ✗ Health check failed (status: $STATUS)"
        echo "[$TIMESTAMP] Attempting restart..."
        
        # Kill existing process
        pkill -f uvicorn
        sleep 2
        
        # Restart
        cd /workspace/IMI
        source venv/bin/activate
        nohup uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 2 \
            > /workspace/umi.log 2>&1 &
        
        echo "[$TIMESTAMP] Restart initiated"
    fi
    
    sleep 60
done
EOF

chmod +x /workspace/health_check.sh

# Run in background
nohup /workspace/health_check.sh > /workspace/health_check.log 2>&1 &
```

---

## Troubleshooting

### Common Issues and Solutions

#### Pod Won't Start

```bash
# Check RunPod console for error messages
# Common causes:
# - Out of disk space → Increase volume size
# - GPU not available → Try different GPU or region
# - Template error → Verify environment variables
```

#### Model Loading Fails

```bash
# Check available VRAM
nvidia-smi

# If OOM (Out of Memory), use smaller/quantized model:
export LLM_MODEL_NAME=TheBloke/Mistral-7B-Instruct-v0.2-GPTQ

# Or use TinyLlama for testing:
export LLM_MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

#### Database Connection Issues

```bash
# Test connection manually
python << 'EOF'
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
import os

async def test():
    url = os.environ.get('DATABASE_URL')
    print(f"Testing: {url[:50]}...")
    
    try:
        engine = create_async_engine(url)
        async with engine.connect() as conn:
            await conn.execute("SELECT 1")
        print("✓ Connection successful!")
    except Exception as e:
        print(f"✗ Connection failed: {e}")

asyncio.run(test())
EOF

# Common fixes:
# - Check DATABASE_URL format (must use postgresql+asyncpg://)
# - Verify Supabase project is running
# - Check IP allowlist in Supabase (allow 0.0.0.0/0 for RunPod)
```

#### Slow Inference

```bash
# Install Flash Attention
pip install flash-attn --no-build-isolation

# Use vLLM for faster inference (already in requirements)
# Verify vLLM is installed:
python -c "import vllm; print('vLLM version:', vllm.__version__)"
```

#### Out of Disk Space

```bash
# Check disk usage
df -h /workspace

# Clean up
# Remove old model caches
rm -rf ~/.cache/huggingface/hub/*

# Remove pip cache
pip cache purge

# If still full, increase volume size in RunPod console
```

#### Application Crashes

```bash
# Check logs for errors
tail -n 200 /workspace/umi.log | grep -i error

# Check system memory
free -h

# Restart with fewer workers
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 1
```

---

## Cost Optimization

### Monthly Cost Estimates

| Configuration | Hours/Month | GPU Cost | Services | Total |
|--------------|-------------|----------|----------|-------|
| RTX 4090 (24/7) | 720 | ~$500 | ~$0 | ~$500 |
| RTX 4090 (12hr/day) | 360 | ~$250 | ~$0 | ~$250 |
| RTX 4090 (Spot, 24/7) | 720 | ~$250 | ~$0 | ~$250 |
| CPU + OpenAI | 720 | ~$50 | ~$100 API | ~$150 |

### Cost-Saving Strategies

#### 1. Use Spot Instances

50-70% cheaper than on-demand:
```
On-Demand: $0.69/hr → Spot: ~$0.25/hr
Monthly savings: ~$300
```

**Trade-off**: Can be interrupted with 30-second notice.

#### 2. Scale Down Off-Hours

Stop pod when not needed:
```bash
# Via RunPod console: Click "Stop" on your pod
# Or via API/CLI

# Restart when needed
# Your /workspace volume persists
```

#### 3. Use OpenAI API Instead

For lower traffic, skip GPU entirely:
```bash
# In environment variables:
USE_LOCAL_LLM=false
OPENAI_API_KEY=sk-your-openai-key

# Use CPU-only pod (~$0.07/hr)
```

#### 4. Optimize Model Loading

Use quantized models for lower VRAM:
```bash
# 4-bit quantized (uses ~4GB instead of ~14GB)
LLM_MODEL_NAME=TheBloke/Mistral-7B-Instruct-v0.2-GPTQ
```

---

## Quick Reference

### Essential Commands

```bash
# === Application Management ===

# Start UMI
cd /workspace/IMI && source venv/bin/activate && \
uvicorn src.main:app --host 0.0.0.0 --port 8000

# Stop UMI
pkill -f uvicorn

# Restart UMI
pkill -f uvicorn && sleep 2 && /workspace/start_umi.sh

# View logs
tail -f /workspace/umi.log

# === Code Updates ===

# Pull latest code
cd /workspace/IMI && git pull origin main

# Update dependencies
pip install -r requirements.txt

# Run migrations
alembic upgrade head

# === Monitoring ===

# Check GPU
nvidia-smi

# Check disk
df -h /workspace

# Check memory
free -h

# Check running processes
ps aux | grep uvicorn

# === Data Ingestion (Optional) ===

# Ingest PubMed articles
python scripts/data_ingestion/ingest_pubmed.py

# Ingest drug information
python scripts/data_ingestion/ingest_drugbank.py

# === Training (Optional) ===

# Download training datasets (includes MedMCQA, BioASQ, ICD-10, LOINC)
python scripts/training/download_datasets.py

# Prepare training data
python scripts/training/prepare_data.py

# Fine-tune model
python scripts/training/fine_tune.py --epochs 3
```

### New Clinical Services API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/clinical/dosage/calculate` | POST | Calculate medication dosage |
| `/api/v1/clinical/dosage/check` | POST | Check if dose is safe |
| `/api/v1/clinical/dosage/drugs` | GET | List available drugs |
| `/api/v1/clinical/labs/interpret` | POST | Interpret lab panel |
| `/api/v1/clinical/labs/interpret/single` | POST | Interpret single lab |
| `/api/v1/clinical/labs/anion-gap` | POST | Calculate anion gap |
| `/api/v1/clinical/labs/corrected-calcium` | POST | Calculate corrected calcium |
| `/api/v1/clinical/labs/tests` | GET | List available lab tests |
| `/api/v1/clinical/icd10/lookup/{code}` | GET | Look up ICD-10 code |
| `/api/v1/clinical/icd10/search` | POST | Search ICD-10 codes |
| `/api/v1/clinical/icd10/suggest` | POST | AI-suggest ICD-10 codes |
| `/api/v1/clinical/icd10/validate/{code}` | GET | Validate ICD-10 code |
| `/api/v1/clinical/icd10/encode-encounter` | POST | Code clinical encounter |

### Important URLs

| Resource | URL |
|----------|-----|
| Your API | `https://[POD_ID]-8000.proxy.runpod.net` |
| API Docs | `https://[POD_ID]-8000.proxy.runpod.net/docs` |
| Health Check | `https://[POD_ID]-8000.proxy.runpod.net/health` |
| RunPod Console | https://runpod.io/console |
| Supabase Dashboard | https://supabase.com/dashboard |
| Qdrant Dashboard | https://cloud.qdrant.io |

### Environment Variables Reference

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection | `postgresql+asyncpg://...` |
| `REDIS_URL` | Redis connection | `rediss://default:...` |
| `QDRANT_URL` | Vector DB URL | `https://xxx.qdrant.io:6333` |
| `QDRANT_API_KEY` | Vector DB API key | `your-api-key` |
| `SECRET_KEY` | App secret (32+ chars) | `random-secure-string` |
| `JWT_SECRET_KEY` | JWT signing key | `another-secure-string` |
| `ENCRYPTION_KEY` | Fernet encryption key | `base64-encoded-key` |
| `USE_LOCAL_LLM` | Use local GPU model | `true` or `false` |
| `LLM_MODEL_NAME` | Hugging Face model ID | `mistralai/Mistral-7B-Instruct-v0.2` |
| `HF_TOKEN` | Hugging Face API token | `hf_xxxxx` |
| `OPENAI_API_KEY` | OpenAI key (if not local) | `sk-xxxxx` |

---

## Launch Checklist

### Pre-Launch

- [ ] All external services configured (Supabase, Upstash, Qdrant)
- [ ] Environment variables set correctly
- [ ] Pod deployed and running
- [ ] AI model downloaded and loaded
- [ ] Database migrations applied
- [ ] Health endpoint responding
- [ ] API authentication working
- [ ] Consultation endpoint tested
- [ ] Drug search working
- [ ] Startup script configured
- [ ] Auto-restart on pod restart configured

### Launch Day

- [ ] Final smoke test all endpoints
- [ ] Monitor GPU utilization
- [ ] Monitor response times
- [ ] Monitor error rates
- [ ] Have scaling plan ready
- [ ] Document your POD_ID and URLs

### Post-Launch

- [ ] Set up monitoring alerts
- [ ] Schedule regular backups
- [ ] Plan for updates/maintenance windows
- [ ] Monitor costs weekly

---

## Support & Resources

- **UMI Repository**: https://github.com/amoufaq5/IMI
- **RunPod Documentation**: https://docs.runpod.io
- **RunPod Discord**: https://discord.gg/runpod
- **Supabase Documentation**: https://supabase.com/docs
- **Qdrant Documentation**: https://qdrant.tech/documentation

---

**Document Version**: 1.0  
**Last Updated**: January 2026  
**Maintainer**: UMI Development Team
