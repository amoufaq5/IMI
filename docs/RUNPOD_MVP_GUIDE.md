# UMI RunPod MVP Guide: GPU Selection, Timeline & Step-by-Step Workflow

Complete guide for launching UMI MVP on RunPod with optimal GPU configuration.

---

## Part 1: GPU Selection

### Recommended GPU for MVP

| Phase | GPU | VRAM | Cost/hr | Why |
|-------|-----|------|---------|-----|
| **Development** | RTX 3090 | 24GB | ~$0.44 | Cheap, sufficient for testing |
| **MVP Launch** | **RTX 4090** | 24GB | ~$0.69 | Best price/performance for 7B models |
| **Scale (if needed)** | A100 40GB | 40GB | ~$1.89 | For 13B+ models or high traffic |

### GPU Comparison for Mistral-7B

| GPU | Inference Speed | Batch Size | Monthly Cost (24/7) | Recommendation |
|-----|-----------------|------------|---------------------|----------------|
| RTX 3090 | ~30 tokens/sec | 1-2 | ~$320 | Development only |
| **RTX 4090** | ~50 tokens/sec | 2-4 | ~$500 | **MVP - Best Value** |
| A100 40GB | ~80 tokens/sec | 4-8 | ~$1,360 | High traffic |
| A100 80GB | ~80 tokens/sec | 8-16 | ~$1,800 | Multiple models |
| H100 | ~150 tokens/sec | 16+ | ~$3,000 | Enterprise scale |

### How Many GPUs?

**For MVP: 1 GPU is sufficient**

```
Traffic Calculation:
- RTX 4090 throughput: ~50 tokens/sec
- Average response: 500 tokens
- Time per response: 10 seconds
- Requests per minute: 6
- Requests per hour: 360
- Daily capacity: 8,640 consultations

MVP Target: 100-500 users/day = 1 GPU handles easily
```

**Scale triggers (add more GPUs when):**
- Response latency > 15 seconds consistently
- Queue depth > 10 requests
- Daily users > 5,000

### GPU Selection Decision Tree

```
START
  │
  ▼
Budget < $500/month?
  │
  ├─ YES → RTX 4090 (1x) ─────────────────────┐
  │                                            │
  └─ NO                                        │
      │                                        │
      ▼                                        │
  Need 13B+ models?                            │
      │                                        │
      ├─ YES → A100 40GB (1x)                  │
      │                                        │
      └─ NO                                    │
          │                                    │
          ▼                                    │
      High traffic (>5K users/day)?            │
          │                                    │
          ├─ YES → RTX 4090 (2x) or A100 (1x)  │
          │                                    │
          └─ NO ───────────────────────────────┘
                          │
                          ▼
              RTX 4090 (1x) - RECOMMENDED FOR MVP
```

---

## Part 2: MVP Timeline

### Total Time to MVP: 4-6 Weeks

```
Week 1: Infrastructure Setup
├── Day 1-2: External services (Supabase, Upstash, Qdrant)
├── Day 3-4: RunPod configuration & deployment
├── Day 5-6: Domain, SSL, basic monitoring
└── Day 7: End-to-end testing

Week 2: AI Model Setup
├── Day 1-2: Fine-tune Mistral on medical QA data
├── Day 3-4: RAG pipeline with medical knowledge
├── Day 5-6: ASMETHOD consultation flow
└── Day 7: AI testing & prompt optimization

Week 3: Core Features
├── Day 1-2: User authentication & profiles
├── Day 3-4: Consultation API completion
├── Day 5-6: Drug information integration
└── Day 7: Integration testing

Week 4: Pharma Module
├── Day 1-2: Facility management
├── Day 3-4: Document generation (SOPs, validation)
├── Day 5-6: Compliance tracking
└── Day 7: Pharma testing

Week 5: Polish & Testing
├── Day 1-2: Bug fixes & optimization
├── Day 3-4: Load testing
├── Day 5-6: Security audit
└── Day 7: Documentation

Week 6: Launch
├── Day 1-2: Soft launch (beta users)
├── Day 3-4: Feedback & fixes
├── Day 5-6: Marketing prep
└── Day 7: Public MVP launch
```

### Accelerated Timeline (3 Weeks)

If you skip fine-tuning and use OpenAI API:

```
Week 1: Infrastructure + Basic API
Week 2: Core Features (Consultation + Pharma)
Week 3: Testing + Launch
```

**Trade-off**: Higher API costs, less control, but faster launch.

---

## Part 3: Step-by-Step RunPod Workflow

### Phase 1: Pre-RunPod Setup (Day 1)

#### Step 1.1: Create External Services

**Supabase (PostgreSQL)**
```
1. Go to https://supabase.com
2. Sign up / Login
3. Click "New Project"
4. Settings:
   - Name: umi-production
   - Database Password: [generate strong password]
   - Region: Choose closest to your users (London for UK)
5. Wait for project creation (~2 minutes)
6. Go to Settings → Database
7. Copy "Connection string" (URI format)
   Example: postgresql://postgres:[PASSWORD]@db.xxxx.supabase.co:5432/postgres
8. For async: Change to postgresql+asyncpg://...
```

**Upstash (Redis)**
```
1. Go to https://upstash.com
2. Sign up / Login
3. Click "Create Database"
4. Settings:
   - Name: umi-redis
   - Type: Regional
   - Region: eu-west-1 (or closest)
5. Copy:
   - UPSTASH_REDIS_REST_URL
   - UPSTASH_REDIS_REST_TOKEN
6. Connection string format: rediss://default:[TOKEN]@[ENDPOINT]:6379
```

**Qdrant Cloud (Vector DB)**
```
1. Go to https://cloud.qdrant.io
2. Sign up / Login
3. Click "Create Cluster"
4. Settings:
   - Name: umi-vectors
   - Cloud: AWS
   - Region: eu-west-1
   - Size: Free tier (1GB) for MVP
5. Copy:
   - Cluster URL: https://xxx.aws.cloud.qdrant.io:6333
   - API Key: [from dashboard]
```

#### Step 1.2: Prepare Environment File

Create `.env.runpod` locally:

```env
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

# Security (generate these)
SECRET_KEY=run-python-secrets-to-generate-this-key-min-32-chars
JWT_SECRET_KEY=another-secure-key-for-jwt-tokens-min-32-chars
ENCRYPTION_KEY=fernet-key-from-cryptography-library

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

Generate security keys:
```python
import secrets
from cryptography.fernet import Fernet

print(f"SECRET_KEY={secrets.token_urlsafe(32)}")
print(f"JWT_SECRET_KEY={secrets.token_urlsafe(32)}")
print(f"ENCRYPTION_KEY={Fernet.generate_key().decode()}")
```

---

### Phase 2: RunPod Setup (Day 2)

#### Step 2.1: Create RunPod Account

```
1. Go to https://runpod.io
2. Sign up with email
3. Add payment method (Settings → Billing)
4. Add $50-100 credits to start
```

#### Step 2.2: Create Pod Template

```
1. Go to RunPod Console → Templates
2. Click "New Template"
3. Fill in:

   Template Name: UMI-Medical-MVP
   
   Container Image: 
   runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
   
   Docker Command: (leave empty)
   
   Volume Mount Path: /workspace
   
   Expose HTTP Ports: 8000
   
   Environment Variables:
   [Paste all variables from .env.runpod]

4. Click "Save Template"
```

#### Step 2.3: Deploy Pod

```
1. Go to RunPod Console → Pods
2. Click "+ Deploy"
3. Select GPU:
   ┌─────────────────────────────────────┐
   │ ✓ NVIDIA RTX 4090                   │
   │   24GB VRAM | ~$0.69/hr             │
   └─────────────────────────────────────┘
4. Select your template: UMI-Medical-MVP
5. Volume Size: 50 GB
6. Click "Deploy On-Demand"
   (or "Deploy Spot" for 50% savings, but can be interrupted)
7. Wait for pod to start (~1-2 minutes)
```

#### Step 2.4: Access Pod

```
1. Click on your running pod
2. Click "Connect" → "Web Terminal"
   OR
   Click "Connect" → "SSH" and copy command:
   ssh root@[IP] -p [PORT] -i ~/.ssh/id_rsa
```

---

### Phase 3: Application Deployment (Day 2-3)

#### Step 3.1: Initial Setup on Pod

```bash
# Connect to pod terminal, then run:

# Update system
apt-get update && apt-get install -y git curl

# Navigate to workspace
cd /workspace

# Clone your repository
git clone https://github.com/amoufaq5/IMI.git
cd IMI

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install CUDA-optimized packages
pip install torch==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install vllm==0.3.2
```

#### Step 3.2: Download AI Models

```bash
# This takes 10-15 minutes for Mistral-7B
python << 'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
print(f"Downloading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Model downloaded and loaded successfully!")
print(f"GPU Memory Used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
EOF
```

#### Step 3.3: Initialize Database

```bash
# Run migrations
alembic upgrade head

# Verify connection
python << 'EOF'
import asyncio
from src.core.database import init_db

async def test():
    await init_db()
    print("Database initialized successfully!")

asyncio.run(test())
EOF
```

#### Step 3.4: Start Application

```bash
# Start in foreground (for testing)
uvicorn src.main:app --host 0.0.0.0 --port 8000

# Or start in background (for production)
nohup uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 2 > /workspace/umi.log 2>&1 &
```

#### Step 3.5: Verify Deployment

```bash
# Test health endpoint
curl http://localhost:8000/health
# Expected: {"status":"healthy","service":"umi"}

# Test from external (use RunPod proxy URL)
# Your URL: https://[POD_ID]-8000.proxy.runpod.net/health
```

---

### Phase 4: Domain & Production Setup (Day 3-4)

#### Step 4.1: Get Your Public URL

RunPod provides automatic HTTPS:
```
Your API URL: https://[POD_ID]-8000.proxy.runpod.net

Example: https://abc123xyz-8000.proxy.runpod.net
```

#### Step 4.2: Custom Domain (Optional)

**Option A: Cloudflare (Recommended)**
```bash
# Install cloudflared
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared-linux-amd64
mv cloudflared-linux-amd64 /usr/local/bin/cloudflared

# Login to Cloudflare
cloudflared tunnel login

# Create tunnel
cloudflared tunnel create umi-api

# Configure
cat > ~/.cloudflared/config.yml << EOF
tunnel: [YOUR_TUNNEL_ID]
credentials-file: /root/.cloudflared/[TUNNEL_ID].json
ingress:
  - hostname: api.yourdomain.com
    service: http://localhost:8000
  - service: http_status:404
EOF

# Run tunnel (in background)
nohup cloudflared tunnel run umi-api > /workspace/cloudflare.log 2>&1 &
```

**Option B: Use RunPod Proxy**
- Just use the provided URL
- Already has HTTPS
- No configuration needed

#### Step 4.3: Create Startup Script

```bash
cat > /workspace/start_umi.sh << 'EOF'
#!/bin/bash
set -e

cd /workspace/IMI
source venv/bin/activate

# Pull latest code
git pull origin main

# Run migrations
alembic upgrade head

# Start application
exec uvicorn src.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 2 \
    --timeout-keep-alive 30
EOF

chmod +x /workspace/start_umi.sh
```

#### Step 4.4: Auto-Start on Pod Restart

Add to RunPod template Docker Command:
```
bash -c "cd /workspace && ./start_umi.sh"
```

---

### Phase 5: Testing & Monitoring (Day 4-5)

#### Step 5.1: Test All Endpoints

```bash
BASE_URL="https://[POD_ID]-8000.proxy.runpod.net"

# 1. Health check
curl $BASE_URL/health

# 2. Register user
curl -X POST $BASE_URL/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"SecurePass123","confirm_password":"SecurePass123"}'

# 3. Login
TOKEN=$(curl -X POST $BASE_URL/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"SecurePass123"}' | jq -r '.access_token')

# 4. Start consultation
curl -X POST $BASE_URL/api/v1/consultations \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"initial_message":"I have a headache"}'

# 5. Test drug search
curl "$BASE_URL/api/v1/drugs/search?q=paracetamol" \
  -H "Authorization: Bearer $TOKEN"
```

#### Step 5.2: Monitor GPU Usage

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Check memory usage
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

#### Step 5.3: View Logs

```bash
# Application logs
tail -f /workspace/umi.log

# System logs
journalctl -f
```

---

### Phase 6: Launch Checklist

#### Pre-Launch (Day 5-6)

```
□ All endpoints tested and working
□ Database migrations applied
□ AI model responding correctly
□ HTTPS working (RunPod proxy or custom domain)
□ Error handling verified
□ Rate limiting configured
□ Monitoring set up
□ Backup strategy in place
□ Documentation updated
```

#### Launch Day (Day 7)

```
□ Final smoke test
□ Announce to beta users
□ Monitor error rates
□ Monitor response times
□ Monitor GPU utilization
□ Be ready to scale if needed
```

---

## Cost Summary for MVP

### Monthly Costs (Estimated)

| Service | Cost | Notes |
|---------|------|-------|
| RunPod RTX 4090 (24/7) | ~$500 | Can reduce with spot instances |
| Supabase | $0-25 | Free tier sufficient for MVP |
| Upstash Redis | $0-10 | Free tier sufficient |
| Qdrant Cloud | $0-25 | Free tier for 1GB |
| Domain | ~$12/year | Optional |
| **Total** | **~$500-560/month** | |

### Cost Optimization Tips

1. **Use Spot Instances**: 50-70% cheaper, good for dev
2. **Scale Down Off-Hours**: Stop pod when not needed
3. **Start with OpenAI**: $0 GPU cost, pay per API call
4. **Serverless Option**: RunPod Serverless for variable traffic

---

## Quick Reference Commands

```bash
# Start UMI
cd /workspace/IMI && source venv/bin/activate && uvicorn src.main:app --host 0.0.0.0 --port 8000

# Stop UMI
pkill -f uvicorn

# Update code
cd /workspace/IMI && git pull && pip install -r requirements.txt

# View logs
tail -f /workspace/umi.log

# Check GPU
nvidia-smi

# Check disk
df -h /workspace

# Restart everything
pkill -f uvicorn && sleep 2 && /workspace/start_umi.sh
```
