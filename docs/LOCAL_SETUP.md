# UMI - Local Development Setup Guide

This guide covers running UMI on your local machine for development and testing.

---

## Prerequisites

### System Requirements
- **OS**: Windows 10/11, macOS, or Linux
- **RAM**: Minimum 16GB (32GB recommended for AI models)
- **Storage**: 50GB free space
- **GPU**: Optional but recommended (NVIDIA with CUDA 11.8+)

### Software Requirements
- Python 3.11+
- Docker Desktop
- Git
- PostgreSQL 16 (via Docker)
- Redis 7 (via Docker)

---

## Step 1: Clone and Setup Environment

```powershell
# Clone repository
git clone https://github.com/your-org/umi.git
cd umi

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Step 2: Environment Configuration

```powershell
# Copy example environment file
copy .env.example .env
```

Edit `.env` with your local settings:

```env
# Application
APP_NAME=UMI
APP_ENV=development
DEBUG=true
HOST=127.0.0.1
PORT=8000

# Database (Docker)
DATABASE_URL=postgresql+asyncpg://umi_user:umi_password@localhost:5432/umi_db

# Redis (Docker)
REDIS_URL=redis://localhost:6379/0

# Security (generate your own)
SECRET_KEY=your-secret-key-min-32-chars-here
JWT_SECRET_KEY=your-jwt-secret-key-here
ENCRYPTION_KEY=your-fernet-encryption-key

# AI/ML - Choose ONE option:

# Option A: OpenAI API (easiest)
OPENAI_API_KEY=sk-your-openai-api-key

# Option B: Local Models (requires GPU)
# LLM_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
# HF_TOKEN=your-huggingface-token

# Vector Database
QDRANT_URL=http://localhost:6333

# Feature Flags
FEATURE_CONSULTATION_ENABLED=true
FEATURE_PHARMA_ENABLED=true
FEATURE_IMAGING_ENABLED=false
```

### Generate Security Keys

```python
# Run in Python to generate keys
import secrets
from cryptography.fernet import Fernet

print(f"SECRET_KEY={secrets.token_urlsafe(32)}")
print(f"JWT_SECRET_KEY={secrets.token_urlsafe(32)}")
print(f"ENCRYPTION_KEY={Fernet.generate_key().decode()}")
```

---

## Step 3: Start Infrastructure Services

```powershell
# Start only infrastructure (database, cache, vector DB)
docker-compose up -d postgres redis qdrant

# Verify services are running
docker-compose ps
```

Expected output:
```
NAME            STATUS    PORTS
umi-postgres    running   0.0.0.0:5432->5432/tcp
umi-redis       running   0.0.0.0:6379->6379/tcp
umi-qdrant      running   0.0.0.0:6333->6333/tcp
```

---

## Step 4: Initialize Database

```powershell
# Run database migrations
alembic upgrade head

# Or create initial migration if none exists
alembic revision --autogenerate -m "Initial migration"
alembic upgrade head
```

---

## Step 5: Run the Application

### Development Mode (with hot reload)

```powershell
# From project root with venv activated
uvicorn src.main:app --reload --host 127.0.0.1 --port 8000
```

### Alternative: Using Python directly

```powershell
python -m src.main
```

---

## Step 6: Verify Installation

### Health Check
```powershell
curl http://localhost:8000/health
# Expected: {"status":"healthy","service":"umi"}
```

### API Documentation
Open in browser: http://localhost:8000/docs

### Test Registration
```powershell
curl -X POST http://localhost:8000/api/v1/auth/register ^
  -H "Content-Type: application/json" ^
  -d "{\"email\":\"test@example.com\",\"password\":\"SecurePass123\",\"confirm_password\":\"SecurePass123\"}"
```

---

## Running with Local AI Models (Optional)

If you want to run AI models locally instead of using OpenAI:

### Install GPU Dependencies

```powershell
# CUDA 11.8 (check your GPU compatibility)
pip install torch==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install vLLM for fast inference
pip install vllm==0.3.2
```

### Download Models

```powershell
# Set Hugging Face token
set HF_TOKEN=your-token

# Models will auto-download on first use, or pre-download:
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')"
```

### Update .env for Local Models

```env
# Disable OpenAI
OPENAI_API_KEY=

# Enable local model
LLM_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
USE_LOCAL_LLM=true
```

---

## Running Tests

```powershell
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_auth.py -v
```

---

## Common Issues

### Port Already in Use
```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill process (replace PID)
taskkill /PID <PID> /F
```

### Database Connection Failed
```powershell
# Check if PostgreSQL is running
docker-compose logs postgres

# Restart PostgreSQL
docker-compose restart postgres
```

### Redis Connection Failed
```powershell
# Check Redis status
docker-compose logs redis

# Test Redis connection
docker exec -it umi-redis redis-cli ping
```

### Out of Memory (AI Models)
- Use smaller models (e.g., `TinyLlama/TinyLlama-1.1B-Chat-v1.0`)
- Use OpenAI API instead of local models
- Reduce batch sizes in config

---

## Development Workflow

```powershell
# 1. Start services
docker-compose up -d postgres redis qdrant

# 2. Activate environment
.\venv\Scripts\activate

# 3. Run with hot reload
uvicorn src.main:app --reload

# 4. Make changes - server auto-restarts

# 5. Run tests before committing
pytest

# 6. Stop services when done
docker-compose down
```

---

## Quick Reference

| Service | URL | Credentials |
|---------|-----|-------------|
| API | http://localhost:8000 | - |
| API Docs | http://localhost:8000/docs | - |
| PostgreSQL | localhost:5432 | umi_user / umi_password |
| Redis | localhost:6379 | - |
| Qdrant | http://localhost:6333 | - |
| Qdrant Dashboard | http://localhost:6333/dashboard | - |
