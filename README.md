# UMI - Universal Medical Intelligence

[![License](https://img.shields.io/badge/license-Proprietary-red.svg)]()
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)]()

> **The first AI platform connecting every stakeholder in healthcare - from patient symptoms to pharmaceutical compliance.**

## 🎯 Overview

UMI is a comprehensive medical AI platform serving:
- **Patients**: Symptom analysis, OTC recommendations, doctor referrals
- **Pharmaceutical Companies**: QA/QC documentation, compliance tracking
- **Hospitals**: ER triage, patient profiling, insurance management
- **Researchers**: Literature review, paper assistance
- **Students**: USMLE prep, medical education

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         UMI Platform                             │
├─────────────────────────────────────────────────────────────────┤
│  Frontend (React/Next.js)  │  Mobile (React Native)            │
├─────────────────────────────────────────────────────────────────┤
│                      API Gateway (FastAPI)                       │
├─────────────────────────────────────────────────────────────────┤
│  Auth  │  Consultation  │  Pharma  │  Research  │  Imaging     │
├─────────────────────────────────────────────────────────────────┤
│                    AI/ML Layer (MoE + RAG)                       │
├─────────────────────────────────────────────────────────────────┤
│  PostgreSQL  │  Redis  │  Qdrant  │  MinIO  │  Kafka           │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Node.js 20+ (for frontend)
- CUDA 12.0+ (for GPU inference)

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/umi.git
cd umi

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Start infrastructure (Docker)
docker-compose up -d

# Run database migrations
alembic upgrade head

# Start development server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test module
pytest tests/test_consultation.py -v
```

## 📁 Project Structure

```
umi/
├── docs/                    # Documentation
│   ├── 01_STRATEGIC_ANALYSIS.md
│   ├── 02_UNIFIED_PITCH.md
│   ├── 03_NEW_HORIZONS.md
│   ├── 04_TECHNICAL_ARCHITECTURE.md
│   └── 05_ROADMAP.md
├── src/                     # Source code
│   ├── api/                 # API routes
│   ├── core/                # Core configuration
│   ├── models/              # Database models
│   ├── schemas/             # Pydantic schemas
│   ├── services/            # Business logic
│   ├── ai/                  # AI/ML components
│   └── main.py              # Application entry
├── tests/                   # Test suite
├── frontend/                # React frontend
├── mobile/                  # React Native app
├── infrastructure/          # Docker, K8s configs
├── scripts/                 # Utility scripts
├── data/                    # Data pipelines
└── ml/                      # ML training code
```

## 🔧 Configuration

Key environment variables:

```env
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/umi
REDIS_URL=redis://localhost:6379

# AI/ML
OPENAI_API_KEY=sk-...  # For development
HF_TOKEN=hf_...        # Hugging Face
QDRANT_URL=http://localhost:6333

# Security
SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256
JWT_EXPIRY_MINUTES=30

# External APIs
PUBMED_API_KEY=...
DRUGBANK_API_KEY=...
```

## 📚 API Documentation

Once running, access:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## 🧪 Testing Strategy

- **Unit Tests**: Individual functions and classes
- **Integration Tests**: API endpoints and database
- **E2E Tests**: Full user workflows
- **Load Tests**: Performance under stress
- **Medical Accuracy Tests**: AI output validation

## 🔐 Security

- OAuth 2.0 / OpenID Connect authentication
- Role-Based Access Control (RBAC)
- Field-level encryption for PII
- GDPR and UAE PDPL compliance
- Regular security audits

## 📊 Monitoring

- **Metrics**: Prometheus + Grafana
- **Logging**: ELK Stack
- **Tracing**: Jaeger
- **Alerts**: PagerDuty integration

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

Proprietary - All rights reserved. See [LICENSE](LICENSE) for details.

## 📞 Contact

- **Email**: team@umi-health.com
- **Website**: https://umi-health.com

---

**Built with ❤️ for better healthcare**
