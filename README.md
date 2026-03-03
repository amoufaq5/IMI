# IMI - Intelligent Medical Interface

A production-grade medical LLM platform with hybrid cognition architecture serving patients, students, doctors, researchers, and pharmaceutical companies.

## Architecture Overview

### 5-Layer Hybrid Cognition Stack

1. **Layer 1 - Knowledge Graph (Truth Layer)**
   - Disease ↔ Symptom ↔ Drug ↔ Interaction ↔ Guideline relationships
   - Neo4j graph database for medical knowledge
   - Sources: Clinical guidelines, drug labels, regulatory texts

2. **Layer 2 - Rule Engine (Safety Layer)**
   - Deterministic logic for OTC eligibility, red-flag symptoms, contraindications
   - ASMETHOD-style triage logic
   - If-then medical reasoning

3. **Layer 3 - LLM (Language + Synthesis)**
   - Meditron-7B medical language model (4-bit QLoRA)
   - 6 domain-specific LoRA adapters (r=16, alpha=32)
   - Parallel multi-GPU training support
   - Never decides alone - always verified

4. **Layer 4 - Verifier/Critic Model**
   - Hallucination detection
   - Guideline conflict checking
   - Overconfidence detection

5. **Layer 5 - Memory & Profiling**
   - Longitudinal patient/pharma profiles
   - Time-aware medical history
   - Outcome feedback loops

## User Types

- **General User**: Medical information queries
- **Patient**: Diagnosis support, OTC recommendations, referrals
- **Student**: USMLE prep, medical education
- **Researcher**: Drug research, patent process support
- **Pharmaceutical**: QA/QC, regulatory compliance, sales tracking
- **Hospital**: ER optimization, patient profiling
- **Doctor**: Diagnosis assistance, case research

## HIPAA Compliance

- End-to-end encryption (AES-256)
- Role-based access control (RBAC)
- Complete audit logging
- Data anonymization
- Secure key management

## Project Structure

```
imi/
├── src/
│   ├── core/                     # Core infrastructure
│   │   ├── config/               # Configuration management (settings.py)
│   │   ├── security/             # HIPAA compliance
│   │   │   ├── encryption.py     # AES-256-GCM encryption
│   │   │   ├── authentication.py # JWT authentication
│   │   │   ├── authorization.py  # RBAC permissions
│   │   │   ├── audit.py          # Audit logging
│   │   │   └── hipaa.py          # PHI handling
│   │   └── database/             # Database connections
│   │       ├── postgres.py       # PostgreSQL async client
│   │       ├── neo4j_client.py   # Neo4j graph client
│   │       └── redis_client.py   # Redis cache client
│   ├── layers/                   # 5-Layer Architecture
│   │   ├── knowledge_graph/      # Layer 1: Truth Layer
│   │   │   ├── schema.py         # Medical entity models
│   │   │   ├── queries.py        # Cypher query builder
│   │   │   └── service.py        # KG service interface
│   │   ├── rule_engine/          # Layer 2: Safety Layer
│   │   │   ├── triage.py         # ASMETHOD triage
│   │   │   ├── otc_eligibility.py# OTC assessment
│   │   │   ├── contraindication_checker.py
│   │   │   ├── red_flags.py      # Critical symptom detection
│   │   │   └── service.py        # Rule engine orchestrator
│   │   ├── llm/                  # Layer 3: Language Layer
│   │   │   ├── meditron.py       # Meditron model wrapper
│   │   │   ├── prompts.py        # Role-specific prompts
│   │   │   ├── adapters.py       # LoRA domain adapters
│   │   │   └── service.py        # LLM service
│   │   ├── verifier/             # Layer 4: Critic Layer
│   │   │   ├── hallucination_detector.py
│   │   │   ├── guideline_checker.py
│   │   │   ├── confidence_calibrator.py
│   │   │   └── service.py        # Verifier orchestrator
│   │   └── memory/               # Layer 5: Profiling Layer
│   │       ├── patient_profile.py# HIPAA-compliant profiles
│   │       ├── entity_profile.py # Pharma/hospital profiles
│   │       ├── conversation_memory.py
│   │       └── service.py        # Memory service
│   ├── domains/                  # Domain-specific services
│   │   ├── patient.py            # Patient triage, diagnosis
│   │   ├── student.py            # USMLE, education
│   │   ├── doctor.py             # Clinical decision support
│   │   ├── researcher.py         # Research, patents
│   │   ├── pharma.py             # QA/QC, regulatory
│   │   ├── hospital.py           # ER, insurance
│   │   └── general.py            # General medical info
│   ├── api/                      # FastAPI REST API
│   │   ├── main.py               # Application entry
│   │   └── routes/               # API endpoints
│   ├── orchestrator.py           # Linear orchestrator
│   └── orchestrator_langgraph.py # Graph-based orchestrator (NEW)
├── scripts/
│   ├── data_collection/          # Dataset collection, PDF ingestion
│   └── training/                 # LoRA training pipeline
├── adapters/                     # Trained LoRA adapters
├── data/                         # Training data, vector store
├── apps/                         # Standalone applications
├── docs/                         # Documentation
├── requirements.txt              # Python dependencies
└── .env.example                  # Environment template
```

### New in v2.0

```
src/layers/
├── rag/                          # RAG Pipeline (NEW)
│   ├── pipeline.py               # Document ingestion & retrieval
│   ├── embeddings.py             # Embedding service
│   └── vector_store.py           # ChromaDB/FAISS storage
├── explainability/               # SHAP Explainability (NEW)
│   └── shap_explainer.py         # Token importance, counterfactuals
└── citation/                     # Citation Tracking (NEW)
    └── tracker.py                # Inline citations, references
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env

# Run database migrations
alembic upgrade head

# Start the server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Patient API (`/api/v1/patient`)
- `POST /assess-symptoms` - Symptom assessment and triage
- `POST /health-info` - Health information queries
- `POST /check-drug-safety` - Drug safety verification
- `POST /analyze-lab-results` - Lab result interpretation

### Doctor API (`/api/v1/doctor`)
- `POST /differential` - Generate differential diagnosis
- `POST /treatment-recommendations` - Evidence-based treatments
- `POST /drug-interactions` - Drug interaction checking
- `GET /guidelines/{condition}` - Clinical guidelines
- `POST /summarize-case` - Case summarization

### Student API (`/api/v1/student`)
- `POST /answer-question` - USMLE question answering
- `POST /explain-concept` - Medical concept explanation
- `POST /generate-practice` - Practice question generation
- `POST /review-essay` - Medical writing review

### Researcher API (`/api/v1/researcher`)
- `POST /literature/search` - Literature search
- `POST /literature/synthesize` - Literature synthesis
- `POST /patent/guidance` - Patent application guidance
- `POST /regulatory-pathway` - Regulatory pathway guidance

### Pharmaceutical API (`/api/v1/pharma`)
- `POST /document/generate` - QA document generation
- `POST /compliance/check` - Regulatory compliance check
- `POST /validation` - Validation record management
- `GET /sales/analytics/{entity_id}` - Sales analytics

### Hospital API (`/api/v1/hospital`)
- `POST /er/triage` - ER patient triage
- `GET /er/queue` - ER queue management
- `POST /appointment` - Appointment scheduling
- `POST /insurance/claim` - Insurance claim processing

### General API (`/api/v1/general`)
- `GET /disease/{name}` - Disease information
- `GET /drug/{name}` - Drug information
- `POST /search` - Medical search
- `POST /drug-interaction` - Drug interaction check

## Environment Variables

See `.env.example` for required configuration.

## Key Features

### Safety-First Architecture
- **LLM never decides alone** on safety-critical matters
- All recommendations pass through deterministic rule engine
- Verifier checks for hallucinations and guideline conflicts
- Red flag detection for emergency symptoms

### HIPAA Compliance
- AES-256-GCM encryption for PHI at rest
- JWT-based authentication with role verification
- Complete audit trail of all PHI access
- Data anonymization for research use

### Domain Adapters (LoRA)
- Patient Triage adapter
- Clinical Pharmacist adapter
- Regulatory QA adapter
- Research adapter
- Education adapter

## Training Your Own Adapters

See `docs/TRAINING_GUIDE.md` for complete instructions.

```bash
# Quick start
python scripts/data_collection/collect_datasets.py      # Download open datasets
python scripts/data_collection/synthetic_generator.py   # Generate synthetic data
python scripts/data_collection/ingest_pdfs.py           # Ingest WHO/FDA PDFs
python scripts/training/prepare_data.py                 # Prepare for training
python scripts/training/train_lora.py --adapter all --parallel  # Train adapters (multi-GPU)
```

## Documentation

| Document | Description |
|----------|-------------|
| `docs/ARCHITECTURE.md` | Complete system architecture |
| `docs/TRAINING_GUIDE.md` | Training pipeline guide |
| `docs/API_REFERENCE.md` | API endpoint documentation |
| `docs/DEPLOYMENT.md` | Deployment instructions |
| `docs/RUNPOD_DEPLOYMENT_GUIDE.md` | RunPod cloud training guide |
| `docs/FEATURES_AND_ROADMAP.md` | Features list and roadmap |
| `docs/PITCH_DECK.md` | Investor pitch deck |
| `docs/REASONING_AND_GOVERNANCE.md` | Decision making and auditing |

## v2.0 Features

- **LangGraph Orchestrator**: Graph-based flow with branching, retry loops, checkpointing
- **RAG Pipeline**: Document ingestion, semantic search, context retrieval
- **SHAP Explainability**: Token importance, feature attribution, counterfactuals
- **Citation Tracking**: Inline citations `[1]`, reference lists, credibility scoring
- **22 Medical Datasets**: 3M+ training examples from open sources

## License

Proprietary - All rights reserved.
