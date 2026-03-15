# IMI — Intelligent Medical Interface

A production-grade medical AI platform built on a **5-Layer Hybrid Cognition Architecture**, fine-tuned on Mistral 7B with 6 domain-specific LoRA adapters, serving patients, students, doctors, researchers, hospitals, and pharmaceutical companies.

## Model

| | |
|---|---|
| **Base model** | `mistralai/Mistral-7B-Instruct-v0.3` (Apache 2.0) |
| **Architecture** | Dense transformer, 7B parameters, 32K context |
| **Training pipeline** | Foundation full FT → ORPO safety alignment → 6 LoRA adapters |
| **Quantization** | QLoRA 4-bit NF4 (inference) / BF16 full (foundation training) |
| **LoRA targets** | q,k,v,o_proj + gate_proj, up_proj, down_proj |
| **Inference** | vLLM with LoRA hot-swapping, <200ms p95 latency |

## Architecture — 5-Layer Hybrid Cognition Stack

```
User Request
     │
     ▼
┌──────────────────────────────────────────┐
│  Layer 0: INPUT GUARDRAILS               │  < 5ms regex — crisis/emergency/scope
│  (before anything else runs)             │  CRISIS → bypass LLM entirely
└──────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────┐
│  Layer 1: KNOWLEDGE GRAPH  (Truth)       │  Neo4j: disease↔symptom↔drug↔guideline
└──────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────┐
│  Layer 2: RULE ENGINE  (Safety)          │  Deterministic triage, OTC, contraindications
└──────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────┐
│  Layer 3: LLM  (Language + Synthesis)    │  Mistral 7B + domain LoRA adapter
│  — never decides alone on safety —       │
└──────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────┐
│  Layer 4: VERIFIER  (Critic)             │  Hallucination detection, guideline check
└──────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────┐
│  Layer 5: MEMORY  (Profiling)            │  HIPAA-compliant patient/entity profiles
└──────────────────────────────────────────┘
     │
     ▼
Verified Response
```

## LoRA Adapters (6 domains)

| Adapter | User Type | Training Data |
|---------|-----------|---------------|
| `patient_triage` | Patient | symptom datasets, triage cases, health advice |
| `clinical_decision` | Doctor | doctor-patient dialogues, clinical notes, SOAP summaries |
| `education` | Student | USMLE, MedMCQA, flashcards, reasoning chains |
| `research` | Researcher | PubMedQA, biomedical literature, clinical trials |
| `clinical_pharmacist` | Pharma/Pharmacist | drug reviews, interactions, pharmacogenomics |
| `regulatory_qa` | Hospital/Regulatory | guidelines, ICD-10, WHO protocols |

## Training Pipeline

```
Step 1 — Data Collection
  python scripts/data_collection/collect_hf_datasets.py   # 8M+ examples, no creds needed
  python scripts/data_collection/collect_datasets.py       # 40+ additional HF datasets
  python scripts/data_collection/collect_biomedical_corpus.py  # PubMed, PMC, CORD-19
  python scripts/data_collection/synthetic_generator.py    # synthetic cases (unlimited)

Step 2 — Foundation Training  (full fine-tuning, all 7B params)
  python scripts/training/train_foundation.py
  # Hardware: 1× A100 80GB + DeepSpeed CPU offload  OR  4× A100 for speed
  # Cost: ~$3–15 depending on dataset size

Step 3 — ORPO Safety Alignment  (replaces DPO — no reference model needed)
  python scripts/training/train_dpo.py train --foundation-path models/foundation
  # Hardware: 1× A100 40GB
  # Cost: ~$1

Step 4 — Domain Adapter Training  (QLoRA, 6 adapters)
  python scripts/training/finetune_mixtral.py --gpu-tier A100_40GB --data-format both
  # Hardware: 1× A100 40GB per adapter (run in parallel)
  # Cost: ~$4 total for all 6

Step 5 — Evaluate
  python scripts/training/evaluate_adapter.py --adapter models/mistral-medical-qlora

Step 6 — Checkpoint Averaging (optional, improves stability)
  python scripts/training/average_checkpoints.py --checkpoint-dir models/mistral-medical-qlora
```

## Dataset Summary

| Source | Examples | Notes |
|--------|----------|-------|
| HuggingFace Hub (100+ datasets) | ~8.2M raw | `collect_hf_datasets.py` |
| Direct URL (GitHub, CDC, CMS) | ~1.9M raw | included above |
| Additional HF catalogue | ~80K | `collect_datasets.py` |
| Biomedical corpus (free tier) | millions more | NCBI API key needed |
| Synthetic generator | unlimited | template-based |
| **Clean usable (deduplicated)** | **~4–5M** | **~2B tokens** |

## Training Cost (Mistral 7B on your 8× A100 80GB)

| Phase | Hardware | Time | Cost |
|-------|----------|------|------|
| Foundation (500K examples, 3 epochs) | 1× A100 80GB | ~2 hrs | ~$4 |
| Foundation (4M examples, 3 epochs) | 8× A100 80GB | ~1.5 hrs | ~$24 |
| ORPO safety alignment | 1× A100 40GB | ~30 min | ~$1 |
| 6 LoRA adapters (QLoRA) | 1× A100 40GB each | ~2 hrs total | ~$4 |
| **Full MVP pipeline** | | | **~$30–50** |

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

## Project Structure

```
IMI/
├── src/
│   ├── core/
│   │   ├── config/settings.py          # App config (model path, DB, etc.)
│   │   ├── security/                   # HIPAA: AES-256, JWT, RBAC, audit
│   │   └── database/                   # PostgreSQL, Neo4j, Redis clients
│   ├── layers/
│   │   ├── knowledge_graph/            # Layer 1 — Neo4j medical facts
│   │   ├── rule_engine/                # Layer 2 — Deterministic safety
│   │   │   └── guardrails.py           # Input/Output pattern guardrails
│   │   ├── llm/
│   │   │   ├── meditron.py             # MistralMedicalModel (was Mixtral)
│   │   │   ├── prompts.py              # Role-specific system prompts
│   │   │   ├── adapters.py             # LoRA domain adapter management
│   │   │   └── service.py              # LLM orchestration service
│   │   ├── verifier/                   # Layer 4 — Hallucination & guideline check
│   │   ├── memory/                     # Layer 5 — HIPAA patient profiles
│   │   ├── rag/                        # RAG pipeline (ChromaDB/FAISS)
│   │   ├── explainability/             # SHAP token importance
│   │   └── citation/                   # Inline citation tracking
│   ├── domains/                        # Per-user-type business logic
│   ├── api/routes/                     # FastAPI endpoints
│   ├── orchestrator.py                 # Linear 5-layer orchestrator
│   └── orchestrator_langgraph.py       # Graph-based orchestrator (LangGraph)
├── scripts/
│   ├── data_collection/
│   │   ├── collect_hf_datasets.py      # 100+ HF datasets, ~8M examples
│   │   ├── collect_datasets.py         # 40+ additional datasets
│   │   ├── collect_biomedical_corpus.py# PubMed, PMC, CORD-19, MIMIC
│   │   ├── synthetic_generator.py      # Synthetic medical cases
│   │   └── ingest_pdfs.py              # WHO/FDA PDF ingestion
│   └── training/
│       ├── train_foundation.py         # Step 2: full FT, all 7B params
│       ├── train_dpo.py                # Step 3: ORPO safety alignment
│       ├── finetune_mixtral.py         # Step 4: QLoRA domain adapters
│       ├── prepare_medical_data.py     # Data preparation
│       ├── evaluate_adapter.py         # Evaluation (9 metrics)
│       └── average_checkpoints.py      # Checkpoint averaging
├── configs/
│   └── deepspeed_zero3.json            # ZeRO-3 + CPU optimizer offload
├── docs/                               # Full documentation
├── requirements.txt
├── requirements-training.txt
└── .env.example
```

## API Endpoints

### Patient `/api/v1/patient`
- `POST /assess-symptoms` — Symptom triage
- `POST /health-info` — Health information
- `POST /check-drug-safety` — Drug safety
- `POST /analyze-lab-results` — Lab interpretation

### Doctor `/api/v1/doctor`
- `POST /differential` — Differential diagnosis
- `POST /treatment-recommendations` — Evidence-based treatment
- `POST /drug-interactions` — Drug interaction check
- `GET /guidelines/{condition}` — Clinical guidelines
- `POST /summarize-case` — Case summary

### Student `/api/v1/student`
- `POST /answer-question` — USMLE Q&A
- `POST /explain-concept` — Medical concept explanation
- `POST /generate-practice` — Practice question generation

### Researcher `/api/v1/researcher`
- `POST /literature/search` — Literature search
- `POST /literature/synthesize` — Literature synthesis
- `POST /patent/guidance` — Patent guidance
- `POST /regulatory-pathway` — Regulatory pathway

### Pharmaceutical `/api/v1/pharma`
- `POST /document/generate` — QA document generation
- `POST /compliance/check` — Regulatory compliance
- `GET /sales/analytics/{entity_id}` — Sales analytics

### Hospital `/api/v1/hospital`
- `POST /er/triage` — ER triage
- `GET /er/queue` — Queue management
- `POST /insurance/claim` — Insurance processing

### General `/api/v1/general`
- `GET /disease/{name}` — Disease information
- `GET /drug/{name}` — Drug information
- `POST /search` — Medical search

## Safety Architecture

- **Guardrails first** — crisis/emergency patterns caught in < 5ms before any LLM call
- **CRISIS** → LLM completely bypassed, hardcoded crisis resources shown
- **EMERGENCY** → emergency banner prepended, LLM generates with urgency context
- **Rule Engine** → deterministic triage/contraindication/OTC logic (not LLM guesswork)
- **ORPO alignment** — model trained to prefer safe, hedged responses
- **Verifier** — hallucination detection, guideline compliance, confidence calibration
- **Output Guardrails** — softens overconfident language, scrubs PHI

## HIPAA Compliance

- AES-256-GCM encryption for all PHI at rest
- JWT-based authentication with role verification
- Complete audit trail (every query, response, safety event)
- PHI anonymization for research use
- Data retention policies (7-year audit log default)

## Documentation

| Doc | Description |
|-----|-------------|
| `docs/ARCHITECTURE.md` | Full 5-layer system architecture |
| `docs/TRAINING_GUIDE.md` | Complete training pipeline guide |
| `docs/TRAINING_QUICKSTART.md` | Quick-start training reference |
| `docs/TRAINING_ARCHITECTURE.md` | Training system design |
| `docs/FINETUNING_GUIDE.md` | Fine-tuning deep dive |
| `docs/API_REFERENCE.md` | API endpoint documentation |
| `docs/DEPLOYMENT.md` | Deployment instructions |
| `docs/RUNPOD_DEPLOYMENT_GUIDE.md` | RunPod cloud training |
| `docs/INVESTOR_DEMO.md` | Investor demo script |
| `docs/PITCH_DECK.md` | Investor pitch deck |
| `docs/REASONING_AND_GOVERNANCE.md` | Decision making and auditing |
| `docs/FEATURES_AND_ROADMAP.md` | Roadmap |
| `docs/RECOMMENDATIONS.md` | Implementation recommendations |

## License

Proprietary — All rights reserved.
