# IMI System Architecture

## Complete System Overview

The IMI (Intelligent Medical Interface) platform is a production-grade medical LLM system built on a **5-Layer Hybrid Cognition Architecture**. This document explains how every module operates and integrates.

---

## System Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER REQUEST                                    │
│                    (Patient, Doctor, Student, etc.)                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           API LAYER (FastAPI)                               │
│  src/api/routes/{patient, doctor, student, researcher, pharma, hospital}   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DOMAIN SERVICES                                      │
│              src/domains/{patient, doctor, student, ...}.py                 │
│         Role-specific business logic and request handling                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ORCHESTRATOR                                       │
│                        src/orchestrator.py                                  │
│              Coordinates all 5 layers for each request                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐         ┌─────────────────┐         ┌─────────────────┐
│   LAYER 5     │         │    LAYER 1      │         │    LAYER 2      │
│    MEMORY     │◄───────►│ KNOWLEDGE GRAPH │◄───────►│  RULE ENGINE    │
│   (Context)   │         │    (Facts)      │         │   (Safety)      │
└───────────────┘         └─────────────────┘         └─────────────────┘
        │                           │                           │
        │                           ▼                           │
        │                 ┌─────────────────┐                   │
        │                 │    LAYER 3      │                   │
        └────────────────►│      LLM        │◄──────────────────┘
                          │  (Generation)   │
                          └─────────────────┘
                                    │
                                    ▼
                          ┌─────────────────┐
                          │    LAYER 4      │
                          │    VERIFIER     │
                          │  (Validation)   │
                          └─────────────────┘
                                    │
                                    ▼
                          ┌─────────────────┐
                          │    RESPONSE     │
                          │   (Verified)    │
                          └─────────────────┘
```

---

## The 5-Layer Architecture Explained

### Layer 1: Knowledge Graph (Truth Layer)
**Location:** `src/layers/knowledge_graph/`

**Purpose:** Provides the factual foundation - medical knowledge that is verifiable and sourced.

**Components:**
- `schema.py` - Pydantic models for medical entities:
  - `Disease`, `Drug`, `Symptom`, `Guideline`, `Condition`, `LabTest`, `Procedure`
  - Relationships: `DrugInteraction`, `Contraindication`, `TreatmentRelation`
  
- `queries.py` - Cypher query builder for Neo4j:
  - Disease-symptom matching
  - Drug interaction detection
  - Differential diagnosis queries
  - Treatment pathway lookups

- `service.py` - High-level interface:
  - `get_disease()`, `get_drug()`, `search_diseases()`
  - `check_drug_interactions()`
  - `differential_diagnosis()`
  - Caching with Redis for performance

**Integration:** 
- Called by Orchestrator to get medical facts before LLM generation
- Used by Rule Engine to validate contraindications
- Provides evidence sources for Verifier

---

### Layer 2: Rule Engine (Safety Layer)
**Location:** `src/layers/rule_engine/`

**Purpose:** Deterministic safety logic that NEVER relies on LLM judgment for critical decisions.

**Components:**
- `triage.py` - ASMETHOD-style patient triage:
  - `TriageEngine` evaluates symptoms against rules
  - Returns urgency levels: EMERGENCY, URGENT, SEMI_URGENT, ROUTINE, SELF_CARE
  - Red flag detection for critical symptoms

- `otc_eligibility.py` - OTC medication assessment:
  - Age restrictions, pregnancy checks
  - Condition-based contraindications
  - Drug interaction warnings
  - Returns eligible products with reasoning

- `contraindication_checker.py` - Drug safety:
  - Drug-condition contraindications
  - Drug-drug interactions
  - Severity classification (ABSOLUTE, RELATIVE, CAUTION)
  - Alternative suggestions

- `red_flags.py` - Emergency detection:
  - Critical symptom patterns
  - Immediate referral triggers
  - Severity scoring

- `service.py` - Orchestrates all rule components:
  - `assess_patient()` - Full safety assessment
  - `check_medication_safety()` - Drug safety check
  - `get_otc_recommendation()` - OTC eligibility

**Integration:**
- Called BEFORE LLM generates any response
- Can BLOCK requests that require immediate medical attention
- Results passed to LLM as safety context
- Logged for audit compliance

---

### Layer 3: LLM (Language Layer)
**Location:** `src/layers/llm/`

**Purpose:** Natural language generation, explanation, and synthesis. NEVER decides alone on safety.

**Components:**
- `meditron.py` - Meditron model wrapper:
  - Model loading with quantization (4-bit, 8-bit)
  - LoRA adapter management
  - Text generation with safety guardrails
  - Streaming support
  - Embedding generation

- `prompts.py` - Role-specific prompts:
  - `RoleType` enum: PATIENT, DOCTOR, STUDENT, RESEARCHER, PHARMA, HOSPITAL
  - System prompts per role
  - Template formatting for different use cases

- `adapters.py` - Domain LoRA adapters:
  - `AdapterType`: PATIENT_TRIAGE, CLINICAL_PHARMACIST, REGULATORY_QA, etc.
  - Adapter selection based on query context
  - Training configuration

- `service.py` - LLM service:
  - `generate()` - Main generation with context
  - Integrates knowledge graph context
  - Integrates safety context from Rule Engine
  - Logs all interactions for audit

**Integration:**
- Receives context from Layer 1 (facts) and Layer 2 (safety)
- Receives conversation history from Layer 5 (memory)
- Output sent to Layer 4 for verification
- Never returns unverified safety-critical information

---

### Layer 4: Verifier (Critic Layer)
**Location:** `src/layers/verifier/`

**Purpose:** Validates LLM outputs before returning to user. Catches hallucinations and errors.

**Components:**
- `hallucination_detector.py`:
  - Checks claims against knowledge graph
  - Detects fabricated statistics
  - Identifies unsupported medical claims
  - Returns confidence scores

- `guideline_checker.py`:
  - Validates against clinical guidelines
  - Detects conflicts with standard of care
  - Checks evidence levels
  - Flags outdated recommendations

- `confidence_calibrator.py`:
  - Detects overconfident language
  - Ensures appropriate uncertainty
  - Adds disclaimers where needed
  - Calibrates probability statements

- `service.py` - Verification orchestrator:
  - `verify()` - Full verification pipeline
  - Aggregates all checks
  - Returns `VerificationResult` with pass/fail and warnings

**Integration:**
- Called after LLM generates response
- Can modify response to add disclaimers
- Can flag response for human review
- Results logged for quality monitoring

---

### Layer 5: Memory (Profiling Layer)
**Location:** `src/layers/memory/`

**Purpose:** Maintains context across sessions. HIPAA-compliant patient profiles.

**Components:**
- `patient_profile.py`:
  - `PatientProfile` - Demographics, conditions, medications, allergies
  - Encrypted PHI storage
  - Longitudinal health history
  - `PatientProfileManager` for CRUD operations

- `entity_profile.py`:
  - `EntityProfile` - Pharma companies, hospitals
  - QA documents, validation records
  - Regulatory accreditations
  - Sales tracking

- `conversation_memory.py`:
  - `Conversation` - Session tracking
  - Message history with metadata
  - Outcome feedback loops
  - Context retrieval for LLM

- `service.py` - Memory service:
  - `create_patient_profile()`, `get_patient_profile()`
  - `start_conversation()`, `add_message()`
  - `get_conversation_context()` - For LLM input
  - All operations audit-logged

**Integration:**
- Provides patient context to Orchestrator at request start
- Stores conversation history for multi-turn interactions
- Enables personalized responses based on history
- Supports outcome tracking for model improvement

---

## Domain Services

**Location:** `src/domains/`

Each domain service combines the 5 layers for specific user types:

| Service | File | Primary Layers Used |
|---------|------|---------------------|
| **PatientService** | `patient.py` | Rule Engine (triage), Memory (profile) |
| **DoctorService** | `doctor.py` | Knowledge Graph (differential), Rule Engine (safety) |
| **StudentService** | `student.py` | LLM (explanations), Knowledge Graph (facts) |
| **ResearcherService** | `researcher.py` | LLM (synthesis), Knowledge Graph (literature) |
| **PharmaService** | `pharma.py` | Memory (entity profiles), Rule Engine (compliance) |
| **HospitalService** | `hospital.py` | Rule Engine (ER triage), Memory (appointments) |
| **GeneralService** | `general.py` | Knowledge Graph (info), LLM (explanations) |

---

## Core Infrastructure

### Security (`src/core/security/`)

| File | Purpose |
|------|---------|
| `encryption.py` | AES-256-GCM encryption for PHI |
| `authentication.py` | JWT token management, password hashing |
| `authorization.py` | RBAC with permissions per role |
| `audit.py` | HIPAA-compliant audit logging |
| `hipaa.py` | PHI detection, anonymization, access control |

### Database (`src/core/database/`)

| File | Purpose |
|------|---------|
| `postgres.py` | Async PostgreSQL for structured data |
| `neo4j_client.py` | Graph database for knowledge graph |
| `redis_client.py` | Caching, sessions, rate limiting |

### Configuration (`src/core/config/`)

| File | Purpose |
|------|---------|
| `settings.py` | Environment-based configuration |

---

## Request Flow Example

### Patient Symptom Assessment

```
1. User submits symptoms via POST /api/v1/patient/assess-symptoms

2. API Route (patient.py) receives request
   └── Validates input with Pydantic

3. PatientService.assess_symptoms() called
   └── Creates PatientAssessment object

4. Layer 5 (Memory): Get patient context
   └── Retrieve existing conditions, medications, allergies

5. Layer 2 (Rule Engine): Safety assessment
   ├── TriageEngine evaluates urgency
   ├── RedFlagDetector checks for emergencies
   └── Returns TriageResult with urgency level

6. Layer 2 (Rule Engine): OTC eligibility (if appropriate)
   └── OTCEligibilityEngine checks if self-care is safe

7. Layer 1 (Knowledge Graph): Get relevant facts
   └── Query disease-symptom relationships

8. Layer 3 (LLM): Generate explanation
   ├── Receives safety context (what to warn about)
   ├── Receives knowledge context (medical facts)
   └── Generates patient-friendly explanation

9. Layer 4 (Verifier): Validate response
   ├── Check for hallucinations
   ├── Verify guideline compliance
   └── Calibrate confidence

10. Layer 5 (Memory): Store interaction
    └── Save to conversation history

11. Audit Logger: Record everything
    └── PHI access, LLM query, safety checks

12. Return SymptomAssessmentResponse to user
```

---

## Safety Guarantees

### The LLM Never Decides Alone

```python
# In orchestrator.py - Safety checks BEFORE LLM
safety_result = await self._run_safety_checks(query, patient_context, user_id)

if safety_result.get("blocked"):
    # Return safety message WITHOUT calling LLM
    return OrchestratorResponse(
        content=safety_result.get("block_message"),
        blocked=True,
        block_reason=safety_result.get("block_reason"),
    )

# Only after safety checks pass does LLM generate
llm_response = await self.llm.generate(...)
```

### Verification Before Response

```python
# In orchestrator.py - Verify AFTER LLM
if self.verifier:
    verification = await self.verifier.verify(response_content, context)
    
    if not verification.is_verified:
        # Add disclaimer to unverified content
        response_content += "\n\n[Note: This response could not be fully verified.]"
```

---

## File Summary

| Directory | Files | Purpose |
|-----------|-------|---------|
| `src/api/` | 10 | FastAPI application and routes |
| `src/core/` | 10 | Infrastructure (security, database, config) |
| `src/domains/` | 8 | Domain-specific services |
| `src/layers/knowledge_graph/` | 4 | Layer 1 - Medical facts |
| `src/layers/rule_engine/` | 6 | Layer 2 - Safety logic |
| `src/layers/llm/` | 5 | Layer 3 - Language model |
| `src/layers/verifier/` | 5 | Layer 4 - Validation |
| `src/layers/memory/` | 5 | Layer 5 - Profiles & memory |
| `src/` | 1 | Orchestrator |
| **Total** | **60** | Complete platform |

---

## Running the System

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your database credentials

# Start the server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Access API docs
open http://localhost:8000/docs
```

---

## Training Pipeline

See `docs/TRAINING_GUIDE.md` for complete training instructions.

### Quick Overview

```bash
# 1. Collect open datasets (MedQA, PubMedQA, etc.)
python scripts/data_collection/collect_datasets.py

# 2. Generate synthetic training data
python scripts/data_collection/synthetic_generator.py --num-examples 5000

# 3. Ingest your WHO/FDA PDFs (place in data/pdfs/)
python scripts/data_collection/ingest_pdfs.py

# 4. Prepare data for training
python scripts/training/prepare_data.py

# 5. Train LoRA adapters
python scripts/training/train_lora.py --adapter patient_triage

# 6. Evaluate
python scripts/training/evaluate_adapter.py --adapter patient_triage
```

### Data Sources

| Source | Script | Output |
|--------|--------|--------|
| Open datasets | `collect_datasets.py` | MedQA, PubMedQA, HealthCareMagic |
| Synthetic cases | `synthetic_generator.py` | Triage, interactions, USMLE |
| Your PDFs | `ingest_pdfs.py` | WHO/FDA regulatory Q&A |
