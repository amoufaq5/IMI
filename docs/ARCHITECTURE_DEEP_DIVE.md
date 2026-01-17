# UMI Architecture Deep Dive

Complete technical documentation explaining how the UMI Medical LLM platform is built, how layers are connected, and the methodology behind each design decision.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Layer Architecture](#layer-architecture)
3. [Component Integration](#component-integration)
4. [AI/ML Pipeline](#aiml-pipeline)
5. [Data Flow](#data-flow)
6. [Methodology Choices](#methodology-choices)
7. [Security Architecture](#security-architecture)
8. [Scalability Design](#scalability-design)

---

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Web App   │  │ Mobile App  │  │  Admin UI   │  │  API Client │        │
│  │  (Next.js)  │  │   (React    │  │  (React)    │  │  (SDK/CLI)  │        │
│  │             │  │   Native)   │  │             │  │             │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
└─────────┼────────────────┼────────────────┼────────────────┼────────────────┘
          │                │                │                │
          └────────────────┴────────────────┴────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API GATEWAY                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  FastAPI Application (src/main.py)                                   │   │
│  │  • Rate Limiting  • Authentication  • Request Validation            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           API ROUTES LAYER                                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │   Auth   │ │  Users   │ │ Consult  │ │  Pharma  │ │ Imaging  │          │
│  │ /auth/*  │ │ /users/* │ │/consult/*│ │/pharma/* │ │/imaging/*│          │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘          │
└───────┼────────────┼────────────┼────────────┼────────────┼─────────────────┘
        │            │            │            │            │
        └────────────┴────────────┴─────┬──────┴────────────┘
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          BUSINESS LOGIC LAYER                                │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐                   │
│  │ Consultation   │ │    Pharma      │ │     Drug       │                   │
│  │   Service      │ │   Service      │ │   Service      │                   │
│  │ • ASMETHOD     │ │ • QA/QC Docs   │ │ • Interactions │                   │
│  │ • Danger Signs │ │ • Compliance   │ │ • OTC Recs     │                   │
│  └───────┬────────┘ └───────┬────────┘ └───────┬────────┘                   │
└──────────┼──────────────────┼──────────────────┼────────────────────────────┘
           │                  │                  │
           └──────────────────┼──────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            AI/ML LAYER                                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │ LLM Service │ │ RAG Service │ │ Vision Svc  │ │ Medical NLP │           │
│  │ • MoE Route │ │ • Embed     │ │ • X-ray     │ │ • Entities  │           │
│  │ • CoT/ToT   │ │ • Retrieve  │ │ • CT/MRI    │ │ • Negation  │           │
│  │ • Generate  │ │ • Rerank    │ │ • Derm      │ │ • Severity  │           │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘           │
└─────────┼───────────────┼───────────────┼───────────────┼───────────────────┘
          │               │               │               │
          └───────────────┴───────┬───────┴───────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │ PostgreSQL  │ │   Redis     │ │   Qdrant    │ │   MinIO     │           │
│  │ • Users     │ │ • Cache     │ │ • Vectors   │ │ • Images    │           │
│  │ • Consults  │ │ • Sessions  │ │ • Embeddings│ │ • Documents │           │
│  │ • Pharma    │ │ • Rate Limit│ │ • RAG Index │ │ • Models    │           │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Layer Architecture

### Layer 1: API Gateway (src/main.py)

**Purpose**: Single entry point for all requests with cross-cutting concerns.

**Components**:
```python
# Entry point: src/main.py
FastAPI Application
├── Lifespan Manager      # Startup/shutdown hooks
├── CORS Middleware       # Cross-origin requests
├── Exception Handlers    # Unified error responses
├── Request Logging       # Structured logging
└── Route Mounting        # API versioning (/api/v1)
```

**How It Works**:
1. Request arrives at FastAPI application
2. CORS middleware validates origin
3. Request logging middleware captures metrics
4. Route is matched to appropriate handler
5. Dependencies (auth, db) are injected
6. Response is formatted and returned

**Methodology Choice**: 
- **FastAPI** over Flask/Django: Async-native, automatic OpenAPI docs, Pydantic validation, type hints
- **Lifespan context manager**: Clean resource management for DB connections

---

### Layer 2: API Routes (src/api/v1/)

**Purpose**: HTTP endpoint definitions with request/response handling.

**Structure**:
```
src/api/v1/
├── router.py          # Aggregates all routes
├── auth.py            # Authentication endpoints
├── users.py           # User management
├── consultations.py   # ASMETHOD consultations
├── pharma.py          # QA/QC documents
├── drugs.py           # Drug information
├── health.py          # Health topics
└── imaging.py         # Medical image analysis
```

**Integration Pattern**:
```python
# Each route file follows this pattern:
from fastapi import APIRouter, Depends
from src.api.deps import get_current_user  # Dependency injection
from src.services.xxx_service import XXXService  # Business logic

router = APIRouter()

@router.post("/endpoint")
async def endpoint(
    data: RequestSchema,                    # Pydantic validation
    current_user: User = Depends(get_current_user),  # Auth
    db: AsyncSession = Depends(get_db),     # Database
) -> ResponseSchema:
    service = XXXService(db)                # Inject DB into service
    result = await service.do_something(data)
    return result
```

**Methodology Choice**:
- **Dependency Injection**: Testable, loosely coupled components
- **Pydantic Schemas**: Automatic validation, serialization, documentation
- **Async handlers**: Non-blocking I/O for high concurrency

---

### Layer 3: Business Logic (src/services/)

**Purpose**: Domain logic separated from HTTP concerns.

**Services**:
```
src/services/
├── consultation_service.py    # ASMETHOD protocol, danger detection
├── pharma_service.py          # Document generation, compliance
├── drug_service.py            # Drug search, interactions
└── medical_knowledge_service.py  # Disease, guidelines, papers
```

**Service Pattern**:
```python
class ConsultationService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.danger_detector = DangerSignsDetector()
        self.asmethod_engine = ASMETHODEngine()
    
    async def create_consultation(self, user_id, data):
        # 1. Create consultation record
        # 2. Initialize ASMETHOD protocol
        # 3. Return with first question
        pass
    
    async def add_message(self, consultation_id, message):
        # 1. Get consultation
        # 2. Check for danger signs
        # 3. Update ASMETHOD data
        # 4. Generate AI response
        # 5. Return updated consultation
        pass
```

**Key Components**:

| Component | Location | Purpose |
|-----------|----------|---------|
| `DangerSignsDetector` | consultation_service.py | Red/amber flag detection |
| `ASMETHODEngine` | consultation_service.py | Protocol state machine |
| `DocumentTemplates` | pharma_service.py | QA/QC document structures |

**Methodology Choice**:
- **Service Layer Pattern**: Separates business logic from presentation
- **Composition over Inheritance**: Services compose specialized components
- **Async Database Operations**: Non-blocking queries with SQLAlchemy async

---

### Layer 4: AI/ML Layer (src/ai/)

**Purpose**: All AI/ML functionality encapsulated in specialized services.

**Components**:
```
src/ai/
├── llm_service.py      # LLM with MoE routing
├── rag_service.py      # Retrieval-Augmented Generation
├── medical_nlp.py      # Medical entity extraction
├── vision_service.py   # Medical image analysis
└── model_loader.py     # Model loading and caching
```

#### 4.1 LLM Service Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      LLMService                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  ExpertRouter                        │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │   │
│  │  │Diagnosis│ │  Drug   │ │Regulatory│ │  QA/QC  │   │   │
│  │  │ Expert  │ │ Expert  │ │ Expert  │ │ Expert  │   │   │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘   │   │
│  │       │           │           │           │         │   │
│  │       └───────────┴─────┬─────┴───────────┘         │   │
│  │                         ▼                           │   │
│  │              ┌─────────────────┐                    │   │
│  │              │ ReasoningEngine │                    │   │
│  │              │ • Chain-of-Thought                   │   │
│  │              │ • Tree-of-Thought                    │   │
│  │              │ • Self-Consistency                   │   │
│  │              └────────┬────────┘                    │   │
│  └───────────────────────┼─────────────────────────────┘   │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Model Backend                           │   │
│  │  ┌──────────────┐  OR  ┌──────────────┐             │   │
│  │  │  OpenAI API  │      │  Local Model │             │   │
│  │  │  (GPT-4)     │      │  (Mistral)   │             │   │
│  │  └──────────────┘      └──────────────┘             │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Expert Routing Logic**:
```python
class ExpertRouter:
    EXPERT_KEYWORDS = {
        ExpertType.DIAGNOSIS: ["symptom", "pain", "fever", ...],
        ExpertType.DRUG: ["medication", "dose", "interaction", ...],
        ExpertType.QA_QC: ["validation", "sop", "batch", ...],
    }
    
    def route(self, query: str, context: Dict) -> Tuple[ExpertType, float]:
        # 1. Tokenize query
        # 2. Match keywords to experts
        # 3. Consider context hints
        # 4. Return best expert with confidence
```

**Methodology Choice**:
- **Mixture of Experts (MoE)**: Different prompts for different domains
- **Chain-of-Thought**: Step-by-step reasoning for medical accuracy
- **Fallback Pattern**: OpenAI API → Local Model → Mock responses

#### 4.2 RAG Service Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      RAGService                              │
│                                                              │
│  Query: "What are symptoms of diabetes?"                     │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              EmbeddingService                        │   │
│  │  sentence-transformers/all-MiniLM-L6-v2              │   │
│  │  Query → [0.12, -0.34, 0.56, ...]  (384 dims)       │   │
│  └────────────────────────┬────────────────────────────┘   │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              VectorStore (Qdrant)                    │   │
│  │  Collections:                                        │   │
│  │  • medical_literature  (PubMed articles)            │   │
│  │  • drug_information    (DrugBank/OpenFDA)           │   │
│  │  • clinical_guidelines (NICE, WHO)                  │   │
│  │  • qa_qc_templates     (Pharma documents)           │   │
│  └────────────────────────┬────────────────────────────┘   │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Reranker                                │   │
│  │  Cross-encoder scoring for relevance                 │   │
│  └────────────────────────┬────────────────────────────┘   │
│                           ▼                                  │
│  Top-K Documents with scores                                 │
└─────────────────────────────────────────────────────────────┘
```

**Methodology Choice**:
- **Dense Retrieval**: Semantic search vs keyword matching
- **Qdrant**: Purpose-built vector DB with filtering
- **Reranking**: Two-stage retrieval for accuracy

#### 4.3 Vision Service Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  MedicalVisionService                        │
│                                                              │
│  Image Input                                                 │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           ImagePreprocessor                          │   │
│  │  • Load (JPEG, PNG, DICOM)                          │   │
│  │  • Resize to model input size                       │   │
│  │  • Normalize (ImageNet stats)                       │   │
│  │  • Convert to tensor                                │   │
│  └────────────────────────┬────────────────────────────┘   │
│                           │                                  │
│           ┌───────────────┼───────────────┐                 │
│           ▼               ▼               ▼                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │ChestXRay    │ │Dermoscopy   │ │LabReport    │           │
│  │Analyzer     │ │Analyzer     │ │OCR          │           │
│  │DenseNet121  │ │ResNet/ViT   │ │Tesseract    │           │
│  │14 pathology │ │7 lesion     │ │Value        │           │
│  │classes      │ │types        │ │extraction   │           │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘           │
│         │               │               │                   │
│         └───────────────┴───────┬───────┘                   │
│                                 ▼                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           ImageAnalysisResult                        │   │
│  │  • findings: [{finding, probability, severity}]     │   │
│  │  • impression: "Findings suggestive of..."          │   │
│  │  • recommendations: ["Consult specialist", ...]     │   │
│  │  • urgency: normal | attention | urgent | critical  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Methodology Choice**:
- **DenseNet121 for Chest X-ray**: Proven architecture for CheXpert/ChestX-ray14
- **DICOM Support**: Native medical imaging format
- **Multi-class Classification**: Detect multiple pathologies simultaneously

---

### Layer 5: Data Layer

**Purpose**: Persistent storage with async access.

#### 5.1 Database Models (src/models/)

```
src/models/
├── user.py       # User, UserProfile, Organization
├── patient.py    # PatientProfile, Consultation, MedicalHistory
├── medical.py    # Disease, Drug, DrugInteraction, Guidelines
└── pharma.py     # Facility, Document, ComplianceCheck, Batch
```

**Model Relationships**:
```
User (1) ──────────────── (N) Consultation
  │                              │
  │                              │
  └── (1) UserProfile            └── (N) ConsultationImage
  
Organization (1) ──────── (N) Facility
                                │
                                ├── (N) Document
                                ├── (N) ComplianceCheck
                                └── (N) ProductionBatch

Drug (1) ──────────────── (N) DrugInteraction ──── (1) Drug
```

**Methodology Choice**:
- **SQLAlchemy 2.0 Async**: Modern async ORM with type hints
- **JSONB Columns**: Flexible schema for medical data (symptoms, medications)
- **UUID Primary Keys**: Distributed-friendly, non-sequential

#### 5.2 Caching Strategy (Redis)

```python
# Cache layers:
1. Session Cache      # User sessions, tokens
2. Query Cache        # Expensive DB queries
3. AI Response Cache  # LLM responses for identical queries
4. Rate Limit Store   # Request counting per user

# TTL Strategy:
- Sessions: 24 hours
- Query cache: 5 minutes
- AI responses: 1 hour (medical info changes)
- Rate limits: 1 minute window
```

#### 5.3 Vector Database (Qdrant)

```python
# Collections:
medical_literature:
  - vectors: 384 dimensions (MiniLM)
  - payload: {title, content, source, pmid, mesh_terms}
  - index: HNSW with cosine similarity

drug_information:
  - vectors: 384 dimensions
  - payload: {name, generic_name, indications, warnings}
  - filters: {drug_class, otc_only}
```

---

## Component Integration

### Request Flow Example: Consultation

```
1. POST /api/v1/consultations
   │
   ▼
2. auth.py: get_current_user()
   │ └── Verify JWT token
   │ └── Load user from DB
   ▼
3. consultations.py: start_consultation()
   │ └── Validate request with Pydantic
   │ └── Inject dependencies (db, user)
   ▼
4. ConsultationService.create_consultation()
   │ └── Create DB record
   │ └── Initialize ASMETHOD state
   │ └── Generate welcome message
   ▼
5. Return ConsultationResponse
```

### Request Flow Example: AI Analysis

```
1. POST /api/v1/consultations/{id}/message
   │
   ▼
2. ConsultationService.add_message()
   │
   ├── DangerSignsDetector.detect()
   │   └── Check for red flags
   │   └── If emergency → immediate referral
   │
   ├── ASMETHODEngine.get_next_question()
   │   └── Determine protocol progress
   │   └── Get next required field
   │
   └── LLMService.generate()
       │
       ├── ExpertRouter.route()
       │   └── Select DIAGNOSIS expert
       │
       ├── RAGService.retrieve()
       │   └── Get relevant medical literature
       │
       └── Generate response with context
           └── Chain-of-Thought reasoning
           └── Include citations
```

### Async Task Flow (Celery)

```
1. API receives image upload
   │
   ▼
2. Create task: analyze_medical_image_task.delay()
   │
   ▼
3. Return task_id immediately
   │
   ▼
4. Celery worker picks up task
   │
   ├── Load image
   ├── Preprocess
   ├── Run model inference (GPU)
   ├── Generate findings
   └── Store results
   │
   ▼
5. Client polls for result or receives webhook
```

---

## AI/ML Pipeline

### Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Pipeline                         │
│                                                              │
│  1. DATA COLLECTION                                          │
│     scripts/training/download_datasets.py                    │
│     ├── PubMedQA (research QA)                              │
│     ├── MedMCQA (medical MCQ)                               │
│     ├── Medical Meadow (flashcards)                         │
│     └── ChatDoctor (conversations)                          │
│                          │                                   │
│                          ▼                                   │
│  2. DATA PREPARATION                                         │
│     scripts/training/prepare_data.py                         │
│     ├── Clean and normalize text                            │
│     ├── Convert to chat format                              │
│     ├── Add ASMETHOD synthetic data                         │
│     ├── Deduplicate                                         │
│     └── Split train/val/test                                │
│                          │                                   │
│                          ▼                                   │
│  3. FINE-TUNING                                              │
│     scripts/training/fine_tune.py                            │
│     ├── Load Mistral-7B-Instruct                            │
│     ├── Apply QLoRA (4-bit quantization)                    │
│     ├── Configure LoRA adapters                             │
│     │   r=64, alpha=128, dropout=0.05                       │
│     ├── Train with SFTTrainer                               │
│     └── Save adapter weights                                │
│                          │                                   │
│                          ▼                                   │
│  4. EVALUATION                                               │
│     scripts/training/evaluate.py                             │
│     ├── Safety tests (crisis, overdose, emergency)          │
│     ├── Medical accuracy (symptoms, drugs)                  │
│     ├── ASMETHOD adherence                                  │
│     └── Drug interaction knowledge                          │
│                          │                                   │
│                          ▼                                   │
│  5. DEPLOYMENT                                               │
│     models/umi-medical-v1/                                   │
│     ├── adapter_config.json                                 │
│     ├── adapter_model.safetensors                           │
│     └── training_config.json                                │
└─────────────────────────────────────────────────────────────┘
```

### Inference Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                   Inference Pipeline                         │
│                                                              │
│  User Query                                                  │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. PREPROCESSING                                     │   │
│  │    • Tokenize input                                  │   │
│  │    • Apply chat template                             │   │
│  │    • Truncate to max length                          │   │
│  └────────────────────────┬────────────────────────────┘   │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 2. EXPERT ROUTING                                    │   │
│  │    • Analyze query keywords                          │   │
│  │    • Select appropriate expert                       │   │
│  │    • Load expert system prompt                       │   │
│  └────────────────────────┬────────────────────────────┘   │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 3. RAG RETRIEVAL (if enabled)                        │   │
│  │    • Embed query                                     │   │
│  │    • Search vector DB                                │   │
│  │    • Rerank results                                  │   │
│  │    • Inject context into prompt                      │   │
│  └────────────────────────┬────────────────────────────┘   │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 4. GENERATION                                        │   │
│  │    • Apply Chain-of-Thought                          │   │
│  │    • Generate with model                             │   │
│  │    • Temperature: 0.3 (medical), 0.7 (general)      │   │
│  └────────────────────────┬────────────────────────────┘   │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 5. POST-PROCESSING                                   │   │
│  │    • Extract citations                               │   │
│  │    • Safety check                                    │   │
│  │    • Format response                                 │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Methodology Choices

### Why Each Technology Was Chosen

| Component | Choice | Alternatives Considered | Reasoning |
|-----------|--------|------------------------|-----------|
| **Web Framework** | FastAPI | Flask, Django | Async-native, auto docs, Pydantic integration |
| **Database** | PostgreSQL | MySQL, MongoDB | JSONB for flexible medical data, mature, reliable |
| **ORM** | SQLAlchemy 2.0 | Django ORM, Tortoise | Async support, type hints, flexibility |
| **Cache** | Redis | Memcached | Pub/sub for real-time, data structures |
| **Vector DB** | Qdrant | Pinecone, Weaviate | Open-source, filtering, self-hosted option |
| **Task Queue** | Celery | RQ, Dramatiq | Mature, Redis backend, monitoring |
| **Base LLM** | Mistral-7B | Llama-2, GPT-4 | Open weights, strong performance, Apache 2.0 |
| **Fine-tuning** | QLoRA | Full fine-tune, LoRA | Memory efficient, preserves base capabilities |
| **Embeddings** | MiniLM-L6 | OpenAI, E5 | Fast, good quality, runs locally |
| **Vision** | DenseNet121 | ResNet, ViT | Proven for medical imaging, efficient |

### Design Patterns Used

| Pattern | Where Used | Purpose |
|---------|------------|---------|
| **Dependency Injection** | API routes | Testability, loose coupling |
| **Repository Pattern** | Services | Abstract data access |
| **Strategy Pattern** | Expert routing | Swappable expert behaviors |
| **Factory Pattern** | Model loader | Centralized model creation |
| **Singleton** | ModelLoader | Cache loaded models |
| **Chain of Responsibility** | Middleware | Request processing pipeline |
| **Observer** | Celery tasks | Async event handling |

### ASMETHOD Protocol Implementation

```python
# Why ASMETHOD for consultations:
# - Structured approach ensures complete information gathering
# - Standardized protocol used by pharmacists
# - Maps directly to clinical decision-making
# - Enables automated danger sign detection

ASMETHOD = {
    "A": "Age",           # Age-specific conditions
    "S": "Self/Other",    # Who is the patient
    "M": "Medications",   # Drug interactions, allergies
    "E": "Exact Symptoms",# Detailed symptom description
    "T": "Time/Duration", # Acute vs chronic
    "H": "History",       # Relevant medical history
    "O": "Other Symptoms",# Associated symptoms
    "D": "Danger Signs",  # Red flags (automated)
}
```

---

## Security Architecture

### Authentication Flow

```
┌─────────────────────────────────────────────────────────────┐
│                   Authentication Flow                        │
│                                                              │
│  1. Login Request                                            │
│     POST /auth/login {email, password}                       │
│                          │                                   │
│                          ▼                                   │
│  2. Verify Credentials                                       │
│     • Hash password with bcrypt                              │
│     • Compare with stored hash                               │
│                          │                                   │
│                          ▼                                   │
│  3. Generate Tokens                                          │
│     • Access token (JWT, 30 min)                            │
│     • Refresh token (JWT, 7 days)                           │
│     • Sign with RS256 or HS256                              │
│                          │                                   │
│                          ▼                                   │
│  4. Subsequent Requests                                      │
│     Authorization: Bearer <access_token>                     │
│                          │                                   │
│                          ▼                                   │
│  5. Token Verification                                       │
│     • Verify signature                                       │
│     • Check expiration                                       │
│     • Load user from DB                                      │
│     • Inject into request                                    │
└─────────────────────────────────────────────────────────────┘
```

### Data Protection

```
┌─────────────────────────────────────────────────────────────┐
│                   Data Protection Layers                     │
│                                                              │
│  Layer 1: Transport                                          │
│  └── TLS 1.3 for all connections                            │
│                                                              │
│  Layer 2: Application                                        │
│  ├── JWT token validation                                   │
│  ├── Role-based access control                              │
│  └── Rate limiting per user                                 │
│                                                              │
│  Layer 3: Data                                               │
│  ├── Field-level encryption (Fernet)                        │
│  │   └── PII: names, addresses, health data                │
│  ├── Password hashing (bcrypt, cost=12)                     │
│  └── Audit logging                                          │
│                                                              │
│  Layer 4: Infrastructure                                     │
│  ├── Database encryption at rest                            │
│  ├── Network isolation                                      │
│  └── Secrets management                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Scalability Design

### Horizontal Scaling

```
                    Load Balancer
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    ┌─────────┐     ┌─────────┐     ┌─────────┐
    │ API Pod │     │ API Pod │     │ API Pod │
    │   #1    │     │   #2    │     │   #3    │
    └────┬────┘     └────┬────┘     └────┬────┘
         │               │               │
         └───────────────┴───────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
         ┌─────────┐          ┌─────────┐
         │  Redis  │          │PostgreSQL│
         │ Cluster │          │ Primary  │
         └─────────┘          └────┬────┘
                                   │
                              ┌────┴────┐
                              ▼         ▼
                         ┌────────┐ ┌────────┐
                         │Replica │ │Replica │
                         │   #1   │ │   #2   │
                         └────────┘ └────────┘
```

### GPU Scaling for AI

```
                    API Pods (CPU)
                         │
                         ▼
                   Redis Queue
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    ┌─────────┐     ┌─────────┐     ┌─────────┐
    │GPU Worker│    │GPU Worker│    │GPU Worker│
    │ RTX 4090 │    │ RTX 4090 │    │   A100   │
    │ Mistral  │    │ Mistral  │    │ Mistral  │
    └─────────┘     └─────────┘     └─────────┘
```

---

## Summary

The UMI platform is built with:

1. **Clean Layer Separation**: API → Services → AI → Data
2. **Async-First Design**: Non-blocking I/O throughout
3. **Modular AI Components**: Swappable LLM, RAG, Vision services
4. **Medical-Specific Patterns**: ASMETHOD, danger detection, drug interactions
5. **Production-Ready Infrastructure**: Docker, Celery, monitoring
6. **Security by Design**: Encryption, RBAC, audit logging

Each layer is independently testable, deployable, and scalable.
