# UMI - Technical Architecture

## System Design with Latest AI/ML Advancements

This document outlines the complete technical architecture for UMI, incorporating cutting-edge AI technologies and production-ready design patterns.

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                   CLIENT LAYER                                       │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │
│  │   Web App   │  │ Mobile App  │  │  Pharma ERP │  │Hospital EHR │                │
│  │   (React)   │  │(React Native│  │ Integration │  │ Integration │                │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                │
│         │                │                │                │                        │
│         └────────────────┴────────────────┴────────────────┘                        │
│                                    │                                                 │
│                          ┌─────────▼─────────┐                                      │
│                          │    API Gateway    │                                      │
│                          │   (Kong/Nginx)    │                                      │
│                          └─────────┬─────────┘                                      │
└────────────────────────────────────┼────────────────────────────────────────────────┘
                                     │
┌────────────────────────────────────┼────────────────────────────────────────────────┐
│                              SERVICE LAYER                                           │
├────────────────────────────────────┼────────────────────────────────────────────────┤
│                          ┌─────────▼─────────┐                                      │
│                          │  Load Balancer    │                                      │
│                          └─────────┬─────────┘                                      │
│                                    │                                                 │
│    ┌───────────────────────────────┼───────────────────────────────┐                │
│    │                               │                               │                │
│    ▼                               ▼                               ▼                │
│ ┌──────────────┐           ┌──────────────┐           ┌──────────────┐             │
│ │   Auth &     │           │   Core API   │           │   Async      │             │
│ │   Identity   │           │   Service    │           │   Workers    │             │
│ │   Service    │           │   (FastAPI)  │           │   (Celery)   │             │
│ └──────────────┘           └──────┬───────┘           └──────────────┘             │
│                                   │                                                 │
└───────────────────────────────────┼─────────────────────────────────────────────────┘
                                    │
┌───────────────────────────────────┼─────────────────────────────────────────────────┐
│                              AI/ML LAYER                                             │
├───────────────────────────────────┼─────────────────────────────────────────────────┤
│                          ┌────────▼────────┐                                        │
│                          │  AI Orchestrator │                                       │
│                          │  (LangChain +    │                                       │
│                          │   Custom Router) │                                       │
│                          └────────┬────────┘                                        │
│                                   │                                                  │
│    ┌──────────────┬───────────────┼───────────────┬──────────────┐                 │
│    │              │               │               │              │                 │
│    ▼              ▼               ▼               ▼              ▼                 │
│ ┌────────┐  ┌──────────┐  ┌────────────┐  ┌──────────┐  ┌────────────┐            │
│ │Medical │  │  Drug    │  │  Imaging   │  │Regulatory│  │  Research  │            │
│ │Diagnosis│  │Interaction│  │  Analysis  │  │Compliance│  │  Assistant │            │
│ │ Expert │  │  Expert  │  │   Expert   │  │  Expert  │  │   Expert   │            │
│ │ (MoE)  │  │  (MoE)   │  │(Vision LLM)│  │  (MoE)   │  │   (MoE)    │            │
│ └────────┘  └──────────┘  └────────────┘  └──────────┘  └────────────┘            │
│                                   │                                                  │
│                          ┌────────▼────────┐                                        │
│                          │   RAG Engine    │                                        │
│                          │ (Vector Search) │                                        │
│                          └─────────────────┘                                        │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                    │
┌───────────────────────────────────┼──────────────────────────────────────────────────┐
│                              DATA LAYER                                               │
├───────────────────────────────────┼──────────────────────────────────────────────────┤
│                                   │                                                   │
│  ┌─────────────┐  ┌─────────────┐ │ ┌─────────────┐  ┌─────────────┐                │
│  │ PostgreSQL  │  │   Redis     │ │ │  Pinecone/  │  │   MinIO     │                │
│  │ (Primary DB)│  │  (Cache)    │ │ │  Qdrant     │  │(Object Store│                │
│  │             │  │             │ │ │(Vector DB)  │  │  for Images)│                │
│  └─────────────┘  └─────────────┘   └─────────────┘  └─────────────┘                │
│                                                                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │ TimescaleDB │  │ Elasticsearch│  │  MongoDB    │  │   Kafka     │                │
│  │(Time Series)│  │  (Search)   │  │ (Documents) │  │  (Events)   │                │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘                 │
│                                                                                       │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. AI/ML Architecture - Latest Advancements

### 2.1 Mixture of Experts (MoE) Architecture

UMI uses a **Sparse Mixture of Experts** model, similar to architectures used in GPT-4 and Mixtral, but specialized for medical domains.

```
                         ┌─────────────────┐
                         │   Input Query   │
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │  Router Network │
                         │  (Learned Gating)│
                         └────────┬────────┘
                                  │
           ┌──────────────────────┼──────────────────────┐
           │                      │                      │
    ┌──────▼──────┐       ┌──────▼──────┐       ┌──────▼──────┐
    │   Expert 1  │       │   Expert 2  │       │   Expert N  │
    │  Diagnosis  │       │    Drugs    │       │  Regulatory │
    │             │       │             │       │             │
    │ Specialized │       │ Specialized │       │ Specialized │
    │   Weights   │       │   Weights   │       │   Weights   │
    └──────┬──────┘       └──────┬──────┘       └──────┬──────┘
           │                      │                      │
           └──────────────────────┼──────────────────────┘
                                  │
                         ┌────────▼────────┐
                         │  Weighted Sum   │
                         │   of Outputs    │
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │  Final Output   │
                         └─────────────────┘
```

**Expert Modules**:
| Expert | Specialization | Training Data |
|--------|---------------|---------------|
| Diagnosis Expert | Disease identification, symptom analysis | Medical textbooks, clinical guidelines |
| Drug Expert | Interactions, dosing, contraindications | DrugBank, RxNorm, FDA labels |
| Imaging Expert | CT, MRI, X-ray interpretation | RadImageNet, CheXpert, MIMIC-CXR |
| Regulatory Expert | MHRA, UAE MOH, FDA compliance | Regulatory documents, guidelines |
| Research Expert | Literature review, paper writing | PubMed, clinical trials |
| QA/QC Expert | Pharma documentation, compliance | GMP guidelines, ISO standards |

### 2.2 Advanced Reasoning Architectures

#### Chain-of-Thought (CoT) Prompting
```python
# Example: Diagnostic reasoning with CoT
prompt = """
Patient presents with: chest pain, shortness of breath, sweating

Let's think step by step:
1. Identify key symptoms and their significance
2. Consider differential diagnoses
3. Evaluate risk factors from patient history
4. Determine urgency level
5. Recommend next steps

Step 1: Key symptoms analysis...
"""
```

#### Tree-of-Thought (ToT) for Complex Diagnoses
```
                    ┌─────────────────┐
                    │ Initial Symptom │
                    │   Assessment    │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
    ┌─────▼─────┐     ┌─────▼─────┐     ┌─────▼─────┐
    │ Cardiac   │     │Respiratory│     │   GI      │
    │ Pathway   │     │  Pathway  │     │ Pathway   │
    └─────┬─────┘     └─────┬─────┘     └─────┬─────┘
          │                 │                 │
    ┌─────▼─────┐     ┌─────▼─────┐     ┌─────▼─────┐
    │  MI/ACS   │     │Pneumonia  │     │  GERD     │
    │  Score:85 │     │ Score:45  │     │ Score:20  │
    └───────────┘     └───────────┘     └───────────┘
          │
          ▼
    [Selected Path: Cardiac - Recommend ECG, Troponin]
```

#### Self-Consistency Decoding
Multiple reasoning paths are generated and the most consistent answer is selected:
```python
def self_consistent_diagnosis(symptoms, n_samples=5):
    diagnoses = []
    for _ in range(n_samples):
        diagnosis = model.generate(symptoms, temperature=0.7)
        diagnoses.append(diagnosis)
    
    # Vote for most consistent diagnosis
    return most_common(diagnoses)
```

### 2.3 Retrieval-Augmented Generation (RAG) Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RAG PIPELINE                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐                                                           │
│  │  User Query  │                                                           │
│  └──────┬───────┘                                                           │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────┐     ┌─────────────────────────────────────────────────┐  │
│  │   Query      │     │              KNOWLEDGE BASES                     │  │
│  │  Embedding   │     ├─────────────────────────────────────────────────┤  │
│  │  (Medical    │     │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌────────┐│  │
│  │   BioBERT)   │────▶│  │ PubMed  │ │DrugBank │ │Clinical │ │  GMP   ││  │
│  └──────────────┘     │  │ Articles│ │  Drugs  │ │ Trials  │ │  Docs  ││  │
│                       │  └─────────┘ └─────────┘ └─────────┘ └────────┘│  │
│                       │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌────────┐│  │
│                       │  │  ICD-11 │ │ SNOMED  │ │  MHRA   │ │  UAE   ││  │
│                       │  │  Codes  │ │   CT    │ │  Regs   │ │  Regs  ││  │
│                       │  └─────────┘ └─────────┘ └─────────┘ └────────┘│  │
│                       └─────────────────────────────────────────────────┘  │
│                                          │                                  │
│                                          ▼                                  │
│                                 ┌─────────────────┐                        │
│                                 │  Top-K Relevant │                        │
│                                 │    Documents    │                        │
│                                 └────────┬────────┘                        │
│                                          │                                  │
│         ┌────────────────────────────────┴────────────────────────────┐    │
│         │                                                              │    │
│         ▼                                                              ▼    │
│  ┌──────────────┐                                              ┌───────────┐│
│  │   Reranker   │                                              │  Context  ││
│  │  (Cross-     │                                              │  Window   ││
│  │   Encoder)   │                                              │ Expansion ││
│  └──────┬───────┘                                              └─────┬─────┘│
│         │                                                            │      │
│         └────────────────────────┬───────────────────────────────────┘      │
│                                  │                                          │
│                                  ▼                                          │
│                         ┌─────────────────┐                                │
│                         │  Augmented      │                                │
│                         │  Prompt with    │                                │
│                         │  Retrieved Docs │                                │
│                         └────────┬────────┘                                │
│                                  │                                          │
│                                  ▼                                          │
│                         ┌─────────────────┐                                │
│                         │   LLM Response  │                                │
│                         │  with Citations │                                │
│                         └─────────────────┘                                │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.4 Multimodal Vision Architecture

For medical imaging analysis (CT, MRI, X-ray, Lab Reports):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MULTIMODAL VISION PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                │
│  │   CT Scan    │     │    X-Ray     │     │  Lab Report  │                │
│  │    Image     │     │    Image     │     │    (PDF)     │                │
│  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘                │
│         │                    │                    │                         │
│         ▼                    ▼                    ▼                         │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                │
│  │  DICOM       │     │   Image      │     │    OCR       │                │
│  │  Parser      │     │   Preprocess │     │  (Tesseract) │                │
│  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘                │
│         │                    │                    │                         │
│         └────────────────────┼────────────────────┘                         │
│                              │                                              │
│                              ▼                                              │
│                    ┌─────────────────┐                                     │
│                    │  Vision Encoder │                                     │
│                    │  (ViT / CLIP    │                                     │
│                    │   Medical)      │                                     │
│                    └────────┬────────┘                                     │
│                             │                                               │
│                             ▼                                               │
│                    ┌─────────────────┐                                     │
│                    │   Projection    │                                     │
│                    │     Layer       │                                     │
│                    └────────┬────────┘                                     │
│                             │                                               │
│              ┌──────────────┴──────────────┐                               │
│              │                             │                               │
│              ▼                             ▼                               │
│     ┌─────────────────┐          ┌─────────────────┐                      │
│     │  Image Tokens   │          │   Text Tokens   │                      │
│     │  (Visual)       │          │   (Query)       │                      │
│     └────────┬────────┘          └────────┬────────┘                      │
│              │                             │                               │
│              └──────────────┬──────────────┘                               │
│                             │                                               │
│                             ▼                                               │
│                    ┌─────────────────┐                                     │
│                    │  Multimodal LLM │                                     │
│                    │  (LLaVA-Med /   │                                     │
│                    │   Custom)       │                                     │
│                    └────────┬────────┘                                     │
│                             │                                               │
│                             ▼                                               │
│                    ┌─────────────────┐                                     │
│                    │  Diagnostic     │                                     │
│                    │  Report with    │                                     │
│                    │  Annotations    │                                     │
│                    └─────────────────┘                                     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.5 ASMETHOD Protocol Engine

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ASMETHOD CONSULTATION ENGINE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    CONVERSATION STATE MACHINE                        │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                      │   │
│  │   [START] ──▶ [A]ge ──▶ [S]elf/Other ──▶ [M]edications ──▶         │   │
│  │                                                                      │   │
│  │   ──▶ [E]xact Symptoms ──▶ [T]ime ──▶ [H]istory ──▶                │   │
│  │                                                                      │   │
│  │   ──▶ [O]ther Symptoms ──▶ [D]anger Signs ──▶ [ASSESSMENT]         │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      DANGER SIGN DETECTION                           │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                      │   │
│  │   RED FLAGS (Immediate Referral):                                   │   │
│  │   • Chest pain + shortness of breath                                │   │
│  │   • Severe headache + neck stiffness                                │   │
│  │   • Sudden vision loss                                              │   │
│  │   • Signs of stroke (FAST)                                          │   │
│  │   • Anaphylaxis symptoms                                            │   │
│  │   • Suicidal ideation                                               │   │
│  │                                                                      │   │
│  │   AMBER FLAGS (Urgent GP/Pharmacist):                               │   │
│  │   • Symptoms > 7 days                                               │   │
│  │   • Fever > 38.5°C for > 3 days                                     │   │
│  │   • Medication interactions detected                                │   │
│  │   • Vulnerable populations (elderly, pregnant, immunocompromised)   │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      DECISION OUTPUT                                 │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                      │   │
│  │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │   │
│  │   │   OTC       │    │   Refer to  │    │  Emergency  │            │   │
│  │   │Recommendation│    │   Doctor    │    │   Referral  │            │   │
│  │   │             │    │             │    │             │            │   │
│  │   │ • Drug name │    │ • Urgency   │    │ • Call 999  │            │   │
│  │   │ • Dosage    │    │ • Specialty │    │ • Symptoms  │            │   │
│  │   │ • Duration  │    │ • Prep info │    │ • Location  │            │   │
│  │   │ • Warnings  │    │             │    │             │            │   │
│  │   └─────────────┘    └─────────────┘    └─────────────┘            │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Architecture

### 3.1 Open Data Sources

| Source | Data Type | Update Frequency | Integration Method |
|--------|-----------|------------------|-------------------|
| **PubMed/MEDLINE** | Research articles | Daily | API + Bulk download |
| **DrugBank** | Drug information | Monthly | API |
| **ClinicalTrials.gov** | Trial data | Daily | API |
| **ICD-11** | Disease codes | Annually | Bulk download |
| **SNOMED CT** | Clinical terms | Bi-annually | Bulk download |
| **RxNorm** | Drug nomenclature | Monthly | API |
| **OpenFDA** | Adverse events | Daily | API |
| **MHRA** | UK drug alerts | Daily | RSS/Scraping |
| **BNF** | UK formulary | Monthly | Licensed access |
| **CheXpert** | Chest X-rays | Static | Download |
| **MIMIC-CXR** | Chest X-rays | Static | PhysioNet access |
| **RadImageNet** | Radiology images | Static | Download |

### 3.2 Database Schema (Simplified)

```sql
-- Core Entities
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATABASE SCHEMA                                    │
├─────────────────────────────────────────────────────────────────────────────┤

-- Users & Authentication
users (id, email, password_hash, role, created_at, updated_at)
user_profiles (id, user_id, first_name, last_name, dob, gender, country)
organizations (id, name, type, country, subscription_tier)
user_organizations (user_id, org_id, role)

-- Patient Data
patient_profiles (id, user_id, blood_type, allergies, chronic_conditions)
medical_history (id, patient_id, condition, diagnosed_date, status)
medications (id, patient_id, drug_id, dosage, frequency, start_date, end_date)
consultations (id, patient_id, asmethod_data, diagnosis, recommendation, created_at)

-- Medical Knowledge
diseases (id, icd_code, name, description, symptoms, treatments)
drugs (id, rxnorm_id, name, generic_name, class, interactions, contraindications)
drug_interactions (drug_id_1, drug_id_2, severity, description)
clinical_guidelines (id, condition, guideline_body, recommendations, version)

-- Pharma QA/QC
facilities (id, org_id, name, type, address, certifications)
documents (id, facility_id, type, content, version, status, created_at)
compliance_checks (id, facility_id, regulation, status, findings, date)
production_batches (id, facility_id, product, batch_number, status, qa_data)

-- Research
research_papers (id, pmid, title, abstract, authors, journal, date)
clinical_trials (id, nct_id, title, phase, status, conditions, interventions)

-- Audit & Compliance
audit_logs (id, user_id, action, entity_type, entity_id, timestamp, ip_address)
ai_decisions (id, consultation_id, model_version, input, output, confidence)

└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Vector Database Schema

```python
# Pinecone/Qdrant Collections

collections = {
    "medical_literature": {
        "dimensions": 768,  # BioBERT embeddings
        "metadata": ["pmid", "title", "journal", "date", "mesh_terms"]
    },
    "drug_information": {
        "dimensions": 768,
        "metadata": ["drug_id", "name", "class", "interactions"]
    },
    "clinical_guidelines": {
        "dimensions": 768,
        "metadata": ["guideline_id", "condition", "source", "version"]
    },
    "regulatory_documents": {
        "dimensions": 768,
        "metadata": ["doc_id", "regulation", "country", "section"]
    },
    "qa_qc_templates": {
        "dimensions": 768,
        "metadata": ["template_id", "type", "regulation", "version"]
    }
}
```

---

## 4. Infrastructure Architecture

### 4.1 Cloud Infrastructure (Hybrid)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        HYBRID CLOUD ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      PUBLIC CLOUD (Azure/AWS)                        │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                      │   │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │   │
│  │   │  Web Apps   │  │  API Layer  │  │  AI Models  │                │   │
│  │   │  (B2C)      │  │  (General)  │  │  (Inference)│                │   │
│  │   └─────────────┘  └─────────────┘  └─────────────┘                │   │
│  │                                                                      │   │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │   │
│  │   │  Vector DB  │  │   Redis     │  │  Blob Store │                │   │
│  │   │  (Pinecone) │  │   Cache     │  │  (Images)   │                │   │
│  │   └─────────────┘  └─────────────┘  └─────────────┘                │   │
│  │                                                                      │   │
│  │   Region: UK South (Primary), UAE North (Secondary)                 │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│                              ▲                                               │
│                              │ VPN / ExpressRoute                           │
│                              ▼                                               │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    ON-PREMISE (Hospital/Pharma)                      │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                      │   │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │   │
│  │   │  Patient    │  │  EHR        │  │  Local AI   │                │   │
│  │   │  Data Store │  │  Integration│  │  Inference  │                │   │
│  │   └─────────────┘  └─────────────┘  └─────────────┘                │   │
│  │                                                                      │   │
│  │   ┌─────────────┐  ┌─────────────┐                                  │   │
│  │   │  Audit      │  │  Backup     │                                  │   │
│  │   │  Logs       │  │  Systems    │                                  │   │
│  │   └─────────────┘  └─────────────┘                                  │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Kubernetes Architecture

```yaml
# Kubernetes Cluster Structure
namespaces:
  - umi-production
  - umi-staging
  - umi-ml
  - umi-monitoring

deployments:
  umi-production:
    - api-gateway (3 replicas, HPA)
    - auth-service (2 replicas)
    - core-api (5 replicas, HPA)
    - consultation-service (3 replicas)
    - pharma-service (2 replicas)
    - async-workers (10 replicas, HPA)
  
  umi-ml:
    - inference-server (GPU nodes, 3 replicas)
    - embedding-service (2 replicas)
    - vision-service (GPU nodes, 2 replicas)
    - rag-service (3 replicas)

services:
  - LoadBalancer: api-gateway
  - ClusterIP: internal services
  - NodePort: monitoring dashboards

storage:
  - PersistentVolumeClaims for databases
  - Azure Files / AWS EFS for shared storage
```

---

## 5. Security Architecture

### 5.1 Security Layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SECURITY ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Layer 1: Network Security                                                  │
│  ├── WAF (Web Application Firewall)                                        │
│  ├── DDoS Protection                                                        │
│  ├── VPN for on-premise connections                                        │
│  └── Network segmentation (VNets/VPCs)                                     │
│                                                                              │
│  Layer 2: Application Security                                              │
│  ├── OAuth 2.0 / OpenID Connect                                            │
│  ├── JWT tokens with short expiry                                          │
│  ├── Role-Based Access Control (RBAC)                                      │
│  ├── API rate limiting                                                      │
│  └── Input validation & sanitization                                       │
│                                                                              │
│  Layer 3: Data Security                                                     │
│  ├── Encryption at rest (AES-256)                                          │
│  ├── Encryption in transit (TLS 1.3)                                       │
│  ├── Field-level encryption for PII                                        │
│  ├── Data masking for non-production                                       │
│  └── Key management (Azure Key Vault / AWS KMS)                            │
│                                                                              │
│  Layer 4: Compliance                                                        │
│  ├── GDPR (UK/EU)                                                          │
│  ├── UAE PDPL                                                               │
│  ├── HIPAA-ready architecture                                              │
│  ├── ISO 27001 controls                                                    │
│  └── SOC 2 Type II audit trail                                             │
│                                                                              │
│  Layer 5: Monitoring & Response                                             │
│  ├── SIEM integration                                                       │
│  ├── Anomaly detection                                                      │
│  ├── Automated incident response                                           │
│  └── 24/7 security monitoring                                              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Data Privacy Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DATA PRIVACY BY DESIGN                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    DATA CLASSIFICATION                               │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                      │   │
│  │   PUBLIC          INTERNAL         CONFIDENTIAL      RESTRICTED     │   │
│  │   ────────        ────────         ────────────      ──────────     │   │
│  │   • Drug info     • Usage stats    • User profiles   • Patient PHI  │   │
│  │   • Guidelines    • Error logs     • Org data        • Medical imgs │   │
│  │   • Research      • Performance    • API keys        • Diagnoses    │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    DATA RESIDENCY                                    │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                      │   │
│  │   UK Users ──────▶ UK South Region (Azure/AWS)                      │   │
│  │   UAE Users ─────▶ UAE North Region (Azure/AWS)                     │   │
│  │                                                                      │   │
│  │   Cross-border transfers: Only with explicit consent + safeguards   │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    CONSENT MANAGEMENT                                │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                      │   │
│  │   • Granular consent for each data use                              │   │
│  │   • Easy withdrawal mechanism                                        │   │
│  │   • Consent audit trail                                              │   │
│  │   • Age verification for minors                                      │   │
│  │   • Parental consent workflow                                        │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. API Architecture

### 6.1 API Design Principles

- **RESTful** for CRUD operations
- **GraphQL** for complex queries (patient profiles, research)
- **WebSocket** for real-time consultations
- **gRPC** for internal service communication

### 6.2 Core API Endpoints

```yaml
# API Structure

/api/v1:
  /auth:
    POST /register
    POST /login
    POST /refresh
    POST /logout
    
  /users:
    GET /me
    PUT /me
    GET /me/profile
    PUT /me/profile
    
  /consultations:
    POST /start                    # Start ASMETHOD consultation
    POST /{id}/message             # Send message in consultation
    GET /{id}                      # Get consultation history
    GET /{id}/summary              # Get AI summary
    POST /{id}/image               # Upload medical image
    
  /diagnoses:
    POST /symptoms                 # Symptom analysis
    POST /image                    # Image analysis
    GET /history                   # User's diagnosis history
    
  /drugs:
    GET /search                    # Search drugs
    GET /{id}                      # Drug details
    POST /interactions             # Check interactions
    GET /alternatives              # Find alternatives
    
  /pharma:
    /facilities:
      GET /                        # List facilities
      POST /                       # Create facility
      GET /{id}                    # Facility details
      
    /documents:
      POST /generate               # Generate QA/QC document
      GET /{id}                    # Get document
      PUT /{id}                    # Update document
      POST /{id}/export            # Export to PDF/Word
      
    /compliance:
      GET /checklist               # Get compliance checklist
      POST /audit                  # Run compliance audit
      GET /reports                 # Compliance reports
      
  /research:
    POST /literature-review        # AI literature review
    POST /paper-assist             # Paper writing assistance
    GET /clinical-trials           # Search clinical trials
    
  /admin:
    /analytics                     # Usage analytics
    /audit-logs                    # Audit trail
    /model-performance             # AI model metrics
```

---

## 7. ML Pipeline Architecture

### 7.1 Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ML TRAINING PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │   Data      │    │   Data      │    │   Feature   │    │   Model     │ │
│  │  Ingestion  │───▶│  Cleaning   │───▶│  Engineering│───▶│  Training   │ │
│  │             │    │             │    │             │    │             │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘ │
│                                                                   │        │
│                                                                   ▼        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │  Model      │    │   A/B       │    │   Model     │    │   Model     │ │
│  │  Serving    │◀───│   Testing   │◀───│  Validation │◀───│  Registry   │ │
│  │             │    │             │    │             │    │             │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘ │
│                                                                              │
│  Tools: MLflow, DVC, Weights & Biases, Kubeflow                            │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Inference Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ML INFERENCE PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Request ──▶ Preprocessing ──▶ Model Router ──▶ Expert Selection          │
│                                                                              │
│                                      │                                       │
│                    ┌─────────────────┼─────────────────┐                    │
│                    │                 │                 │                    │
│                    ▼                 ▼                 ▼                    │
│              ┌──────────┐     ┌──────────┐     ┌──────────┐                │
│              │ Expert 1 │     │ Expert 2 │     │ Expert N │                │
│              │ (GPU)    │     │ (GPU)    │     │ (GPU)    │                │
│              └────┬─────┘     └────┬─────┘     └────┬─────┘                │
│                   │                │                │                       │
│                   └────────────────┼────────────────┘                       │
│                                    │                                        │
│                                    ▼                                        │
│                            ┌──────────────┐                                │
│                            │   Ensemble   │                                │
│                            │   + RAG      │                                │
│                            └──────┬───────┘                                │
│                                   │                                         │
│                                   ▼                                         │
│                            ┌──────────────┐                                │
│                            │   Response   │                                │
│                            │   + Logging  │                                │
│                            └──────────────┘                                │
│                                                                              │
│  Serving: vLLM, TensorRT-LLM, Triton Inference Server                      │
│  Latency Target: < 2 seconds for text, < 5 seconds for images              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Technology Stack Summary

### 8.1 Complete Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Frontend** | React, Next.js, TailwindCSS | Web application |
| **Mobile** | React Native | iOS/Android apps |
| **API Gateway** | Kong / Nginx | Routing, rate limiting |
| **Backend** | FastAPI (Python) | Core API services |
| **Auth** | Keycloak / Auth0 | Identity management |
| **Queue** | Celery + Redis | Async task processing |
| **Events** | Apache Kafka | Event streaming |
| **Primary DB** | PostgreSQL | Relational data |
| **Document DB** | MongoDB | Unstructured documents |
| **Cache** | Redis | Session, caching |
| **Vector DB** | Pinecone / Qdrant | Embeddings, RAG |
| **Search** | Elasticsearch | Full-text search |
| **Object Store** | MinIO / S3 | Images, files |
| **ML Framework** | PyTorch, Transformers | Model development |
| **LLM Serving** | vLLM, TensorRT-LLM | Fast inference |
| **ML Ops** | MLflow, Kubeflow | Pipeline management |
| **Container** | Docker, Kubernetes | Orchestration |
| **CI/CD** | GitHub Actions, ArgoCD | Deployment |
| **Monitoring** | Prometheus, Grafana | Metrics |
| **Logging** | ELK Stack | Log aggregation |
| **Tracing** | Jaeger | Distributed tracing |

---

## 9. Scalability Considerations

### 9.1 Horizontal Scaling

```
Load Balancer
     │
     ├──▶ API Server 1 ──▶ DB Read Replica 1
     ├──▶ API Server 2 ──▶ DB Read Replica 2
     ├──▶ API Server 3 ──▶ DB Read Replica 3
     └──▶ API Server N ──▶ DB Primary (Writes)
```

### 9.2 GPU Scaling for Inference

```
Inference Router
     │
     ├──▶ GPU Node 1 (A100) ──▶ Text Models
     ├──▶ GPU Node 2 (A100) ──▶ Text Models
     ├──▶ GPU Node 3 (A100) ──▶ Vision Models
     └──▶ GPU Node 4 (A100) ──▶ Vision Models
```

### 9.3 Performance Targets

| Metric | Target | Strategy |
|--------|--------|----------|
| API Latency (p99) | < 200ms | Caching, CDN |
| AI Response (text) | < 2s | vLLM, batching |
| AI Response (image) | < 5s | GPU optimization |
| Concurrent Users | 100,000 | Horizontal scaling |
| Uptime | 99.9% | Multi-region, failover |

---

**Next Document**: [05_ROADMAP.md](./05_ROADMAP.md) - Development timeline and milestones
