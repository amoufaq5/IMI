# IMI Platform - Reasoning, Governance & Auditing

## Overview

IMI implements a **Safety-First AI Architecture** where no single component makes critical decisions alone. Every medical response passes through multiple verification layers with complete audit trails.

---

## 1. Decision Making Flow

### Complete Query Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER QUERY                                        │
│                "I have chest pain and shortness of breath"                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: AUTHENTICATION & AUTHORIZATION                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  • JWT token validation                                                     │
│  • User role verification (patient/doctor/pharmacist)                       │
│  • Rate limiting check                                                      │
│  • AUDIT: Log access attempt with user_id, timestamp, IP                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: ORCHESTRATOR RECEIVES QUERY                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Assigns unique query_id (UUID)                                           │
│  • Determines domain (patient_triage, clinical, pharmacy, etc.)             │
│  • Selects appropriate LoRA adapter                                         │
│  • AUDIT: Log query receipt with query_id, domain, adapter                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: LAYER 1 - KNOWLEDGE GRAPH LOOKUP                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  Query: "chest pain", "shortness of breath"                                 │
│                                                                             │
│  Returns:                                                                   │
│  • Related conditions: MI, angina, PE, pneumonia, anxiety                   │
│  • Risk factors: age, smoking, diabetes, hypertension                       │
│  • Red flags: radiation to arm, diaphoresis, syncope                        │
│  • Urgency indicators: HIGH for cardiac symptoms                            │
│                                                                             │
│  AUDIT: Log KG queries, nodes accessed, relationships traversed             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: LAYER 2 - RULE ENGINE EVALUATION                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  Deterministic Rules Applied:                                               │
│                                                                             │
│  RULE: chest_pain_triage                                                    │
│  IF symptoms CONTAIN ["chest pain", "shortness of breath"]                  │
│  AND age > 40                                                               │
│  THEN urgency = "EMERGENCY"                                                 │
│  ACTION: recommend_immediate_care = TRUE                                    │
│                                                                             │
│  RULE: cardiac_risk_assessment                                              │
│  IF symptoms MATCH cardiac_pattern                                          │
│  THEN flag_for_verification = TRUE                                          │
│  THEN require_disclaimer = TRUE                                             │
│                                                                             │
│  Output:                                                                    │
│  • urgency_level: EMERGENCY                                                 │
│  • safety_flags: [cardiac_risk, requires_immediate_attention]               │
│  • required_disclaimers: [seek_emergency_care, call_911]                    │
│  • llm_constraints: [must_recommend_emergency, no_home_remedies]            │
│                                                                             │
│  AUDIT: Log all rules evaluated, conditions matched, actions triggered      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: LAYER 5 - MEMORY/PROFILE LOOKUP                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  Patient Profile (encrypted, decrypted for this query):                     │
│                                                                             │
│  • Age: 52                                                                  │
│  • Medical history: hypertension, type 2 diabetes                           │
│  • Medications: metformin, lisinopril                                       │
│  • Allergies: penicillin                                                    │
│  • Previous queries: back pain (2 weeks ago)                                │
│                                                                             │
│  Context Enhancement:                                                       │
│  • Risk multiplier: HIGH (age + comorbidities)                              │
│  • Relevant history: cardiovascular risk factors present                    │
│                                                                             │
│  AUDIT: Log profile access (not content), fields accessed, purpose          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 6: LAYER 3 - LLM GENERATION                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  Adapter: patient_triage                                                    │
│                                                                             │
│  Prompt Construction:                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ SYSTEM: You are a medical triage assistant.                         │   │
│  │                                                                     │   │
│  │ CONTEXT FROM KNOWLEDGE GRAPH:                                       │   │
│  │ - Chest pain + SOB associated with: MI, PE, angina                  │   │
│  │ - Red flags: arm radiation, diaphoresis, syncope                    │   │
│  │                                                                     │   │
│  │ PATIENT CONTEXT:                                                    │   │
│  │ - 52yo with hypertension, diabetes                                  │   │
│  │ - Current medications: metformin, lisinopril                        │   │
│  │                                                                     │   │
│  │ CONSTRAINTS (from Rule Engine):                                     │   │
│  │ - MUST recommend emergency evaluation                               │   │
│  │ - MUST NOT suggest home remedies for these symptoms                 │   │
│  │ - MUST include emergency disclaimer                                 │   │
│  │                                                                     │   │
│  │ USER QUERY: I have chest pain and shortness of breath               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  LLM Output:                                                                │
│  "Based on your symptoms of chest pain and shortness of breath,            │
│   combined with your history of hypertension and diabetes, this            │
│   requires IMMEDIATE medical evaluation. Please call 911 or go to          │
│   the nearest emergency room right away. These symptoms could              │
│   indicate a serious cardiac event..."                                     │
│                                                                             │
│  AUDIT: Log prompt hash, adapter used, generation params, token count      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 7: LAYER 4 - VERIFICATION                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Verification Checks:                                                       │
│                                                                             │
│  ✓ HALLUCINATION CHECK                                                      │
│    • Extract medical entities from response                                 │
│    • Verify each entity exists in Knowledge Graph                           │
│    • Check relationships are accurate                                       │
│    • Result: PASSED (all entities verified)                                 │
│                                                                             │
│  ✓ GUIDELINE COMPLIANCE                                                     │
│    • Check response against clinical guidelines                             │
│    • Verify emergency recommendation present (REQUIRED by Rule Engine)      │
│    • Verify no contraindicated advice given                                 │
│    • Result: PASSED (emergency care recommended)                            │
│                                                                             │
│  ✓ SAFETY CHECK                                                             │
│    • Verify required disclaimers present                                    │
│    • Check for dangerous recommendations                                    │
│    • Verify no medication advice conflicts with patient allergies           │
│    • Result: PASSED                                                         │
│                                                                             │
│  ✓ CONSTRAINT COMPLIANCE                                                    │
│    • Verify LLM followed Rule Engine constraints                            │
│    • Check: emergency recommended? YES                                      │
│    • Check: home remedies suggested? NO                                     │
│    • Result: PASSED                                                         │
│                                                                             │
│  Verification Score: 100% (4/4 checks passed)                               │
│                                                                             │
│  AUDIT: Log each check, result, confidence score, any modifications         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 8: RESPONSE ASSEMBLY                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  Final Response:                                                            │
│  {                                                                          │
│    "response": "Based on your symptoms...",                                 │
│    "urgency": "EMERGENCY",                                                  │
│    "confidence": 0.95,                                                      │
│    "sources": ["AHA Guidelines 2024", "ACC Chest Pain Protocol"],           │
│    "disclaimers": ["This is not a diagnosis. Seek immediate care."],        │
│    "recommended_actions": ["Call 911", "Go to ER immediately"],             │
│    "reasoning_trace": {                                                     │
│      "knowledge_graph": ["MI", "angina", "PE identified as possibilities"], │
│      "rules_applied": ["chest_pain_triage", "cardiac_risk_assessment"],     │
│      "patient_factors": ["age 52", "hypertension", "diabetes"],             │
│      "verification": "4/4 checks passed"                                    │
│    }                                                                        │
│  }                                                                          │
│                                                                             │
│  AUDIT: Log complete response, reasoning trace, final confidence            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 9: DELIVERY & LOGGING                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Response delivered to user                                               │
│  • Complete audit record written to database                                │
│  • Metrics updated (latency, token usage, verification scores)              │
│  • Memory updated with this interaction (encrypted)                         │
│                                                                             │
│  AUDIT: Final log with response_id, total_latency, all layer timings        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Governance Model

### 2.1 Decision Authority Matrix

| Decision Type | Authority | Override Possible? |
|---------------|-----------|-------------------|
| Emergency triage level | Rule Engine (deterministic) | NO |
| Drug contraindications | Rule Engine + Knowledge Graph | NO |
| Drug interactions | Knowledge Graph (verified data) | NO |
| Response content | LLM (constrained by rules) | YES (by Verifier) |
| Disclaimers | Rule Engine (mandatory) | NO |
| Data access | RBAC + Audit | NO |

### 2.2 Safety Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                    SAFETY HIERARCHY                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LEVEL 1: ABSOLUTE RULES (Never Overridden)                     │
│  ├── Emergency symptoms → Always recommend immediate care       │
│  ├── Known allergies → Never recommend allergen                 │
│  ├── Dangerous interactions → Always warn                       │
│  └── Suicide/self-harm → Crisis protocol activated              │
│                                                                 │
│  LEVEL 2: CLINICAL GUIDELINES (Evidence-Based)                  │
│  ├── Treatment protocols from AHA, ACC, etc.                    │
│  ├── Dosing guidelines from FDA                                 │
│  └── Diagnostic criteria from DSM, ICD                          │
│                                                                 │
│  LEVEL 3: LLM JUDGMENT (Verified)                               │
│  ├── Explanation generation                                     │
│  ├── Patient education content                                  │
│  └── Conversational responses                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 When LLM is Overridden

The Verifier can modify or reject LLM output when:

```python
OVERRIDE_CONDITIONS = {
    "hallucination_detected": {
        "action": "remove_or_correct",
        "example": "LLM mentions drug that doesn't exist → removed"
    },
    "safety_violation": {
        "action": "reject_and_regenerate", 
        "example": "LLM suggests home remedy for chest pain → regenerate with stricter constraints"
    },
    "missing_disclaimer": {
        "action": "append_disclaimer",
        "example": "Emergency case without 911 recommendation → add it"
    },
    "contraindication_missed": {
        "action": "add_warning",
        "example": "Drug suggested conflicts with patient allergy → add warning"
    },
    "confidence_too_low": {
        "action": "add_uncertainty_language",
        "example": "Confidence < 0.7 → add 'consult your doctor' language"
    }
}
```

---

## 3. Reasoning Transparency

### 3.1 Explainable AI Output

Every response includes a reasoning trace:

```json
{
  "response": "Based on your symptoms...",
  "reasoning": {
    "step_1_knowledge_graph": {
      "entities_identified": ["chest pain", "shortness of breath"],
      "conditions_matched": ["myocardial infarction", "pulmonary embolism", "angina"],
      "risk_factors_found": ["hypertension", "diabetes", "age > 50"],
      "confidence": 0.92
    },
    "step_2_rules_applied": [
      {
        "rule_id": "TRIAGE-001",
        "rule_name": "chest_pain_emergency",
        "condition": "chest_pain AND (age > 40 OR cardiac_risk_factors)",
        "result": "EMERGENCY",
        "action": "recommend_immediate_care"
      },
      {
        "rule_id": "SAFETY-012",
        "rule_name": "cardiac_disclaimer",
        "condition": "urgency == EMERGENCY",
        "result": "TRIGGERED",
        "action": "append_911_recommendation"
      }
    ],
    "step_3_patient_context": {
      "relevant_history": ["hypertension", "diabetes"],
      "medications_checked": ["metformin", "lisinopril"],
      "allergies_verified": ["penicillin - not relevant"],
      "risk_multiplier": "HIGH"
    },
    "step_4_llm_generation": {
      "adapter_used": "patient_triage",
      "prompt_tokens": 512,
      "response_tokens": 156,
      "temperature": 0.7
    },
    "step_5_verification": {
      "hallucination_check": "PASSED",
      "guideline_compliance": "PASSED",
      "safety_check": "PASSED",
      "constraint_compliance": "PASSED",
      "overall_score": 1.0
    }
  },
  "sources": [
    {"name": "AHA Chest Pain Guidelines 2024", "section": "Initial Assessment"},
    {"name": "ACC/AHA STEMI Protocol", "relevance": 0.89}
  ]
}
```

### 3.2 Source Attribution

All factual claims are attributed:

```
Response: "Chest pain with shortness of breath in patients over 40 
          with cardiovascular risk factors requires immediate evaluation."

Sources:
├── Knowledge Graph: node_id=COND_MI_001, relationship=PRESENTS_WITH
├── Guideline: AHA 2024 Chest Pain Guidelines, Section 3.2
└── Rule: TRIAGE-001 (deterministic, not LLM-generated)
```

---

## 4. Audit System

### 4.1 What Gets Logged

```python
AUDIT_EVENTS = {
    # Access Events
    "USER_LOGIN": ["user_id", "timestamp", "ip_address", "success"],
    "USER_LOGOUT": ["user_id", "timestamp", "session_duration"],
    "API_REQUEST": ["user_id", "endpoint", "method", "timestamp"],
    
    # Query Events
    "QUERY_RECEIVED": ["query_id", "user_id", "domain", "timestamp"],
    "QUERY_COMPLETED": ["query_id", "latency_ms", "token_count"],
    
    # Layer Events
    "KG_LOOKUP": ["query_id", "nodes_accessed", "relationships", "latency_ms"],
    "RULE_EVALUATION": ["query_id", "rules_checked", "rules_triggered", "actions"],
    "LLM_GENERATION": ["query_id", "adapter", "prompt_hash", "tokens", "latency_ms"],
    "VERIFICATION": ["query_id", "checks_run", "checks_passed", "modifications"],
    
    # Data Access Events
    "PHI_ACCESS": ["user_id", "patient_id", "fields_accessed", "purpose", "timestamp"],
    "PHI_MODIFICATION": ["user_id", "patient_id", "fields_modified", "old_hash", "new_hash"],
    
    # Safety Events
    "SAFETY_OVERRIDE": ["query_id", "override_type", "original", "modified", "reason"],
    "EMERGENCY_FLAGGED": ["query_id", "user_id", "symptoms", "urgency_level"],
    "CONTRAINDICATION_BLOCKED": ["query_id", "drug", "condition", "severity"],
    
    # System Events
    "MODEL_LOADED": ["model_name", "adapter", "timestamp"],
    "ADAPTER_SWITCHED": ["from_adapter", "to_adapter", "reason"],
    "ERROR_OCCURRED": ["error_type", "stack_trace", "query_id"]
}
```

### 4.2 Audit Log Schema

```sql
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    user_id UUID REFERENCES users(id),
    query_id UUID,
    
    -- Event Details (JSONB for flexibility)
    event_data JSONB NOT NULL,
    
    -- Security
    ip_address INET,
    user_agent TEXT,
    
    -- Integrity
    previous_hash VARCHAR(64),  -- Chain for tamper detection
    current_hash VARCHAR(64) NOT NULL,
    
    -- Indexing
    INDEX idx_timestamp (timestamp),
    INDEX idx_user_id (user_id),
    INDEX idx_query_id (query_id),
    INDEX idx_event_type (event_type)
);
```

### 4.3 Audit Trail Example

```
Query ID: 550e8400-e29b-41d4-a716-446655440000
User: patient_12345
Timestamp: 2026-01-24T14:30:00Z

AUDIT TRAIL:
─────────────────────────────────────────────────────────────────
14:30:00.000 | API_REQUEST      | POST /api/v1/patient/query
14:30:00.005 | AUTH_VALIDATED   | user_id=patient_12345, role=patient
14:30:00.010 | QUERY_RECEIVED   | domain=patient_triage, adapter=patient_triage
14:30:00.015 | KG_LOOKUP        | nodes=["chest_pain", "dyspnea"], relations=12
14:30:00.045 | RULE_EVALUATION  | rules_checked=8, triggered=2 [TRIAGE-001, SAFETY-012]
14:30:00.050 | PHI_ACCESS       | patient_id=12345, fields=["age","conditions","medications"]
14:30:00.055 | MEMORY_LOOKUP    | history_items=3, relevant=1
14:30:00.060 | LLM_GENERATION   | adapter=patient_triage, tokens=668, latency=1200ms
14:30:01.260 | VERIFICATION     | checks=4, passed=4, modifications=0
14:30:01.265 | RESPONSE_SENT    | urgency=EMERGENCY, confidence=0.95
14:30:01.270 | QUERY_COMPLETED  | total_latency=1270ms, success=true
─────────────────────────────────────────────────────────────────
```

### 4.4 Compliance Reports

```python
# Generate HIPAA compliance report
def generate_compliance_report(start_date, end_date):
    return {
        "period": f"{start_date} to {end_date}",
        "phi_access_summary": {
            "total_accesses": 15420,
            "unique_patients": 3201,
            "by_role": {
                "doctor": 8500,
                "nurse": 4200,
                "patient_self": 2720
            }
        },
        "security_events": {
            "failed_logins": 23,
            "unauthorized_access_attempts": 0,
            "data_exports": 5
        },
        "safety_events": {
            "emergency_flags": 142,
            "contraindications_blocked": 89,
            "llm_overrides": 34
        },
        "audit_integrity": {
            "total_records": 245000,
            "hash_chain_valid": True,
            "gaps_detected": 0
        }
    }
```

---

## 5. Governance Policies

### 5.1 Data Retention

| Data Type | Retention Period | Reason |
|-----------|------------------|--------|
| Audit logs | 7 years | HIPAA requirement |
| PHI | Per patient consent | Legal requirement |
| Query logs | 3 years | Quality improvement |
| Model outputs | 1 year | Model evaluation |
| Error logs | 90 days | Debugging |

### 5.2 Access Control (RBAC)

```python
ROLES = {
    "patient": {
        "can_access": ["own_profile", "own_history", "patient_triage"],
        "cannot_access": ["other_patients", "clinical_tools", "admin"]
    },
    "doctor": {
        "can_access": ["assigned_patients", "clinical_decision", "prescribing"],
        "cannot_access": ["unassigned_patients", "admin", "billing"]
    },
    "pharmacist": {
        "can_access": ["medication_queries", "drug_interactions", "patient_allergies"],
        "cannot_access": ["full_medical_history", "diagnosis", "admin"]
    },
    "admin": {
        "can_access": ["user_management", "audit_logs", "system_config"],
        "cannot_access": ["phi_content"]  # Can see access logs, not actual PHI
    }
}
```

### 5.3 Model Governance

```python
MODEL_GOVERNANCE = {
    "deployment_approval": {
        "required_checks": [
            "accuracy_threshold_met",      # >95% on test set
            "safety_evaluation_passed",    # No dangerous outputs in red team
            "bias_audit_completed",        # Demographic parity checked
            "clinical_review_signed"       # CMO approval
        ]
    },
    "monitoring": {
        "hallucination_rate": {"threshold": 0.01, "alert": "immediate"},
        "safety_override_rate": {"threshold": 0.05, "alert": "daily"},
        "user_satisfaction": {"threshold": 4.0, "alert": "weekly"}
    },
    "rollback_triggers": [
        "hallucination_rate > 0.02",
        "safety_incident_reported",
        "regulatory_concern_raised"
    ]
}
```

---

## 6. Summary: Why This Architecture is Safe

### The LLM Never Decides Alone

```
┌─────────────────────────────────────────────────────────────────┐
│                    DECISION AUTHORITY                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  "Is this an emergency?"                                        │
│  └── RULE ENGINE decides (deterministic, not LLM)               │
│                                                                 │
│  "Is this drug safe for this patient?"                          │
│  └── KNOWLEDGE GRAPH + RULE ENGINE decide (verified data)       │
│                                                                 │
│  "What should I tell the patient?"                              │
│  └── LLM generates, VERIFIER checks, RULES constrain            │
│                                                                 │
│  "Should I recommend emergency care?"                           │
│  └── RULE ENGINE mandates, LLM cannot override                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Complete Accountability

Every decision can be traced:
1. **Who** made the query (authenticated user)
2. **What** data was accessed (audit log)
3. **How** the decision was made (reasoning trace)
4. **Why** this response was given (rules + sources)
5. **When** it happened (timestamp)
6. **What checks** were performed (verification log)

### Fail-Safe Defaults

When uncertain, the system:
- Recommends professional consultation
- Adds uncertainty language
- Increases disclaimer prominence
- Logs for human review
- Never guesses on safety-critical decisions
