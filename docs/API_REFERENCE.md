# IMI API Reference

Complete API documentation for the IMI Medical LLM Platform.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

All endpoints require JWT authentication (except health checks).

```bash
Authorization: Bearer <token>
```

---

## Health Endpoints

### GET /health
Basic health check.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-22T14:00:00Z",
  "service": "IMI Medical LLM Platform"
}
```

### GET /health/detailed
Detailed component status.

**Response:**
```json
{
  "status": "healthy",
  "components": {
    "api": "healthy",
    "database": "healthy",
    "knowledge_graph": "healthy",
    "llm": "healthy",
    "cache": "healthy"
  }
}
```

---

## Patient API

### POST /patient/assess-symptoms
Assess patient symptoms and provide triage recommendation.

**Request:**
```json
{
  "symptoms": ["headache", "fever", "fatigue"],
  "chief_complaint": "Severe headache for 2 days",
  "duration_hours": 48,
  "severity": 7,
  "age": 35,
  "gender": "female",
  "is_pregnant": false,
  "is_breastfeeding": false,
  "medical_conditions": ["hypertension"],
  "current_medications": ["lisinopril"],
  "allergies": ["penicillin"]
}
```

**Response:**
```json
{
  "triage_result": {
    "urgency": "semi_urgent",
    "recommendation": "See a doctor within 2-3 days",
    "red_flags_detected": []
  },
  "otc_recommendation": {
    "eligible": true,
    "products": ["acetaminophen", "ibuprofen"]
  },
  "explanation": "Based on your symptoms...",
  "next_steps": ["Schedule appointment", "Monitor symptoms"],
  "disclaimer": "This is not a medical diagnosis..."
}
```

### POST /patient/check-drug-safety
Check if a drug is safe for the patient.

**Request:**
```json
{
  "drug_name": "ibuprofen",
  "conditions": ["hypertension", "kidney disease"],
  "current_medications": ["lisinopril"],
  "allergies": [],
  "age": 65,
  "is_pregnant": false
}
```

**Response:**
```json
{
  "drug": "ibuprofen",
  "is_safe": false,
  "contraindications": ["kidney disease"],
  "interactions": ["lisinopril - may reduce effectiveness"],
  "warnings": ["Use with caution in elderly"],
  "alternatives": ["acetaminophen"]
}
```

---

## Doctor API

### POST /doctor/differential
Generate differential diagnosis.

**Request:**
```json
{
  "chief_complaint": "Chest pain",
  "history_of_present_illness": "45yo male with sudden onset chest pain...",
  "past_medical_history": ["hypertension", "diabetes"],
  "medications": ["metformin", "lisinopril"],
  "vital_signs": {
    "heart_rate": 95,
    "blood_pressure": "150/90",
    "temperature": 98.6
  },
  "patient_age": 45,
  "patient_sex": "male"
}
```

**Response:**
```json
{
  "differentials": [
    {
      "diagnosis": "Acute Coronary Syndrome",
      "probability": "high",
      "supporting_features": ["chest pain", "hypertension", "diabetes"],
      "recommended_workup": ["ECG", "Troponin", "CXR"]
    },
    {
      "diagnosis": "GERD",
      "probability": "moderate",
      "supporting_features": ["chest pain"],
      "recommended_workup": ["Trial PPI"]
    }
  ]
}
```

### POST /doctor/drug-interactions
Check for drug-drug interactions.

**Request:**
```json
{
  "medications": ["warfarin", "aspirin", "ibuprofen"]
}
```

**Response:**
```json
{
  "medications_checked": ["warfarin", "aspirin", "ibuprofen"],
  "interactions_found": 2,
  "interactions": [
    {
      "drugs": ["warfarin", "aspirin"],
      "severity": "major",
      "description": "Increased bleeding risk"
    }
  ],
  "is_safe": false
}
```

---

## Student API

### POST /student/answer-question
Answer a USMLE-style question.

**Request:**
```json
{
  "question": "A 55-year-old man presents with crushing chest pain...",
  "options": ["A. Aspirin", "B. Morphine", "C. Oxygen", "D. Nitroglycerin"],
  "exam_type": "usmle_step2_ck"
}
```

**Response:**
```json
{
  "analysis": {
    "likely_topic": "Cardiology",
    "question_type": "management"
  },
  "explanation": "This question tests your understanding of ACS management...",
  "high_yield_points": [
    "MONA protocol for ACS",
    "Aspirin is first-line antiplatelet"
  ]
}
```

### POST /student/explain-concept
Explain a medical concept.

**Request:**
```json
{
  "concept": "heart failure pathophysiology",
  "depth": "intermediate",
  "include_clinical": true
}
```

---

## Researcher API

### POST /researcher/patent/guidance
Get patent application guidance.

**Request:**
```json
{
  "compound_info": {
    "name": "Novel ACE inhibitor",
    "structure": "..."
  },
  "patent_type": "composition_of_matter"
}
```

### POST /researcher/regulatory-pathway
Get regulatory pathway guidance.

**Request:**
```json
{
  "drug_type": "small_molecule",
  "indication": "hypertension",
  "target_markets": ["US", "EU", "SA"]
}
```

---

## Pharmaceutical API

### POST /pharma/document/generate
Generate QA document.

**Request:**
```json
{
  "document_type": "sop",
  "template_data": {
    "purpose": "Define cleaning validation procedures",
    "scope": "Manufacturing equipment"
  }
}
```

### POST /pharma/compliance/check
Check regulatory compliance.

**Request:**
```json
{
  "entity_id": "uuid",
  "regulatory_body": "fda",
  "check_areas": ["validation", "documentation"]
}
```

---

## Hospital API

### POST /hospital/er/triage
Triage ER patient.

**Request:**
```json
{
  "patient_id": "uuid",
  "chief_complaint": "Chest pain",
  "symptoms": ["chest pain", "shortness of breath"],
  "vital_signs": {
    "heart_rate": 110,
    "bp_systolic": 90,
    "oxygen_saturation": 92
  },
  "age": 65
}
```

**Response:**
```json
{
  "id": "uuid",
  "priority": "emergent",
  "triage_score": 2,
  "arrival_time": "2024-01-22T14:00:00Z",
  "status": "waiting"
}
```

### POST /hospital/appointment
Schedule appointment.

**Request:**
```json
{
  "patient_id": "uuid",
  "provider_id": "uuid",
  "department": "Cardiology",
  "scheduled_time": "2024-01-25T10:00:00Z",
  "duration_minutes": 30,
  "reason": "Follow-up"
}
```

---

## General API

### GET /general/disease/{name}
Get disease information.

**Response:**
```json
{
  "name": "Hypertension",
  "found": true,
  "description": "Persistently elevated blood pressure",
  "symptoms": ["headache", "dizziness"],
  "risk_factors": ["obesity", "high sodium diet"],
  "treatment_options": [
    {"name": "Lisinopril", "line": 1}
  ]
}
```

### GET /general/drug/{name}
Get drug information.

### POST /general/drug-interaction
Check interaction between two drugs.

**Request:**
```json
{
  "drug1": "warfarin",
  "drug2": "aspirin"
}
```

---

## Error Responses

All errors follow this format:

```json
{
  "error": "Error description",
  "status_code": 400
}
```

### Common Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 500 | Internal Server Error |
