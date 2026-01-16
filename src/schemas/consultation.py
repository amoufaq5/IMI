"""
UMI Consultation Schemas
Pydantic models for ASMETHOD consultation API
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from src.models.patient import ConsultationStatus, ConsultationOutcome


class ASMETHODData(BaseModel):
    """ASMETHOD protocol data structure."""
    
    # A - Age
    age: Optional[int] = Field(None, ge=0, le=150, description="Patient age in years")
    
    # S - Self or Someone Else
    self_or_other: Optional[str] = Field(
        None,
        pattern="^(self|other)$",
        description="Is the consultation for self or someone else?"
    )
    patient_relation: Optional[str] = Field(
        None,
        description="If other, relationship to patient (e.g., parent, spouse)"
    )
    
    # M - Medications
    current_medications: Optional[List[str]] = Field(
        default_factory=list,
        description="List of current medications"
    )
    allergies: Optional[List[str]] = Field(
        default_factory=list,
        description="Known allergies"
    )
    
    # E - Exact Symptoms
    exact_symptoms: Optional[str] = Field(
        None,
        max_length=2000,
        description="Detailed description of symptoms"
    )
    symptom_location: Optional[str] = Field(None, description="Where is the symptom located?")
    symptom_severity: Optional[int] = Field(
        None,
        ge=1,
        le=10,
        description="Severity on scale of 1-10"
    )
    
    # T - Time/Duration
    symptom_duration: Optional[str] = Field(
        None,
        description="How long have symptoms been present?"
    )
    symptom_onset: Optional[str] = Field(
        None,
        description="When did symptoms start? (sudden/gradual)"
    )
    symptom_pattern: Optional[str] = Field(
        None,
        description="Pattern of symptoms (constant/intermittent/worsening)"
    )
    
    # H - History
    medical_history: Optional[str] = Field(
        None,
        max_length=2000,
        description="Relevant medical history"
    )
    previous_episodes: Optional[bool] = Field(
        None,
        description="Has this happened before?"
    )
    previous_treatment: Optional[str] = Field(
        None,
        description="What was tried before?"
    )
    
    # O - Other Symptoms
    other_symptoms: Optional[List[str]] = Field(
        default_factory=list,
        description="Associated symptoms"
    )
    
    # D - Danger Signs
    danger_signs: Optional[List[str]] = Field(
        default_factory=list,
        description="Red flag symptoms detected"
    )
    
    # Additional context
    lifestyle_factors: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Relevant lifestyle factors"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "self_or_other": "self",
                "current_medications": ["Metformin 500mg", "Lisinopril 10mg"],
                "allergies": ["Penicillin"],
                "exact_symptoms": "Persistent headache on the right side of my head",
                "symptom_location": "Right temple",
                "symptom_severity": 6,
                "symptom_duration": "3 days",
                "symptom_onset": "gradual",
                "symptom_pattern": "worsening",
                "medical_history": "Type 2 diabetes, hypertension",
                "previous_episodes": True,
                "previous_treatment": "Paracetamol - partial relief",
                "other_symptoms": ["Nausea", "Light sensitivity"],
                "danger_signs": [],
            }
        }


class ConsultationMessage(BaseModel):
    """Single message in consultation conversation."""
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., max_length=10000)
    timestamp: Optional[datetime] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "I have a headache that started 3 days ago",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class ConsultationCreate(BaseModel):
    """Schema for starting a new consultation."""
    initial_message: Optional[str] = Field(
        None,
        max_length=2000,
        description="Initial symptom description"
    )
    language: str = Field(default="en", description="Preferred language")


class ConsultationMessageRequest(BaseModel):
    """Schema for sending a message in consultation."""
    message: str = Field(..., max_length=5000, description="User message")
    
    # Optional structured data updates
    asmethod_update: Optional[Dict[str, Any]] = Field(
        None,
        description="Partial ASMETHOD data update"
    )


class DiagnosisItem(BaseModel):
    """Single diagnosis in differential."""
    condition: str
    probability: float = Field(..., ge=0, le=1)
    icd_code: Optional[str] = None
    reasoning: Optional[str] = None


class DrugRecommendation(BaseModel):
    """OTC drug recommendation."""
    drug_name: str
    generic_name: Optional[str] = None
    dosage: str
    frequency: str
    duration: str
    warnings: Optional[List[str]] = None
    contraindications: Optional[List[str]] = None


class ConsultationResponse(BaseModel):
    """Schema for consultation response."""
    id: UUID
    status: ConsultationStatus
    
    # ASMETHOD data
    asmethod_data: ASMETHODData
    
    # Conversation
    messages: List[ConsultationMessage]
    
    # Analysis (if completed)
    symptoms_analysis: Optional[Dict[str, Any]] = None
    differential_diagnosis: Optional[List[DiagnosisItem]] = None
    
    # Outcome
    outcome: Optional[ConsultationOutcome] = None
    recommendation: Optional[str] = None
    recommended_drugs: Optional[List[DrugRecommendation]] = None
    referral_specialty: Optional[str] = None
    referral_urgency: Optional[str] = None
    
    # Safety
    danger_signs_detected: bool = False
    requires_physician_review: bool = False
    
    # Metadata
    confidence_score: Optional[float] = None
    model_version: Optional[str] = None
    
    # Timestamps
    started_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ConsultationSummary(BaseModel):
    """Brief consultation summary for lists."""
    id: UUID
    status: ConsultationStatus
    outcome: Optional[ConsultationOutcome] = None
    primary_symptom: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ConsultationListResponse(BaseModel):
    """Paginated consultation list."""
    items: List[ConsultationSummary]
    total: int
    page: int
    page_size: int
    pages: int


class ConsultationFeedback(BaseModel):
    """User feedback on consultation."""
    rating: int = Field(..., ge=1, le=5, description="Rating 1-5")
    feedback: Optional[str] = Field(None, max_length=1000)
    was_helpful: Optional[bool] = None
    followed_recommendation: Optional[bool] = None


class ImageUploadResponse(BaseModel):
    """Response after uploading medical image."""
    id: UUID
    filename: str
    file_type: str
    analysis_status: str
    analysis_result: Optional[Dict[str, Any]] = None


class SymptomCheckRequest(BaseModel):
    """Quick symptom check request (non-consultation)."""
    symptoms: List[str] = Field(..., min_length=1, max_length=10)
    age: Optional[int] = Field(None, ge=0, le=150)
    gender: Optional[str] = None


class SymptomCheckResponse(BaseModel):
    """Quick symptom check response."""
    possible_conditions: List[DiagnosisItem]
    urgency_level: str = Field(..., pattern="^(low|medium|high|emergency)$")
    recommendation: str
    should_see_doctor: bool
    disclaimer: str
