"""
UMI Patient Models
Patient profile, medical history, and consultation models
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, JSON, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.core.database import Base


class BloodType(str, Enum):
    """Blood types."""
    A_POSITIVE = "A+"
    A_NEGATIVE = "A-"
    B_POSITIVE = "B+"
    B_NEGATIVE = "B-"
    AB_POSITIVE = "AB+"
    AB_NEGATIVE = "AB-"
    O_POSITIVE = "O+"
    O_NEGATIVE = "O-"
    UNKNOWN = "unknown"


class PatientProfile(Base):
    """Extended patient profile with medical information."""
    
    __tablename__ = "patient_profiles"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    
    # Basic medical info
    blood_type: Mapped[Optional[BloodType]] = mapped_column(SQLEnum(BloodType))
    height_cm: Mapped[Optional[float]] = mapped_column(Float)
    weight_kg: Mapped[Optional[float]] = mapped_column(Float)
    
    # Allergies (stored as JSON array)
    allergies: Mapped[Optional[List[str]]] = mapped_column(JSONB, default=list)
    
    # Chronic conditions (stored as JSON array)
    chronic_conditions: Mapped[Optional[List[str]]] = mapped_column(JSONB, default=list)
    
    # Family history (stored as JSON)
    family_history: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, default=dict)
    
    # Lifestyle factors
    smoking_status: Mapped[Optional[str]] = mapped_column(String(50))  # never, former, current
    alcohol_consumption: Mapped[Optional[str]] = mapped_column(String(50))  # none, occasional, moderate, heavy
    exercise_frequency: Mapped[Optional[str]] = mapped_column(String(50))  # sedentary, light, moderate, active
    
    # Emergency contact
    emergency_contact_name: Mapped[Optional[str]] = mapped_column(String(200))
    emergency_contact_phone: Mapped[Optional[str]] = mapped_column(String(20))
    emergency_contact_relation: Mapped[Optional[str]] = mapped_column(String(50))
    
    # Insurance info (encrypted in production)
    insurance_provider: Mapped[Optional[str]] = mapped_column(String(200))
    insurance_policy_number: Mapped[Optional[str]] = mapped_column(String(100))
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="patient_profile")
    medical_history: Mapped[List["MedicalHistory"]] = relationship(
        "MedicalHistory",
        back_populates="patient",
        cascade="all, delete-orphan",
    )
    medications: Mapped[List["Medication"]] = relationship(
        "Medication",
        back_populates="patient",
        cascade="all, delete-orphan",
    )

    @property
    def bmi(self) -> Optional[float]:
        """Calculate BMI if height and weight are available."""
        if self.height_cm and self.weight_kg:
            height_m = self.height_cm / 100
            return round(self.weight_kg / (height_m ** 2), 1)
        return None

    def __repr__(self) -> str:
        return f"<PatientProfile user_id={self.user_id}>"


class ConditionStatus(str, Enum):
    """Medical condition status."""
    ACTIVE = "active"
    RESOLVED = "resolved"
    CHRONIC = "chronic"
    IN_REMISSION = "in_remission"


class MedicalHistory(Base):
    """Patient medical history entries."""
    
    __tablename__ = "medical_history"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("patient_profiles.id", ondelete="CASCADE"),
        nullable=False,
    )
    
    # Condition details
    condition_name: Mapped[str] = mapped_column(String(255), nullable=False)
    icd_code: Mapped[Optional[str]] = mapped_column(String(20))  # ICD-11 code
    description: Mapped[Optional[str]] = mapped_column(Text)
    
    status: Mapped[ConditionStatus] = mapped_column(
        SQLEnum(ConditionStatus),
        default=ConditionStatus.ACTIVE,
    )
    
    diagnosed_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    resolved_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Treatment info
    treatment_notes: Mapped[Optional[str]] = mapped_column(Text)
    treating_physician: Mapped[Optional[str]] = mapped_column(String(200))
    
    # Source of record
    source: Mapped[Optional[str]] = mapped_column(String(100))  # self-reported, imported, consultation
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # Relationships
    patient: Mapped["PatientProfile"] = relationship("PatientProfile", back_populates="medical_history")

    def __repr__(self) -> str:
        return f"<MedicalHistory {self.condition_name}>"


class MedicationStatus(str, Enum):
    """Medication status."""
    ACTIVE = "active"
    COMPLETED = "completed"
    DISCONTINUED = "discontinued"
    ON_HOLD = "on_hold"


class Medication(Base):
    """Patient medication records."""
    
    __tablename__ = "medications"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("patient_profiles.id", ondelete="CASCADE"),
        nullable=False,
    )
    drug_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("drugs.id"),
        nullable=True,
    )
    
    # Medication details
    drug_name: Mapped[str] = mapped_column(String(255), nullable=False)
    generic_name: Mapped[Optional[str]] = mapped_column(String(255))
    dosage: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g., "500mg"
    frequency: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g., "twice daily"
    route: Mapped[Optional[str]] = mapped_column(String(50))  # oral, topical, injection, etc.
    
    status: Mapped[MedicationStatus] = mapped_column(
        SQLEnum(MedicationStatus),
        default=MedicationStatus.ACTIVE,
    )
    
    # Dates
    start_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    end_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Prescriber info
    prescribed_by: Mapped[Optional[str]] = mapped_column(String(200))
    prescription_number: Mapped[Optional[str]] = mapped_column(String(100))
    
    # Notes
    reason: Mapped[Optional[str]] = mapped_column(Text)  # Why prescribed
    notes: Mapped[Optional[str]] = mapped_column(Text)
    
    # Adherence tracking
    adherence_score: Mapped[Optional[float]] = mapped_column(Float)  # 0-100%
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # Relationships
    patient: Mapped["PatientProfile"] = relationship("PatientProfile", back_populates="medications")
    drug: Mapped[Optional["Drug"]] = relationship("Drug")

    def __repr__(self) -> str:
        return f"<Medication {self.drug_name} {self.dosage}>"


class ConsultationStatus(str, Enum):
    """Consultation status."""
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REFERRED = "referred"
    CANCELLED = "cancelled"


class ConsultationOutcome(str, Enum):
    """Consultation outcome type."""
    OTC_RECOMMENDATION = "otc_recommendation"
    LIFESTYLE_ADVICE = "lifestyle_advice"
    DOCTOR_REFERRAL = "doctor_referral"
    EMERGENCY_REFERRAL = "emergency_referral"
    INFORMATION_ONLY = "information_only"
    FOLLOW_UP_NEEDED = "follow_up_needed"


class Consultation(Base):
    """AI consultation session with ASMETHOD protocol."""
    
    __tablename__ = "consultations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    
    # Session info
    status: Mapped[ConsultationStatus] = mapped_column(
        SQLEnum(ConsultationStatus),
        default=ConsultationStatus.IN_PROGRESS,
    )
    
    # ASMETHOD data (stored as JSON)
    asmethod_data: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )
    # Structure:
    # {
    #     "age": int,
    #     "self_or_other": "self" | "other",
    #     "medications": [...],
    #     "exact_symptoms": "...",
    #     "time_duration": "...",
    #     "history": "...",
    #     "other_symptoms": "...",
    #     "danger_signs": [...]
    # }
    
    # Conversation history
    messages: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )
    # Structure: [{"role": "user"|"assistant", "content": "...", "timestamp": "..."}]
    
    # AI Analysis
    symptoms_analysis: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    differential_diagnosis: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSONB)
    
    # Outcome
    outcome: Mapped[Optional[ConsultationOutcome]] = mapped_column(SQLEnum(ConsultationOutcome))
    recommendation: Mapped[Optional[str]] = mapped_column(Text)
    recommended_drugs: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSONB)
    referral_specialty: Mapped[Optional[str]] = mapped_column(String(100))
    referral_urgency: Mapped[Optional[str]] = mapped_column(String(50))  # routine, urgent, emergency
    
    # Safety flags
    danger_signs_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    requires_physician_review: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Feedback
    user_rating: Mapped[Optional[int]] = mapped_column(Integer)  # 1-5
    user_feedback: Mapped[Optional[str]] = mapped_column(Text)
    
    # AI metadata
    model_version: Mapped[Optional[str]] = mapped_column(String(100))
    confidence_score: Mapped[Optional[float]] = mapped_column(Float)
    
    # Timestamps
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="consultations")
    images: Mapped[List["ConsultationImage"]] = relationship(
        "ConsultationImage",
        back_populates="consultation",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Consultation {self.id} status={self.status}>"


class ConsultationImage(Base):
    """Medical images uploaded during consultation."""
    
    __tablename__ = "consultation_images"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    consultation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("consultations.id", ondelete="CASCADE"),
        nullable=False,
    )
    
    # File info
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    file_type: Mapped[str] = mapped_column(String(50), nullable=False)  # xray, ct, mri, lab_report, photo
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # AI Analysis
    analysis_status: Mapped[str] = mapped_column(String(50), default="pending")  # pending, processing, completed, failed
    analysis_result: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    analysis_confidence: Mapped[Optional[float]] = mapped_column(Float)
    
    # Metadata
    dicom_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)  # For DICOM files
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )

    # Relationships
    consultation: Mapped["Consultation"] = relationship("Consultation", back_populates="images")

    def __repr__(self) -> str:
        return f"<ConsultationImage {self.filename}>"


# Import for type hints
from src.models.user import User
from src.models.medical import Drug
