"""
UMI Pharmaceutical Company Models
Facility, QA/QC Documentation, Compliance, and Production models
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.core.database import Base


class FacilityType(str, Enum):
    """Pharmaceutical facility types."""
    MANUFACTURING = "manufacturing"
    PACKAGING = "packaging"
    LABORATORY = "laboratory"
    WAREHOUSE = "warehouse"
    RESEARCH = "research"
    HEADQUARTERS = "headquarters"


class FacilityStatus(str, Enum):
    """Facility operational status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    UNDER_INSPECTION = "under_inspection"
    SUSPENDED = "suspended"


class Facility(Base):
    """Pharmaceutical manufacturing/production facility."""
    
    __tablename__ = "facilities"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
    )
    
    # Basic info
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    facility_code: Mapped[Optional[str]] = mapped_column(String(50), unique=True)
    type: Mapped[FacilityType] = mapped_column(SQLEnum(FacilityType), nullable=False)
    status: Mapped[FacilityStatus] = mapped_column(
        SQLEnum(FacilityStatus),
        default=FacilityStatus.ACTIVE,
    )
    
    # Location
    address: Mapped[str] = mapped_column(Text, nullable=False)
    city: Mapped[str] = mapped_column(String(100), nullable=False)
    country: Mapped[str] = mapped_column(String(100), nullable=False)
    postal_code: Mapped[Optional[str]] = mapped_column(String(20))
    
    # Contact
    phone: Mapped[Optional[str]] = mapped_column(String(20))
    email: Mapped[Optional[str]] = mapped_column(String(255))
    facility_manager: Mapped[Optional[str]] = mapped_column(String(200))
    
    # Regulatory
    license_number: Mapped[Optional[str]] = mapped_column(String(100))
    license_expiry: Mapped[Optional[datetime]] = mapped_column(DateTime)
    regulatory_body: Mapped[Optional[str]] = mapped_column(String(100))  # MHRA, UAE MOH, FDA
    
    # Certifications (stored as JSON)
    certifications: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSONB, default=list)
    # Structure: [{"name": "GMP", "issued_by": "MHRA", "valid_until": "2025-12-31", "certificate_number": "..."}]
    
    # Facility specifications
    total_area_sqm: Mapped[Optional[float]] = mapped_column(Float)
    production_capacity: Mapped[Optional[str]] = mapped_column(String(200))
    cleanroom_classes: Mapped[Optional[List[str]]] = mapped_column(JSONB, default=list)  # ISO 5, ISO 7, etc.
    
    # Systems info
    hvac_system: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    water_system: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    
    # Products manufactured
    product_types: Mapped[Optional[List[str]]] = mapped_column(JSONB, default=list)
    
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
    organization: Mapped["Organization"] = relationship("Organization", back_populates="facilities")
    documents: Mapped[List["Document"]] = relationship(
        "Document",
        back_populates="facility",
        cascade="all, delete-orphan",
    )
    compliance_checks: Mapped[List["ComplianceCheck"]] = relationship(
        "ComplianceCheck",
        back_populates="facility",
        cascade="all, delete-orphan",
    )
    production_batches: Mapped[List["ProductionBatch"]] = relationship(
        "ProductionBatch",
        back_populates="facility",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Facility {self.name}>"


class DocumentType(str, Enum):
    """QA/QC document types."""
    # Validation Documents
    CLEANING_VALIDATION = "cleaning_validation"
    PROCESS_VALIDATION = "process_validation"
    EQUIPMENT_VALIDATION = "equipment_validation"
    METHOD_VALIDATION = "method_validation"
    COMPUTER_SYSTEM_VALIDATION = "computer_system_validation"
    
    # HVAC & Utilities
    HVAC_QUALIFICATION = "hvac_qualification"
    WATER_SYSTEM_VALIDATION = "water_system_validation"
    COMPRESSED_AIR_VALIDATION = "compressed_air_validation"
    
    # Manufacturing
    BATCH_RECORD = "batch_record"
    MANUFACTURING_LOGBOOK = "manufacturing_logbook"
    EQUIPMENT_LOGBOOK = "equipment_logbook"
    
    # Quality Control
    SPECIFICATION = "specification"
    TEST_METHOD = "test_method"
    CERTIFICATE_OF_ANALYSIS = "certificate_of_analysis"
    STABILITY_STUDY = "stability_study"
    
    # Quality Assurance
    SOP = "sop"  # Standard Operating Procedure
    DEVIATION_REPORT = "deviation_report"
    CAPA = "capa"  # Corrective and Preventive Action
    CHANGE_CONTROL = "change_control"
    AUDIT_REPORT = "audit_report"
    
    # Regulatory
    TECHNICAL_DOSSIER = "technical_dossier"
    ANNUAL_PRODUCT_REVIEW = "annual_product_review"
    REGULATORY_SUBMISSION = "regulatory_submission"


class DocumentStatus(str, Enum):
    """Document lifecycle status."""
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    EFFECTIVE = "effective"
    SUPERSEDED = "superseded"
    OBSOLETE = "obsolete"


class Document(Base):
    """QA/QC documentation for pharmaceutical facilities."""
    
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    facility_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("facilities.id", ondelete="CASCADE"),
        nullable=False,
    )
    
    # Document identification
    document_number: Mapped[str] = mapped_column(String(100), nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    type: Mapped[DocumentType] = mapped_column(SQLEnum(DocumentType), nullable=False)
    
    # Version control
    version: Mapped[str] = mapped_column(String(20), nullable=False, default="1.0")
    status: Mapped[DocumentStatus] = mapped_column(
        SQLEnum(DocumentStatus),
        default=DocumentStatus.DRAFT,
    )
    
    # Content
    content: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)  # Structured content
    content_text: Mapped[Optional[str]] = mapped_column(Text)  # Plain text version
    
    # Template used
    template_id: Mapped[Optional[str]] = mapped_column(String(100))
    
    # Regulatory reference
    regulation_reference: Mapped[Optional[str]] = mapped_column(String(200))  # e.g., "21 CFR Part 211"
    
    # Approval workflow
    prepared_by: Mapped[Optional[str]] = mapped_column(String(200))
    prepared_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    reviewed_by: Mapped[Optional[str]] = mapped_column(String(200))
    reviewed_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    approved_by: Mapped[Optional[str]] = mapped_column(String(200))
    approved_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Effective dates
    effective_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    review_date: Mapped[Optional[datetime]] = mapped_column(DateTime)  # Next review
    expiry_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Related documents
    related_documents: Mapped[Optional[List[str]]] = mapped_column(JSONB, default=list)
    supersedes: Mapped[Optional[str]] = mapped_column(String(100))  # Previous version doc number
    
    # File attachment
    file_path: Mapped[Optional[str]] = mapped_column(String(500))
    file_type: Mapped[Optional[str]] = mapped_column(String(50))  # pdf, docx, etc.
    
    # AI generation metadata
    ai_generated: Mapped[bool] = mapped_column(Boolean, default=False)
    ai_model_version: Mapped[Optional[str]] = mapped_column(String(100))
    generation_prompt: Mapped[Optional[str]] = mapped_column(Text)
    
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
    facility: Mapped["Facility"] = relationship("Facility", back_populates="documents")

    def __repr__(self) -> str:
        return f"<Document {self.document_number}: {self.title}>"


class ComplianceStatus(str, Enum):
    """Compliance check status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_APPLICABLE = "not_applicable"
    PENDING = "pending"


class ComplianceCheck(Base):
    """Regulatory compliance checks and audits."""
    
    __tablename__ = "compliance_checks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    facility_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("facilities.id", ondelete="CASCADE"),
        nullable=False,
    )
    
    # Check identification
    check_number: Mapped[str] = mapped_column(String(100), nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    
    # Regulation
    regulation: Mapped[str] = mapped_column(String(200), nullable=False)  # GMP, ISO, etc.
    regulation_section: Mapped[Optional[str]] = mapped_column(String(100))
    regulatory_body: Mapped[str] = mapped_column(String(100), nullable=False)  # MHRA, UAE MOH
    
    # Check details
    check_type: Mapped[str] = mapped_column(String(100), nullable=False)  # self-inspection, external audit
    scope: Mapped[Optional[str]] = mapped_column(Text)
    
    # Status
    status: Mapped[ComplianceStatus] = mapped_column(
        SQLEnum(ComplianceStatus),
        default=ComplianceStatus.PENDING,
    )
    
    # Findings
    findings: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSONB, default=list)
    # Structure: [{"finding": "...", "severity": "critical/major/minor", "area": "...", "evidence": "..."}]
    
    observations: Mapped[Optional[str]] = mapped_column(Text)
    recommendations: Mapped[Optional[str]] = mapped_column(Text)
    
    # Scores
    compliance_score: Mapped[Optional[float]] = mapped_column(Float)  # 0-100
    
    # Dates
    scheduled_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    conducted_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    due_date: Mapped[Optional[datetime]] = mapped_column(DateTime)  # For corrective actions
    
    # Personnel
    auditor: Mapped[Optional[str]] = mapped_column(String(200))
    auditee: Mapped[Optional[str]] = mapped_column(String(200))
    
    # Follow-up
    capa_required: Mapped[bool] = mapped_column(Boolean, default=False)
    capa_reference: Mapped[Optional[str]] = mapped_column(String(100))
    follow_up_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
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
    facility: Mapped["Facility"] = relationship("Facility", back_populates="compliance_checks")

    def __repr__(self) -> str:
        return f"<ComplianceCheck {self.check_number}: {self.status}>"


class BatchStatus(str, Enum):
    """Production batch status."""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    UNDER_QC = "under_qc"
    APPROVED = "approved"
    REJECTED = "rejected"
    RELEASED = "released"
    QUARANTINE = "quarantine"


class ProductionBatch(Base):
    """Production batch records."""
    
    __tablename__ = "production_batches"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    facility_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("facilities.id", ondelete="CASCADE"),
        nullable=False,
    )
    
    # Batch identification
    batch_number: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    product_name: Mapped[str] = mapped_column(String(255), nullable=False)
    product_code: Mapped[Optional[str]] = mapped_column(String(100))
    
    # Status
    status: Mapped[BatchStatus] = mapped_column(
        SQLEnum(BatchStatus),
        default=BatchStatus.PLANNED,
    )
    
    # Quantities
    batch_size: Mapped[float] = mapped_column(Float, nullable=False)
    batch_size_unit: Mapped[str] = mapped_column(String(50), nullable=False)  # kg, L, units
    theoretical_yield: Mapped[Optional[float]] = mapped_column(Float)
    actual_yield: Mapped[Optional[float]] = mapped_column(Float)
    yield_percentage: Mapped[Optional[float]] = mapped_column(Float)
    
    # Dates
    planned_start: Mapped[Optional[datetime]] = mapped_column(DateTime)
    actual_start: Mapped[Optional[datetime]] = mapped_column(DateTime)
    planned_end: Mapped[Optional[datetime]] = mapped_column(DateTime)
    actual_end: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Expiry
    manufacturing_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    expiry_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Raw materials
    raw_materials: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSONB, default=list)
    # Structure: [{"material": "...", "batch": "...", "quantity": 100, "unit": "kg"}]
    
    # Equipment used
    equipment_used: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSONB, default=list)
    # Structure: [{"equipment_id": "...", "name": "...", "start_time": "...", "end_time": "..."}]
    
    # Process parameters
    process_parameters: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    
    # QC data
    qc_tests: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSONB, default=list)
    # Structure: [{"test": "...", "specification": "...", "result": "...", "status": "pass/fail"}]
    
    qc_release_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    qc_released_by: Mapped[Optional[str]] = mapped_column(String(200))
    
    # Deviations
    deviations: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSONB, default=list)
    
    # Personnel
    production_supervisor: Mapped[Optional[str]] = mapped_column(String(200))
    qa_reviewer: Mapped[Optional[str]] = mapped_column(String(200))
    
    # Notes
    notes: Mapped[Optional[str]] = mapped_column(Text)
    
    # Related documents
    batch_record_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True))
    
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
    facility: Mapped["Facility"] = relationship("Facility", back_populates="production_batches")

    def __repr__(self) -> str:
        return f"<ProductionBatch {self.batch_number}: {self.product_name}>"


# Import for type hints
from src.models.user import Organization
