"""
UMI Pharmaceutical Schemas
Pydantic models for Pharma QA/QC API
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from src.models.pharma import (
    FacilityType,
    FacilityStatus,
    DocumentType,
    DocumentStatus,
    ComplianceStatus,
    BatchStatus,
)


# ============================================================================
# Facility Schemas
# ============================================================================

class CertificationInfo(BaseModel):
    """Certification information."""
    name: str = Field(..., description="Certification name (e.g., GMP, ISO 9001)")
    issued_by: str = Field(..., description="Issuing body")
    certificate_number: Optional[str] = None
    issue_date: Optional[datetime] = None
    valid_until: Optional[datetime] = None


class FacilityBase(BaseModel):
    """Base facility schema."""
    name: str = Field(..., max_length=255)
    type: FacilityType
    address: str
    city: str = Field(..., max_length=100)
    country: str = Field(..., max_length=100)
    postal_code: Optional[str] = Field(None, max_length=20)
    phone: Optional[str] = Field(None, max_length=20)
    email: Optional[str] = Field(None, max_length=255)
    facility_manager: Optional[str] = Field(None, max_length=200)


class FacilityCreate(FacilityBase):
    """Schema for creating a facility."""
    facility_code: Optional[str] = Field(None, max_length=50)
    license_number: Optional[str] = Field(None, max_length=100)
    license_expiry: Optional[datetime] = None
    regulatory_body: Optional[str] = Field(None, max_length=100)
    certifications: Optional[List[CertificationInfo]] = None
    total_area_sqm: Optional[float] = None
    production_capacity: Optional[str] = None
    cleanroom_classes: Optional[List[str]] = None
    product_types: Optional[List[str]] = None


class FacilityUpdate(BaseModel):
    """Schema for updating a facility."""
    name: Optional[str] = Field(None, max_length=255)
    status: Optional[FacilityStatus] = None
    address: Optional[str] = None
    city: Optional[str] = Field(None, max_length=100)
    phone: Optional[str] = Field(None, max_length=20)
    email: Optional[str] = Field(None, max_length=255)
    facility_manager: Optional[str] = Field(None, max_length=200)
    license_number: Optional[str] = Field(None, max_length=100)
    license_expiry: Optional[datetime] = None
    certifications: Optional[List[CertificationInfo]] = None


class FacilityResponse(FacilityBase):
    """Schema for facility response."""
    id: UUID
    organization_id: UUID
    facility_code: Optional[str] = None
    status: FacilityStatus
    license_number: Optional[str] = None
    license_expiry: Optional[datetime] = None
    regulatory_body: Optional[str] = None
    certifications: Optional[List[CertificationInfo]] = None
    total_area_sqm: Optional[float] = None
    production_capacity: Optional[str] = None
    cleanroom_classes: Optional[List[str]] = None
    product_types: Optional[List[str]] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class FacilityListResponse(BaseModel):
    """Paginated facility list."""
    items: List[FacilityResponse]
    total: int
    page: int
    page_size: int
    pages: int


# ============================================================================
# Document Schemas
# ============================================================================

class DocumentBase(BaseModel):
    """Base document schema."""
    title: str = Field(..., max_length=500)
    type: DocumentType


class DocumentCreate(DocumentBase):
    """Schema for creating a document manually."""
    document_number: str = Field(..., max_length=100)
    content: Optional[Dict[str, Any]] = None
    content_text: Optional[str] = None
    regulation_reference: Optional[str] = Field(None, max_length=200)
    effective_date: Optional[datetime] = None
    review_date: Optional[datetime] = None


class DocumentGenerateRequest(BaseModel):
    """Schema for AI-generated document request."""
    type: DocumentType
    title: Optional[str] = Field(None, max_length=500)
    
    # Context for generation
    facility_id: UUID
    
    # Type-specific parameters
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Type-specific generation parameters"
    )
    
    # Regulatory context
    regulation: str = Field(
        default="GMP",
        description="Target regulation (GMP, ISO, etc.)"
    )
    regulatory_body: str = Field(
        default="MHRA",
        description="Target regulatory body"
    )
    
    # Output preferences
    language: str = Field(default="en")
    include_appendices: bool = Field(default=True)
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": "cleaning_validation",
                "facility_id": "123e4567-e89b-12d3-a456-426614174000",
                "parameters": {
                    "equipment_name": "Tablet Press Model XYZ",
                    "equipment_id": "EQ-001",
                    "product_changeover": True,
                    "cleaning_method": "CIP",
                    "acceptance_criteria": {
                        "visual": "No visible residue",
                        "chemical": "< 10 ppm active",
                        "microbial": "< 100 CFU/swab"
                    }
                },
                "regulation": "GMP",
                "regulatory_body": "MHRA"
            }
        }


class DocumentUpdate(BaseModel):
    """Schema for updating a document."""
    title: Optional[str] = Field(None, max_length=500)
    content: Optional[Dict[str, Any]] = None
    content_text: Optional[str] = None
    status: Optional[DocumentStatus] = None
    reviewed_by: Optional[str] = Field(None, max_length=200)
    reviewed_date: Optional[datetime] = None
    approved_by: Optional[str] = Field(None, max_length=200)
    approved_date: Optional[datetime] = None
    effective_date: Optional[datetime] = None
    review_date: Optional[datetime] = None


class DocumentResponse(DocumentBase):
    """Schema for document response."""
    id: UUID
    facility_id: UUID
    document_number: str
    version: str
    status: DocumentStatus
    content: Optional[Dict[str, Any]] = None
    content_text: Optional[str] = None
    template_id: Optional[str] = None
    regulation_reference: Optional[str] = None
    
    # Approval info
    prepared_by: Optional[str] = None
    prepared_date: Optional[datetime] = None
    reviewed_by: Optional[str] = None
    reviewed_date: Optional[datetime] = None
    approved_by: Optional[str] = None
    approved_date: Optional[datetime] = None
    
    # Dates
    effective_date: Optional[datetime] = None
    review_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    
    # AI metadata
    ai_generated: bool = False
    
    # File
    file_path: Optional[str] = None
    file_type: Optional[str] = None
    
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    """Paginated document list."""
    items: List[DocumentResponse]
    total: int
    page: int
    page_size: int
    pages: int


class DocumentExportRequest(BaseModel):
    """Request to export document."""
    format: str = Field(
        default="pdf",
        pattern="^(pdf|docx|html)$",
        description="Export format"
    )
    include_signatures: bool = Field(default=True)
    include_watermark: bool = Field(default=False)
    watermark_text: Optional[str] = Field(None, max_length=100)


# ============================================================================
# Compliance Schemas
# ============================================================================

class ComplianceFinding(BaseModel):
    """Single compliance finding."""
    finding: str
    severity: str = Field(..., pattern="^(critical|major|minor|observation)$")
    area: str
    evidence: Optional[str] = None
    recommendation: Optional[str] = None


class ComplianceCheckCreate(BaseModel):
    """Schema for creating compliance check."""
    title: str = Field(..., max_length=500)
    regulation: str = Field(..., max_length=200)
    regulation_section: Optional[str] = Field(None, max_length=100)
    regulatory_body: str = Field(..., max_length=100)
    check_type: str = Field(..., max_length=100)
    scope: Optional[str] = None
    scheduled_date: Optional[datetime] = None
    auditor: Optional[str] = Field(None, max_length=200)


class ComplianceCheckUpdate(BaseModel):
    """Schema for updating compliance check."""
    status: Optional[ComplianceStatus] = None
    findings: Optional[List[ComplianceFinding]] = None
    observations: Optional[str] = None
    recommendations: Optional[str] = None
    compliance_score: Optional[float] = Field(None, ge=0, le=100)
    conducted_date: Optional[datetime] = None
    capa_required: Optional[bool] = None
    capa_reference: Optional[str] = Field(None, max_length=100)


class ComplianceCheckResponse(BaseModel):
    """Schema for compliance check response."""
    id: UUID
    facility_id: UUID
    check_number: str
    title: str
    regulation: str
    regulation_section: Optional[str] = None
    regulatory_body: str
    check_type: str
    scope: Optional[str] = None
    status: ComplianceStatus
    findings: Optional[List[ComplianceFinding]] = None
    observations: Optional[str] = None
    recommendations: Optional[str] = None
    compliance_score: Optional[float] = None
    scheduled_date: Optional[datetime] = None
    conducted_date: Optional[datetime] = None
    due_date: Optional[datetime] = None
    auditor: Optional[str] = None
    capa_required: bool = False
    capa_reference: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ============================================================================
# Production Batch Schemas
# ============================================================================

class RawMaterialInput(BaseModel):
    """Raw material used in batch."""
    material: str
    batch_number: str
    quantity: float
    unit: str


class QCTestResult(BaseModel):
    """QC test result."""
    test: str
    specification: str
    result: str
    status: str = Field(..., pattern="^(pass|fail|pending)$")
    tested_by: Optional[str] = None
    tested_date: Optional[datetime] = None


class ProductionBatchCreate(BaseModel):
    """Schema for creating production batch."""
    batch_number: str = Field(..., max_length=100)
    product_name: str = Field(..., max_length=255)
    product_code: Optional[str] = Field(None, max_length=100)
    batch_size: float
    batch_size_unit: str = Field(..., max_length=50)
    theoretical_yield: Optional[float] = None
    planned_start: Optional[datetime] = None
    planned_end: Optional[datetime] = None
    production_supervisor: Optional[str] = Field(None, max_length=200)


class ProductionBatchUpdate(BaseModel):
    """Schema for updating production batch."""
    status: Optional[BatchStatus] = None
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None
    actual_yield: Optional[float] = None
    manufacturing_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    raw_materials: Optional[List[RawMaterialInput]] = None
    qc_tests: Optional[List[QCTestResult]] = None
    notes: Optional[str] = None


class ProductionBatchResponse(BaseModel):
    """Schema for production batch response."""
    id: UUID
    facility_id: UUID
    batch_number: str
    product_name: str
    product_code: Optional[str] = None
    status: BatchStatus
    batch_size: float
    batch_size_unit: str
    theoretical_yield: Optional[float] = None
    actual_yield: Optional[float] = None
    yield_percentage: Optional[float] = None
    planned_start: Optional[datetime] = None
    actual_start: Optional[datetime] = None
    planned_end: Optional[datetime] = None
    actual_end: Optional[datetime] = None
    manufacturing_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    raw_materials: Optional[List[RawMaterialInput]] = None
    qc_tests: Optional[List[QCTestResult]] = None
    production_supervisor: Optional[str] = None
    qa_reviewer: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ============================================================================
# Financial Analysis Schemas
# ============================================================================

class SalesDataInput(BaseModel):
    """Sales data for analysis."""
    product_name: str
    period: str  # e.g., "2024-Q1"
    units_sold: int
    revenue: float
    cost: float
    region: Optional[str] = None


class FinancialAnalysisRequest(BaseModel):
    """Request for financial analysis."""
    facility_id: Optional[UUID] = None
    period_start: datetime
    period_end: datetime
    analysis_type: str = Field(
        ...,
        pattern="^(sales|production|compliance|comprehensive)$"
    )
    include_projections: bool = Field(default=False)


class FinancialAnalysisResponse(BaseModel):
    """Financial analysis response."""
    analysis_type: str
    period: str
    summary: Dict[str, Any]
    metrics: Dict[str, float]
    trends: Optional[List[Dict[str, Any]]] = None
    projections: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[str]] = None
    generated_at: datetime
