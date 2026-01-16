"""
UMI Pharmaceutical API
Facility management, QA/QC documentation, and compliance tracking
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_db
from src.core.exceptions import NotFoundError, ValidationError
from src.api.deps import get_current_user, require_roles
from src.models.user import User, UserRole
from src.models.pharma import DocumentType, DocumentStatus
from src.services.pharma_service import PharmaService
from src.schemas.pharma import (
    FacilityCreate,
    FacilityUpdate,
    FacilityResponse,
    FacilityListResponse,
    DocumentCreate,
    DocumentGenerateRequest,
    DocumentUpdate,
    DocumentResponse,
    DocumentListResponse,
    DocumentExportRequest,
    ComplianceCheckCreate,
    ComplianceCheckUpdate,
    ComplianceCheckResponse,
    ProductionBatchCreate,
    ProductionBatchUpdate,
    ProductionBatchResponse,
    FinancialAnalysisRequest,
    FinancialAnalysisResponse,
)

router = APIRouter()


# =============================================================================
# Facility Endpoints
# =============================================================================

@router.post("/facilities", response_model=FacilityResponse, status_code=status.HTTP_201_CREATED)
async def create_facility(
    data: FacilityCreate,
    current_user: User = Depends(require_roles(UserRole.PHARMA_ADMIN, UserRole.SYSTEM_ADMIN)),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Create a new pharmaceutical facility.
    
    Requires PHARMA_ADMIN or SYSTEM_ADMIN role.
    """
    # Get user's organization
    if not current_user.organizations:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User must belong to an organization",
        )
    
    org_id = current_user.organizations[0].organization_id
    
    service = PharmaService(db)
    facility = await service.create_facility(
        organization_id=org_id,
        data=data,
    )
    
    return facility


@router.get("/facilities", response_model=FacilityListResponse)
async def list_facilities(
    page: int = 1,
    page_size: int = 20,
    current_user: User = Depends(require_roles(UserRole.PHARMA_ADMIN, UserRole.SYSTEM_ADMIN)),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """List facilities for the user's organization."""
    if not current_user.organizations:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User must belong to an organization",
        )
    
    org_id = current_user.organizations[0].organization_id
    
    service = PharmaService(db)
    result = await service.list_facilities(
        organization_id=org_id,
        page=page,
        page_size=min(page_size, 100),
    )
    
    return result


@router.get("/facilities/{facility_id}", response_model=FacilityResponse)
async def get_facility(
    facility_id: UUID,
    current_user: User = Depends(require_roles(UserRole.PHARMA_ADMIN, UserRole.SYSTEM_ADMIN)),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Get a specific facility by ID."""
    if not current_user.organizations:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User must belong to an organization",
        )
    
    org_id = current_user.organizations[0].organization_id
    
    service = PharmaService(db)
    try:
        facility = await service.get_facility(
            facility_id=facility_id,
            organization_id=org_id,
        )
        return facility
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Facility not found",
        )


@router.put("/facilities/{facility_id}", response_model=FacilityResponse)
async def update_facility(
    facility_id: UUID,
    data: FacilityUpdate,
    current_user: User = Depends(require_roles(UserRole.PHARMA_ADMIN, UserRole.SYSTEM_ADMIN)),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Update a facility."""
    if not current_user.organizations:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User must belong to an organization",
        )
    
    org_id = current_user.organizations[0].organization_id
    
    service = PharmaService(db)
    try:
        facility = await service.update_facility(
            facility_id=facility_id,
            organization_id=org_id,
            data=data,
        )
        return facility
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Facility not found",
        )


# =============================================================================
# Document Endpoints
# =============================================================================

@router.post("/facilities/{facility_id}/documents", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def create_document(
    facility_id: UUID,
    data: DocumentCreate,
    current_user: User = Depends(require_roles(UserRole.PHARMA_ADMIN, UserRole.SYSTEM_ADMIN)),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Create a QA/QC document manually."""
    if not current_user.organizations:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User must belong to an organization",
        )
    
    org_id = current_user.organizations[0].organization_id
    
    service = PharmaService(db)
    try:
        document = await service.create_document(
            facility_id=facility_id,
            organization_id=org_id,
            data=data,
        )
        return document
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Facility not found",
        )


@router.post("/documents/generate", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def generate_document(
    data: DocumentGenerateRequest,
    current_user: User = Depends(require_roles(UserRole.PHARMA_ADMIN, UserRole.SYSTEM_ADMIN)),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Generate a QA/QC document using AI.
    
    Supported document types:
    - **cleaning_validation**: Cleaning validation protocol
    - **process_validation**: Process validation protocol
    - **hvac_qualification**: HVAC system qualification
    - **water_system_validation**: Water system validation
    - **sop**: Standard Operating Procedure
    - **batch_record**: Batch manufacturing record
    - **deviation_report**: Deviation report
    - **capa**: Corrective and Preventive Action
    
    The AI will generate a complete document based on the provided
    parameters and regulatory requirements.
    """
    if not current_user.organizations:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User must belong to an organization",
        )
    
    org_id = current_user.organizations[0].organization_id
    
    service = PharmaService(db)
    try:
        document = await service.generate_document(
            organization_id=org_id,
            request=data,
        )
        return document
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Facility not found",
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )


@router.get("/facilities/{facility_id}/documents", response_model=DocumentListResponse)
async def list_documents(
    facility_id: UUID,
    doc_type: Optional[DocumentType] = None,
    doc_status: Optional[DocumentStatus] = None,
    page: int = 1,
    page_size: int = 20,
    current_user: User = Depends(require_roles(UserRole.PHARMA_ADMIN, UserRole.SYSTEM_ADMIN)),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    List documents for a facility.
    
    Filter by:
    - **doc_type**: Document type (cleaning_validation, sop, etc.)
    - **doc_status**: Document status (draft, approved, etc.)
    """
    service = PharmaService(db)
    result = await service.list_documents(
        facility_id=facility_id,
        doc_type=doc_type,
        status=doc_status,
        page=page,
        page_size=min(page_size, 100),
    )
    
    return result


@router.get("/facilities/{facility_id}/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    facility_id: UUID,
    document_id: UUID,
    current_user: User = Depends(require_roles(UserRole.PHARMA_ADMIN, UserRole.SYSTEM_ADMIN)),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Get a specific document."""
    service = PharmaService(db)
    try:
        document = await service.get_document(
            document_id=document_id,
            facility_id=facility_id,
        )
        return document
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )


@router.put("/facilities/{facility_id}/documents/{document_id}", response_model=DocumentResponse)
async def update_document(
    facility_id: UUID,
    document_id: UUID,
    data: DocumentUpdate,
    current_user: User = Depends(require_roles(UserRole.PHARMA_ADMIN, UserRole.SYSTEM_ADMIN)),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Update a document."""
    service = PharmaService(db)
    try:
        document = await service.update_document(
            document_id=document_id,
            facility_id=facility_id,
            data=data,
        )
        return document
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )


@router.post("/facilities/{facility_id}/documents/{document_id}/approve", response_model=DocumentResponse)
async def approve_document(
    facility_id: UUID,
    document_id: UUID,
    current_user: User = Depends(require_roles(UserRole.PHARMA_ADMIN, UserRole.SYSTEM_ADMIN)),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Approve a document."""
    service = PharmaService(db)
    try:
        document = await service.approve_document(
            document_id=document_id,
            facility_id=facility_id,
            approved_by=current_user.profile.full_name if current_user.profile else current_user.email,
        )
        return document
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/facilities/{facility_id}/documents/{document_id}/export")
async def export_document(
    facility_id: UUID,
    document_id: UUID,
    data: DocumentExportRequest,
    current_user: User = Depends(require_roles(UserRole.PHARMA_ADMIN, UserRole.SYSTEM_ADMIN)),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Export document to PDF, DOCX, or HTML.
    
    Returns a download URL for the exported file.
    """
    # TODO: Implement document export
    return {
        "download_url": f"/api/v1/pharma/downloads/{document_id}.{data.format}",
        "expires_in": 3600,
    }


# =============================================================================
# Compliance Endpoints
# =============================================================================

@router.post("/facilities/{facility_id}/compliance", response_model=ComplianceCheckResponse, status_code=status.HTTP_201_CREATED)
async def create_compliance_check(
    facility_id: UUID,
    data: ComplianceCheckCreate,
    current_user: User = Depends(require_roles(UserRole.PHARMA_ADMIN, UserRole.SYSTEM_ADMIN)),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Create a new compliance check."""
    if not current_user.organizations:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User must belong to an organization",
        )
    
    org_id = current_user.organizations[0].organization_id
    
    service = PharmaService(db)
    try:
        check = await service.create_compliance_check(
            facility_id=facility_id,
            organization_id=org_id,
            data=data,
        )
        return check
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Facility not found",
        )


# =============================================================================
# Production Batch Endpoints
# =============================================================================

@router.post("/facilities/{facility_id}/batches", response_model=ProductionBatchResponse, status_code=status.HTTP_201_CREATED)
async def create_batch(
    facility_id: UUID,
    data: ProductionBatchCreate,
    current_user: User = Depends(require_roles(UserRole.PHARMA_ADMIN, UserRole.SYSTEM_ADMIN)),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Create a new production batch record."""
    if not current_user.organizations:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User must belong to an organization",
        )
    
    org_id = current_user.organizations[0].organization_id
    
    service = PharmaService(db)
    try:
        batch = await service.create_batch(
            facility_id=facility_id,
            organization_id=org_id,
            data=data,
        )
        return batch
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Facility not found",
        )


@router.put("/facilities/{facility_id}/batches/{batch_id}", response_model=ProductionBatchResponse)
async def update_batch(
    facility_id: UUID,
    batch_id: UUID,
    data: ProductionBatchUpdate,
    current_user: User = Depends(require_roles(UserRole.PHARMA_ADMIN, UserRole.SYSTEM_ADMIN)),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Update a production batch."""
    service = PharmaService(db)
    try:
        batch = await service.update_batch(
            batch_id=batch_id,
            facility_id=facility_id,
            data=data,
        )
        return batch
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Batch not found",
        )


# =============================================================================
# Financial Analysis Endpoints
# =============================================================================

@router.post("/analysis/financial", response_model=FinancialAnalysisResponse)
async def run_financial_analysis(
    data: FinancialAnalysisRequest,
    current_user: User = Depends(require_roles(UserRole.PHARMA_ADMIN, UserRole.SYSTEM_ADMIN)),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Run financial analysis for pharmaceutical operations.
    
    Analysis types:
    - **sales**: Sales performance analysis
    - **production**: Production efficiency analysis
    - **compliance**: Compliance cost analysis
    - **comprehensive**: Full financial overview
    """
    from datetime import datetime
    
    # TODO: Implement actual financial analysis with AI
    return {
        "analysis_type": data.analysis_type,
        "period": f"{data.period_start.date()} to {data.period_end.date()}",
        "summary": {
            "total_revenue": 0,
            "total_cost": 0,
            "gross_margin": 0,
            "compliance_cost": 0,
        },
        "metrics": {
            "revenue_growth": 0.0,
            "cost_efficiency": 0.0,
            "compliance_rate": 0.0,
        },
        "trends": [],
        "projections": None,
        "recommendations": [
            "Implement automated QA/QC documentation to reduce compliance costs",
            "Optimize production scheduling to improve efficiency",
        ],
        "generated_at": datetime.utcnow(),
    }
