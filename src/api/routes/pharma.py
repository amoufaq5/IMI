"""Pharmaceutical API endpoints"""
from fastapi import APIRouter, Depends
from typing import Optional, List
from datetime import date
from pydantic import BaseModel, Field

from src.domains.pharma import PharmaService, RegulatoryBody, DocumentType, ValidationType
from src.layers.memory.entity_profile import EntityType
from src.core.security.authentication import get_current_user, UserContext

router = APIRouter()

pharma_service = PharmaService()


class DocumentGenerationRequest(BaseModel):
    """Document generation request"""
    document_type: DocumentType
    template_data: dict
    entity_id: Optional[str] = None


class ComplianceCheckRequest(BaseModel):
    """Compliance check request"""
    entity_id: str
    regulatory_body: RegulatoryBody
    check_areas: Optional[List[str]] = None


class ValidationRequest(BaseModel):
    """Validation record request"""
    entity_id: str
    validation_type: ValidationType
    protocol_number: str
    equipment_or_process: str


class ValidationCompleteRequest(BaseModel):
    """Validation completion request"""
    entity_id: str
    validation_id: str
    result: str
    next_revalidation: date
    deviations: Optional[List[str]] = None


class SalesRecordRequest(BaseModel):
    """Sales record request"""
    entity_id: str
    product_id: str
    period: str
    region: str
    units_sold: int
    revenue: float
    marketing_spend: Optional[float] = None


class EntityCreateRequest(BaseModel):
    """Entity creation request"""
    name: str
    entity_type: EntityType = EntityType.PHARMACEUTICAL_COMPANY
    address: Optional[str] = None
    country: Optional[str] = None


@router.post("/entity")
async def create_entity(
    request: EntityCreateRequest,
    user: UserContext = Depends(get_current_user),
):
    """Create a pharmaceutical entity profile"""
    profile = pharma_service.entity_manager.create_profile(
        entity_type=request.entity_type,
        name=request.name,
        address=request.address,
        country=request.country,
    )
    return {"entity_id": profile.id, "name": profile.name}


@router.get("/entity/{entity_id}")
async def get_entity(
    entity_id: str,
    user: UserContext = Depends(get_current_user),
):
    """Get entity profile"""
    entity = pharma_service.entity_manager.get_profile(entity_id)
    if not entity:
        return {"error": "Entity not found"}
    return entity.model_dump()


@router.get("/entity/{entity_id}/status")
async def get_facility_status(
    entity_id: str,
    user: UserContext = Depends(get_current_user),
):
    """Get comprehensive facility status"""
    return pharma_service.get_facility_status(entity_id)


@router.post("/document/generate")
async def generate_document(
    request: DocumentGenerationRequest,
    user: UserContext = Depends(get_current_user),
):
    """Generate a QA document from template"""
    return await pharma_service.generate_document(
        document_type=request.document_type,
        template_data=request.template_data,
        entity_id=request.entity_id,
        user_id=user.user_id if user else None,
    )


@router.post("/compliance/check")
async def check_compliance(
    request: ComplianceCheckRequest,
    user: UserContext = Depends(get_current_user),
):
    """Check compliance against regulatory requirements"""
    result = await pharma_service.check_compliance(
        entity_id=request.entity_id,
        regulatory_body=request.regulatory_body,
        check_areas=request.check_areas,
        user_id=user.user_id if user else None,
    )
    return result.model_dump()


@router.post("/validation")
async def create_validation_record(
    request: ValidationRequest,
    user: UserContext = Depends(get_current_user),
):
    """Create a new validation record"""
    validation = pharma_service.create_validation_record(
        entity_id=request.entity_id,
        validation_type=request.validation_type,
        protocol_number=request.protocol_number,
        equipment_or_process=request.equipment_or_process,
        user_id=user.user_id if user else None,
    )
    if not validation:
        return {"error": "Failed to create validation record"}
    return {"validation_id": validation.id, "status": validation.status}


@router.post("/validation/complete")
async def complete_validation(
    request: ValidationCompleteRequest,
    user: UserContext = Depends(get_current_user),
):
    """Complete a validation record"""
    success = pharma_service.complete_validation(
        entity_id=request.entity_id,
        validation_id=request.validation_id,
        result=request.result,
        next_revalidation=request.next_revalidation,
        deviations=request.deviations,
        user_id=user.user_id if user else None,
    )
    return {"success": success}


@router.post("/sales")
async def add_sales_record(
    request: SalesRecordRequest,
    user: UserContext = Depends(get_current_user),
):
    """Add a sales record"""
    sales = pharma_service.add_sales_record(
        entity_id=request.entity_id,
        product_id=request.product_id,
        period=request.period,
        region=request.region,
        units_sold=request.units_sold,
        revenue=request.revenue,
        marketing_spend=request.marketing_spend,
        user_id=user.user_id if user else None,
    )
    if not sales:
        return {"error": "Failed to add sales record"}
    return {"sales_id": sales.id}


@router.get("/sales/analytics/{entity_id}")
async def get_sales_analytics(
    entity_id: str,
    product_id: Optional[str] = None,
    period_start: Optional[str] = None,
    period_end: Optional[str] = None,
    user: UserContext = Depends(get_current_user),
):
    """Get sales analytics"""
    return pharma_service.get_sales_analytics(
        entity_id=entity_id,
        product_id=product_id,
        period_start=period_start,
        period_end=period_end,
    )


@router.get("/regulatory/guidance")
async def get_regulatory_guidance(
    topic: str,
    regulatory_body: RegulatoryBody,
    user: UserContext = Depends(get_current_user),
):
    """Get regulatory guidance on a topic"""
    return await pharma_service.get_regulatory_guidance(
        topic=topic,
        regulatory_body=regulatory_body,
        user_id=user.user_id if user else None,
    )
