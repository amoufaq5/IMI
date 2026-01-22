"""General API endpoints"""
from fastapi import APIRouter, Depends
from typing import Optional, List
from pydantic import BaseModel

from src.domains.general import GeneralService
from src.core.security.authentication import get_current_user, UserContext

router = APIRouter()

general_service = GeneralService()


class SearchRequest(BaseModel):
    """Medical search request"""
    query: str
    category: Optional[str] = None
    limit: int = 10


class DrugInteractionCheckRequest(BaseModel):
    """Drug interaction check request"""
    drug1: str
    drug2: str


class TermExplanationRequest(BaseModel):
    """Medical term explanation request"""
    term: str
    audience: str = "general"


@router.get("/disease/{disease_name}")
async def get_disease_info(
    disease_name: str,
    user: UserContext = Depends(get_current_user),
):
    """Get comprehensive information about a disease"""
    return await general_service.get_disease_info(
        disease_name=disease_name,
        user_id=user.user_id if user else None,
    )


@router.get("/drug/{drug_name}")
async def get_drug_info(
    drug_name: str,
    user: UserContext = Depends(get_current_user),
):
    """Get comprehensive information about a drug"""
    return await general_service.get_drug_info(
        drug_name=drug_name,
        user_id=user.user_id if user else None,
    )


@router.post("/search")
async def search_medical_info(
    request: SearchRequest,
    user: UserContext = Depends(get_current_user),
):
    """Search for medical information"""
    return await general_service.search_medical_info(
        query=request.query,
        category=request.category,
        limit=request.limit,
        user_id=user.user_id if user else None,
    )


@router.post("/drug-interaction")
async def check_drug_interaction(
    request: DrugInteractionCheckRequest,
    user: UserContext = Depends(get_current_user),
):
    """Check for interaction between two drugs"""
    return await general_service.check_drug_interaction(
        drug1=request.drug1,
        drug2=request.drug2,
        user_id=user.user_id if user else None,
    )


@router.post("/explain-term")
async def explain_medical_term(
    request: TermExplanationRequest,
    user: UserContext = Depends(get_current_user),
):
    """Explain a medical term in plain language"""
    return await general_service.explain_medical_term(
        term=request.term,
        audience=request.audience,
        user_id=user.user_id if user else None,
    )


@router.get("/health-tips")
async def get_health_tips(
    category: str = "general",
    user: UserContext = Depends(get_current_user),
):
    """Get health tips by category"""
    tips = await general_service.get_health_tips(
        category=category,
        user_id=user.user_id if user else None,
    )
    return {"category": category, "tips": tips}
