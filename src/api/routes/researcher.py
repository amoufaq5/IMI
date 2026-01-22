"""Researcher API endpoints"""
from fastapi import APIRouter, Depends
from typing import Optional, List
from datetime import date
from pydantic import BaseModel, Field

from src.domains.researcher import ResearcherService, PatentType, ClinicalTrialPhase
from src.core.security.authentication import get_current_user, UserContext

router = APIRouter()

researcher_service = ResearcherService()


class LiteratureSearchRequest(BaseModel):
    """Literature search request"""
    query: str
    filters: Optional[dict] = None
    max_results: int = 20


class SynthesisRequest(BaseModel):
    """Literature synthesis request"""
    topic: str
    papers: List[dict] = Field(default_factory=list)
    synthesis_type: str = "systematic_review"


class PatentGuidanceRequest(BaseModel):
    """Patent guidance request"""
    compound_info: dict
    patent_type: PatentType


class ProjectRequest(BaseModel):
    """Research project creation request"""
    title: str
    description: str
    principal_investigator: str
    phase: Optional[ClinicalTrialPhase] = None


class RegulatoryPathwayRequest(BaseModel):
    """Regulatory pathway request"""
    drug_type: str
    indication: str
    target_markets: List[str]


@router.post("/literature/search")
async def search_literature(
    request: LiteratureSearchRequest,
    user: UserContext = Depends(get_current_user),
):
    """Search medical literature"""
    return await researcher_service.search_literature(
        query=request.query,
        filters=request.filters,
        max_results=request.max_results,
        user_id=user.user_id if user else None,
    )


@router.post("/literature/synthesize")
async def synthesize_literature(
    request: SynthesisRequest,
    user: UserContext = Depends(get_current_user),
):
    """Synthesize findings from multiple papers"""
    return await researcher_service.synthesize_literature(
        topic=request.topic,
        papers=request.papers,
        synthesis_type=request.synthesis_type,
        user_id=user.user_id if user else None,
    )


@router.post("/patent/guidance")
async def get_patent_guidance(
    request: PatentGuidanceRequest,
    user: UserContext = Depends(get_current_user),
):
    """Get guidance on patent application"""
    return await researcher_service.get_patent_guidance(
        compound_info=request.compound_info,
        patent_type=request.patent_type,
        user_id=user.user_id if user else None,
    )


@router.post("/project")
async def create_research_project(
    request: ProjectRequest,
    user: UserContext = Depends(get_current_user),
):
    """Create a new research project"""
    project = researcher_service.create_research_project(
        title=request.title,
        description=request.description,
        principal_investigator=request.principal_investigator,
        phase=request.phase,
        user_id=user.user_id if user else None,
    )
    return {"project_id": project.id, "title": project.title}


@router.get("/project/{project_id}")
async def get_research_project(
    project_id: str,
    user: UserContext = Depends(get_current_user),
):
    """Get a research project"""
    project = researcher_service.get_research_project(project_id)
    if not project:
        return {"error": "Project not found"}
    return project.model_dump()


@router.post("/regulatory-pathway")
async def get_regulatory_pathway(
    request: RegulatoryPathwayRequest,
    user: UserContext = Depends(get_current_user),
):
    """Get regulatory pathway guidance"""
    return await researcher_service.get_regulatory_pathway(
        drug_type=request.drug_type,
        indication=request.indication,
        target_markets=request.target_markets,
        user_id=user.user_id if user else None,
    )
