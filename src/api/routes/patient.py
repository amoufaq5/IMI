"""Patient API endpoints"""
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List
from pydantic import BaseModel, Field

from src.domains.patient import PatientService, SymptomAssessmentRequest, SymptomAssessmentResponse
from src.core.security.authentication import get_current_user, UserContext
from src.core.security.authorization import require_permission, Permission

router = APIRouter()

# Initialize service
patient_service = PatientService()


class HealthQueryRequest(BaseModel):
    """Health information query"""
    query: str
    context: Optional[str] = None


class DrugSafetyRequest(BaseModel):
    """Drug safety check request"""
    drug_name: str
    conditions: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    age: int
    is_pregnant: bool = False


@router.post("/assess-symptoms", response_model=SymptomAssessmentResponse)
async def assess_symptoms(
    request: SymptomAssessmentRequest,
    user: UserContext = Depends(get_current_user),
):
    """
    Assess patient symptoms and provide triage recommendation
    
    Uses the 5-layer architecture:
    - Knowledge Graph for medical facts
    - Rule Engine for safety checks
    - LLM for explanation
    - Verifier for accuracy
    - Memory for patient context
    """
    return await patient_service.assess_symptoms(
        request=request,
        user_id=user.user_id if user else None,
    )


@router.post("/health-info")
async def get_health_information(
    request: HealthQueryRequest,
    user: UserContext = Depends(get_current_user),
):
    """Get general health information"""
    return await patient_service.get_health_information(
        query=request.query,
        user_id=user.user_id if user else None,
    )


@router.post("/check-drug-safety")
async def check_drug_safety(
    request: DrugSafetyRequest,
    user: UserContext = Depends(get_current_user),
):
    """Check if a drug is safe for the patient"""
    return await patient_service.check_drug_safety(
        drug_name=request.drug_name,
        patient_conditions=request.conditions,
        current_medications=request.current_medications,
        allergies=request.allergies,
        age=request.age,
        is_pregnant=request.is_pregnant,
        user_id=user.user_id if user else None,
    )


@router.post("/analyze-lab-results")
async def analyze_lab_results(
    lab_results: dict,
    user: UserContext = Depends(get_current_user),
):
    """Analyze lab results and provide interpretation"""
    return await patient_service.analyze_lab_results(
        lab_results=lab_results,
        user_id=user.user_id if user else None,
    )
