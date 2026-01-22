"""Doctor API endpoints"""
from fastapi import APIRouter, Depends, HTTPException
from typing import Optional, List
from pydantic import BaseModel, Field

from src.domains.doctor import DoctorService, ClinicalCase
from src.core.security.authentication import get_current_user, UserContext
from src.core.security.authorization import require_role, UserRole

router = APIRouter()

doctor_service = DoctorService()


class DifferentialRequest(BaseModel):
    """Differential diagnosis request"""
    chief_complaint: str
    history_of_present_illness: str
    past_medical_history: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    vital_signs: Optional[dict] = None
    lab_results: Optional[dict] = None
    patient_age: Optional[int] = None
    patient_sex: Optional[str] = None


class TreatmentRequest(BaseModel):
    """Treatment recommendation request"""
    diagnosis: str
    patient_conditions: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)


class DrugInteractionRequest(BaseModel):
    """Drug interaction check request"""
    medications: List[str]


@router.post("/differential")
async def generate_differential(
    request: DifferentialRequest,
    user: UserContext = Depends(get_current_user),
):
    """Generate differential diagnosis for a clinical case"""
    case = ClinicalCase(
        chief_complaint=request.chief_complaint,
        history_of_present_illness=request.history_of_present_illness,
        past_medical_history=request.past_medical_history,
        medications=request.medications,
        allergies=request.allergies,
        vital_signs=request.vital_signs,
        lab_results=request.lab_results,
        patient_age=request.patient_age,
        patient_sex=request.patient_sex,
    )
    
    differentials = await doctor_service.generate_differential(
        case=case,
        user_id=user.user_id if user else None,
    )
    
    return {"differentials": [d.model_dump() for d in differentials]}


@router.post("/treatment-recommendations")
async def get_treatment_recommendations(
    request: TreatmentRequest,
    user: UserContext = Depends(get_current_user),
):
    """Get evidence-based treatment recommendations"""
    recommendations = await doctor_service.get_treatment_recommendations(
        diagnosis=request.diagnosis,
        patient_conditions=request.patient_conditions,
        current_medications=request.current_medications,
        allergies=request.allergies,
        user_id=user.user_id if user else None,
    )
    
    return {"recommendations": [r.model_dump() for r in recommendations]}


@router.post("/drug-interactions")
async def check_drug_interactions(
    request: DrugInteractionRequest,
    user: UserContext = Depends(get_current_user),
):
    """Check for drug-drug interactions"""
    return await doctor_service.check_drug_interactions(
        medications=request.medications,
        user_id=user.user_id if user else None,
    )


@router.get("/guidelines/{condition}")
async def get_clinical_guidelines(
    condition: str,
    user: UserContext = Depends(get_current_user),
):
    """Get clinical guidelines for a condition"""
    guidelines = await doctor_service.get_clinical_guidelines(
        condition=condition,
        user_id=user.user_id if user else None,
    )
    return {"guidelines": guidelines}


@router.post("/summarize-case")
async def summarize_case(
    request: DifferentialRequest,
    include_assessment: bool = True,
    user: UserContext = Depends(get_current_user),
):
    """Generate clinical case summary"""
    case = ClinicalCase(
        chief_complaint=request.chief_complaint,
        history_of_present_illness=request.history_of_present_illness,
        past_medical_history=request.past_medical_history,
        medications=request.medications,
        allergies=request.allergies,
        vital_signs=request.vital_signs,
        lab_results=request.lab_results,
        patient_age=request.patient_age,
        patient_sex=request.patient_sex,
    )
    
    return await doctor_service.summarize_case(
        case=case,
        include_assessment=include_assessment,
        user_id=user.user_id if user else None,
    )
