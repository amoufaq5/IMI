"""Hospital API endpoints"""
from fastapi import APIRouter, Depends
from typing import Optional, List
from datetime import datetime, date
from pydantic import BaseModel, Field

from src.domains.hospital import HospitalService, ERPriority
from src.core.security.authentication import get_current_user, UserContext

router = APIRouter()

hospital_service = HospitalService()


class TriageRequest(BaseModel):
    """ER triage request"""
    patient_id: str
    chief_complaint: str
    symptoms: List[str]
    vital_signs: dict
    age: int
    medical_history: Optional[List[str]] = None


class AppointmentRequest(BaseModel):
    """Appointment scheduling request"""
    patient_id: str
    provider_id: str
    department: str
    scheduled_time: datetime
    duration_minutes: int = 30
    reason: Optional[str] = None


class InsuranceClaimRequest(BaseModel):
    """Insurance claim request"""
    patient_id: str
    encounter_id: str
    insurance_provider: str
    policy_number: str
    diagnosis_codes: List[str]
    procedure_codes: List[str]
    total_charges: float


class BedAssignmentRequest(BaseModel):
    """Bed assignment request"""
    patient_id: str
    bed_id: str


class DischargeRequest(BaseModel):
    """Patient discharge request"""
    patient_id: str
    disposition: str


@router.post("/er/triage")
async def triage_patient(
    request: TriageRequest,
    user: UserContext = Depends(get_current_user),
):
    """Triage a patient arriving at ER"""
    patient = await hospital_service.triage_patient(
        patient_id=request.patient_id,
        chief_complaint=request.chief_complaint,
        symptoms=request.symptoms,
        vital_signs=request.vital_signs,
        age=request.age,
        medical_history=request.medical_history,
        user_id=user.user_id if user else None,
    )
    return patient.model_dump()


@router.get("/er/queue")
async def get_er_queue(
    user: UserContext = Depends(get_current_user),
):
    """Get current ER queue sorted by priority"""
    queue = hospital_service.get_er_queue()
    return {"queue": [p.model_dump() for p in queue]}


@router.get("/er/metrics")
async def get_er_metrics(
    user: UserContext = Depends(get_current_user),
):
    """Get ER performance metrics"""
    return hospital_service.get_er_metrics()


@router.post("/er/assign-bed")
async def assign_bed(
    request: BedAssignmentRequest,
    user: UserContext = Depends(get_current_user),
):
    """Assign a bed to an ER patient"""
    success = hospital_service.assign_bed(
        patient_id=request.patient_id,
        bed_id=request.bed_id,
        user_id=user.user_id if user else None,
    )
    return {"success": success}


@router.post("/er/discharge")
async def discharge_patient(
    request: DischargeRequest,
    user: UserContext = Depends(get_current_user),
):
    """Discharge an ER patient"""
    success = hospital_service.discharge_patient(
        patient_id=request.patient_id,
        disposition=request.disposition,
        user_id=user.user_id if user else None,
    )
    return {"success": success}


@router.post("/appointment")
async def schedule_appointment(
    request: AppointmentRequest,
    user: UserContext = Depends(get_current_user),
):
    """Schedule a patient appointment"""
    appointment = hospital_service.schedule_appointment(
        patient_id=request.patient_id,
        provider_id=request.provider_id,
        department=request.department,
        scheduled_time=request.scheduled_time,
        duration_minutes=request.duration_minutes,
        reason=request.reason,
        user_id=user.user_id if user else None,
    )
    return {"appointment_id": appointment.id, "scheduled_time": appointment.scheduled_time.isoformat()}


@router.get("/appointments")
async def get_appointments(
    patient_id: Optional[str] = None,
    provider_id: Optional[str] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    user: UserContext = Depends(get_current_user),
):
    """Get appointments with filters"""
    appointments = hospital_service.get_appointments(
        patient_id=patient_id,
        provider_id=provider_id,
        date_from=date_from,
        date_to=date_to,
    )
    return {"appointments": [a.model_dump() for a in appointments]}


@router.delete("/appointment/{appointment_id}")
async def cancel_appointment(
    appointment_id: str,
    reason: Optional[str] = None,
    user: UserContext = Depends(get_current_user),
):
    """Cancel an appointment"""
    success = hospital_service.cancel_appointment(
        appointment_id=appointment_id,
        reason=reason,
        user_id=user.user_id if user else None,
    )
    return {"success": success}


@router.post("/insurance/claim")
async def create_insurance_claim(
    request: InsuranceClaimRequest,
    user: UserContext = Depends(get_current_user),
):
    """Create an insurance claim"""
    claim = hospital_service.create_insurance_claim(
        patient_id=request.patient_id,
        encounter_id=request.encounter_id,
        insurance_provider=request.insurance_provider,
        policy_number=request.policy_number,
        diagnosis_codes=request.diagnosis_codes,
        procedure_codes=request.procedure_codes,
        total_charges=request.total_charges,
        user_id=user.user_id if user else None,
    )
    return {"claim_id": claim.id, "status": claim.status}


@router.post("/insurance/claim/{claim_id}/submit")
async def submit_claim(
    claim_id: str,
    user: UserContext = Depends(get_current_user),
):
    """Submit an insurance claim"""
    success = hospital_service.submit_claim(
        claim_id=claim_id,
        user_id=user.user_id if user else None,
    )
    return {"success": success}


@router.get("/insurance/claims")
async def get_claims(
    patient_id: Optional[str] = None,
    status: Optional[str] = None,
    user: UserContext = Depends(get_current_user),
):
    """Get insurance claims with filters"""
    claims = hospital_service.get_claims(
        patient_id=patient_id,
        status=status,
    )
    return {"claims": [c.model_dump() for c in claims]}


@router.get("/insurance/analytics")
async def get_claim_analytics(
    user: UserContext = Depends(get_current_user),
):
    """Get insurance claim analytics"""
    return hospital_service.get_claim_analytics()
