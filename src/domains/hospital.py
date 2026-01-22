"""
Hospital Domain Service

Provides hospital-focused functionality:
- ER triage optimization
- Patient flow management
- Insurance processing
- Booking and scheduling
"""
from typing import Optional, List, Dict, Any
from datetime import datetime, date, timedelta
from enum import Enum
from pydantic import BaseModel, Field

from src.layers.rule_engine import RuleEngineService, get_rule_engine_service
from src.layers.rule_engine.triage import PatientAssessment, TriageUrgency
from src.layers.memory import MemoryService
from src.core.security.audit import AuditLogger, get_audit_logger, AuditAction


class ERPriority(str, Enum):
    """ER priority levels"""
    IMMEDIATE = "immediate"  # Life-threatening
    EMERGENT = "emergent"    # High risk
    URGENT = "urgent"        # Serious
    LESS_URGENT = "less_urgent"
    NON_URGENT = "non_urgent"


class BedStatus(str, Enum):
    """Hospital bed status"""
    AVAILABLE = "available"
    OCCUPIED = "occupied"
    CLEANING = "cleaning"
    MAINTENANCE = "maintenance"
    RESERVED = "reserved"


class ERPatient(BaseModel):
    """ER patient record"""
    id: str
    arrival_time: datetime
    chief_complaint: str
    priority: ERPriority
    triage_score: int = Field(ge=1, le=5)
    assigned_bed: Optional[str] = None
    assigned_provider: Optional[str] = None
    status: str = "waiting"  # waiting, in_treatment, admitted, discharged
    wait_time_minutes: int = 0
    notes: List[str] = Field(default_factory=list)


class Appointment(BaseModel):
    """Patient appointment"""
    id: str
    patient_id: str
    provider_id: str
    department: str
    scheduled_time: datetime
    duration_minutes: int = 30
    status: str = "scheduled"  # scheduled, confirmed, checked_in, completed, cancelled
    reason: Optional[str] = None
    notes: Optional[str] = None


class InsuranceClaim(BaseModel):
    """Insurance claim"""
    id: str
    patient_id: str
    encounter_id: str
    insurance_provider: str
    policy_number: str
    diagnosis_codes: List[str]
    procedure_codes: List[str]
    total_charges: float
    status: str = "pending"  # pending, submitted, approved, denied, appealed
    submitted_date: Optional[date] = None
    response_date: Optional[date] = None
    approved_amount: Optional[float] = None
    denial_reason: Optional[str] = None


class HospitalService:
    """
    Hospital operations support service
    
    Capabilities:
    - ER queue management and triage
    - Patient flow optimization
    - Appointment scheduling
    - Insurance claim processing
    - Bed management
    """
    
    # ESI (Emergency Severity Index) criteria
    ESI_CRITERIA = {
        1: {
            "description": "Immediate life-saving intervention required",
            "examples": ["Cardiac arrest", "Respiratory failure", "Severe trauma"],
            "target_time": "Immediate",
        },
        2: {
            "description": "High risk situation or confused/lethargic/disoriented",
            "examples": ["Chest pain", "Stroke symptoms", "Severe pain"],
            "target_time": "10 minutes",
        },
        3: {
            "description": "Stable but needs multiple resources",
            "examples": ["Abdominal pain", "Moderate asthma", "Lacerations"],
            "target_time": "30 minutes",
        },
        4: {
            "description": "Stable, needs one resource",
            "examples": ["Simple laceration", "Urinary symptoms", "Minor injury"],
            "target_time": "60 minutes",
        },
        5: {
            "description": "Stable, needs no resources",
            "examples": ["Prescription refill", "Minor cold symptoms"],
            "target_time": "120 minutes",
        },
    }
    
    def __init__(
        self,
        rule_engine: Optional[RuleEngineService] = None,
        memory_service: Optional[MemoryService] = None,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self.rule_engine = rule_engine or get_rule_engine_service()
        self.memory = memory_service
        self.audit = audit_logger or get_audit_logger()
        
        # In-memory storage for demo
        self._er_queue: Dict[str, ERPatient] = {}
        self._appointments: Dict[str, Appointment] = {}
        self._claims: Dict[str, InsuranceClaim] = {}
        self._beds: Dict[str, BedStatus] = {}
    
    # ER Management
    async def triage_patient(
        self,
        patient_id: str,
        chief_complaint: str,
        symptoms: List[str],
        vital_signs: Dict[str, Any],
        age: int,
        medical_history: List[str] = None,
        user_id: Optional[str] = None,
    ) -> ERPatient:
        """
        Triage a patient arriving at ER
        
        Uses rule engine for safety assessment and ESI scoring.
        """
        # Create assessment for rule engine
        assessment = PatientAssessment(
            age=age,
            chief_complaint=chief_complaint,
            symptoms=symptoms,
            medical_conditions=medical_history or [],
            temperature_f=vital_signs.get("temperature"),
            heart_rate=vital_signs.get("heart_rate"),
            blood_pressure_systolic=vital_signs.get("bp_systolic"),
            blood_pressure_diastolic=vital_signs.get("bp_diastolic"),
            respiratory_rate=vital_signs.get("respiratory_rate"),
            oxygen_saturation=vital_signs.get("oxygen_saturation"),
        )
        
        # Get triage result from rule engine
        safety = self.rule_engine.assess_patient(assessment, user_id)
        triage = safety.triage
        
        # Map to ESI score
        esi_score = self._map_to_esi(triage.urgency)
        priority = self._esi_to_priority(esi_score)
        
        # Create ER patient record
        import uuid
        er_patient = ERPatient(
            id=str(uuid.uuid4()),
            arrival_time=datetime.utcnow(),
            chief_complaint=chief_complaint,
            priority=priority,
            triage_score=esi_score,
        )
        
        # Add to queue
        self._er_queue[er_patient.id] = er_patient
        
        # Log triage
        self.audit.log(
            action=AuditAction.CREATE,
            description="ER patient triaged",
            user_id=user_id,
            resource_type="er_patient",
            resource_id=er_patient.id,
            details={
                "esi_score": esi_score,
                "priority": priority.value,
                "red_flags": triage.red_flags_detected,
            },
            contains_phi=True,
        )
        
        return er_patient
    
    def _map_to_esi(self, urgency: TriageUrgency) -> int:
        """Map triage urgency to ESI score"""
        mapping = {
            TriageUrgency.EMERGENCY: 1,
            TriageUrgency.URGENT: 2,
            TriageUrgency.SEMI_URGENT: 3,
            TriageUrgency.ROUTINE: 4,
            TriageUrgency.SELF_CARE: 5,
        }
        return mapping.get(urgency, 3)
    
    def _esi_to_priority(self, esi_score: int) -> ERPriority:
        """Map ESI score to ER priority"""
        mapping = {
            1: ERPriority.IMMEDIATE,
            2: ERPriority.EMERGENT,
            3: ERPriority.URGENT,
            4: ERPriority.LESS_URGENT,
            5: ERPriority.NON_URGENT,
        }
        return mapping.get(esi_score, ERPriority.URGENT)
    
    def get_er_queue(self) -> List[ERPatient]:
        """Get current ER queue sorted by priority"""
        patients = list(self._er_queue.values())
        
        # Update wait times
        now = datetime.utcnow()
        for p in patients:
            if p.status == "waiting":
                p.wait_time_minutes = int((now - p.arrival_time).total_seconds() / 60)
        
        # Sort by ESI score (lower is more urgent), then by arrival time
        return sorted(patients, key=lambda p: (p.triage_score, p.arrival_time))
    
    def get_er_metrics(self) -> Dict[str, Any]:
        """Get ER performance metrics"""
        patients = list(self._er_queue.values())
        waiting = [p for p in patients if p.status == "waiting"]
        
        return {
            "total_patients": len(patients),
            "waiting_count": len(waiting),
            "in_treatment": len([p for p in patients if p.status == "in_treatment"]),
            "average_wait_minutes": (
                sum(p.wait_time_minutes for p in waiting) / len(waiting)
                if waiting else 0
            ),
            "by_priority": {
                priority.value: len([p for p in patients if p.priority == priority])
                for priority in ERPriority
            },
            "longest_wait_minutes": max((p.wait_time_minutes for p in waiting), default=0),
        }
    
    def assign_bed(
        self,
        patient_id: str,
        bed_id: str,
        user_id: Optional[str] = None,
    ) -> bool:
        """Assign a bed to an ER patient"""
        patient = self._er_queue.get(patient_id)
        if not patient:
            return False
        
        if self._beds.get(bed_id) != BedStatus.AVAILABLE:
            return False
        
        patient.assigned_bed = bed_id
        patient.status = "in_treatment"
        self._beds[bed_id] = BedStatus.OCCUPIED
        
        self.audit.log(
            action=AuditAction.UPDATE,
            description="Bed assigned to patient",
            user_id=user_id,
            resource_type="er_patient",
            resource_id=patient_id,
            details={"bed_id": bed_id},
        )
        
        return True
    
    def discharge_patient(
        self,
        patient_id: str,
        disposition: str,
        user_id: Optional[str] = None,
    ) -> bool:
        """Discharge an ER patient"""
        patient = self._er_queue.get(patient_id)
        if not patient:
            return False
        
        # Free up bed
        if patient.assigned_bed:
            self._beds[patient.assigned_bed] = BedStatus.CLEANING
        
        patient.status = "discharged"
        
        self.audit.log(
            action=AuditAction.UPDATE,
            description="Patient discharged",
            user_id=user_id,
            resource_type="er_patient",
            resource_id=patient_id,
            details={"disposition": disposition},
        )
        
        return True
    
    # Appointment Scheduling
    def schedule_appointment(
        self,
        patient_id: str,
        provider_id: str,
        department: str,
        scheduled_time: datetime,
        duration_minutes: int = 30,
        reason: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Appointment:
        """Schedule a patient appointment"""
        import uuid
        appointment = Appointment(
            id=str(uuid.uuid4()),
            patient_id=patient_id,
            provider_id=provider_id,
            department=department,
            scheduled_time=scheduled_time,
            duration_minutes=duration_minutes,
            reason=reason,
        )
        
        self._appointments[appointment.id] = appointment
        
        self.audit.log(
            action=AuditAction.CREATE,
            description="Appointment scheduled",
            user_id=user_id,
            resource_type="appointment",
            resource_id=appointment.id,
            details={
                "department": department,
                "scheduled_time": scheduled_time.isoformat(),
            },
        )
        
        return appointment
    
    def get_appointments(
        self,
        patient_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
    ) -> List[Appointment]:
        """Get appointments with filters"""
        appointments = list(self._appointments.values())
        
        if patient_id:
            appointments = [a for a in appointments if a.patient_id == patient_id]
        if provider_id:
            appointments = [a for a in appointments if a.provider_id == provider_id]
        if date_from:
            appointments = [a for a in appointments if a.scheduled_time.date() >= date_from]
        if date_to:
            appointments = [a for a in appointments if a.scheduled_time.date() <= date_to]
        
        return sorted(appointments, key=lambda a: a.scheduled_time)
    
    def cancel_appointment(
        self,
        appointment_id: str,
        reason: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """Cancel an appointment"""
        appointment = self._appointments.get(appointment_id)
        if not appointment:
            return False
        
        appointment.status = "cancelled"
        appointment.notes = reason
        
        self.audit.log(
            action=AuditAction.UPDATE,
            description="Appointment cancelled",
            user_id=user_id,
            resource_type="appointment",
            resource_id=appointment_id,
        )
        
        return True
    
    # Insurance Processing
    def create_insurance_claim(
        self,
        patient_id: str,
        encounter_id: str,
        insurance_provider: str,
        policy_number: str,
        diagnosis_codes: List[str],
        procedure_codes: List[str],
        total_charges: float,
        user_id: Optional[str] = None,
    ) -> InsuranceClaim:
        """Create an insurance claim"""
        import uuid
        claim = InsuranceClaim(
            id=str(uuid.uuid4()),
            patient_id=patient_id,
            encounter_id=encounter_id,
            insurance_provider=insurance_provider,
            policy_number=policy_number,
            diagnosis_codes=diagnosis_codes,
            procedure_codes=procedure_codes,
            total_charges=total_charges,
        )
        
        self._claims[claim.id] = claim
        
        self.audit.log(
            action=AuditAction.CREATE,
            description="Insurance claim created",
            user_id=user_id,
            resource_type="insurance_claim",
            resource_id=claim.id,
            details={
                "total_charges": total_charges,
                "insurance_provider": insurance_provider,
            },
        )
        
        return claim
    
    def submit_claim(
        self,
        claim_id: str,
        user_id: Optional[str] = None,
    ) -> bool:
        """Submit an insurance claim"""
        claim = self._claims.get(claim_id)
        if not claim:
            return False
        
        claim.status = "submitted"
        claim.submitted_date = date.today()
        
        self.audit.log(
            action=AuditAction.UPDATE,
            description="Insurance claim submitted",
            user_id=user_id,
            resource_type="insurance_claim",
            resource_id=claim_id,
        )
        
        return True
    
    def get_claims(
        self,
        patient_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[InsuranceClaim]:
        """Get insurance claims with filters"""
        claims = list(self._claims.values())
        
        if patient_id:
            claims = [c for c in claims if c.patient_id == patient_id]
        if status:
            claims = [c for c in claims if c.status == status]
        
        return claims
    
    def get_claim_analytics(self) -> Dict[str, Any]:
        """Get insurance claim analytics"""
        claims = list(self._claims.values())
        
        return {
            "total_claims": len(claims),
            "by_status": {
                status: len([c for c in claims if c.status == status])
                for status in ["pending", "submitted", "approved", "denied"]
            },
            "total_charges": sum(c.total_charges for c in claims),
            "total_approved": sum(c.approved_amount or 0 for c in claims),
            "approval_rate": (
                len([c for c in claims if c.status == "approved"]) / len(claims)
                if claims else 0
            ),
        }
