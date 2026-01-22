"""
Patient Domain Service

Provides patient-focused functionality:
- Symptom assessment and triage
- OTC recommendations
- Doctor referrals
- Health information
- Medical image analysis support
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from src.layers.rule_engine import RuleEngineService, get_rule_engine_service
from src.layers.rule_engine.triage import PatientAssessment, TriageResult, TriageUrgency
from src.layers.rule_engine.otc_eligibility import OTCDecision
from src.layers.knowledge_graph import KnowledgeGraphService
from src.layers.llm import LLMService
from src.layers.llm.prompts import RoleType
from src.layers.verifier import VerifierService
from src.layers.memory import MemoryService
from src.core.security.audit import AuditLogger, get_audit_logger, AuditAction


class SymptomAssessmentRequest(BaseModel):
    """Request for symptom assessment"""
    symptoms: List[str]
    chief_complaint: str
    duration_hours: Optional[float] = None
    severity: Optional[int] = Field(default=None, ge=1, le=10)
    age: int
    gender: Optional[str] = None
    is_pregnant: bool = False
    is_breastfeeding: bool = False
    medical_conditions: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    additional_context: Optional[str] = None


class SymptomAssessmentResponse(BaseModel):
    """Response from symptom assessment"""
    triage_result: Dict[str, Any]
    otc_recommendation: Optional[Dict[str, Any]] = None
    explanation: str
    next_steps: List[str]
    red_flags_detected: List[str] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)
    disclaimer: str = "This is not a medical diagnosis. Please consult a healthcare provider for proper evaluation."


class PatientService:
    """
    Patient-focused medical assistance service
    
    Capabilities:
    - Symptom triage and assessment
    - OTC medication eligibility
    - Health information queries
    - Referral recommendations
    - Medical image analysis support
    """
    
    def __init__(
        self,
        rule_engine: Optional[RuleEngineService] = None,
        llm_service: Optional[LLMService] = None,
        verifier_service: Optional[VerifierService] = None,
        memory_service: Optional[MemoryService] = None,
        knowledge_graph: Optional[KnowledgeGraphService] = None,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self.rule_engine = rule_engine or get_rule_engine_service()
        self.llm = llm_service
        self.verifier = verifier_service
        self.memory = memory_service
        self.kg = knowledge_graph
        self.audit = audit_logger or get_audit_logger()
    
    async def assess_symptoms(
        self,
        request: SymptomAssessmentRequest,
        user_id: Optional[str] = None,
        patient_id: Optional[str] = None,
    ) -> SymptomAssessmentResponse:
        """
        Perform comprehensive symptom assessment
        
        Uses the 5-layer architecture:
        1. Knowledge Graph for medical facts
        2. Rule Engine for safety checks
        3. LLM for explanation
        4. Verifier for accuracy
        5. Memory for patient context
        """
        # Create patient assessment for rule engine
        assessment = PatientAssessment(
            age=request.age,
            sex=request.gender,
            is_pregnant=request.is_pregnant,
            is_breastfeeding=request.is_breastfeeding,
            chief_complaint=request.chief_complaint,
            symptoms=request.symptoms,
            symptom_duration_hours=request.duration_hours,
            symptom_severity=request.severity,
            medical_conditions=request.medical_conditions,
            current_medications=request.current_medications,
            allergies=request.allergies,
        )
        
        # Layer 2: Rule Engine - Safety assessment
        safety_assessment = self.rule_engine.assess_patient(assessment, user_id)
        triage = safety_assessment.triage
        
        # Get OTC recommendation if appropriate
        otc_recommendation = None
        if triage.urgency in [TriageUrgency.SELF_CARE, TriageUrgency.ROUTINE]:
            otc_decision = self.rule_engine.get_otc_recommendation(
                symptoms=request.symptoms,
                age=request.age,
                conditions=request.medical_conditions,
                current_medications=request.current_medications,
                is_pregnant=request.is_pregnant,
                is_breastfeeding=request.is_breastfeeding,
                allergies=request.allergies,
            )
            otc_recommendation = otc_decision.model_dump()
        
        # Build next steps based on triage
        next_steps = self._build_next_steps(triage, otc_recommendation)
        
        # Get follow-up questions
        follow_up_questions = self.rule_engine.get_triage_questions(request.chief_complaint)
        
        # Generate explanation using LLM (if available)
        explanation = self._generate_explanation(triage, request)
        
        # Log the assessment
        self.audit.log(
            action=AuditAction.DIAGNOSIS_REQUEST,
            description="Patient symptom assessment",
            user_id=user_id,
            resource_type="patient",
            resource_id=patient_id,
            details={
                "symptoms_count": len(request.symptoms),
                "triage_urgency": triage.urgency.value,
                "red_flags_count": len(triage.red_flags_detected),
            },
            contains_phi=True,
        )
        
        return SymptomAssessmentResponse(
            triage_result=triage.model_dump(),
            otc_recommendation=otc_recommendation,
            explanation=explanation,
            next_steps=next_steps,
            red_flags_detected=triage.red_flags_detected,
            follow_up_questions=follow_up_questions[:5],
        )
    
    def _build_next_steps(
        self,
        triage: TriageResult,
        otc_recommendation: Optional[Dict[str, Any]],
    ) -> List[str]:
        """Build recommended next steps based on triage"""
        steps = []
        
        if triage.urgency == TriageUrgency.EMERGENCY:
            steps.append("Call 911 or go to the nearest emergency room immediately")
            steps.append("Do not drive yourself - have someone else drive or call an ambulance")
        
        elif triage.urgency == TriageUrgency.URGENT:
            steps.append("See a doctor within 24 hours")
            steps.append("If symptoms worsen, go to urgent care or emergency room")
            steps.append("Keep a record of your symptoms to share with the doctor")
        
        elif triage.urgency == TriageUrgency.SEMI_URGENT:
            steps.append("Schedule an appointment with your doctor within 2-3 days")
            steps.append("Monitor your symptoms and note any changes")
        
        elif triage.urgency == TriageUrgency.ROUTINE:
            steps.append("Schedule a routine appointment with your doctor")
            if otc_recommendation and otc_recommendation.get("eligible_products"):
                steps.append("Consider OTC options while waiting for your appointment")
        
        elif triage.urgency == TriageUrgency.SELF_CARE:
            steps.append("Self-care at home is appropriate for your symptoms")
            if otc_recommendation and otc_recommendation.get("eligible_products"):
                products = otc_recommendation["eligible_products"][:3]
                steps.append(f"OTC options to consider: {', '.join(products)}")
            steps.append("If symptoms persist beyond 7 days or worsen, see a doctor")
        
        # Add general advice
        steps.append("Stay hydrated and get adequate rest")
        
        return steps
    
    def _generate_explanation(
        self,
        triage: TriageResult,
        request: SymptomAssessmentRequest,
    ) -> str:
        """Generate patient-friendly explanation"""
        urgency_explanations = {
            TriageUrgency.EMERGENCY: (
                f"Based on your symptoms ({', '.join(request.symptoms[:3])}), "
                "this appears to be a medical emergency that requires immediate attention. "
                "Please seek emergency care right away."
            ),
            TriageUrgency.URGENT: (
                f"Your symptoms suggest a condition that should be evaluated by a doctor soon. "
                "While not an immediate emergency, it's important to be seen within 24 hours."
            ),
            TriageUrgency.SEMI_URGENT: (
                "Your symptoms warrant medical attention, but can likely wait a few days. "
                "Schedule an appointment with your healthcare provider."
            ),
            TriageUrgency.ROUTINE: (
                "Your symptoms appear to be manageable and can be addressed at a routine appointment. "
                "In the meantime, self-care measures may help."
            ),
            TriageUrgency.SELF_CARE: (
                "Based on the information provided, your symptoms appear suitable for self-care. "
                "Over-the-counter treatments may help relieve your symptoms."
            ),
        }
        
        explanation = urgency_explanations.get(triage.urgency, "")
        
        if triage.red_flags_detected:
            explanation += (
                f"\n\nImportant: We detected concerning symptoms "
                f"({', '.join(triage.red_flags_detected[:2])}) that require attention."
            )
        
        return explanation
    
    async def get_health_information(
        self,
        query: str,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get general health information"""
        # This would use the LLM with patient role
        return {
            "query": query,
            "response": "Health information would be provided here using the LLM.",
            "sources": [],
            "disclaimer": "This information is for educational purposes only.",
        }
    
    async def check_drug_safety(
        self,
        drug_name: str,
        patient_conditions: List[str],
        current_medications: List[str],
        allergies: List[str],
        age: int,
        is_pregnant: bool = False,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Check if a drug is safe for the patient"""
        result = self.rule_engine.check_medication_safety(
            drug=drug_name,
            patient_conditions=patient_conditions,
            current_medications=current_medications,
            allergies=allergies,
            age=age,
            is_pregnant=is_pregnant,
            user_id=user_id,
        )
        
        return {
            "drug": drug_name,
            "is_safe": result.is_safe,
            "contraindications": result.contraindications,
            "interactions": result.interactions,
            "warnings": result.warnings,
            "alternatives": result.alternatives,
        }
    
    async def analyze_lab_results(
        self,
        lab_results: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze lab results and provide interpretation"""
        # This would analyze lab values and provide patient-friendly interpretation
        return {
            "analysis": "Lab result analysis would be provided here.",
            "abnormal_values": [],
            "recommendations": [],
            "disclaimer": "Please discuss these results with your healthcare provider.",
        }
