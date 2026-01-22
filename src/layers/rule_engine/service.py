"""
Rule Engine Service - Layer 2 Safety Layer

Orchestrates all deterministic safety logic:
- Triage assessment
- OTC eligibility
- Contraindication checking
- Red flag detection
"""
from typing import Optional, List, Dict, Any
from datetime import datetime

from src.core.security.audit import AuditLogger, get_audit_logger, AuditAction

from .triage import TriageEngine, TriageResult, PatientAssessment, TriageUrgency
from .otc_eligibility import OTCEligibilityEngine, OTCDecision
from .contraindication_checker import ContraindicationChecker, ContraindicationResult
from .red_flags import RedFlagDetector, RedFlagResult


class SafetyAssessment:
    """Complete safety assessment result"""
    
    def __init__(
        self,
        triage: Optional[TriageResult] = None,
        otc_decision: Optional[OTCDecision] = None,
        contraindications: Optional[ContraindicationResult] = None,
        red_flags: Optional[RedFlagResult] = None,
    ):
        self.triage = triage
        self.otc_decision = otc_decision
        self.contraindications = contraindications
        self.red_flags = red_flags
        self.timestamp = datetime.utcnow()
        self.is_safe_to_proceed = True
        self.blocking_issues: List[str] = []
        self.warnings: List[str] = []
        self.recommendations: List[str] = []
        
        self._evaluate_safety()
    
    def _evaluate_safety(self) -> None:
        """Evaluate overall safety based on all assessments"""
        # Check red flags
        if self.red_flags and self.red_flags.has_red_flags:
            from .red_flags import RedFlagSeverity
            if self.red_flags.highest_severity == RedFlagSeverity.CRITICAL:
                self.is_safe_to_proceed = False
                self.blocking_issues.append("Critical red flags detected - emergency referral required")
                if self.red_flags.immediate_action:
                    self.recommendations.append(self.red_flags.immediate_action)
        
        # Check triage
        if self.triage:
            if self.triage.urgency == TriageUrgency.EMERGENCY:
                self.is_safe_to_proceed = False
                self.blocking_issues.append("Emergency triage level - immediate medical attention required")
            elif self.triage.urgency == TriageUrgency.URGENT:
                self.warnings.append("Urgent triage level - see doctor within 24 hours")
            self.recommendations.extend(self.triage.recommendations)
        
        # Check contraindications
        if self.contraindications and not self.contraindications.is_safe:
            self.is_safe_to_proceed = False
            for contra in self.contraindications.contraindications:
                self.blocking_issues.append(
                    f"Contraindication: {contra.get('drug')} with {contra.get('condition')}"
                )
            self.recommendations.extend(
                [f"Consider alternative: {alt}" for alt in self.contraindications.alternatives]
            )
        
        # Check OTC eligibility
        if self.otc_decision:
            from .otc_eligibility import OTCDecisionType
            if self.otc_decision.decision == OTCDecisionType.REFER_TO_DOCTOR:
                self.warnings.append("OTC not appropriate - doctor referral recommended")
            elif self.otc_decision.decision == OTCDecisionType.ELIGIBLE_WITH_CAUTION:
                self.warnings.extend(self.otc_decision.warnings)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            "is_safe_to_proceed": self.is_safe_to_proceed,
            "blocking_issues": self.blocking_issues,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp.isoformat(),
            "triage": self.triage.model_dump() if self.triage else None,
            "otc_decision": self.otc_decision.model_dump() if self.otc_decision else None,
            "contraindications": self.contraindications.model_dump() if self.contraindications else None,
            "red_flags": self.red_flags.model_dump() if self.red_flags else None,
        }
    
    def get_explainability_report(self) -> Dict[str, Any]:
        """Generate explainability report for audit/review"""
        rules_triggered = []
        
        if self.triage:
            rules_triggered.extend(self.triage.rules_triggered)
        if self.otc_decision:
            rules_triggered.extend(self.otc_decision.rules_applied)
        if self.contraindications:
            rules_triggered.extend(self.contraindications.rules_triggered)
        if self.red_flags:
            rules_triggered.extend(self.red_flags.rules_triggered)
        
        return {
            "assessment_timestamp": self.timestamp.isoformat(),
            "rules_triggered": rules_triggered,
            "decision_path": {
                "red_flag_check": bool(self.red_flags and self.red_flags.has_red_flags),
                "triage_urgency": self.triage.urgency.value if self.triage else None,
                "contraindication_safe": self.contraindications.is_safe if self.contraindications else None,
                "otc_eligible": self.otc_decision.decision.value if self.otc_decision else None,
            },
            "final_decision": {
                "safe_to_proceed": self.is_safe_to_proceed,
                "blocking_issues_count": len(self.blocking_issues),
                "warnings_count": len(self.warnings),
            },
        }


class RuleEngineService:
    """
    Layer 2: Rule Engine Service
    
    The Safety Layer - provides deterministic medical reasoning:
    - If-then medical logic
    - ASMETHOD-style triage
    - OTC eligibility assessment
    - Contraindication checking
    - Red flag detection
    
    This layer NEVER allows the LLM to make safety decisions alone.
    All safety-critical decisions go through deterministic rules.
    """
    
    def __init__(
        self,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self.triage_engine = TriageEngine()
        self.otc_engine = OTCEligibilityEngine()
        self.contraindication_checker = ContraindicationChecker()
        self.red_flag_detector = RedFlagDetector()
        self.audit = audit_logger or get_audit_logger()
    
    def assess_patient(
        self,
        assessment: PatientAssessment,
        user_id: Optional[str] = None,
    ) -> SafetyAssessment:
        """
        Perform complete safety assessment for a patient
        
        This is the main entry point for patient safety evaluation.
        """
        # Detect red flags first
        red_flags = self.red_flag_detector.detect(
            symptoms=assessment.symptoms,
            chief_complaint=assessment.chief_complaint,
            age=assessment.age,
        )
        
        # Perform triage
        triage = self.triage_engine.assess(assessment)
        
        # Check OTC eligibility
        otc_decision = self.otc_engine.assess_eligibility(
            symptoms=assessment.symptoms,
            age=assessment.age,
            conditions=assessment.medical_conditions,
            current_medications=assessment.current_medications,
            is_pregnant=assessment.is_pregnant,
            is_breastfeeding=assessment.is_breastfeeding,
            allergies=assessment.allergies,
        )
        
        # Create safety assessment
        safety = SafetyAssessment(
            triage=triage,
            otc_decision=otc_decision,
            red_flags=red_flags,
        )
        
        # Log the assessment
        self.audit.log(
            action=AuditAction.RULE_ENGINE_EVALUATION,
            description="Patient safety assessment",
            user_id=user_id,
            details={
                "triage_urgency": triage.urgency.value,
                "red_flags_detected": red_flags.has_red_flags,
                "otc_eligible": otc_decision.decision.value,
                "is_safe": safety.is_safe_to_proceed,
            },
            contains_phi=True,
        )
        
        return safety
    
    def check_medication_safety(
        self,
        drug: str,
        patient_conditions: List[str],
        current_medications: List[str],
        allergies: List[str] = None,
        age: Optional[int] = None,
        is_pregnant: bool = False,
        is_breastfeeding: bool = False,
        user_id: Optional[str] = None,
    ) -> ContraindicationResult:
        """
        Check if a medication is safe for a patient
        
        Evaluates:
        - Drug-condition contraindications
        - Drug-drug interactions
        - Allergy checks
        - Age/pregnancy restrictions
        """
        result = self.contraindication_checker.check_all(
            drug=drug,
            conditions=patient_conditions,
            current_medications=current_medications,
            allergies=allergies or [],
            age=age,
            is_pregnant=is_pregnant,
            is_breastfeeding=is_breastfeeding,
        )
        
        # Log the check
        self.audit.log(
            action=AuditAction.RULE_ENGINE_EVALUATION,
            description="Medication safety check",
            user_id=user_id,
            details={
                "drug": drug,
                "is_safe": result.is_safe,
                "contraindications_count": len(result.contraindications),
                "interactions_count": len(result.interactions),
            },
        )
        
        return result
    
    def check_drug_interactions(
        self,
        drugs: List[str],
        user_id: Optional[str] = None,
    ) -> ContraindicationResult:
        """Check for interactions between multiple drugs"""
        result = self.contraindication_checker.check_drug_drug(drugs)
        
        self.audit.log(
            action=AuditAction.RULE_ENGINE_EVALUATION,
            description="Drug interaction check",
            user_id=user_id,
            details={
                "drugs_checked": len(drugs),
                "interactions_found": len(result.interactions),
            },
        )
        
        return result
    
    def detect_red_flags(
        self,
        symptoms: List[str],
        chief_complaint: Optional[str] = None,
        age: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> RedFlagResult:
        """Detect red flag symptoms"""
        result = self.red_flag_detector.detect(
            symptoms=symptoms,
            chief_complaint=chief_complaint,
            age=age,
        )
        
        if result.has_red_flags:
            self.audit.log(
                action=AuditAction.RULE_ENGINE_EVALUATION,
                description="Red flags detected",
                user_id=user_id,
                severity_override="warning",
                details={
                    "red_flags_count": len(result.red_flags),
                    "highest_severity": result.highest_severity.value if result.highest_severity else None,
                },
            )
        
        return result
    
    def get_triage_questions(self, chief_complaint: str) -> List[str]:
        """Get ASMETHOD assessment questions for a complaint"""
        return self.triage_engine.get_assessment_questions(chief_complaint)
    
    def get_otc_recommendation(
        self,
        symptoms: List[str],
        age: int,
        conditions: List[str],
        current_medications: List[str],
        is_pregnant: bool = False,
        is_breastfeeding: bool = False,
        allergies: List[str] = None,
    ) -> OTCDecision:
        """Get OTC medication recommendation"""
        return self.otc_engine.assess_eligibility(
            symptoms=symptoms,
            age=age,
            conditions=conditions,
            current_medications=current_medications,
            is_pregnant=is_pregnant,
            is_breastfeeding=is_breastfeeding,
            allergies=allergies,
        )
    
    def validate_llm_recommendation(
        self,
        recommendation: Dict[str, Any],
        patient_context: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate an LLM-generated recommendation against safety rules
        
        This is a critical function - the LLM's recommendations must
        pass through the rule engine before being presented to users.
        """
        validation_result = {
            "is_valid": True,
            "blocked_recommendations": [],
            "modified_recommendations": [],
            "warnings": [],
            "original_recommendation": recommendation,
        }
        
        # Extract recommended medications if any
        recommended_drugs = recommendation.get("medications", [])
        
        for drug in recommended_drugs:
            drug_name = drug.get("name") if isinstance(drug, dict) else drug
            
            # Check safety
            safety_check = self.check_medication_safety(
                drug=drug_name,
                patient_conditions=patient_context.get("conditions", []),
                current_medications=patient_context.get("current_medications", []),
                allergies=patient_context.get("allergies", []),
                age=patient_context.get("age"),
                is_pregnant=patient_context.get("is_pregnant", False),
                is_breastfeeding=patient_context.get("is_breastfeeding", False),
                user_id=user_id,
            )
            
            if not safety_check.is_safe:
                validation_result["is_valid"] = False
                validation_result["blocked_recommendations"].append({
                    "drug": drug_name,
                    "reason": safety_check.contraindications,
                    "alternatives": safety_check.alternatives,
                })
            elif safety_check.warnings:
                validation_result["warnings"].extend(safety_check.warnings)
        
        # Log validation
        self.audit.log(
            action=AuditAction.VERIFICATION_CHECK,
            description="LLM recommendation validation",
            user_id=user_id,
            details={
                "is_valid": validation_result["is_valid"],
                "blocked_count": len(validation_result["blocked_recommendations"]),
                "warnings_count": len(validation_result["warnings"]),
            },
        )
        
        return validation_result


# Singleton
_rule_engine: Optional[RuleEngineService] = None


def get_rule_engine_service() -> RuleEngineService:
    """Get or create rule engine service singleton"""
    global _rule_engine
    if _rule_engine is None:
        _rule_engine = RuleEngineService()
    return _rule_engine
