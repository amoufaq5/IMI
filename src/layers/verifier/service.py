"""
Verifier Service - Layer 4 Critic/Verification Layer

Orchestrates all verification checks:
- Hallucination detection
- Guideline compliance
- Confidence calibration
- Safety validation
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from src.core.security.audit import AuditLogger, get_audit_logger, AuditAction

from .hallucination_detector import HallucinationDetector, HallucinationResult
from .guideline_checker import GuidelineChecker, GuidelineCheckResult
from .confidence_calibrator import ConfidenceCalibrator, CalibrationResult


class VerificationResult(BaseModel):
    """Complete verification result"""
    is_verified: bool = True
    verification_score: float = Field(ge=0.0, le=1.0, default=1.0)
    
    hallucination_result: Optional[HallucinationResult] = None
    guideline_result: Optional[GuidelineCheckResult] = None
    calibration_result: Optional[CalibrationResult] = None
    
    blocking_issues: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    
    verified_at: datetime = Field(default_factory=datetime.utcnow)
    verification_time_ms: float = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_verified": self.is_verified,
            "verification_score": self.verification_score,
            "blocking_issues": self.blocking_issues,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
            "verified_at": self.verified_at.isoformat(),
            "verification_time_ms": self.verification_time_ms,
        }


class VerifierService:
    """
    Layer 4: Verifier Service
    
    The Critic Layer - ensures LLM outputs are:
    - Factually accurate (no hallucinations)
    - Guideline-compliant
    - Appropriately calibrated (not overconfident)
    - Safe for the intended use
    
    This layer acts as a second model checking the primary LLM's output.
    """
    
    VERIFICATION_THRESHOLD = 0.85
    
    def __init__(
        self,
        knowledge_graph=None,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self.hallucination_detector = HallucinationDetector(knowledge_graph)
        self.guideline_checker = GuidelineChecker(knowledge_graph)
        self.confidence_calibrator = ConfidenceCalibrator()
        self.audit = audit_logger or get_audit_logger()
        self.knowledge_graph = knowledge_graph
    
    async def verify(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        condition: Optional[str] = None,
        source_documents: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        strict_mode: bool = False,
    ) -> VerificationResult:
        """
        Perform complete verification of LLM output
        
        Args:
            text: The LLM-generated text to verify
            context: Original query context
            condition: Medical condition context
            source_documents: Source documents for attribution
            user_id: User ID for audit logging
            strict_mode: If True, apply stricter verification
        
        Returns:
            VerificationResult with all checks
        """
        import time
        start_time = time.time()
        
        result = VerificationResult()
        
        # Run all verification checks
        hallucination_result = await self.hallucination_detector.detect(
            text=text,
            context=context,
            source_documents=source_documents,
        )
        result.hallucination_result = hallucination_result
        
        guideline_result = await self.guideline_checker.check(
            text=text,
            condition=condition,
            patient_context=context,
        )
        result.guideline_result = guideline_result
        
        calibration_result = self.confidence_calibrator.calibrate(
            text=text,
            context_type=context.get("type", "general") if context else "general",
        )
        result.calibration_result = calibration_result
        
        # Aggregate results
        self._aggregate_results(result, strict_mode)
        
        result.verification_time_ms = (time.time() - start_time) * 1000
        
        # Log verification
        self.audit.log(
            action=AuditAction.VERIFICATION_CHECK,
            description="LLM output verification",
            user_id=user_id,
            details={
                "is_verified": result.is_verified,
                "verification_score": result.verification_score,
                "blocking_issues_count": len(result.blocking_issues),
                "warnings_count": len(result.warnings),
                "hallucinations_found": hallucination_result.has_hallucinations,
                "guideline_compliant": guideline_result.is_compliant,
                "well_calibrated": calibration_result.is_well_calibrated,
            },
        )
        
        return result
    
    def _aggregate_results(
        self,
        result: VerificationResult,
        strict_mode: bool,
    ) -> None:
        """Aggregate all verification results into final decision"""
        scores = []
        
        # Process hallucination results
        if result.hallucination_result:
            hr = result.hallucination_result
            scores.append(hr.overall_reliability_score)
            
            for h in hr.hallucinations:
                if h.severity.value in ["critical", "high"]:
                    result.blocking_issues.append(
                        f"Hallucination detected: {h.explanation}"
                    )
                    if h.suggested_correction:
                        result.suggestions.append(h.suggested_correction)
                else:
                    result.warnings.append(f"Potential issue: {h.explanation}")
        
        # Process guideline results
        if result.guideline_result:
            gr = result.guideline_result
            scores.append(gr.compliance_score)
            
            for conflict in gr.conflicts:
                if conflict.severity.value in ["critical", "major"]:
                    result.blocking_issues.append(
                        f"Guideline conflict ({conflict.guideline_source}): {conflict.conflict_description}"
                    )
                    if conflict.resolution_suggestion:
                        result.suggestions.append(conflict.resolution_suggestion)
                else:
                    result.warnings.append(
                        f"Guideline note: {conflict.conflict_description}"
                    )
        
        # Process calibration results
        if result.calibration_result:
            cr = result.calibration_result
            scores.append(cr.calibration_score)
            
            for issue in cr.overconfidence_issues:
                if issue.type.value in ["definitive_diagnosis", "absolute_recommendation"]:
                    result.warnings.append(
                        f"Overconfidence: {issue.explanation}"
                    )
                    result.suggestions.append(issue.suggested_revision)
        
        # Calculate overall score
        if scores:
            result.verification_score = sum(scores) / len(scores)
        
        # Determine if verified
        threshold = self.VERIFICATION_THRESHOLD if not strict_mode else 0.95
        result.is_verified = (
            result.verification_score >= threshold and
            len(result.blocking_issues) == 0
        )
    
    async def quick_verify(
        self,
        text: str,
        user_id: Optional[str] = None,
    ) -> bool:
        """Quick verification check - returns True/False only"""
        result = await self.verify(text=text, user_id=user_id)
        return result.is_verified
    
    async def verify_medication_recommendation(
        self,
        drug_name: str,
        indication: str,
        dosage: Optional[str] = None,
        patient_context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> VerificationResult:
        """Specialized verification for medication recommendations"""
        text = f"Recommending {drug_name} for {indication}"
        if dosage:
            text += f" at {dosage}"
        
        result = await self.verify(
            text=text,
            context=patient_context,
            user_id=user_id,
            strict_mode=True,
        )
        
        # Additional medication-specific checks
        if patient_context:
            conditions = patient_context.get("conditions", [])
            medications = patient_context.get("medications", [])
            
            # These would be checked against knowledge graph in production
            result.suggestions.append(
                f"Verify {drug_name} against patient's {len(conditions)} conditions"
            )
            if medications:
                result.suggestions.append(
                    f"Check interactions with {len(medications)} current medications"
                )
        
        return result
    
    async def verify_diagnosis_suggestion(
        self,
        symptoms: List[str],
        suggested_diagnosis: str,
        confidence_expressed: str,
        user_id: Optional[str] = None,
    ) -> VerificationResult:
        """Specialized verification for diagnosis suggestions"""
        text = f"Based on symptoms ({', '.join(symptoms)}), suggesting {suggested_diagnosis}"
        
        result = await self.verify(
            text=text,
            context={"symptoms": symptoms, "type": "diagnosis"},
            condition=suggested_diagnosis,
            user_id=user_id,
            strict_mode=True,
        )
        
        # Check confidence appropriateness
        overconfident_terms = ["definitely", "certainly", "is", "you have"]
        if any(term in confidence_expressed.lower() for term in overconfident_terms):
            result.warnings.append(
                "Diagnosis expressed with inappropriate certainty"
            )
            result.suggestions.append(
                "Use hedging language: 'symptoms are consistent with' or 'consider'"
            )
        
        return result
    
    def get_verification_summary(
        self,
        result: VerificationResult,
    ) -> str:
        """Generate human-readable verification summary"""
        lines = []
        
        if result.is_verified:
            lines.append("VERIFIED - Content passed all safety checks")
        else:
            lines.append("NOT VERIFIED - Issues detected")
        
        lines.append(f"Verification Score: {result.verification_score:.2%}")
        
        if result.blocking_issues:
            lines.append("\nBlocking Issues:")
            for issue in result.blocking_issues:
                lines.append(f"  - {issue}")
        
        if result.warnings:
            lines.append("\nWarnings:")
            for warning in result.warnings:
                lines.append(f"  - {warning}")
        
        if result.suggestions:
            lines.append("\nSuggestions:")
            for suggestion in result.suggestions:
                lines.append(f"  - {suggestion}")
        
        return "\n".join(lines)


_verifier_service: Optional[VerifierService] = None


async def get_verifier_service() -> VerifierService:
    """Get or create verifier service singleton"""
    global _verifier_service
    if _verifier_service is None:
        _verifier_service = VerifierService()
    return _verifier_service
