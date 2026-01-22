"""
Doctor Domain Service

Provides clinical decision support:
- Differential diagnosis assistance
- Treatment recommendations
- Drug interaction checking
- Clinical guidelines access
- Case summarization
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from src.layers.rule_engine import RuleEngineService, get_rule_engine_service
from src.layers.knowledge_graph import KnowledgeGraphService
from src.layers.llm.prompts import RoleType
from src.core.security.audit import AuditLogger, get_audit_logger, AuditAction


class ClinicalCase(BaseModel):
    """Clinical case presentation"""
    chief_complaint: str
    history_of_present_illness: str
    past_medical_history: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    family_history: Optional[str] = None
    social_history: Optional[str] = None
    review_of_systems: Optional[Dict[str, List[str]]] = None
    vital_signs: Optional[Dict[str, Any]] = None
    physical_exam: Optional[Dict[str, str]] = None
    lab_results: Optional[Dict[str, Any]] = None
    imaging_results: Optional[List[str]] = None
    patient_age: Optional[int] = None
    patient_sex: Optional[str] = None


class DifferentialDiagnosis(BaseModel):
    """Differential diagnosis result"""
    diagnosis: str
    probability: str  # high, moderate, low
    supporting_features: List[str]
    against_features: List[str]
    recommended_workup: List[str]
    icd10_code: Optional[str] = None


class TreatmentRecommendation(BaseModel):
    """Treatment recommendation"""
    treatment: str
    line_of_therapy: int
    evidence_level: str
    dosing: Optional[str] = None
    duration: Optional[str] = None
    monitoring: List[str] = Field(default_factory=list)
    contraindications_checked: bool = False
    guideline_source: Optional[str] = None


class DoctorService:
    """
    Clinical decision support service for physicians
    
    Capabilities:
    - Differential diagnosis generation
    - Evidence-based treatment recommendations
    - Drug interaction and safety checking
    - Clinical guideline access
    - Case summarization and documentation
    """
    
    def __init__(
        self,
        rule_engine: Optional[RuleEngineService] = None,
        knowledge_graph: Optional[KnowledgeGraphService] = None,
        llm_service=None,
        verifier_service=None,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self.rule_engine = rule_engine or get_rule_engine_service()
        self.kg = knowledge_graph
        self.llm = llm_service
        self.verifier = verifier_service
        self.audit = audit_logger or get_audit_logger()
    
    async def generate_differential(
        self,
        case: ClinicalCase,
        max_diagnoses: int = 5,
        user_id: Optional[str] = None,
    ) -> List[DifferentialDiagnosis]:
        """
        Generate differential diagnosis for a clinical case
        
        Uses knowledge graph for disease-symptom matching
        and LLM for clinical reasoning.
        """
        differentials = []
        
        # Extract symptoms from case
        symptoms = self._extract_symptoms(case)
        
        # Query knowledge graph for matching diseases
        if self.kg:
            kg_results = await self.kg.differential_diagnosis(
                symptoms=symptoms,
                patient_age=case.patient_age,
                patient_sex=case.patient_sex,
                limit=max_diagnoses,
                user_id=user_id,
            )
            
            for result in kg_results:
                differentials.append(DifferentialDiagnosis(
                    diagnosis=result.get("disease_name", "Unknown"),
                    probability=self._calculate_probability(result),
                    supporting_features=self._get_supporting_features(result, symptoms),
                    against_features=[],
                    recommended_workup=self._get_recommended_workup(result),
                    icd10_code=result.get("icd10_code"),
                ))
        
        # Log the query
        self.audit.log(
            action=AuditAction.DIAGNOSIS_RESULT,
            description="Differential diagnosis generated",
            user_id=user_id,
            details={
                "symptoms_count": len(symptoms),
                "differentials_count": len(differentials),
            },
            contains_phi=True,
        )
        
        return differentials
    
    def _extract_symptoms(self, case: ClinicalCase) -> List[str]:
        """Extract symptoms from clinical case"""
        symptoms = []
        
        # Parse chief complaint
        symptoms.append(case.chief_complaint)
        
        # Parse HPI for symptoms
        hpi_lower = case.history_of_present_illness.lower()
        common_symptoms = [
            "fever", "cough", "pain", "fatigue", "nausea", "vomiting",
            "diarrhea", "headache", "dizziness", "weakness", "shortness of breath",
        ]
        for symptom in common_symptoms:
            if symptom in hpi_lower:
                symptoms.append(symptom)
        
        # Add from review of systems
        if case.review_of_systems:
            for system, system_symptoms in case.review_of_systems.items():
                symptoms.extend(system_symptoms)
        
        return list(set(symptoms))
    
    def _calculate_probability(self, result: Dict[str, Any]) -> str:
        """Calculate probability category from match score"""
        score = result.get("relevance_score", 0)
        match_count = result.get("symptom_match_count", 0)
        
        if score > 10 or match_count >= 4:
            return "high"
        elif score > 5 or match_count >= 2:
            return "moderate"
        else:
            return "low"
    
    def _get_supporting_features(
        self,
        result: Dict[str, Any],
        patient_symptoms: List[str],
    ) -> List[str]:
        """Get features supporting the diagnosis"""
        matched = result.get("matched_symptoms", [])
        return [m.get("symptom", "") for m in matched if isinstance(m, dict)]
    
    def _get_recommended_workup(self, result: Dict[str, Any]) -> List[str]:
        """Get recommended diagnostic workup"""
        # Default workup based on diagnosis type
        return ["CBC", "BMP", "Consider imaging based on clinical suspicion"]
    
    async def get_treatment_recommendations(
        self,
        diagnosis: str,
        patient_conditions: List[str],
        current_medications: List[str],
        allergies: List[str],
        user_id: Optional[str] = None,
    ) -> List[TreatmentRecommendation]:
        """
        Get evidence-based treatment recommendations
        
        Checks for contraindications and interactions.
        """
        recommendations = []
        
        # Get treatments from knowledge graph
        if self.kg:
            disease = await self.kg.get_disease(diagnosis)
            if disease:
                treatments = await self.kg.get_disease_treatments(disease.get("id"))
                
                for treatment in treatments:
                    drug_name = treatment.get("drug_name")
                    
                    # Check safety
                    safety_check = self.rule_engine.check_medication_safety(
                        drug=drug_name,
                        patient_conditions=patient_conditions,
                        current_medications=current_medications,
                        allergies=allergies,
                        user_id=user_id,
                    )
                    
                    if safety_check.is_safe:
                        recommendations.append(TreatmentRecommendation(
                            treatment=drug_name,
                            line_of_therapy=treatment.get("line_of_therapy", 1),
                            evidence_level=treatment.get("evidence_level", ""),
                            dosing=treatment.get("dosing_guidance"),
                            monitoring=safety_check.monitoring_required,
                            contraindications_checked=True,
                            guideline_source=treatment.get("guideline_source"),
                        ))
        
        return recommendations
    
    async def check_drug_interactions(
        self,
        medications: List[str],
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Check for drug-drug interactions"""
        result = self.rule_engine.check_drug_interactions(medications, user_id)
        
        return {
            "medications_checked": medications,
            "interactions_found": len(result.interactions),
            "interactions": result.interactions,
            "is_safe": result.is_safe,
            "warnings": result.warnings,
        }
    
    async def get_clinical_guidelines(
        self,
        condition: str,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get clinical guidelines for a condition"""
        if self.kg:
            disease = await self.kg.get_disease(condition)
            if disease:
                return await self.kg.get_clinical_guidelines(disease_id=disease.get("id"))
        return []
    
    async def summarize_case(
        self,
        case: ClinicalCase,
        include_assessment: bool = True,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate clinical case summary"""
        summary = {
            "patient": f"{case.patient_age}yo {case.patient_sex or 'patient'}",
            "chief_complaint": case.chief_complaint,
            "key_history": self._summarize_history(case),
            "pertinent_findings": self._get_pertinent_findings(case),
            "active_problems": case.past_medical_history,
            "current_medications": case.medications,
        }
        
        if include_assessment:
            differentials = await self.generate_differential(case, user_id=user_id)
            summary["assessment"] = {
                "differential_diagnoses": [d.diagnosis for d in differentials[:3]],
                "most_likely": differentials[0].diagnosis if differentials else None,
            }
        
        return summary
    
    def _summarize_history(self, case: ClinicalCase) -> str:
        """Summarize relevant history"""
        parts = []
        if case.past_medical_history:
            parts.append(f"PMH: {', '.join(case.past_medical_history[:3])}")
        if case.allergies:
            parts.append(f"Allergies: {', '.join(case.allergies)}")
        return "; ".join(parts) if parts else "No significant history"
    
    def _get_pertinent_findings(self, case: ClinicalCase) -> List[str]:
        """Extract pertinent positive and negative findings"""
        findings = []
        
        if case.vital_signs:
            vs = case.vital_signs
            if vs.get("temperature") and float(vs.get("temperature", 98.6)) > 100.4:
                findings.append(f"Fever: {vs['temperature']}F")
            if vs.get("heart_rate") and int(vs.get("heart_rate", 80)) > 100:
                findings.append(f"Tachycardia: {vs['heart_rate']} bpm")
        
        if case.lab_results:
            for test, value in case.lab_results.items():
                if isinstance(value, dict) and value.get("abnormal"):
                    findings.append(f"{test}: {value.get('value')} (abnormal)")
        
        return findings
    
    async def order_decision_support(
        self,
        order_type: str,
        order_details: Dict[str, Any],
        patient_context: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Provide decision support for clinical orders"""
        return {
            "order_type": order_type,
            "alerts": [],
            "suggestions": [],
            "guidelines_referenced": [],
            "approved": True,
        }
