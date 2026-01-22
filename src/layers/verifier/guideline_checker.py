"""
Guideline Checker - Verifies LLM outputs against clinical guidelines

Ensures recommendations align with:
- Clinical practice guidelines
- Drug labeling
- Regulatory requirements
- Evidence-based medicine standards
"""
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class GuidelineConflictSeverity(str, Enum):
    """Severity of guideline conflicts"""
    CRITICAL = "critical"  # Direct contradiction of guideline
    MAJOR = "major"        # Significant deviation
    MINOR = "minor"        # Minor inconsistency
    INFO = "info"          # Informational note


class GuidelineConflict(BaseModel):
    """A detected conflict with guidelines"""
    guideline_name: str
    guideline_source: str
    conflict_description: str
    severity: GuidelineConflictSeverity
    recommendation_text: str
    guideline_recommendation: str
    resolution_suggestion: Optional[str] = None


class GuidelineCheckResult(BaseModel):
    """Result of guideline checking"""
    is_compliant: bool = True
    conflicts: List[GuidelineConflict] = Field(default_factory=list)
    aligned_guidelines: List[str] = Field(default_factory=list)
    applicable_guidelines: List[str] = Field(default_factory=list)
    compliance_score: float = Field(ge=0.0, le=1.0, default=1.0)
    notes: List[str] = Field(default_factory=list)


class ClinicalGuideline(BaseModel):
    """Clinical guideline definition"""
    id: str
    name: str
    source: str  # AHA, ACC, NICE, etc.
    condition: str
    recommendations: List[Dict[str, Any]]
    contraindications: List[str] = Field(default_factory=list)
    first_line_treatments: List[str] = Field(default_factory=list)
    avoid_treatments: List[str] = Field(default_factory=list)
    monitoring_requirements: List[str] = Field(default_factory=list)


class GuidelineChecker:
    """
    Checks LLM outputs against clinical guidelines
    
    Ensures medical recommendations align with established
    clinical practice guidelines and evidence-based standards.
    """
    
    def __init__(self, knowledge_graph=None):
        self.knowledge_graph = knowledge_graph
        self.guidelines: Dict[str, ClinicalGuideline] = self._load_default_guidelines()
    
    def _load_default_guidelines(self) -> Dict[str, ClinicalGuideline]:
        """Load default clinical guidelines"""
        return {
            "hypertension_jnc8": ClinicalGuideline(
                id="hypertension_jnc8",
                name="JNC 8 Hypertension Guidelines",
                source="JNC 8",
                condition="hypertension",
                recommendations=[
                    {"population": "general", "target_bp": "140/90"},
                    {"population": "diabetes", "target_bp": "140/90"},
                    {"population": "ckd", "target_bp": "140/90"},
                    {"population": "age_60_plus", "target_bp": "150/90"},
                ],
                first_line_treatments=["thiazide", "ace_inhibitor", "arb", "ccb"],
                avoid_treatments=[],
                monitoring_requirements=["blood_pressure", "renal_function", "electrolytes"],
            ),
            "diabetes_ada": ClinicalGuideline(
                id="diabetes_ada",
                name="ADA Standards of Medical Care in Diabetes",
                source="ADA",
                condition="diabetes_type2",
                recommendations=[
                    {"metric": "hba1c", "target": "<7%", "population": "general"},
                    {"metric": "hba1c", "target": "<8%", "population": "elderly_frail"},
                ],
                first_line_treatments=["metformin"],
                avoid_treatments=[],
                contraindications=["metformin_in_egfr_below_30"],
                monitoring_requirements=["hba1c", "renal_function", "lipids"],
            ),
            "heart_failure_acc": ClinicalGuideline(
                id="heart_failure_acc",
                name="ACC/AHA Heart Failure Guidelines",
                source="ACC/AHA",
                condition="heart_failure",
                recommendations=[
                    {"class": "hfref", "treatment": "ace_inhibitor_or_arb"},
                    {"class": "hfref", "treatment": "beta_blocker"},
                    {"class": "hfref", "treatment": "mineralocorticoid_antagonist"},
                ],
                first_line_treatments=["ace_inhibitor", "beta_blocker", "diuretic"],
                avoid_treatments=["nsaid", "thiazolidinedione", "nondihydropyridine_ccb"],
                contraindications=["nsaid_in_heart_failure"],
                monitoring_requirements=["weight", "renal_function", "electrolytes"],
            ),
            "anticoagulation_chest": ClinicalGuideline(
                id="anticoagulation_chest",
                name="CHEST Antithrombotic Guidelines",
                source="CHEST",
                condition="atrial_fibrillation",
                recommendations=[
                    {"indication": "afib_stroke_prevention", "treatment": "doac_preferred"},
                    {"indication": "vte_treatment", "duration": "3_months_minimum"},
                ],
                first_line_treatments=["apixaban", "rivaroxaban", "dabigatran", "edoxaban"],
                avoid_treatments=[],
                monitoring_requirements=["bleeding_signs", "renal_function"],
            ),
            "asthma_gina": ClinicalGuideline(
                id="asthma_gina",
                name="GINA Asthma Guidelines",
                source="GINA",
                condition="asthma",
                recommendations=[
                    {"step": 1, "treatment": "saba_prn"},
                    {"step": 2, "treatment": "low_dose_ics"},
                    {"step": 3, "treatment": "low_dose_ics_laba"},
                ],
                first_line_treatments=["inhaled_corticosteroid", "saba"],
                avoid_treatments=["beta_blocker_nonselective"],
                monitoring_requirements=["peak_flow", "symptom_control"],
            ),
        }
    
    async def check(
        self,
        text: str,
        condition: Optional[str] = None,
        patient_context: Optional[Dict[str, Any]] = None,
    ) -> GuidelineCheckResult:
        """
        Check text against applicable clinical guidelines
        
        Args:
            text: LLM-generated text to check
            condition: Specific condition to check guidelines for
            patient_context: Patient-specific context
        
        Returns:
            GuidelineCheckResult with conflicts and compliance info
        """
        result = GuidelineCheckResult()
        
        # Find applicable guidelines
        applicable = self._find_applicable_guidelines(text, condition)
        result.applicable_guidelines = [g.name for g in applicable]
        
        if not applicable:
            result.notes.append("No specific guidelines found for this content")
            return result
        
        # Check against each guideline
        for guideline in applicable:
            conflicts = self._check_against_guideline(text, guideline, patient_context)
            
            if conflicts:
                result.conflicts.extend(conflicts)
                result.is_compliant = False
            else:
                result.aligned_guidelines.append(guideline.name)
        
        # Calculate compliance score
        if result.applicable_guidelines:
            aligned_count = len(result.aligned_guidelines)
            total_count = len(result.applicable_guidelines)
            
            # Penalize for conflicts
            conflict_penalty = sum(
                0.3 if c.severity == GuidelineConflictSeverity.CRITICAL else
                0.2 if c.severity == GuidelineConflictSeverity.MAJOR else
                0.1 if c.severity == GuidelineConflictSeverity.MINOR else 0.05
                for c in result.conflicts
            )
            
            base_score = aligned_count / total_count if total_count > 0 else 1.0
            result.compliance_score = max(0.0, base_score - conflict_penalty)
        
        return result
    
    def _find_applicable_guidelines(
        self,
        text: str,
        condition: Optional[str],
    ) -> List[ClinicalGuideline]:
        """Find guidelines applicable to the text content"""
        applicable = []
        text_lower = text.lower()
        
        for guideline in self.guidelines.values():
            # Check if condition matches
            if condition and condition.lower() in guideline.condition.lower():
                applicable.append(guideline)
                continue
            
            # Check if guideline condition mentioned in text
            if guideline.condition.replace("_", " ") in text_lower:
                applicable.append(guideline)
                continue
            
            # Check if treatments mentioned
            for treatment in guideline.first_line_treatments:
                if treatment.replace("_", " ") in text_lower:
                    applicable.append(guideline)
                    break
        
        return applicable
    
    def _check_against_guideline(
        self,
        text: str,
        guideline: ClinicalGuideline,
        patient_context: Optional[Dict[str, Any]],
    ) -> List[GuidelineConflict]:
        """Check text against a specific guideline"""
        conflicts = []
        text_lower = text.lower()
        
        # Check for avoid treatments being recommended
        for avoid in guideline.avoid_treatments:
            avoid_normalized = avoid.replace("_", " ")
            if avoid_normalized in text_lower:
                # Check if it's being recommended (not just mentioned)
                recommend_patterns = [
                    f"recommend {avoid_normalized}",
                    f"prescribe {avoid_normalized}",
                    f"start {avoid_normalized}",
                    f"use {avoid_normalized}",
                    f"{avoid_normalized} is indicated",
                ]
                
                for pattern in recommend_patterns:
                    if pattern in text_lower:
                        conflicts.append(GuidelineConflict(
                            guideline_name=guideline.name,
                            guideline_source=guideline.source,
                            conflict_description=f"Recommends {avoid} which should be avoided per guidelines",
                            severity=GuidelineConflictSeverity.CRITICAL,
                            recommendation_text=pattern,
                            guideline_recommendation=f"Avoid {avoid} in {guideline.condition}",
                            resolution_suggestion=f"Consider alternatives: {', '.join(guideline.first_line_treatments)}",
                        ))
                        break
        
        # Check for contraindications
        for contra in guideline.contraindications:
            contra_normalized = contra.replace("_", " ")
            if contra_normalized in text_lower:
                conflicts.append(GuidelineConflict(
                    guideline_name=guideline.name,
                    guideline_source=guideline.source,
                    conflict_description=f"Contains contraindicated recommendation: {contra}",
                    severity=GuidelineConflictSeverity.CRITICAL,
                    recommendation_text=contra_normalized,
                    guideline_recommendation=f"Contraindicated: {contra}",
                ))
        
        # Check if first-line treatments are bypassed
        mentions_treatment = any(
            t.replace("_", " ") in text_lower
            for t in guideline.first_line_treatments
        )
        
        mentions_second_line = False  # Would need second-line list
        
        if mentions_second_line and not mentions_treatment:
            conflicts.append(GuidelineConflict(
                guideline_name=guideline.name,
                guideline_source=guideline.source,
                conflict_description="Recommends second-line treatment without mentioning first-line options",
                severity=GuidelineConflictSeverity.MAJOR,
                recommendation_text="",
                guideline_recommendation=f"First-line treatments: {', '.join(guideline.first_line_treatments)}",
            ))
        
        return conflicts
    
    def add_guideline(self, guideline: ClinicalGuideline) -> None:
        """Add a clinical guideline"""
        self.guidelines[guideline.id] = guideline
    
    def get_guideline(self, guideline_id: str) -> Optional[ClinicalGuideline]:
        """Get a specific guideline"""
        return self.guidelines.get(guideline_id)
    
    def list_guidelines(self) -> List[Dict[str, str]]:
        """List all available guidelines"""
        return [
            {"id": g.id, "name": g.name, "source": g.source, "condition": g.condition}
            for g in self.guidelines.values()
        ]
