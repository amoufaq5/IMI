"""
Confidence Calibrator - Detects and corrects overconfidence in LLM outputs

Ensures the model appropriately expresses uncertainty when:
- Evidence is limited
- Multiple valid interpretations exist
- Recommendations require professional judgment
"""
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
import re


class ConfidenceLevel(str, Enum):
    """Confidence levels for medical statements"""
    HIGH = "high"          # Strong evidence, well-established
    MODERATE = "moderate"  # Good evidence, generally accepted
    LOW = "low"            # Limited evidence, emerging
    UNCERTAIN = "uncertain"  # Insufficient evidence


class OverconfidenceType(str, Enum):
    """Types of overconfidence issues"""
    DEFINITIVE_DIAGNOSIS = "definitive_diagnosis"
    ABSOLUTE_RECOMMENDATION = "absolute_recommendation"
    MISSING_UNCERTAINTY = "missing_uncertainty"
    OVERGENERALIZATION = "overgeneralization"
    UNSUPPORTED_CERTAINTY = "unsupported_certainty"


class OverconfidenceIssue(BaseModel):
    """A detected overconfidence issue"""
    type: OverconfidenceType
    text_span: str
    explanation: str
    suggested_revision: str
    original_confidence: ConfidenceLevel
    appropriate_confidence: ConfidenceLevel


class CalibrationResult(BaseModel):
    """Result of confidence calibration"""
    is_well_calibrated: bool = True
    overconfidence_issues: List[OverconfidenceIssue] = Field(default_factory=list)
    underconfidence_issues: List[str] = Field(default_factory=list)
    calibration_score: float = Field(ge=0.0, le=1.0, default=1.0)
    suggested_hedging_phrases: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class ConfidenceCalibrator:
    """
    Calibrates confidence levels in LLM medical outputs
    
    Detects:
    - Overconfident diagnostic statements
    - Absolute recommendations without appropriate caveats
    - Missing uncertainty language
    - Overgeneralization of limited evidence
    """
    
    # Patterns indicating overconfidence
    OVERCONFIDENT_PATTERNS = [
        (r'\b(definitely|certainly|absolutely|always|never)\b', OverconfidenceType.ABSOLUTE_RECOMMENDATION),
        (r'\byou have\b(?! to consult)', OverconfidenceType.DEFINITIVE_DIAGNOSIS),
        (r'\bthis is\b (?:clearly |obviously )?(?:a case of|caused by)', OverconfidenceType.DEFINITIVE_DIAGNOSIS),
        (r'\bthe diagnosis is\b', OverconfidenceType.DEFINITIVE_DIAGNOSIS),
        (r'\bwill (?:definitely |certainly )?(?:cure|fix|solve)', OverconfidenceType.ABSOLUTE_RECOMMENDATION),
        (r'\bguaranteed to\b', OverconfidenceType.ABSOLUTE_RECOMMENDATION),
        (r'\b100%\b', OverconfidenceType.UNSUPPORTED_CERTAINTY),
        (r'\beveryone should\b', OverconfidenceType.OVERGENERALIZATION),
        (r'\ball patients (?:must|should|need to)\b', OverconfidenceType.OVERGENERALIZATION),
    ]
    
    # Appropriate hedging phrases
    HEDGING_PHRASES = {
        "diagnosis": [
            "This could potentially be",
            "The symptoms are consistent with",
            "This presentation suggests",
            "One possibility to consider is",
            "Based on the information provided, this may indicate",
        ],
        "recommendation": [
            "You may want to consider",
            "It would be advisable to",
            "A healthcare provider might recommend",
            "Options to discuss with your doctor include",
            "Evidence suggests that",
        ],
        "prognosis": [
            "In many cases",
            "Outcomes can vary, but typically",
            "With appropriate treatment, many patients",
            "Individual results may vary",
        ],
        "general": [
            "Based on current evidence",
            "Generally speaking",
            "In most cases",
            "It's important to note that",
            "While this is often the case",
        ],
    }
    
    # Phrases that appropriately express uncertainty
    UNCERTAINTY_PHRASES = [
        "may", "might", "could", "possibly", "potentially",
        "suggests", "indicates", "consistent with",
        "consider", "discuss with", "consult",
        "in some cases", "often", "typically", "generally",
        "based on", "according to", "evidence suggests",
    ]
    
    def __init__(self):
        self.overconfident_regex = [
            (re.compile(pattern, re.IGNORECASE), issue_type)
            for pattern, issue_type in self.OVERCONFIDENT_PATTERNS
        ]
    
    def calibrate(
        self,
        text: str,
        context_type: str = "general",
        evidence_level: Optional[str] = None,
    ) -> CalibrationResult:
        """
        Calibrate confidence in generated text
        
        Args:
            text: The generated text to calibrate
            context_type: Type of content (diagnosis, recommendation, etc.)
            evidence_level: Known evidence level for the topic
        
        Returns:
            CalibrationResult with issues and suggestions
        """
        result = CalibrationResult()
        
        # Check for overconfident patterns
        for pattern, issue_type in self.overconfident_regex:
            matches = pattern.finditer(text)
            for match in matches:
                # Get surrounding context
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                issue = self._create_overconfidence_issue(
                    match.group(),
                    context,
                    issue_type,
                )
                result.overconfidence_issues.append(issue)
                result.is_well_calibrated = False
        
        # Check for missing uncertainty language
        sentences = text.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if sentence makes a claim without hedging
            if self._is_medical_claim(sentence) and not self._has_uncertainty(sentence):
                result.overconfidence_issues.append(OverconfidenceIssue(
                    type=OverconfidenceType.MISSING_UNCERTAINTY,
                    text_span=sentence[:100],
                    explanation="Medical claim without appropriate uncertainty language",
                    suggested_revision=self._add_hedging(sentence),
                    original_confidence=ConfidenceLevel.HIGH,
                    appropriate_confidence=ConfidenceLevel.MODERATE,
                ))
                result.is_well_calibrated = False
        
        # Add suggested hedging phrases
        result.suggested_hedging_phrases = self.HEDGING_PHRASES.get(
            context_type,
            self.HEDGING_PHRASES["general"]
        )
        
        # Calculate calibration score
        total_sentences = len([s for s in sentences if s.strip()])
        if total_sentences > 0:
            issues_ratio = len(result.overconfidence_issues) / total_sentences
            result.calibration_score = max(0.0, 1.0 - issues_ratio)
        
        # Add notes based on evidence level
        if evidence_level:
            if evidence_level in ["limited", "emerging", "low"]:
                result.notes.append(
                    "Given limited evidence, ensure appropriate uncertainty is expressed"
                )
            elif evidence_level in ["strong", "high", "established"]:
                result.notes.append(
                    "Strong evidence supports more confident statements, but avoid absolutes"
                )
        
        return result
    
    def _is_medical_claim(self, sentence: str) -> bool:
        """Check if sentence makes a medical claim"""
        claim_indicators = [
            "causes", "treats", "cures", "prevents",
            "is effective", "works by", "indicated for",
            "recommended for", "should be used",
            "diagnosis", "prognosis", "treatment",
        ]
        sentence_lower = sentence.lower()
        return any(indicator in sentence_lower for indicator in claim_indicators)
    
    def _has_uncertainty(self, sentence: str) -> bool:
        """Check if sentence has appropriate uncertainty language"""
        sentence_lower = sentence.lower()
        return any(phrase in sentence_lower for phrase in self.UNCERTAINTY_PHRASES)
    
    def _create_overconfidence_issue(
        self,
        matched_text: str,
        context: str,
        issue_type: OverconfidenceType,
    ) -> OverconfidenceIssue:
        """Create an overconfidence issue with suggested revision"""
        
        revisions = {
            OverconfidenceType.DEFINITIVE_DIAGNOSIS: (
                f"Replace '{matched_text}' with uncertainty language",
                "Consider: 'The symptoms are consistent with...' or 'This may indicate...'"
            ),
            OverconfidenceType.ABSOLUTE_RECOMMENDATION: (
                f"Avoid absolute term '{matched_text}'",
                "Consider: 'In many cases...' or 'Evidence suggests...'"
            ),
            OverconfidenceType.UNSUPPORTED_CERTAINTY: (
                f"Remove unsupported certainty '{matched_text}'",
                "Avoid percentage claims without supporting data"
            ),
            OverconfidenceType.OVERGENERALIZATION: (
                f"Avoid overgeneralization '{matched_text}'",
                "Consider: 'Many patients...' or 'In most cases...'"
            ),
            OverconfidenceType.MISSING_UNCERTAINTY: (
                "Add appropriate hedging language",
                "Consider adding 'may', 'could', or 'suggests'"
            ),
        }
        
        explanation, suggestion = revisions.get(
            issue_type,
            ("Review for appropriate confidence level", "Add hedging language")
        )
        
        return OverconfidenceIssue(
            type=issue_type,
            text_span=context,
            explanation=explanation,
            suggested_revision=suggestion,
            original_confidence=ConfidenceLevel.HIGH,
            appropriate_confidence=ConfidenceLevel.MODERATE,
        )
    
    def _add_hedging(self, sentence: str) -> str:
        """Add hedging to a sentence"""
        # Simple hedging - prepend with uncertainty phrase
        hedges = ["This may indicate that", "Evidence suggests that", "It appears that"]
        import random
        hedge = random.choice(hedges)
        
        # Lowercase first letter of original
        if sentence and sentence[0].isupper():
            sentence = sentence[0].lower() + sentence[1:]
        
        return f"{hedge} {sentence}"
    
    def get_appropriate_confidence(
        self,
        evidence_level: str,
        claim_type: str,
    ) -> ConfidenceLevel:
        """Get appropriate confidence level for a claim"""
        evidence_map = {
            "1a": ConfidenceLevel.HIGH,      # Systematic review of RCTs
            "1b": ConfidenceLevel.HIGH,      # Individual RCT
            "2a": ConfidenceLevel.MODERATE,  # Systematic review of cohort
            "2b": ConfidenceLevel.MODERATE,  # Individual cohort
            "3": ConfidenceLevel.LOW,        # Case-control
            "4": ConfidenceLevel.LOW,        # Case series
            "5": ConfidenceLevel.UNCERTAIN,  # Expert opinion
        }
        
        return evidence_map.get(evidence_level, ConfidenceLevel.MODERATE)
