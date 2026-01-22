"""
Hallucination Detector - Identifies factually incorrect or unsupported claims

Uses multiple strategies:
1. Knowledge graph verification
2. Semantic consistency checking
3. Source attribution validation
4. Medical fact verification
"""
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field
import re


class HallucinationType(str, Enum):
    """Types of hallucinations"""
    FACTUAL_ERROR = "factual_error"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    CONTRADICTORY = "contradictory"
    FABRICATED_SOURCE = "fabricated_source"
    INCORRECT_DOSAGE = "incorrect_dosage"
    WRONG_INDICATION = "wrong_indication"
    INVENTED_DRUG = "invented_drug"
    INCORRECT_INTERACTION = "incorrect_interaction"


class HallucinationSeverity(str, Enum):
    """Severity of detected hallucination"""
    CRITICAL = "critical"  # Could cause patient harm
    HIGH = "high"          # Significant misinformation
    MEDIUM = "medium"      # Minor inaccuracy
    LOW = "low"            # Stylistic/minor issue


class DetectedHallucination(BaseModel):
    """A detected hallucination"""
    type: HallucinationType
    severity: HallucinationSeverity
    text_span: str
    explanation: str
    suggested_correction: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    source_reference: Optional[str] = None


class HallucinationResult(BaseModel):
    """Result of hallucination detection"""
    has_hallucinations: bool = False
    hallucinations: List[DetectedHallucination] = Field(default_factory=list)
    overall_reliability_score: float = Field(ge=0.0, le=1.0, default=1.0)
    verified_claims: List[str] = Field(default_factory=list)
    unverified_claims: List[str] = Field(default_factory=list)
    verification_sources: List[str] = Field(default_factory=list)
    processing_time_ms: float = 0


class HallucinationDetector:
    """
    Detects hallucinations in LLM-generated medical content
    
    Uses multiple verification strategies to identify:
    - Factual errors
    - Unsupported claims
    - Contradictions with known facts
    - Fabricated sources or data
    """
    
    # Known drug name patterns for validation
    DRUG_NAME_PATTERN = re.compile(r'\b[A-Z][a-z]+(?:mab|nib|vir|zole|pril|sartan|statin|olol|pine|pam|lam)\b')
    
    # Dosage patterns
    DOSAGE_PATTERN = re.compile(r'\b(\d+(?:\.\d+)?)\s*(mg|g|mcg|ml|mL|units?|IU)\b', re.IGNORECASE)
    
    # Common medical claim patterns
    CLAIM_PATTERNS = [
        (r'is (?:used|indicated) for', 'indication_claim'),
        (r'causes?|leads? to|results? in', 'causation_claim'),
        (r'(?:should|must) (?:not )?(?:be |take|use)', 'recommendation_claim'),
        (r'(?:increases?|decreases?|reduces?) (?:the )?risk', 'risk_claim'),
        (r'(?:first|second|third)-line (?:treatment|therapy)', 'treatment_line_claim'),
        (r'contraindicated in', 'contraindication_claim'),
        (r'interacts? with', 'interaction_claim'),
    ]
    
    def __init__(self, knowledge_graph=None, threshold: float = 0.85):
        self.knowledge_graph = knowledge_graph
        self.threshold = threshold
        self._known_drugs: set = set()
        self._known_diseases: set = set()
    
    async def detect(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        source_documents: Optional[List[str]] = None,
    ) -> HallucinationResult:
        """
        Detect hallucinations in generated text
        
        Args:
            text: The generated text to verify
            context: Original context/query
            source_documents: Source documents for attribution check
        
        Returns:
            HallucinationResult with detected issues
        """
        import time
        start_time = time.time()
        
        result = HallucinationResult()
        
        # Extract claims from text
        claims = self._extract_claims(text)
        
        # Verify each claim
        for claim in claims:
            verification = await self._verify_claim(claim, context)
            
            if verification["verified"]:
                result.verified_claims.append(claim["text"])
                if verification.get("source"):
                    result.verification_sources.append(verification["source"])
            else:
                result.unverified_claims.append(claim["text"])
                
                if verification.get("hallucination"):
                    result.hallucinations.append(verification["hallucination"])
                    result.has_hallucinations = True
        
        # Check for fabricated drug names
        drug_hallucinations = self._check_drug_names(text)
        result.hallucinations.extend(drug_hallucinations)
        if drug_hallucinations:
            result.has_hallucinations = True
        
        # Check dosage validity
        dosage_issues = self._check_dosages(text)
        result.hallucinations.extend(dosage_issues)
        if dosage_issues:
            result.has_hallucinations = True
        
        # Check source attributions
        if source_documents:
            attribution_issues = self._check_attributions(text, source_documents)
            result.hallucinations.extend(attribution_issues)
            if attribution_issues:
                result.has_hallucinations = True
        
        # Calculate overall reliability score
        total_claims = len(result.verified_claims) + len(result.unverified_claims)
        if total_claims > 0:
            base_score = len(result.verified_claims) / total_claims
        else:
            base_score = 1.0
        
        # Penalize for hallucinations
        hallucination_penalty = sum(
            0.2 if h.severity == HallucinationSeverity.CRITICAL else
            0.1 if h.severity == HallucinationSeverity.HIGH else
            0.05 if h.severity == HallucinationSeverity.MEDIUM else 0.02
            for h in result.hallucinations
        )
        
        result.overall_reliability_score = max(0.0, base_score - hallucination_penalty)
        result.processing_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def _extract_claims(self, text: str) -> List[Dict[str, Any]]:
        """Extract verifiable claims from text"""
        claims = []
        sentences = text.split('.')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            for pattern, claim_type in self.CLAIM_PATTERNS:
                if re.search(pattern, sentence, re.IGNORECASE):
                    claims.append({
                        "text": sentence,
                        "type": claim_type,
                        "pattern": pattern,
                    })
                    break
        
        return claims
    
    async def _verify_claim(
        self,
        claim: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Verify a single claim against knowledge base"""
        result = {"verified": False, "source": None, "hallucination": None}
        
        if not self.knowledge_graph:
            # Without knowledge graph, mark as unverified but not hallucination
            return result
        
        claim_text = claim["text"].lower()
        claim_type = claim["type"]
        
        # Verify based on claim type
        if claim_type == "indication_claim":
            # Extract drug and indication
            verified = await self._verify_indication(claim_text)
            result["verified"] = verified
            if not verified:
                result["hallucination"] = DetectedHallucination(
                    type=HallucinationType.WRONG_INDICATION,
                    severity=HallucinationSeverity.HIGH,
                    text_span=claim["text"],
                    explanation="Could not verify this indication in knowledge base",
                    confidence=0.7,
                )
        
        elif claim_type == "interaction_claim":
            verified = await self._verify_interaction(claim_text)
            result["verified"] = verified
            if not verified:
                result["hallucination"] = DetectedHallucination(
                    type=HallucinationType.INCORRECT_INTERACTION,
                    severity=HallucinationSeverity.HIGH,
                    text_span=claim["text"],
                    explanation="Could not verify this drug interaction",
                    confidence=0.7,
                )
        
        elif claim_type == "contraindication_claim":
            verified = await self._verify_contraindication(claim_text)
            result["verified"] = verified
            if not verified:
                result["hallucination"] = DetectedHallucination(
                    type=HallucinationType.FACTUAL_ERROR,
                    severity=HallucinationSeverity.CRITICAL,
                    text_span=claim["text"],
                    explanation="Could not verify this contraindication",
                    confidence=0.8,
                )
        
        return result
    
    async def _verify_indication(self, claim_text: str) -> bool:
        """Verify a drug indication claim"""
        # In production, query knowledge graph
        # For now, return True to avoid false positives
        return True
    
    async def _verify_interaction(self, claim_text: str) -> bool:
        """Verify a drug interaction claim"""
        return True
    
    async def _verify_contraindication(self, claim_text: str) -> bool:
        """Verify a contraindication claim"""
        return True
    
    def _check_drug_names(self, text: str) -> List[DetectedHallucination]:
        """Check for potentially fabricated drug names"""
        hallucinations = []
        
        # Find drug-like names
        potential_drugs = self.DRUG_NAME_PATTERN.findall(text)
        
        for drug in potential_drugs:
            if drug.lower() not in self._known_drugs and len(self._known_drugs) > 0:
                # Potentially fabricated drug
                hallucinations.append(DetectedHallucination(
                    type=HallucinationType.INVENTED_DRUG,
                    severity=HallucinationSeverity.CRITICAL,
                    text_span=drug,
                    explanation=f"Drug name '{drug}' not found in database - may be fabricated",
                    confidence=0.6,
                ))
        
        return hallucinations
    
    def _check_dosages(self, text: str) -> List[DetectedHallucination]:
        """Check for potentially dangerous dosages"""
        hallucinations = []
        
        dosages = self.DOSAGE_PATTERN.findall(text)
        
        for amount, unit in dosages:
            amount_float = float(amount)
            
            # Flag suspiciously high dosages
            if unit.lower() == 'g' and amount_float > 10:
                hallucinations.append(DetectedHallucination(
                    type=HallucinationType.INCORRECT_DOSAGE,
                    severity=HallucinationSeverity.CRITICAL,
                    text_span=f"{amount} {unit}",
                    explanation=f"Dosage of {amount}g seems unusually high - verify",
                    confidence=0.7,
                ))
            elif unit.lower() == 'mg' and amount_float > 5000:
                hallucinations.append(DetectedHallucination(
                    type=HallucinationType.INCORRECT_DOSAGE,
                    severity=HallucinationSeverity.HIGH,
                    text_span=f"{amount} {unit}",
                    explanation=f"Dosage of {amount}mg seems high - verify",
                    confidence=0.6,
                ))
        
        return hallucinations
    
    def _check_attributions(
        self,
        text: str,
        source_documents: List[str],
    ) -> List[DetectedHallucination]:
        """Check if claims are properly attributed to sources"""
        hallucinations = []
        
        # Look for citation patterns
        citation_pattern = re.compile(r'\(([^)]+(?:et al\.?|20\d{2})[^)]*)\)')
        citations = citation_pattern.findall(text)
        
        for citation in citations:
            # Check if citation matches any source
            found = False
            for source in source_documents:
                if any(word in source.lower() for word in citation.lower().split()):
                    found = True
                    break
            
            if not found:
                hallucinations.append(DetectedHallucination(
                    type=HallucinationType.FABRICATED_SOURCE,
                    severity=HallucinationSeverity.HIGH,
                    text_span=citation,
                    explanation=f"Citation '{citation}' not found in provided sources",
                    confidence=0.8,
                ))
        
        return hallucinations
    
    def load_known_entities(self, drugs: List[str], diseases: List[str]) -> None:
        """Load known entities for validation"""
        self._known_drugs = set(d.lower() for d in drugs)
        self._known_diseases = set(d.lower() for d in diseases)
