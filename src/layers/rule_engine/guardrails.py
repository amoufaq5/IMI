"""
Input/Output Guardrails — Integrated with IMI Rule Engine (Layer 2)

Adds Doc1-style pattern-matching guardrails ON TOP of the existing deterministic
rule engine. These catch crisis/emergency signals at the text level before and
after LLM generation.

Architecture integration:
  User query → InputGuardrail.scan()  → Rule Engine (triage, red flags, etc.)
             → LLM generation
             → OutputGuardrail.scan() → Verifier (Layer 4)
             → Final response

Three signal types:
  CRISIS    — Suicide/self-harm → full-screen overlay, model NEVER responds
  EMERGENCY — Acute medical danger → orange banner prepended to response
  SCOPE     — Out-of-scope request → blocked with explanation
"""
import re
import logging
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class GuardrailSignal(str, Enum):
    """Signal types detected by guardrails"""
    CRISIS = "crisis"
    EMERGENCY = "emergency"
    SCOPE_VIOLATION = "scope_violation"
    PHI_DETECTED = "phi_detected"
    OVERCONFIDENT = "overconfident"
    NONE = "none"


@dataclass
class GuardrailResult:
    """Result of a guardrail scan"""
    signal: GuardrailSignal = GuardrailSignal.NONE
    triggered: bool = False
    matched_patterns: List[str] = field(default_factory=list)
    override_response: Optional[str] = None
    prepend_banner: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# CRISIS PATTERNS — Suicide, self-harm, violence
# If ANY match: block LLM response, show crisis resources only
# ============================================================================

CRISIS_PATTERNS = [
    # Suicide ideation
    r"\b(want|going|plan(?:ning)?|think(?:ing)?(?:\s+about)?)\s+(to\s+)?(kill\s+my\s*self|end\s+(my\s+)?life|die|suicide|commit\s+suicide)\b",
    r"\b(suicid(?:e|al)|kill\s+my\s*self|end\s+it\s+all|take\s+my\s+(?:own\s+)?life)\b",
    r"\bnot\s+(?:want(?:ing)?|worth)\s+(?:to\s+)?liv(?:e|ing)\b",
    r"\bbetter\s+off\s+dead\b",
    r"\bwish\s+I\s+(?:was|were)\s+dead\b",
    r"\bno\s+reason\s+to\s+live\b",
    r"\bdon'?t\s+want\s+to\s+(?:be\s+here|wake\s+up|exist)\b",
    # Self-harm
    r"\b(cutting|self[- ]?harm|hurt(?:ing)?\s+my\s*self|burn(?:ing)?\s+my\s*self)\b",
    r"\bhow\s+(?:to|can\s+I)\s+(?:hurt|harm|cut|injure)\s+my\s*self\b",
    # Methods
    r"\b(?:how\s+(?:to|can\s+I|many\s+pills?\s+to))\s+(?:overdose|OD)\b",
    r"\blethal\s+dose\b",
    r"\bhow\s+(?:to|can)\s+(?:hang|drown|shoot|poison)\s+(?:my\s*self|yourself)\b",
    # Harm to others
    r"\b(?:want|going|plan(?:ning)?)\s+to\s+(?:kill|hurt|harm)\s+(?:someone|people|them|him|her)\b",
    r"\bhomicid(?:e|al)\b",
]

CRISIS_RESPONSE = """I'm concerned about your safety. Please reach out for help immediately:

• **988 Suicide & Crisis Lifeline**: Call or text **988** (available 24/7)
• **Crisis Text Line**: Text **HOME** to **741741**
• **Emergency**: Call **911**

You don't have to face this alone. Trained counselors are ready to help right now."""


# ============================================================================
# EMERGENCY PATTERNS — Acute medical danger
# If match: allow LLM response but prepend emergency banner
# ============================================================================

EMERGENCY_PATTERNS = [
    r"\b(chest\s+pain|crushing\s+chest|heart\s+attack)\b",
    r"\b(can'?t\s+breathe|difficulty\s+breathing|severe\s+shortness\s+of\s+breath|choking)\b",
    r"\b(stroke|sudden\s+weakness|face\s+droop(?:ing)?|slurred\s+speech)\b",
    r"\b(anaphylax(?:is|tic)|severe\s+allergic\s+reaction|throat\s+(?:closing|swelling))\b",
    r"\b(seizure|convuls(?:ion|ing)|unresponsive|unconscious|passed\s+out)\b",
    r"\b(severe\s+bleed(?:ing)?|hemorrhag(?:e|ing)|cough(?:ing)?\s+(?:up\s+)?blood)\b",
    r"\b(swallow(?:ed)?\s+(?:a\s+)?(?:battery|poison|bleach|chemicals?))\b",
    r"\b(overdos(?:e|ed|ing)|took\s+too\s+(?:many|much)\s+(?:pills?|medication))\b",
    r"\b(head\s+injur(?:y|ies)|severe\s+head\s+trauma|skull\s+fracture)\b",
    r"\b(high\s+fever|fever\s+(?:over|above)\s+(?:103|104|105)|infant\s+fever)\b",
    r"\b(diabetic\s+(?:coma|emergency|ketoacidosis|DKA))\b",
    r"\b(severe\s+(?:abdominal|stomach)\s+pain)\b",
    r"\b(loss\s+of\s+(?:consciousness|vision)|sudden\s+(?:blindness|vision\s+loss))\b",
]

EMERGENCY_BANNER = """⚠️ **IMPORTANT**: The symptoms described may require immediate medical attention.
**Call 911** or go to your nearest **Emergency Room** now.
Do not delay seeking emergency care based on any information provided here.

---

"""


# ============================================================================
# SCOPE PATTERNS — Requests outside medical AI scope
# If match: block and explain
# ============================================================================

SCOPE_PATTERNS = [
    (r"\b(?:write|give)\s+(?:me\s+)?(?:a\s+)?prescription\b", "I cannot write prescriptions. Only a licensed healthcare provider can prescribe medications after proper evaluation."),
    (r"\b(?:diagnos(?:e|is)\s+me|what\s+(?:do|is)\s+I\s+have|tell\s+me\s+(?:my|what)\s+(?:diagnosis|disease|condition))\b", "I cannot provide a medical diagnosis. Diagnosis requires physical examination, lab tests, and clinical judgment from a licensed provider."),
    (r"\brefill\s+(?:my\s+)?(?:prescription|medication|rx)\b", "I cannot process prescription refills. Please contact your pharmacy or healthcare provider directly."),
    (r"\b(?:order|schedule|book)\s+(?:a\s+)?(?:lab|blood|imaging|x-?ray|MRI|CT|scan|test)\b", "I cannot order medical tests or imaging. Your healthcare provider can determine which tests are appropriate and order them for you."),
    (r"\b(?:legal|law(?:suit)?|malpractice|sue|attorney|lawyer)\s+(?:advice|help|question)\b", "I cannot provide legal advice. Please consult with a healthcare attorney for legal matters."),
]


# ============================================================================
# OUTPUT GUARDRAIL PATTERNS — Applied to LLM output
# ============================================================================

# Overconfident language to soften
OVERCONFIDENT_PATTERNS = [
    (r"\byou\s+(?:definitely|certainly|clearly)\s+have\b", "the symptoms you describe may be consistent with"),
    (r"\bthis\s+is\s+(?:definitely|certainly|clearly|without\s+(?:a\s+)?doubt)\b", "this may be"),
    (r"\byou\s+should\s+(?:definitely|immediately)\s+(?:take|start)\b", "you may want to discuss with your doctor whether to"),
    (r"\bI\s+(?:can\s+)?confirm\s+(?:that\s+)?you\s+have\b", "based on what you've described, a possibility to discuss with your doctor is"),
    (r"\b(?:take|use)\s+(\d+\s*(?:mg|mcg|ml|g|units?))\s+(?:of\s+)?(\w+)\b", r"discuss with your doctor the appropriate dose of \2"),
    (r"\bthere\s+is\s+no\s+(?:risk|danger|concern)\b", "the risk appears to be low, but discuss with your doctor"),
]

# PHI patterns to scrub from output
PHI_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",           # SSN
    r"\b[A-Z]{2}\d{7,8}\b",              # Passport
    r"\b\d{10,}\b",                       # Long numbers (potential MRN)
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
]


class InputGuardrail:
    """
    Scans user input BEFORE it reaches the rule engine or LLM.
    
    Catches crisis signals, emergency symptoms, and out-of-scope requests
    using regex pattern matching. This is a fast first-pass filter that
    complements the more thorough rule engine analysis.
    """

    def __init__(self):
        self._crisis_patterns = [re.compile(p, re.IGNORECASE) for p in CRISIS_PATTERNS]
        self._emergency_patterns = [re.compile(p, re.IGNORECASE) for p in EMERGENCY_PATTERNS]
        self._scope_patterns = [(re.compile(p, re.IGNORECASE), msg) for p, msg in SCOPE_PATTERNS]

    def scan(self, text: str) -> GuardrailResult:
        """
        Scan input text for guardrail signals.
        Priority: CRISIS > EMERGENCY > SCOPE > NONE
        """
        text_clean = text.strip()

        # 1. Crisis check (highest priority — blocks everything)
        crisis_matches = []
        for pattern in self._crisis_patterns:
            match = pattern.search(text_clean)
            if match:
                crisis_matches.append(match.group())

        if crisis_matches:
            logger.warning(f"CRISIS signal detected: {crisis_matches}")
            return GuardrailResult(
                signal=GuardrailSignal.CRISIS,
                triggered=True,
                matched_patterns=crisis_matches,
                override_response=CRISIS_RESPONSE,
                metadata={"action": "block_and_show_crisis_resources"},
            )

        # 2. Emergency check
        emergency_matches = []
        for pattern in self._emergency_patterns:
            match = pattern.search(text_clean)
            if match:
                emergency_matches.append(match.group())

        if emergency_matches:
            logger.warning(f"EMERGENCY signal detected: {emergency_matches}")
            return GuardrailResult(
                signal=GuardrailSignal.EMERGENCY,
                triggered=True,
                matched_patterns=emergency_matches,
                prepend_banner=EMERGENCY_BANNER,
                metadata={"action": "prepend_emergency_banner"},
            )

        # 3. Scope check
        for pattern, message in self._scope_patterns:
            match = pattern.search(text_clean)
            if match:
                logger.info(f"SCOPE violation detected: {match.group()}")
                return GuardrailResult(
                    signal=GuardrailSignal.SCOPE_VIOLATION,
                    triggered=True,
                    matched_patterns=[match.group()],
                    override_response=message,
                    metadata={"action": "block_with_scope_explanation"},
                )

        return GuardrailResult()


class OutputGuardrail:
    """
    Scans LLM output BEFORE it's returned to the user.
    
    Softens overconfident language, scrubs potential PHI,
    and adds safety disclaimers. Works alongside the Verifier (Layer 4)
    but catches text-level issues the verifier doesn't handle.
    """

    def __init__(self):
        self._overconfident_patterns = [
            (re.compile(p, re.IGNORECASE), replacement)
            for p, replacement in OVERCONFIDENT_PATTERNS
        ]
        self._phi_patterns = [re.compile(p) for p in PHI_PATTERNS]

    def scan(self, text: str, user_type: str = "general") -> GuardrailResult:
        """
        Scan and optionally modify LLM output.
        Returns result with cleaned text in override_response if modifications were made.
        """
        modified_text = text
        warnings = []
        matched = []

        # 1. Soften overconfident language
        for pattern, replacement in self._overconfident_patterns:
            if pattern.search(modified_text):
                matched.append(f"overconfident: {pattern.pattern[:50]}...")
                modified_text = pattern.sub(replacement, modified_text)

        # 2. Scrub PHI
        for pattern in self._phi_patterns:
            if pattern.search(modified_text):
                matched.append(f"phi: {pattern.pattern[:30]}...")
                modified_text = pattern.sub("[REDACTED]", modified_text)
                warnings.append("Potential PHI detected and scrubbed from output")

        # 3. Check for crisis signals in output (LLM should never generate these)
        crisis_check = InputGuardrail()
        crisis_result = crisis_check.scan(modified_text)
        if crisis_result.signal == GuardrailSignal.CRISIS:
            logger.error("LLM generated crisis-related content — blocking output")
            return GuardrailResult(
                signal=GuardrailSignal.CRISIS,
                triggered=True,
                matched_patterns=crisis_result.matched_patterns,
                override_response=CRISIS_RESPONSE,
                warnings=["LLM attempted to generate crisis-related content"],
                metadata={"action": "block_llm_crisis_output"},
            )

        was_modified = modified_text != text

        if was_modified:
            return GuardrailResult(
                signal=GuardrailSignal.OVERCONFIDENT if not warnings else GuardrailSignal.PHI_DETECTED,
                triggered=True,
                matched_patterns=matched,
                override_response=modified_text,
                warnings=warnings,
                metadata={"action": "output_modified", "modifications": len(matched)},
            )

        return GuardrailResult()
