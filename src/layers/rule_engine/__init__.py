"""Layer 2: Rule Engine - Safety Layer"""
from .service import RuleEngineService, get_rule_engine_service
from .triage import TriageEngine, TriageResult, TriageUrgency
from .otc_eligibility import OTCEligibilityEngine, OTCDecision
from .contraindication_checker import ContraindicationChecker
from .red_flags import RedFlagDetector, RedFlag

__all__ = [
    "RuleEngineService",
    "get_rule_engine_service",
    "TriageEngine",
    "TriageResult",
    "TriageUrgency",
    "OTCEligibilityEngine",
    "OTCDecision",
    "ContraindicationChecker",
    "RedFlagDetector",
    "RedFlag",
]
