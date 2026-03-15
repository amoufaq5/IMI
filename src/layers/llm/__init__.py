"""Layer 3: LLM - Language and Synthesis Layer"""
from .service import LLMService, get_llm_service
from .meditron import MistralMedicalModel
# Backward compatibility aliases
MixtralMedicalModel = MistralMedicalModel
MeditronModel = MistralMedicalModel
from .prompts import PromptTemplates, RolePrompts
from .adapters import DomainAdapter, AdapterType

__all__ = [
    "LLMService",
    "get_llm_service",
    "MistralMedicalModel",
    "MixtralMedicalModel",   # backward compat alias
    "MeditronModel",          # backward compat alias
    "PromptTemplates",
    "RolePrompts",
    "DomainAdapter",
    "AdapterType",
]
