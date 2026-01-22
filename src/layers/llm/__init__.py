"""Layer 3: LLM - Language and Synthesis Layer"""
from .service import LLMService, get_llm_service
from .meditron import MeditronModel
from .prompts import PromptTemplates, RolePrompts
from .adapters import DomainAdapter, AdapterType

__all__ = [
    "LLMService",
    "get_llm_service",
    "MeditronModel",
    "PromptTemplates",
    "RolePrompts",
    "DomainAdapter",
    "AdapterType",
]
