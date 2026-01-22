"""
LLM Service - Layer 3 Language and Synthesis Layer

Orchestrates LLM interactions with safety guardrails.
"""
from typing import Optional, List, Dict, Any, Generator
from datetime import datetime
import logging
import time

from src.core.config import settings
from src.core.security.audit import AuditLogger, get_audit_logger, AuditAction

from .meditron import MeditronModel
from .prompts import PromptTemplates, RolePrompts, RoleType, ConversationFormatter
from .adapters import DomainAdapter, AdapterType

logger = logging.getLogger(__name__)


class LLMResponse:
    """Structured LLM response"""
    
    def __init__(
        self,
        content: str,
        role: RoleType,
        adapter_used: Optional[AdapterType] = None,
        tokens_used: int = 0,
        latency_ms: float = 0,
        verified: bool = False,
        verification_notes: Optional[str] = None,
        knowledge_sources: List[str] = None,
        safety_checked: bool = False,
    ):
        self.content = content
        self.role = role
        self.adapter_used = adapter_used
        self.tokens_used = tokens_used
        self.latency_ms = latency_ms
        self.verified = verified
        self.verification_notes = verification_notes
        self.knowledge_sources = knowledge_sources or []
        self.safety_checked = safety_checked
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "role": self.role.value,
            "adapter_used": self.adapter_used.value if self.adapter_used else None,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
            "verified": self.verified,
            "verification_notes": self.verification_notes,
            "knowledge_sources": self.knowledge_sources,
            "safety_checked": self.safety_checked,
            "timestamp": self.timestamp.isoformat(),
        }


class LLMService:
    """
    Layer 3: LLM Service - Language and Synthesis Layer
    
    CRITICAL: This layer NEVER makes safety decisions alone.
    All outputs are verified by Layer 4 (Verifier).
    """
    
    def __init__(
        self,
        model: Optional[MeditronModel] = None,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self.model = model or MeditronModel()
        self.adapter_manager = DomainAdapter()
        self.audit = audit_logger or get_audit_logger()
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the LLM service"""
        if self._initialized:
            return
        logger.info("Initializing LLM service...")
        self.model.load()
        self._initialized = True
        logger.info("LLM service initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the LLM service"""
        self.model.unload()
        self._initialized = False
    
    def _select_adapter(self, task: str, role: RoleType) -> Optional[AdapterType]:
        """Select appropriate adapter based on task and role"""
        role_adapter_map = {
            RoleType.PATIENT: AdapterType.PATIENT_TRIAGE,
            RoleType.PHARMACIST: AdapterType.CLINICAL_PHARMACIST,
            RoleType.PHARMA_QA: AdapterType.REGULATORY_QA,
            RoleType.RESEARCHER: AdapterType.RESEARCH,
            RoleType.STUDENT: AdapterType.EDUCATION,
        }
        if role in role_adapter_map:
            return role_adapter_map[role]
        return self.adapter_manager.select_adapter_for_task(task)
    
    def _build_prompt(
        self,
        query: str,
        role: RoleType,
        context: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        knowledge_context: Optional[str] = None,
        safety_context: Optional[str] = None,
    ) -> str:
        """Build the complete prompt for the model"""
        system_prompt = RolePrompts.get_prompt(role)
        
        parts = []
        parts.append("[SYSTEM]")
        parts.append(system_prompt)
        parts.append("[/SYSTEM]")
        
        if knowledge_context:
            parts.append("[CONTEXT]")
            parts.append(knowledge_context)
            parts.append("[/CONTEXT]")
        
        if safety_context:
            parts.append("[SAFETY]")
            parts.append(safety_context)
            parts.append("[/SAFETY]")
        
        if chat_history:
            for msg in chat_history[-10:]:
                if msg["role"] == "user":
                    parts.append("[USER]")
                else:
                    parts.append("[ASSISTANT]")
                parts.append(msg["content"])
        
        parts.append("[USER]")
        parts.append(query)
        parts.append("[ASSISTANT]")
        
        return "\n".join(parts)
    
    async def generate(
        self,
        query: str,
        role: RoleType = RoleType.GENERAL,
        context: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        knowledge_context: Optional[str] = None,
        safety_context: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        user_id: Optional[str] = None,
    ) -> LLMResponse:
        """Generate a response from the LLM"""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        adapter = self._select_adapter(query, role)
        
        prompt = self._build_prompt(
            query=query,
            role=role,
            context=context,
            chat_history=chat_history,
            knowledge_context=knowledge_context,
            safety_context=safety_context,
        )
        
        content = self.model.generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        
        latency_ms = (time.time() - start_time) * 1000
        tokens_used = self.model.count_tokens(prompt) + self.model.count_tokens(content)
        
        response = LLMResponse(
            content=content,
            role=role,
            adapter_used=adapter,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
        )
        
        self.audit.log_llm_interaction(
            user_id=user_id or "anonymous",
            user_role=role.value,
            query=query,
            response=content,
            model_name="meditron",
            verification_passed=False,
            latency_ms=latency_ms,
        )
        
        return response
    
    async def generate_stream(
        self,
        query: str,
        role: RoleType = RoleType.GENERAL,
        context: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        knowledge_context: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Generate a streaming response"""
        if not self._initialized:
            await self.initialize()
        
        prompt = self._build_prompt(
            query=query,
            role=role,
            context=context,
            chat_history=chat_history,
            knowledge_context=knowledge_context,
        )
        
        for token in self.model.generate_stream(prompt):
            yield token
    
    async def explain_medical_concept(
        self,
        concept: str,
        audience: str = "patient",
        user_id: Optional[str] = None,
    ) -> LLMResponse:
        """Explain a medical concept"""
        role = RoleType.PATIENT if audience == "patient" else RoleType.STUDENT
        query = f"Explain the following medical concept in terms appropriate for a {audience}: {concept}"
        return await self.generate(query=query, role=role, user_id=user_id)
    
    async def summarize_clinical_data(
        self,
        clinical_data: str,
        user_id: Optional[str] = None,
    ) -> LLMResponse:
        """Summarize clinical data"""
        query = f"Summarize the following clinical information:\n\n{clinical_data}"
        return await self.generate(query=query, role=RoleType.DOCTOR, user_id=user_id)
    
    async def draft_document(
        self,
        document_type: str,
        content_requirements: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> LLMResponse:
        """Draft a medical or regulatory document"""
        query = f"Draft a {document_type} document with the following requirements:\n"
        for key, value in content_requirements.items():
            query += f"- {key}: {value}\n"
        
        role = RoleType.PHARMA_QA if "regulatory" in document_type.lower() else RoleType.DOCTOR
        return await self.generate(query=query, role=role, user_id=user_id)
    
    async def answer_usmle_question(
        self,
        question: str,
        options: Optional[List[str]] = None,
        user_id: Optional[str] = None,
    ) -> LLMResponse:
        """Answer a USMLE-style question"""
        query = f"USMLE Question: {question}"
        if options:
            query += "\n\nOptions:\n"
            for i, opt in enumerate(options):
                query += f"{chr(65+i)}. {opt}\n"
        query += "\nProvide the answer with detailed explanation."
        
        return await self.generate(query=query, role=RoleType.STUDENT, user_id=user_id)


_llm_service: Optional[LLMService] = None


async def get_llm_service() -> LLMService:
    """Get or create LLM service singleton"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
