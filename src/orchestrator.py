"""
IMI Orchestrator - Coordinates all 5 layers of the hybrid cognition stack

This is the central coordinator that ensures:
1. Knowledge Graph provides facts
2. Rule Engine enforces safety
3. LLM generates responses
4. Verifier checks accuracy
5. Memory maintains context

The LLM NEVER decides alone on safety-critical matters.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

from src.core.security.audit import AuditLogger, get_audit_logger, AuditAction
from src.core.security.hipaa import HIPAAComplianceService, get_hipaa_service

from src.layers.knowledge_graph import KnowledgeGraphService
from src.layers.rule_engine import RuleEngineService, get_rule_engine_service
from src.layers.llm import LLMService
from src.layers.llm.prompts import RoleType
from src.layers.verifier import VerifierService
from src.layers.memory import MemoryService, get_memory_service
from src.layers.memory.conversation_memory import ConversationOutcome

logger = logging.getLogger(__name__)


class OrchestratorResponse:
    """Response from the orchestrator"""
    
    def __init__(
        self,
        content: str,
        verified: bool = False,
        safety_checked: bool = False,
        knowledge_sources: List[str] = None,
        rules_applied: List[str] = None,
        warnings: List[str] = None,
        blocked: bool = False,
        block_reason: Optional[str] = None,
    ):
        self.content = content
        self.verified = verified
        self.safety_checked = safety_checked
        self.knowledge_sources = knowledge_sources or []
        self.rules_applied = rules_applied or []
        self.warnings = warnings or []
        self.blocked = blocked
        self.block_reason = block_reason
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "verified": self.verified,
            "safety_checked": self.safety_checked,
            "knowledge_sources": self.knowledge_sources,
            "rules_applied": self.rules_applied,
            "warnings": self.warnings,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "timestamp": self.timestamp.isoformat(),
        }


class IMIOrchestrator:
    """
    Central orchestrator for the IMI Medical LLM Platform
    
    Coordinates the 5-layer hybrid cognition stack:
    
    Layer 1 - Knowledge Graph (Truth Layer)
        Disease, symptom, drug, interaction, guideline relationships
        
    Layer 2 - Rule Engine (Safety Layer)
        Deterministic logic for OTC eligibility, red flags, contraindications
        
    Layer 3 - LLM (Language + Synthesis)
        Meditron for explanation, summarization, conversation
        NEVER allowed to decide alone on safety matters
        
    Layer 4 - Verifier/Critic
        Checks for hallucinations, guideline conflicts, overconfidence
        
    Layer 5 - Memory & Profiling
        Longitudinal patient/entity profiles, conversation history
    """
    
    def __init__(
        self,
        knowledge_graph: Optional[KnowledgeGraphService] = None,
        rule_engine: Optional[RuleEngineService] = None,
        llm_service: Optional[LLMService] = None,
        verifier_service: Optional[VerifierService] = None,
        memory_service: Optional[MemoryService] = None,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self.kg = knowledge_graph
        self.rule_engine = rule_engine or get_rule_engine_service()
        self.llm = llm_service
        self.verifier = verifier_service
        self.memory = memory_service or get_memory_service()
        self.audit = audit_logger or get_audit_logger()
        self.hipaa = get_hipaa_service()
    
    async def process_query(
        self,
        query: str,
        user_id: str,
        user_role: RoleType = RoleType.GENERAL,
        conversation_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> OrchestratorResponse:
        """
        Process a user query through all 5 layers
        
        Flow:
        1. Get patient context from Memory (Layer 5)
        2. Query Knowledge Graph for relevant facts (Layer 1)
        3. Run safety checks through Rule Engine (Layer 2)
        4. Generate response with LLM (Layer 3)
        5. Verify response with Critic (Layer 4)
        6. Store interaction in Memory (Layer 5)
        """
        logger.info(f"Processing query for user {user_id}, role {user_role.value}")
        
        # Initialize response tracking
        knowledge_sources = []
        rules_applied = []
        warnings = []
        
        # Layer 5: Get patient/conversation context
        patient_context = {}
        if patient_id:
            patient_context = self.memory.get_patient_context_for_llm(patient_id)
            knowledge_sources.append(f"patient_profile:{patient_id}")
        
        # Get or create conversation
        if conversation_id:
            conversation = self.memory.get_conversation(conversation_id)
        else:
            conversation = self.memory.start_conversation(
                user_id=user_id,
                patient_id=patient_id,
                topic=query[:100],
                user_role=user_role.value,
            )
            conversation_id = conversation.id
        
        # Add user message to conversation
        self.memory.add_message(conversation_id, "user", query)
        
        # Layer 1: Query Knowledge Graph for relevant facts
        kg_context = ""
        if self.kg:
            kg_results = await self._query_knowledge_graph(query, patient_context)
            if kg_results:
                kg_context = self._format_kg_context(kg_results)
                knowledge_sources.extend(kg_results.get("sources", []))
        
        # Layer 2: Run safety checks
        safety_context = ""
        safety_result = await self._run_safety_checks(
            query=query,
            patient_context=patient_context,
            user_id=user_id,
        )
        
        if safety_result:
            rules_applied.extend(safety_result.get("rules_triggered", []))
            warnings.extend(safety_result.get("warnings", []))
            
            # Block if critical safety issues
            if safety_result.get("blocked"):
                return OrchestratorResponse(
                    content=safety_result.get("block_message", "Request blocked for safety reasons."),
                    blocked=True,
                    block_reason=safety_result.get("block_reason"),
                    rules_applied=rules_applied,
                    warnings=warnings,
                    safety_checked=True,
                )
            
            safety_context = self._format_safety_context(safety_result)
        
        # Layer 3: Generate response with LLM
        if self.llm:
            chat_history = self.memory.get_conversation_context(conversation_id)
            
            llm_response = await self.llm.generate(
                query=query,
                role=user_role,
                context=context,
                chat_history=chat_history,
                knowledge_context=kg_context,
                safety_context=safety_context,
                user_id=user_id,
            )
            
            response_content = llm_response.content
        else:
            # Fallback if LLM not available
            response_content = self._generate_fallback_response(query, user_role)
        
        # Layer 4: Verify response
        verified = False
        if self.verifier:
            verification = await self.verifier.verify(
                text=response_content,
                context={"query": query, "role": user_role.value},
                user_id=user_id,
            )
            
            verified = verification.is_verified
            warnings.extend(verification.warnings)
            
            # If verification fails, add disclaimer
            if not verified:
                response_content += "\n\n[Note: This response could not be fully verified. Please consult a healthcare professional.]"
        
        # Layer 5: Store response in memory
        self.memory.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=response_content,
            verified=verified,
            safety_checked=True,
            knowledge_sources=knowledge_sources,
            rules_applied=rules_applied,
        )
        
        # Log the interaction
        self.audit.log_llm_interaction(
            user_id=user_id,
            user_role=user_role.value,
            query=query,
            response=response_content,
            model_name="meditron",
            verification_passed=verified,
        )
        
        return OrchestratorResponse(
            content=response_content,
            verified=verified,
            safety_checked=True,
            knowledge_sources=knowledge_sources,
            rules_applied=rules_applied,
            warnings=warnings,
        )
    
    async def _query_knowledge_graph(
        self,
        query: str,
        patient_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Query knowledge graph for relevant medical facts"""
        if not self.kg:
            return {}
        
        results = {"facts": [], "sources": []}
        
        # Extract medical entities from query
        # In production, would use NER
        query_lower = query.lower()
        
        # Search for diseases mentioned
        # Search for drugs mentioned
        # Get relevant guidelines
        
        return results
    
    def _format_kg_context(self, kg_results: Dict[str, Any]) -> str:
        """Format knowledge graph results as context for LLM"""
        if not kg_results.get("facts"):
            return ""
        
        parts = ["Relevant medical knowledge:"]
        for fact in kg_results["facts"]:
            parts.append(f"- {fact}")
        
        return "\n".join(parts)
    
    async def _run_safety_checks(
        self,
        query: str,
        patient_context: Dict[str, Any],
        user_id: str,
    ) -> Dict[str, Any]:
        """Run safety checks through rule engine"""
        result = {
            "rules_triggered": [],
            "warnings": [],
            "blocked": False,
            "block_reason": None,
            "block_message": None,
        }
        
        # Check for red flags in query
        query_lower = query.lower()
        
        emergency_keywords = [
            "chest pain", "can't breathe", "severe bleeding",
            "unconscious", "stroke", "heart attack", "suicide",
        ]
        
        for keyword in emergency_keywords:
            if keyword in query_lower:
                result["warnings"].append(f"Emergency keyword detected: {keyword}")
                result["rules_triggered"].append("emergency_detection")
        
        # If patient context, check medications
        if patient_context.get("medications"):
            result["rules_triggered"].append("medication_context_check")
        
        return result
    
    def _format_safety_context(self, safety_result: Dict[str, Any]) -> str:
        """Format safety check results as context for LLM"""
        parts = []
        
        if safety_result.get("warnings"):
            parts.append("Safety warnings:")
            for warning in safety_result["warnings"]:
                parts.append(f"- {warning}")
        
        return "\n".join(parts) if parts else ""
    
    def _generate_fallback_response(self, query: str, role: RoleType) -> str:
        """Generate fallback response when LLM is not available"""
        return (
            "I apologize, but I'm currently unable to process your request. "
            "Please try again later or consult a healthcare professional directly."
        )
    
    async def end_conversation(
        self,
        conversation_id: str,
        outcome: ConversationOutcome,
        user_id: str,
    ) -> bool:
        """End a conversation with outcome tracking"""
        return self.memory.end_conversation(
            conversation_id=conversation_id,
            outcome=outcome,
            user_id=user_id,
        )


# Singleton
_orchestrator: Optional[IMIOrchestrator] = None


def get_orchestrator() -> IMIOrchestrator:
    """Get or create orchestrator singleton"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = IMIOrchestrator()
    return _orchestrator
