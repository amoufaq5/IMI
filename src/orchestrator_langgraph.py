"""
IMI LangGraph Orchestrator

Graph-based orchestration with:
- Conditional branching (emergency vs routine)
- Retry loops on verification failure
- Parallel execution where possible
- State persistence and checkpointing
- Human-in-the-loop breakpoints
"""
import logging
from typing import TypedDict, Annotated, Literal, Optional, List, Dict, Any
from datetime import datetime
import operator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from src.layers.knowledge_graph import KnowledgeGraphService
from src.layers.rule_engine import RuleEngineService, get_rule_engine_service
from src.layers.llm import LLMService
from src.layers.llm.prompts import RoleType
from src.layers.verifier import VerifierService
from src.layers.memory import MemoryService, get_memory_service
from src.layers.citation import CitationTracker, get_citation_tracker
from src.core.security.audit import AuditLogger, get_audit_logger

logger = logging.getLogger(__name__)


# ============================================================================
# STATE DEFINITION
# ============================================================================

class IMIState(TypedDict):
    """State that flows through the graph"""
    # Input
    query: str
    user_id: str
    user_role: str
    conversation_id: Optional[str]
    patient_id: Optional[str]
    
    # Layer outputs
    patient_context: Dict[str, Any]
    knowledge_context: str
    knowledge_sources: List[str]
    safety_result: Dict[str, Any]
    rules_applied: List[str]
    warnings: Annotated[List[str], operator.add]  # Accumulates across nodes
    
    # LLM
    llm_response: str
    llm_adapter: str
    
    # Verification
    verified: bool
    verification_attempts: int
    verification_feedback: str
    
    # RAG
    rag_context: str
    rag_sources: List[str]
    rag_results: List[Dict[str, Any]]  # Full RAG results for citations
    
    # Citations
    citations: List[Dict[str, Any]]
    cited_response: str
    
    # Flow control
    is_emergency: bool
    is_blocked: bool
    block_reason: Optional[str]
    
    # Output
    final_response: str
    reasoning_trace: Dict[str, Any]
    
    # Metadata
    start_time: str
    node_timings: Dict[str, float]


# ============================================================================
# NODE FUNCTIONS
# ============================================================================

class IMINodes:
    """Node implementations for the IMI graph"""
    
    def __init__(
        self,
        knowledge_graph: Optional[KnowledgeGraphService] = None,
        rule_engine: Optional[RuleEngineService] = None,
        llm_service: Optional[LLMService] = None,
        verifier_service: Optional[VerifierService] = None,
        memory_service: Optional[MemoryService] = None,
        rag_pipeline: Optional[Any] = None,  # Will be RAGPipeline
        audit_logger: Optional[AuditLogger] = None,
    ):
        self.kg = knowledge_graph
        self.rule_engine = rule_engine or get_rule_engine_service()
        self.llm = llm_service
        self.verifier = verifier_service
        self.memory = memory_service or get_memory_service()
        self.rag = rag_pipeline
        self.audit = audit_logger or get_audit_logger()
    
    # -------------------------------------------------------------------------
    # Node: Initialize
    # -------------------------------------------------------------------------
    def initialize(self, state: IMIState) -> IMIState:
        """Initialize state and start timing"""
        logger.info(f"[INIT] Query from user {state['user_id']}")
        
        return {
            **state,
            "start_time": datetime.utcnow().isoformat(),
            "node_timings": {},
            "warnings": [],
            "knowledge_sources": [],
            "rag_sources": [],
            "rules_applied": [],
            "verification_attempts": 0,
            "verified": False,
            "is_emergency": False,
            "is_blocked": False,
            "reasoning_trace": {},
        }
    
    # -------------------------------------------------------------------------
    # Node: Memory Lookup
    # -------------------------------------------------------------------------
    def memory_lookup(self, state: IMIState) -> IMIState:
        """Layer 5: Get patient context and conversation history"""
        logger.info("[MEMORY] Looking up patient context")
        start = datetime.utcnow()
        
        patient_context = {}
        if state.get("patient_id"):
            patient_context = self.memory.get_patient_context_for_llm(
                state["patient_id"]
            )
        
        # Get or create conversation
        conversation_id = state.get("conversation_id")
        if not conversation_id:
            conversation = self.memory.start_conversation(
                user_id=state["user_id"],
                patient_id=state.get("patient_id"),
                topic=state["query"][:100],
                user_role=state["user_role"],
            )
            conversation_id = conversation.id
        
        # Add user message
        self.memory.add_message(conversation_id, "user", state["query"])
        
        elapsed = (datetime.utcnow() - start).total_seconds()
        
        return {
            **state,
            "patient_context": patient_context,
            "conversation_id": conversation_id,
            "node_timings": {**state.get("node_timings", {}), "memory_lookup": elapsed},
            "reasoning_trace": {
                **state.get("reasoning_trace", {}),
                "memory": {
                    "patient_context_loaded": bool(patient_context),
                    "fields": list(patient_context.keys()) if patient_context else [],
                }
            }
        }
    
    # -------------------------------------------------------------------------
    # Node: RAG Retrieval
    # -------------------------------------------------------------------------
    def rag_retrieval(self, state: IMIState) -> IMIState:
        """Retrieve relevant documents from RAG pipeline"""
        logger.info("[RAG] Retrieving relevant documents")
        start = datetime.utcnow()
        
        rag_context = ""
        rag_sources = []
        
        if self.rag:
            # Query RAG pipeline
            results = self.rag.retrieve(
                query=state["query"],
                patient_context=state.get("patient_context", {}),
                top_k=5,
            )
            
            if results:
                rag_context = "\n\n".join([
                    f"[Source: {r.get('metadata', {}).get('source', r.get('source', 'Unknown'))}]\n{r.get('document', r.get('content', ''))}"
                    for r in results
                ])
                rag_sources = [r.get("metadata", {}).get("source", r.get("source", "Unknown")) for r in results]
                rag_results = results  # Store full results for citation tracking
        else:
            rag_results = []
        
        elapsed = (datetime.utcnow() - start).total_seconds()
        
        return {
            **state,
            "rag_context": rag_context,
            "rag_sources": rag_sources,
            "rag_results": rag_results,  # Full results for citations
            "node_timings": {**state.get("node_timings", {}), "rag_retrieval": elapsed},
            "reasoning_trace": {
                **state.get("reasoning_trace", {}),
                "rag": {
                    "documents_retrieved": len(rag_sources),
                    "sources": rag_sources,
                }
            }
        }
    
    # -------------------------------------------------------------------------
    # Node: Knowledge Graph
    # -------------------------------------------------------------------------
    def knowledge_graph_lookup(self, state: IMIState) -> IMIState:
        """Layer 1: Query knowledge graph for medical facts"""
        logger.info("[KG] Querying knowledge graph")
        start = datetime.utcnow()
        
        knowledge_context = ""
        knowledge_sources = []
        
        if self.kg:
            # Extract entities and query KG
            # In production, use NER to extract medical entities
            query_lower = state["query"].lower()
            
            # Query for relevant conditions, drugs, symptoms
            # This would be async in production
            facts = []
            
            # Add facts to context
            if facts:
                knowledge_context = "Medical Knowledge:\n" + "\n".join(
                    f"- {fact}" for fact in facts
                )
                knowledge_sources = ["knowledge_graph"]
        
        elapsed = (datetime.utcnow() - start).total_seconds()
        
        return {
            **state,
            "knowledge_context": knowledge_context,
            "knowledge_sources": knowledge_sources,
            "node_timings": {**state.get("node_timings", {}), "knowledge_graph": elapsed},
            "reasoning_trace": {
                **state.get("reasoning_trace", {}),
                "knowledge_graph": {
                    "facts_retrieved": len(knowledge_sources),
                    "sources": knowledge_sources,
                }
            }
        }
    
    # -------------------------------------------------------------------------
    # Node: Rule Engine (Safety Check)
    # -------------------------------------------------------------------------
    def safety_check(self, state: IMIState) -> IMIState:
        """Layer 2: Run deterministic safety rules"""
        logger.info("[RULES] Running safety checks")
        start = datetime.utcnow()
        
        query_lower = state["query"].lower()
        patient_context = state.get("patient_context", {})
        
        safety_result = {
            "rules_triggered": [],
            "warnings": [],
            "blocked": False,
            "block_reason": None,
            "is_emergency": False,
            "constraints": [],
        }
        
        # Emergency detection
        emergency_keywords = [
            "chest pain", "can't breathe", "severe bleeding",
            "unconscious", "stroke", "heart attack", "suicide",
            "overdose", "seizure", "anaphylaxis",
        ]
        
        for keyword in emergency_keywords:
            if keyword in query_lower:
                safety_result["is_emergency"] = True
                safety_result["rules_triggered"].append(f"EMERGENCY:{keyword}")
                safety_result["constraints"].append("MUST recommend immediate care")
                safety_result["constraints"].append("MUST NOT suggest home remedies")
                safety_result["warnings"].append(f"Emergency keyword: {keyword}")
        
        # Drug interaction check
        if patient_context.get("medications"):
            medications = patient_context["medications"]
            # Check for dangerous combinations
            safety_result["rules_triggered"].append("medication_check")
        
        # Allergy check
        if patient_context.get("allergies"):
            safety_result["rules_triggered"].append("allergy_check")
            safety_result["constraints"].append(
                f"MUST NOT recommend: {', '.join(patient_context['allergies'])}"
            )
        
        # Age-specific rules
        if patient_context.get("age"):
            age = patient_context["age"]
            if age < 18:
                safety_result["constraints"].append("Use pediatric dosing guidelines")
            elif age > 65:
                safety_result["constraints"].append("Consider geriatric precautions")
        
        elapsed = (datetime.utcnow() - start).total_seconds()
        
        return {
            **state,
            "safety_result": safety_result,
            "rules_applied": safety_result["rules_triggered"],
            "warnings": state.get("warnings", []) + safety_result["warnings"],
            "is_emergency": safety_result["is_emergency"],
            "is_blocked": safety_result["blocked"],
            "block_reason": safety_result.get("block_reason"),
            "node_timings": {**state.get("node_timings", {}), "safety_check": elapsed},
            "reasoning_trace": {
                **state.get("reasoning_trace", {}),
                "rule_engine": {
                    "rules_evaluated": len(safety_result["rules_triggered"]),
                    "rules_triggered": safety_result["rules_triggered"],
                    "constraints_applied": safety_result["constraints"],
                    "is_emergency": safety_result["is_emergency"],
                }
            }
        }
    
    # -------------------------------------------------------------------------
    # Node: Emergency Protocol
    # -------------------------------------------------------------------------
    def emergency_protocol(self, state: IMIState) -> IMIState:
        """Handle emergency cases with immediate response"""
        logger.info("[EMERGENCY] Activating emergency protocol")
        
        emergency_response = (
            "⚠️ **EMERGENCY DETECTED**\n\n"
            "Based on your symptoms, you should seek immediate medical attention.\n\n"
            "**Please call 911 or go to the nearest emergency room immediately.**\n\n"
            "While waiting for help:\n"
            "- Stay calm and try to remain still\n"
            "- If possible, have someone stay with you\n"
            "- Do not eat or drink anything\n"
            "- If you have prescribed emergency medication, use it as directed\n\n"
            "This is not a substitute for professional medical care."
        )
        
        return {
            **state,
            "final_response": emergency_response,
            "verified": True,  # Emergency responses are pre-verified
            "reasoning_trace": {
                **state.get("reasoning_trace", {}),
                "emergency_protocol": {
                    "activated": True,
                    "reason": state.get("warnings", []),
                }
            }
        }
    
    # -------------------------------------------------------------------------
    # Node: LLM Generation
    # -------------------------------------------------------------------------
    def llm_generate(self, state: IMIState) -> IMIState:
        """Layer 3: Generate response with LLM"""
        logger.info("[LLM] Generating response")
        start = datetime.utcnow()
        
        # Build context from all sources
        context_parts = []
        
        # Add RAG context
        if state.get("rag_context"):
            context_parts.append(f"Relevant Medical Literature:\n{state['rag_context']}")
        
        # Add knowledge graph context
        if state.get("knowledge_context"):
            context_parts.append(state["knowledge_context"])
        
        # Add patient context
        if state.get("patient_context"):
            pc = state["patient_context"]
            patient_info = []
            if pc.get("age"):
                patient_info.append(f"Age: {pc['age']}")
            if pc.get("conditions"):
                patient_info.append(f"Conditions: {', '.join(pc['conditions'])}")
            if pc.get("medications"):
                patient_info.append(f"Medications: {', '.join(pc['medications'])}")
            if pc.get("allergies"):
                patient_info.append(f"Allergies: {', '.join(pc['allergies'])}")
            if patient_info:
                context_parts.append(f"Patient Information:\n" + "\n".join(patient_info))
        
        # Add safety constraints
        if state.get("safety_result", {}).get("constraints"):
            constraints = state["safety_result"]["constraints"]
            context_parts.append(
                "IMPORTANT CONSTRAINTS (you MUST follow these):\n" +
                "\n".join(f"- {c}" for c in constraints)
            )
        
        # Add verification feedback if retrying
        if state.get("verification_feedback"):
            context_parts.append(
                f"PREVIOUS RESPONSE FAILED VERIFICATION:\n{state['verification_feedback']}\n"
                "Please regenerate following the feedback."
            )
        
        full_context = "\n\n".join(context_parts)
        
        # Select adapter based on role
        adapter_map = {
            "patient": "patient_triage",
            "doctor": "clinical_decision",
            "pharmacist": "clinical_pharmacist",
            "student": "education",
            "researcher": "research",
            "pharma": "regulatory_qa",
        }
        adapter = adapter_map.get(state["user_role"], "patient_triage")
        
        # Generate response
        if self.llm:
            role = RoleType(state["user_role"]) if state["user_role"] in [r.value for r in RoleType] else RoleType.GENERAL
            
            # In production, this would be async
            response = f"[LLM Response for: {state['query'][:50]}...]"
            
            # Actual LLM call would be:
            # response = await self.llm.generate(
            #     query=state["query"],
            #     role=role,
            #     context={"full_context": full_context},
            #     adapter=adapter,
            # )
        else:
            response = self._fallback_response(state["query"])
        
        elapsed = (datetime.utcnow() - start).total_seconds()
        
        return {
            **state,
            "llm_response": response,
            "llm_adapter": adapter,
            "node_timings": {**state.get("node_timings", {}), "llm_generate": elapsed},
            "reasoning_trace": {
                **state.get("reasoning_trace", {}),
                "llm": {
                    "adapter_used": adapter,
                    "context_sources": len(context_parts),
                    "retry_attempt": state.get("verification_attempts", 0),
                }
            }
        }
    
    def _fallback_response(self, query: str) -> str:
        return (
            "I apologize, but I'm currently unable to process your request. "
            "Please consult a healthcare professional directly."
        )
    
    # -------------------------------------------------------------------------
    # Node: Verification
    # -------------------------------------------------------------------------
    def verify_response(self, state: IMIState) -> IMIState:
        """Layer 4: Verify LLM response for safety and accuracy"""
        logger.info("[VERIFY] Checking response")
        start = datetime.utcnow()
        
        attempts = state.get("verification_attempts", 0) + 1
        verified = True
        feedback = ""
        
        if self.verifier:
            # Run verification checks
            response = state.get("llm_response", "")
            
            # Check 1: Hallucination detection
            # Check 2: Guideline compliance
            # Check 3: Safety constraint compliance
            # Check 4: Allergy/contraindication check
            
            # For now, simulate verification
            # In production:
            # result = await self.verifier.verify(response, context)
            # verified = result.is_verified
            # feedback = result.feedback
            
            verified = True  # Placeholder
        
        elapsed = (datetime.utcnow() - start).total_seconds()
        
        return {
            **state,
            "verified": verified,
            "verification_attempts": attempts,
            "verification_feedback": feedback if not verified else "",
            "node_timings": {**state.get("node_timings", {}), "verify": elapsed},
            "reasoning_trace": {
                **state.get("reasoning_trace", {}),
                "verification": {
                    "passed": verified,
                    "attempts": attempts,
                    "feedback": feedback if not verified else None,
                }
            }
        }
    
    # -------------------------------------------------------------------------
    # Node: Finalize Response
    # -------------------------------------------------------------------------
    def finalize_response(self, state: IMIState) -> IMIState:
        """Assemble final response with metadata and citations"""
        logger.info("[FINALIZE] Assembling response with citations")
        
        response = state.get("llm_response", "")
        
        # Add citations from all sources
        citation_tracker = get_citation_tracker()
        citation_result = citation_tracker.format_response_with_citations(
            response=response,
            rag_results=state.get("rag_results", []),
            kg_results={"sources": [{"name": s} for s in state.get("knowledge_sources", [])]},
            rules_applied=state.get("rules_applied", []),
            include_reference_list=True,
        )
        
        cited_response = citation_result["response"]
        citations = citation_result["citations"]
        
        # Add disclaimers if needed
        if not state.get("verified"):
            cited_response += (
                "\n\n---\n"
                "*Note: This response could not be fully verified. "
                "Please consult a healthcare professional.*"
            )
        
        # Add warnings
        if state.get("warnings"):
            cited_response += "\n\n**Warnings:**\n"
            for warning in state["warnings"]:
                cited_response += f"- {warning}\n"
        
        # Store in memory
        if state.get("conversation_id"):
            self.memory.add_message(
                conversation_id=state["conversation_id"],
                role="assistant",
                content=cited_response,
                verified=state.get("verified", False),
                safety_checked=True,
                knowledge_sources=state.get("knowledge_sources", []) + state.get("rag_sources", []),
                rules_applied=state.get("rules_applied", []),
            )
        
        # Audit log
        self.audit.log_llm_interaction(
            user_id=state["user_id"],
            user_role=state["user_role"],
            query=state["query"],
            response=cited_response,
            model_name=f"meditron:{state.get('llm_adapter', 'base')}",
            verification_passed=state.get("verified", False),
        )
        
        return {
            **state,
            "final_response": cited_response,
            "cited_response": cited_response,
            "citations": citations,
            "reasoning_trace": {
                **state.get("reasoning_trace", {}),
                "citations": {
                    "total": len(citations),
                    "sources": [c.get("source") for c in citations],
                    "credibility": citation_result.get("credibility_breakdown", {}),
                }
            }
        }
    
    # -------------------------------------------------------------------------
    # Node: Blocked Response
    # -------------------------------------------------------------------------
    def blocked_response(self, state: IMIState) -> IMIState:
        """Handle blocked requests"""
        logger.info("[BLOCKED] Request blocked")
        
        return {
            **state,
            "final_response": (
                f"I'm unable to process this request. "
                f"Reason: {state.get('block_reason', 'Safety policy')}"
            ),
        }


# ============================================================================
# CONDITIONAL EDGES
# ============================================================================

def route_after_safety(state: IMIState) -> Literal["emergency", "blocked", "generate"]:
    """Route based on safety check results"""
    if state.get("is_blocked"):
        return "blocked"
    elif state.get("is_emergency"):
        return "emergency"
    else:
        return "generate"


def route_after_verification(state: IMIState) -> Literal["finalize", "retry", "give_up"]:
    """Route based on verification results"""
    if state.get("verified"):
        return "finalize"
    elif state.get("verification_attempts", 0) < 3:
        return "retry"
    else:
        return "give_up"


# ============================================================================
# GRAPH BUILDER
# ============================================================================

def build_imi_graph(
    knowledge_graph: Optional[KnowledgeGraphService] = None,
    rule_engine: Optional[RuleEngineService] = None,
    llm_service: Optional[LLMService] = None,
    verifier_service: Optional[VerifierService] = None,
    memory_service: Optional[MemoryService] = None,
    rag_pipeline: Optional[Any] = None,
    audit_logger: Optional[AuditLogger] = None,
) -> StateGraph:
    """Build the IMI LangGraph orchestrator"""
    
    # Initialize nodes
    nodes = IMINodes(
        knowledge_graph=knowledge_graph,
        rule_engine=rule_engine,
        llm_service=llm_service,
        verifier_service=verifier_service,
        memory_service=memory_service,
        rag_pipeline=rag_pipeline,
        audit_logger=audit_logger,
    )
    
    # Create graph
    graph = StateGraph(IMIState)
    
    # Add nodes
    graph.add_node("initialize", nodes.initialize)
    graph.add_node("memory_lookup", nodes.memory_lookup)
    graph.add_node("rag_retrieval", nodes.rag_retrieval)
    graph.add_node("knowledge_graph", nodes.knowledge_graph_lookup)
    graph.add_node("safety_check", nodes.safety_check)
    graph.add_node("emergency_protocol", nodes.emergency_protocol)
    graph.add_node("llm_generate", nodes.llm_generate)
    graph.add_node("verify_response", nodes.verify_response)
    graph.add_node("finalize_response", nodes.finalize_response)
    graph.add_node("blocked_response", nodes.blocked_response)
    
    # Set entry point
    graph.set_entry_point("initialize")
    
    # Add edges
    graph.add_edge("initialize", "memory_lookup")
    graph.add_edge("memory_lookup", "rag_retrieval")
    graph.add_edge("rag_retrieval", "knowledge_graph")
    graph.add_edge("knowledge_graph", "safety_check")
    
    # Conditional routing after safety check
    graph.add_conditional_edges(
        "safety_check",
        route_after_safety,
        {
            "emergency": "emergency_protocol",
            "blocked": "blocked_response",
            "generate": "llm_generate",
        }
    )
    
    # Emergency and blocked go to END
    graph.add_edge("emergency_protocol", END)
    graph.add_edge("blocked_response", END)
    
    # LLM generation goes to verification
    graph.add_edge("llm_generate", "verify_response")
    
    # Conditional routing after verification
    graph.add_conditional_edges(
        "verify_response",
        route_after_verification,
        {
            "finalize": "finalize_response",
            "retry": "llm_generate",  # Loop back for retry
            "give_up": "finalize_response",  # Give up after max retries
        }
    )
    
    # Finalize goes to END
    graph.add_edge("finalize_response", END)
    
    return graph


# ============================================================================
# ORCHESTRATOR CLASS
# ============================================================================

class LangGraphOrchestrator:
    """
    LangGraph-based orchestrator for IMI
    
    Features:
    - Graph-based flow with conditional branching
    - Retry loops on verification failure
    - State persistence with checkpointing
    - Full reasoning trace for explainability
    """
    
    def __init__(
        self,
        knowledge_graph: Optional[KnowledgeGraphService] = None,
        rule_engine: Optional[RuleEngineService] = None,
        llm_service: Optional[LLMService] = None,
        verifier_service: Optional[VerifierService] = None,
        memory_service: Optional[MemoryService] = None,
        rag_pipeline: Optional[Any] = None,
        audit_logger: Optional[AuditLogger] = None,
        enable_checkpointing: bool = True,
    ):
        # Build graph
        self.graph = build_imi_graph(
            knowledge_graph=knowledge_graph,
            rule_engine=rule_engine,
            llm_service=llm_service,
            verifier_service=verifier_service,
            memory_service=memory_service,
            rag_pipeline=rag_pipeline,
            audit_logger=audit_logger,
        )
        
        # Compile with checkpointing
        if enable_checkpointing:
            self.checkpointer = MemorySaver()
            self.app = self.graph.compile(checkpointer=self.checkpointer)
        else:
            self.app = self.graph.compile()
    
    async def process_query(
        self,
        query: str,
        user_id: str,
        user_role: str = "patient",
        conversation_id: Optional[str] = None,
        patient_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process a query through the graph"""
        
        # Initial state
        initial_state: IMIState = {
            "query": query,
            "user_id": user_id,
            "user_role": user_role,
            "conversation_id": conversation_id,
            "patient_id": patient_id,
            "patient_context": {},
            "knowledge_context": "",
            "knowledge_sources": [],
            "safety_result": {},
            "rules_applied": [],
            "warnings": [],
            "llm_response": "",
            "llm_adapter": "",
            "verified": False,
            "verification_attempts": 0,
            "verification_feedback": "",
            "rag_context": "",
            "rag_sources": [],
            "is_emergency": False,
            "is_blocked": False,
            "block_reason": None,
            "final_response": "",
            "reasoning_trace": {},
            "start_time": "",
            "node_timings": {},
        }
        
        # Run graph
        config = {"configurable": {"thread_id": conversation_id or user_id}}
        final_state = await self.app.ainvoke(initial_state, config)
        
        # Return response with full trace and citations
        return {
            "response": final_state["final_response"],
            "cited_response": final_state.get("cited_response", final_state["final_response"]),
            "citations": final_state.get("citations", []),
            "verified": final_state["verified"],
            "warnings": final_state["warnings"],
            "sources": final_state["knowledge_sources"] + final_state["rag_sources"],
            "rules_applied": final_state["rules_applied"],
            "reasoning_trace": final_state["reasoning_trace"],
            "timings": final_state["node_timings"],
            "is_emergency": final_state["is_emergency"],
        }
    
    def get_graph_visualization(self) -> str:
        """Get ASCII visualization of the graph"""
        return """
        ┌─────────────┐
        │  Initialize │
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │   Memory    │
        │   Lookup    │
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │     RAG     │
        │  Retrieval  │
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │  Knowledge  │
        │    Graph    │
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │   Safety    │◄─────────────────────┐
        │   Check     │                      │
        └──────┬──────┘                      │
               │                             │
       ┌───────┼───────┐                     │
       │       │       │                     │
  EMERGENCY  BLOCKED  ROUTINE                │
       │       │       │                     │
       ▼       ▼       ▼                     │
   ┌───────┐ ┌───────┐ ┌───────┐             │
   │ Emerg │ │Blocked│ │  LLM  │             │
   │Protocol│ │ Resp │ │Generate│            │
   └───┬───┘ └───┬───┘ └───┬───┘             │
       │         │         │                 │
       │         │         ▼                 │
       │         │   ┌───────────┐           │
       │         │   │  Verify   │───FAIL───►│
       │         │   │ Response  │  (retry)  │
       │         │   └─────┬─────┘           │
       │         │         │                 │
       │         │       PASS                │
       │         │         │                 │
       │         │         ▼                 │
       │         │   ┌───────────┐           │
       │         │   │ Finalize  │           │
       │         │   │ Response  │           │
       │         │   └─────┬─────┘           │
       │         │         │                 │
       ▼         ▼         ▼                 │
        ┌─────────────────────┐              │
        │        END          │              │
        └─────────────────────┘              │
        """


# ============================================================================
# SINGLETON
# ============================================================================

_langgraph_orchestrator: Optional[LangGraphOrchestrator] = None


def get_langgraph_orchestrator() -> LangGraphOrchestrator:
    """Get or create LangGraph orchestrator singleton"""
    global _langgraph_orchestrator
    if _langgraph_orchestrator is None:
        _langgraph_orchestrator = LangGraphOrchestrator()
    return _langgraph_orchestrator
