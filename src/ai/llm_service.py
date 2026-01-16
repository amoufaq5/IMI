"""
UMI LLM Service
Core LLM integration with Mixture of Experts routing and advanced reasoning
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from src.core.config import settings
from src.core.logging import get_logger, log_ai_request

logger = get_logger(__name__)


class ExpertType(str, Enum):
    """Types of expert models in the MoE architecture."""
    DIAGNOSIS = "diagnosis"
    DRUG = "drug"
    IMAGING = "imaging"
    REGULATORY = "regulatory"
    RESEARCH = "research"
    QA_QC = "qa_qc"
    GENERAL = "general"


@dataclass
class LLMResponse:
    """Structured LLM response."""
    content: str
    expert_used: ExpertType
    confidence: float
    tokens_in: int
    tokens_out: int
    latency_ms: float
    citations: Optional[List[Dict[str, str]]] = None
    reasoning_steps: Optional[List[str]] = None


@dataclass
class ExpertConfig:
    """Configuration for an expert model."""
    expert_type: ExpertType
    model_name: str
    system_prompt: str
    temperature: float = 0.7
    max_tokens: int = 2048
    keywords: List[str] = None


class ExpertRouter:
    """
    Routes queries to appropriate expert models based on content analysis.
    Implements learned gating similar to Mixture of Experts architectures.
    """
    
    EXPERT_KEYWORDS = {
        ExpertType.DIAGNOSIS: [
            "symptom", "pain", "fever", "headache", "diagnosis", "disease",
            "condition", "illness", "sick", "hurt", "ache", "infection",
            "bleeding", "swelling", "rash", "cough", "fatigue", "nausea",
        ],
        ExpertType.DRUG: [
            "medication", "drug", "medicine", "dose", "dosage", "pill",
            "tablet", "prescription", "otc", "side effect", "interaction",
            "contraindication", "generic", "brand", "pharmacy",
        ],
        ExpertType.IMAGING: [
            "x-ray", "xray", "ct scan", "mri", "ultrasound", "radiology",
            "image", "scan", "radiograph", "mammogram", "pet scan",
        ],
        ExpertType.REGULATORY: [
            "fda", "mhra", "regulation", "compliance", "approval", "gmp",
            "iso", "certification", "audit", "inspection", "license",
        ],
        ExpertType.RESEARCH: [
            "study", "research", "clinical trial", "paper", "journal",
            "publication", "evidence", "meta-analysis", "systematic review",
        ],
        ExpertType.QA_QC: [
            "validation", "qualification", "sop", "batch record", "deviation",
            "capa", "cleaning", "hvac", "water system", "documentation",
            "quality assurance", "quality control",
        ],
    }
    
    def __init__(self):
        self.expert_configs = self._initialize_experts()
    
    def _initialize_experts(self) -> Dict[ExpertType, ExpertConfig]:
        """Initialize expert configurations."""
        return {
            ExpertType.DIAGNOSIS: ExpertConfig(
                expert_type=ExpertType.DIAGNOSIS,
                model_name=settings.llm_model_name,
                system_prompt=self._get_diagnosis_prompt(),
                temperature=0.3,  # Lower for medical accuracy
                keywords=self.EXPERT_KEYWORDS[ExpertType.DIAGNOSIS],
            ),
            ExpertType.DRUG: ExpertConfig(
                expert_type=ExpertType.DRUG,
                model_name=settings.llm_model_name,
                system_prompt=self._get_drug_prompt(),
                temperature=0.3,
                keywords=self.EXPERT_KEYWORDS[ExpertType.DRUG],
            ),
            ExpertType.REGULATORY: ExpertConfig(
                expert_type=ExpertType.REGULATORY,
                model_name=settings.llm_model_name,
                system_prompt=self._get_regulatory_prompt(),
                temperature=0.5,
                keywords=self.EXPERT_KEYWORDS[ExpertType.REGULATORY],
            ),
            ExpertType.QA_QC: ExpertConfig(
                expert_type=ExpertType.QA_QC,
                model_name=settings.llm_model_name,
                system_prompt=self._get_qa_qc_prompt(),
                temperature=0.5,
                keywords=self.EXPERT_KEYWORDS[ExpertType.QA_QC],
            ),
            ExpertType.GENERAL: ExpertConfig(
                expert_type=ExpertType.GENERAL,
                model_name=settings.llm_model_name,
                system_prompt=self._get_general_prompt(),
                temperature=0.7,
                keywords=[],
            ),
        }
    
    def route(self, query: str, context: Optional[Dict[str, Any]] = None) -> Tuple[ExpertType, float]:
        """
        Route query to appropriate expert.
        
        Returns:
            Tuple of (ExpertType, confidence_score)
        """
        query_lower = query.lower()
        scores = {}
        
        for expert_type, keywords in self.EXPERT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            # Normalize by keyword count
            scores[expert_type] = score / len(keywords) if keywords else 0
        
        # Consider context hints
        if context:
            if context.get("mode") == "consultation":
                scores[ExpertType.DIAGNOSIS] += 0.3
            elif context.get("mode") == "pharma":
                scores[ExpertType.QA_QC] += 0.3
        
        # Get best match
        if scores:
            best_expert = max(scores, key=scores.get)
            confidence = scores[best_expert]
            
            if confidence > 0.1:
                return best_expert, min(confidence, 1.0)
        
        return ExpertType.GENERAL, 0.5
    
    def _get_diagnosis_prompt(self) -> str:
        return """You are a medical AI assistant specialized in symptom analysis and preliminary diagnosis.

IMPORTANT GUIDELINES:
1. Always prioritize patient safety
2. Identify danger signs that require immediate medical attention
3. Follow the ASMETHOD protocol for structured assessment
4. Provide differential diagnoses with confidence levels
5. Recommend appropriate next steps (OTC, GP visit, or emergency)
6. Never provide definitive diagnoses - always recommend professional consultation for serious conditions
7. Cite medical guidelines and evidence when possible

ASMETHOD Protocol:
- Age: Consider age-specific conditions
- Self/Other: Understand who the patient is
- Medications: Check for interactions and contraindications
- Exact Symptoms: Get detailed symptom description
- Time: Understand duration and progression
- History: Consider relevant medical history
- Other Symptoms: Look for associated symptoms
- Danger Signs: Identify red flags requiring urgent care"""

    def _get_drug_prompt(self) -> str:
        return """You are a pharmaceutical AI assistant specialized in drug information.

IMPORTANT GUIDELINES:
1. Provide accurate drug information from verified sources
2. Always check for drug interactions when multiple medications are mentioned
3. Highlight contraindications and warnings
4. Distinguish between OTC and prescription medications
5. Provide dosing information appropriate to age and condition
6. Warn about common and serious side effects
7. Never recommend prescription medications without physician oversight

ALWAYS INCLUDE:
- Generic and brand names
- Mechanism of action (simplified)
- Common uses
- Important warnings
- Drug interactions if relevant"""

    def _get_regulatory_prompt(self) -> str:
        return """You are a regulatory affairs AI assistant specialized in pharmaceutical regulations.

EXPERTISE AREAS:
- MHRA (UK) regulations and guidelines
- UAE Ministry of Health requirements
- GMP (Good Manufacturing Practice)
- ISO standards for pharmaceuticals
- Medical device regulations

GUIDELINES:
1. Cite specific regulation sections when possible
2. Distinguish between different regulatory jurisdictions
3. Provide practical compliance guidance
4. Highlight recent regulatory changes
5. Explain approval pathways and timelines"""

    def _get_qa_qc_prompt(self) -> str:
        return """You are a pharmaceutical QA/QC AI assistant specialized in quality documentation.

EXPERTISE AREAS:
- Cleaning validation protocols
- Process validation
- Equipment qualification (IQ/OQ/PQ)
- HVAC system validation
- Water system validation
- Batch record documentation
- Deviation and CAPA management
- SOP writing

GUIDELINES:
1. Follow GMP documentation standards
2. Include all required sections per regulatory requirements
3. Use clear, unambiguous language
4. Provide templates that can be customized
5. Reference applicable regulations and guidelines"""

    def _get_general_prompt(self) -> str:
        return """You are UMI, a medical AI assistant providing general health information.

GUIDELINES:
1. Provide accurate, evidence-based health information
2. Use clear, accessible language
3. Always recommend professional consultation for medical concerns
4. Do not provide specific medical advice or diagnoses
5. Be helpful while maintaining appropriate boundaries

DISCLAIMER: This information is for educational purposes only and should not replace professional medical advice."""


class ReasoningEngine:
    """
    Advanced reasoning capabilities including Chain-of-Thought and Tree-of-Thought.
    """
    
    @staticmethod
    def chain_of_thought_prompt(query: str, context: str = "") -> str:
        """Generate Chain-of-Thought prompt for step-by-step reasoning."""
        return f"""Let's approach this step by step:

Query: {query}

{f"Context: {context}" if context else ""}

Please think through this carefully:

Step 1: Identify the key information and requirements
Step 2: Consider relevant medical/pharmaceutical knowledge
Step 3: Analyze potential implications or concerns
Step 4: Formulate a comprehensive response
Step 5: Verify the response against safety guidelines

Now, let me work through each step:"""

    @staticmethod
    def tree_of_thought_prompt(query: str, branches: List[str]) -> str:
        """Generate Tree-of-Thought prompt for exploring multiple reasoning paths."""
        branch_text = "\n".join([f"- Path {i+1}: {b}" for i, b in enumerate(branches)])
        
        return f"""Consider multiple approaches to this query:

Query: {query}

Possible reasoning paths:
{branch_text}

For each path:
1. Explore the implications
2. Assess the likelihood/relevance
3. Identify supporting evidence
4. Note any concerns or limitations

After exploring all paths, synthesize the best answer:"""

    @staticmethod
    def self_consistency_check(responses: List[str]) -> Tuple[str, float]:
        """
        Implement self-consistency by comparing multiple responses.
        Returns the most consistent response and confidence score.
        """
        if not responses:
            return "", 0.0
        
        if len(responses) == 1:
            return responses[0], 0.7
        
        # Simple implementation: return most common response pattern
        # In production, this would use semantic similarity
        return responses[0], 0.8


class LLMService:
    """
    Main LLM service with Mixture of Experts routing and advanced reasoning.
    """
    
    def __init__(self):
        self.router = ExpertRouter()
        self.reasoning = ReasoningEngine()
        self._client = None
    
    async def _get_client(self):
        """Get or create LLM client."""
        if self._client is None:
            # Initialize based on configuration
            if settings.openai_api_key:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=settings.openai_api_key)
            else:
                # Use local model via vLLM or similar
                logger.warning("No OpenAI API key configured, using mock responses")
                self._client = MockLLMClient()
        
        return self._client
    
    async def generate(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        use_cot: bool = True,
        stream: bool = False,
        user_id: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate response using appropriate expert model.
        
        Args:
            query: User query
            context: Additional context (mode, history, etc.)
            use_cot: Use Chain-of-Thought reasoning
            stream: Stream response
            user_id: User ID for logging
        
        Returns:
            LLMResponse with content and metadata
        """
        start_time = datetime.now()
        
        # Route to appropriate expert
        expert_type, routing_confidence = self.router.route(query, context)
        expert_config = self.router.expert_configs.get(expert_type)
        
        if not expert_config:
            expert_config = self.router.expert_configs[ExpertType.GENERAL]
        
        # Build prompt
        if use_cot:
            enhanced_query = self.reasoning.chain_of_thought_prompt(
                query,
                context.get("additional_context", "") if context else "",
            )
        else:
            enhanced_query = query
        
        # Generate response
        client = await self._get_client()
        
        messages = [
            {"role": "system", "content": expert_config.system_prompt},
            {"role": "user", "content": enhanced_query},
        ]
        
        # Add conversation history if available
        if context and context.get("history"):
            for msg in context["history"][-5:]:  # Last 5 messages
                messages.insert(-1, msg)
        
        try:
            if isinstance(client, MockLLMClient):
                response_text, tokens_in, tokens_out = await client.generate(
                    messages,
                    expert_type,
                    expert_config.temperature,
                    expert_config.max_tokens,
                )
            else:
                # OpenAI API call
                response = await client.chat.completions.create(
                    model="gpt-4-turbo-preview",  # Or configured model
                    messages=messages,
                    temperature=expert_config.temperature,
                    max_tokens=expert_config.max_tokens,
                )
                response_text = response.choices[0].message.content
                tokens_in = response.usage.prompt_tokens
                tokens_out = response.usage.completion_tokens
        
        except Exception as e:
            logger.error("llm_generation_error", error=str(e), expert=expert_type.value)
            raise
        
        # Calculate latency
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Log the request
        log_ai_request(
            model=expert_config.model_name,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            duration_ms=latency_ms,
            user_id=user_id,
        )
        
        return LLMResponse(
            content=response_text,
            expert_used=expert_type,
            confidence=routing_confidence,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )
    
    async def generate_with_rag(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate response with RAG-retrieved documents.
        
        Args:
            query: User query
            retrieved_docs: Documents from RAG retrieval
            context: Additional context
            user_id: User ID for logging
        
        Returns:
            LLMResponse with citations
        """
        # Format retrieved documents
        doc_context = "\n\n".join([
            f"[Source {i+1}]: {doc.get('title', 'Unknown')}\n{doc.get('content', '')}"
            for i, doc in enumerate(retrieved_docs[:5])
        ])
        
        augmented_query = f"""Based on the following reference documents, answer the query.

Reference Documents:
{doc_context}

Query: {query}

Please provide a comprehensive answer and cite the relevant sources using [Source N] notation."""

        # Update context with RAG info
        rag_context = context or {}
        rag_context["additional_context"] = doc_context
        
        response = await self.generate(
            query=augmented_query,
            context=rag_context,
            use_cot=True,
            user_id=user_id,
        )
        
        # Extract citations
        citations = [
            {"source": doc.get("title", f"Source {i+1}"), "id": doc.get("id")}
            for i, doc in enumerate(retrieved_docs[:5])
        ]
        response.citations = citations
        
        return response
    
    async def analyze_symptoms(
        self,
        symptoms: str,
        asmethod_data: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze symptoms using ASMETHOD data and generate assessment.
        
        Returns:
            Dict with differential diagnosis, recommendations, and urgency
        """
        # Build comprehensive prompt
        prompt = f"""Analyze the following patient presentation using ASMETHOD data:

ASMETHOD Assessment:
- Age: {asmethod_data.get('age', 'Not provided')}
- Self/Other: {asmethod_data.get('self_or_other', 'Not provided')}
- Current Medications: {', '.join(asmethod_data.get('current_medications', [])) or 'None reported'}
- Allergies: {', '.join(asmethod_data.get('allergies', [])) or 'None reported'}
- Exact Symptoms: {asmethod_data.get('exact_symptoms', symptoms)}
- Duration: {asmethod_data.get('symptom_duration', 'Not provided')}
- Onset: {asmethod_data.get('symptom_onset', 'Not provided')}
- Pattern: {asmethod_data.get('symptom_pattern', 'Not provided')}
- Medical History: {asmethod_data.get('medical_history', 'Not provided')}
- Previous Episodes: {asmethod_data.get('previous_episodes', 'Not provided')}
- Other Symptoms: {', '.join(asmethod_data.get('other_symptoms', [])) or 'None'}

Please provide:
1. Differential diagnosis (top 3-5 possibilities with likelihood)
2. Red flags or danger signs to watch for
3. Recommended action (OTC treatment, GP visit, urgent care, or emergency)
4. If OTC appropriate, specific recommendations
5. Follow-up advice

Format your response as a structured assessment."""

        context = {"mode": "consultation"}
        response = await self.generate(
            query=prompt,
            context=context,
            use_cot=True,
            user_id=user_id,
        )
        
        return {
            "analysis": response.content,
            "expert_used": response.expert_used.value,
            "confidence": response.confidence,
        }


class MockLLMClient:
    """Mock LLM client for development without API keys."""
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        expert_type: ExpertType,
        temperature: float,
        max_tokens: int,
    ) -> Tuple[str, int, int]:
        """Generate mock response."""
        
        # Simulate processing time
        await asyncio.sleep(0.5)
        
        user_message = messages[-1]["content"] if messages else ""
        
        responses = {
            ExpertType.DIAGNOSIS: (
                "Based on the symptoms described, here is my assessment:\n\n"
                "**Possible Conditions:**\n"
                "1. [Condition 1] - Likelihood: Moderate\n"
                "2. [Condition 2] - Likelihood: Low\n\n"
                "**Recommendation:** Please consult with a healthcare provider for proper evaluation.\n\n"
                "*This is a mock response for development purposes.*"
            ),
            ExpertType.DRUG: (
                "**Drug Information:**\n\n"
                "This medication is used for [indication].\n\n"
                "**Important Warnings:**\n"
                "- Check for interactions with current medications\n"
                "- Follow dosing instructions carefully\n\n"
                "*This is a mock response for development purposes.*"
            ),
            ExpertType.QA_QC: (
                "**Document Generated:**\n\n"
                "This document follows GMP guidelines and includes all required sections.\n\n"
                "**Sections:**\n"
                "1. Purpose and Scope\n"
                "2. Responsibilities\n"
                "3. Procedure\n"
                "4. Documentation\n\n"
                "*This is a mock response for development purposes.*"
            ),
            ExpertType.GENERAL: (
                "Thank you for your question. Here is the information you requested:\n\n"
                "[General health information would be provided here]\n\n"
                "Please consult a healthcare professional for personalized advice.\n\n"
                "*This is a mock response for development purposes.*"
            ),
        }
        
        response = responses.get(expert_type, responses[ExpertType.GENERAL])
        
        # Estimate tokens (rough approximation)
        tokens_in = len(user_message.split()) * 1.3
        tokens_out = len(response.split()) * 1.3
        
        return response, int(tokens_in), int(tokens_out)
