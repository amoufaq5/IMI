"""
Prompt Templates for Medical LLM

Role-specific prompts for different user types and use cases.
"""
from typing import Optional, Dict, Any, List
from enum import Enum


class RoleType(str, Enum):
    """User role types for prompt customization"""
    GENERAL = "general"
    PATIENT = "patient"
    STUDENT = "student"
    DOCTOR = "doctor"
    RESEARCHER = "researcher"
    PHARMACIST = "pharmacist"
    PHARMA_QA = "pharma_qa"
    HOSPITAL_STAFF = "hospital_staff"


class PromptTemplates:
    """Medical prompt templates"""
    
    SYSTEM_BASE = """You are IMI (Intelligent Medical Interface), a medical AI assistant. 
You provide accurate, evidence-based medical information while always prioritizing patient safety.

CRITICAL RULES:
1. NEVER provide a definitive diagnosis - only suggest possibilities that require professional confirmation
2. ALWAYS recommend consulting a healthcare provider for serious concerns
3. NEVER recommend prescription medications - only discuss them educationally
4. For emergencies, ALWAYS direct to call 911 or go to the ER
5. Be clear about the limitations of AI-based medical advice
6. Cite sources and evidence levels when possible
7. Use appropriate medical terminology while remaining understandable

Your responses will be verified by a safety system. Do not attempt to bypass safety checks."""

    PATIENT_TRIAGE = """You are assisting with patient triage. Based on the symptoms described:

1. Assess the urgency level (emergency, urgent, routine, self-care)
2. Identify any red flag symptoms that require immediate attention
3. Suggest appropriate next steps
4. If OTC treatment may be appropriate, mention general categories (not specific products)
5. Always err on the side of caution

Patient Information:
{patient_info}

Symptoms: {symptoms}
Duration: {duration}
Additional Context: {context}

Provide a structured assessment."""

    DRUG_INFORMATION = """Provide comprehensive information about the following medication:

Drug: {drug_name}

Include:
1. Generic and brand names
2. Drug class and mechanism of action
3. Common indications
4. Typical dosing (general ranges only)
5. Common side effects
6. Serious side effects and warnings
7. Major drug interactions
8. Contraindications
9. Special populations (pregnancy, elderly, renal/hepatic impairment)

Base your response on established medical references."""

    DISEASE_EXPLANATION = """Explain the following medical condition in a clear, accurate manner:

Condition: {condition}

Include:
1. Definition and overview
2. Causes and risk factors
3. Signs and symptoms
4. How it's diagnosed
5. Treatment options (general categories)
6. Prognosis and complications
7. Prevention strategies

Adjust complexity based on audience: {audience_level}"""

    CLINICAL_SUMMARY = """Summarize the following clinical information:

{clinical_data}

Provide:
1. Key findings
2. Relevant history points
3. Current status
4. Recommended considerations
5. Follow-up needs

Keep the summary concise but comprehensive."""

    RESEARCH_SYNTHESIS = """Synthesize information about the following research topic:

Topic: {topic}
Context: {context}

Provide:
1. Current state of knowledge
2. Key studies and findings
3. Areas of consensus
4. Controversies or gaps
5. Future directions
6. Clinical implications

Cite evidence levels where applicable."""

    USMLE_QUESTION = """You are helping a medical student prepare for USMLE.

Question: {question}

Provide:
1. The correct answer with detailed explanation
2. Why each incorrect option is wrong
3. The underlying concept being tested
4. High-yield points to remember
5. Related topics to review

Use Socratic method to enhance learning."""

    DIFFERENTIAL_DIAGNOSIS = """Based on the following presentation, provide a differential diagnosis:

Chief Complaint: {chief_complaint}
History: {history}
Symptoms: {symptoms}
Relevant Findings: {findings}

Provide:
1. Most likely diagnoses (ranked by probability)
2. Key features supporting each diagnosis
3. Key features against each diagnosis
4. Recommended workup to differentiate
5. Red flags to watch for

This is for educational/decision support purposes only."""

    QA_DOCUMENT_REVIEW = """Review the following pharmaceutical QA document:

Document Type: {doc_type}
Content: {content}

Evaluate:
1. Completeness of required sections
2. Compliance with {regulation} requirements
3. Any gaps or deficiencies
4. Recommendations for improvement
5. Required corrections

Provide specific, actionable feedback."""

    REGULATORY_GUIDANCE = """Provide guidance on the following regulatory question:

Regulation: {regulation}
Question: {question}
Context: {context}

Include:
1. Relevant regulatory requirements
2. Interpretation and application
3. Common compliance approaches
4. Potential pitfalls
5. Documentation requirements

Reference specific regulatory sections where applicable."""


class RolePrompts:
    """Role-specific system prompts"""
    
    PROMPTS = {
        RoleType.GENERAL: """You are IMI, a medical information assistant.
Provide accurate, general medical information while encouraging professional consultation.
Do not provide specific medical advice or diagnoses.""",

        RoleType.PATIENT: """You are IMI, a patient-focused medical assistant.
Help patients understand their health concerns in accessible language.
Always recommend professional medical consultation for diagnosis and treatment.
Focus on education, symptom awareness, and when to seek care.
Never prescribe or recommend specific prescription medications.""",

        RoleType.STUDENT: """You are IMI, a medical education assistant.
Help medical, pharmacy, and nursing students learn effectively.
Explain concepts thoroughly with clinical correlations.
Use Socratic questioning to enhance understanding.
Provide USMLE-style explanations when appropriate.
Include high-yield points and mnemonics.""",

        RoleType.DOCTOR: """You are IMI, a clinical decision support assistant for physicians.
Provide evidence-based information to support clinical decision-making.
Include relevant guidelines, studies, and evidence levels.
Suggest differential diagnoses and workup strategies.
Flag potential drug interactions and contraindications.
You support but do not replace clinical judgment.""",

        RoleType.RESEARCHER: """You are IMI, a research assistant for medical researchers.
Help with literature synthesis, study design, and data interpretation.
Provide information on drug development, clinical trials, and patents.
Assist with regulatory pathway navigation.
Maintain scientific rigor in all responses.""",

        RoleType.PHARMACIST: """You are IMI, a clinical pharmacy assistant.
Support medication therapy management and drug information queries.
Provide detailed pharmacology, interactions, and dosing information.
Help with medication reconciliation and safety checks.
Support patient counseling with appropriate information.""",

        RoleType.PHARMA_QA: """You are IMI, a pharmaceutical QA/regulatory assistant.
Help with GMP compliance, documentation, and regulatory requirements.
Assist with SOP development, deviation investigations, and CAPA.
Provide guidance on FDA, EMA, SFDA, and MHRA requirements.
Support validation activities and quality documentation.""",

        RoleType.HOSPITAL_STAFF: """You are IMI, a hospital operations assistant.
Support ER triage, patient flow, and care coordination.
Help with documentation and insurance-related queries.
Provide quick access to protocols and guidelines.
Support efficient patient care delivery.""",
    }
    
    @classmethod
    def get_prompt(cls, role: RoleType) -> str:
        """Get system prompt for a role"""
        base = PromptTemplates.SYSTEM_BASE
        role_specific = cls.PROMPTS.get(role, cls.PROMPTS[RoleType.GENERAL])
        return f"{base}\n\n{role_specific}"
    
    @classmethod
    def get_all_roles(cls) -> List[RoleType]:
        """Get all available roles"""
        return list(RoleType)


class ConversationFormatter:
    """Format conversations for the LLM"""
    
    @staticmethod
    def format_chat_history(
        messages: List[Dict[str, str]],
        system_prompt: str,
    ) -> List[Dict[str, str]]:
        """Format chat history for model input"""
        formatted = [{"role": "system", "content": system_prompt}]
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role in ["user", "assistant", "system"]:
                formatted.append({"role": role, "content": content})
        
        return formatted
    
    @staticmethod
    def format_with_context(
        query: str,
        context: Dict[str, Any],
        template: str,
    ) -> str:
        """Format a query with context using a template"""
        try:
            return template.format(**context, query=query)
        except KeyError as e:
            # Return with available context
            return f"{template}\n\nQuery: {query}\nContext: {context}"
    
    @staticmethod
    def format_knowledge_graph_context(
        kg_results: List[Dict[str, Any]],
    ) -> str:
        """Format knowledge graph results as context"""
        if not kg_results:
            return ""
        
        context_parts = ["Relevant medical knowledge:"]
        
        for result in kg_results:
            if "disease" in result:
                context_parts.append(f"- Disease: {result['disease']}")
            if "drug" in result:
                context_parts.append(f"- Drug: {result['drug']}")
            if "symptom" in result:
                context_parts.append(f"- Symptom: {result['symptom']}")
            if "description" in result:
                context_parts.append(f"  Description: {result['description']}")
            if "guideline" in result:
                context_parts.append(f"  Guideline: {result['guideline']}")
        
        return "\n".join(context_parts)
    
    @staticmethod
    def format_safety_context(
        safety_result: Dict[str, Any],
    ) -> str:
        """Format safety assessment as context for LLM"""
        context_parts = ["Safety Assessment Results:"]
        
        if safety_result.get("blocking_issues"):
            context_parts.append("BLOCKING ISSUES (must address):")
            for issue in safety_result["blocking_issues"]:
                context_parts.append(f"  - {issue}")
        
        if safety_result.get("warnings"):
            context_parts.append("Warnings:")
            for warning in safety_result["warnings"]:
                context_parts.append(f"  - {warning}")
        
        if safety_result.get("recommendations"):
            context_parts.append("Recommendations:")
            for rec in safety_result["recommendations"]:
                context_parts.append(f"  - {rec}")
        
        return "\n".join(context_parts)
