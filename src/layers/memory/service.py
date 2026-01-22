"""
Memory Service - Layer 5 Memory and Profiling Layer

Orchestrates all memory and profiling operations:
- Patient profiles
- Entity profiles
- Conversation memory
- Outcome feedback loops
"""
from typing import Optional, List, Dict, Any
from datetime import datetime

from src.core.security.audit import AuditLogger, get_audit_logger, AuditAction
from src.core.security.hipaa import HIPAAComplianceService, get_hipaa_service

from .patient_profile import PatientProfile, PatientProfileManager, Medication, Allergy, Condition
from .entity_profile import EntityProfile, EntityProfileManager, EntityType
from .conversation_memory import ConversationMemory, Conversation, ConversationOutcome


class MemoryService:
    """
    Layer 5: Memory Service
    
    The Profiling Layer - provides:
    - Longitudinal patient profiles
    - Entity (pharma, hospital) profiles
    - Conversation memory
    - Time-aware medical history
    - Outcome feedback loops
    """
    
    def __init__(
        self,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self.patient_manager = PatientProfileManager()
        self.entity_manager = EntityProfileManager()
        self.conversation_memory = ConversationMemory()
        self.audit = audit_logger or get_audit_logger()
        self.hipaa = get_hipaa_service()
    
    # Patient Profile Operations
    def create_patient_profile(
        self,
        first_name: str,
        last_name: str,
        date_of_birth,
        user_id: Optional[str] = None,
        **kwargs,
    ) -> PatientProfile:
        """Create a new patient profile"""
        profile = self.patient_manager.create_profile(
            first_name=first_name,
            last_name=last_name,
            date_of_birth=date_of_birth,
            **kwargs,
        )
        
        self.audit.log(
            action=AuditAction.CREATE,
            description="Patient profile created",
            user_id=user_id,
            resource_type="patient_profile",
            resource_id=profile.id,
            contains_phi=True,
            phi_types=["name", "dob"],
        )
        
        return profile
    
    def get_patient_profile(
        self,
        profile_id: str,
        user_id: Optional[str] = None,
        user_role: Optional[str] = None,
    ) -> Optional[PatientProfile]:
        """Get a patient profile with audit logging"""
        profile = self.patient_manager.get_profile(profile_id)
        
        if profile:
            self.audit.log_phi_access(
                user_id=user_id or "system",
                user_role=user_role or "system",
                patient_id=profile_id,
                phi_types=["demographics", "medical_history"],
                action=AuditAction.PHI_ACCESS,
                description="Patient profile accessed",
            )
        
        return profile
    
    def get_patient_summary(
        self,
        profile_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get patient summary for clinical use"""
        return self.patient_manager.get_patient_summary(profile_id)
    
    def get_patient_context_for_llm(
        self,
        profile_id: str,
    ) -> Dict[str, Any]:
        """Get patient context formatted for LLM input"""
        profile = self.patient_manager.get_profile(profile_id)
        if not profile:
            return {}
        
        return {
            "age": profile.age,
            "gender": profile.gender.value if profile.gender else None,
            "conditions": profile.get_condition_list(),
            "medications": profile.get_medication_list(),
            "allergies": profile.get_allergy_list(),
        }
    
    def add_patient_medication(
        self,
        profile_id: str,
        medication: Medication,
        user_id: Optional[str] = None,
    ) -> bool:
        """Add medication to patient profile"""
        success = self.patient_manager.add_medication(profile_id, medication)
        
        if success:
            self.audit.log(
                action=AuditAction.PHI_MODIFY,
                description="Medication added to patient profile",
                user_id=user_id,
                resource_type="patient_profile",
                resource_id=profile_id,
                details={"medication": medication.name},
                contains_phi=True,
            )
        
        return success
    
    def add_patient_condition(
        self,
        profile_id: str,
        condition: Condition,
        user_id: Optional[str] = None,
    ) -> bool:
        """Add condition to patient profile"""
        success = self.patient_manager.add_condition(profile_id, condition)
        
        if success:
            self.audit.log(
                action=AuditAction.PHI_MODIFY,
                description="Condition added to patient profile",
                user_id=user_id,
                resource_type="patient_profile",
                resource_id=profile_id,
                details={"condition": condition.name},
                contains_phi=True,
            )
        
        return success
    
    # Entity Profile Operations
    def create_entity_profile(
        self,
        entity_type: EntityType,
        name: str,
        user_id: Optional[str] = None,
        **kwargs,
    ) -> EntityProfile:
        """Create a new entity profile"""
        profile = self.entity_manager.create_profile(
            entity_type=entity_type,
            name=name,
            **kwargs,
        )
        
        self.audit.log(
            action=AuditAction.CREATE,
            description=f"{entity_type.value} profile created",
            user_id=user_id,
            resource_type="entity_profile",
            resource_id=profile.id,
        )
        
        return profile
    
    def get_entity_profile(
        self,
        profile_id: str,
    ) -> Optional[EntityProfile]:
        """Get an entity profile"""
        return self.entity_manager.get_profile(profile_id)
    
    def get_entity_compliance_summary(
        self,
        profile_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get compliance summary for an entity"""
        return self.entity_manager.get_compliance_summary(profile_id)
    
    # Conversation Operations
    def start_conversation(
        self,
        user_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        topic: Optional[str] = None,
        user_role: Optional[str] = None,
    ) -> Conversation:
        """Start a new conversation"""
        # Get patient context if available
        medical_context = {}
        if patient_id:
            medical_context = self.get_patient_context_for_llm(patient_id)
        
        conv = self.conversation_memory.create_conversation(
            user_id=user_id,
            patient_id=patient_id,
            topic=topic,
            user_role=user_role,
            medical_context=medical_context,
        )
        
        self.audit.log(
            action=AuditAction.CREATE,
            description="Conversation started",
            user_id=user_id,
            resource_type="conversation",
            resource_id=conv.id,
            details={"topic": topic, "has_patient_context": bool(patient_id)},
        )
        
        return conv
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation"""
        return self.conversation_memory.get_conversation(conversation_id)
    
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        response_time_ms: Optional[float] = None,
        **kwargs,
    ):
        """Add a message to conversation"""
        if role == "user":
            return self.conversation_memory.add_user_message(
                conversation_id, content, **kwargs
            )
        else:
            return self.conversation_memory.add_assistant_message(
                conversation_id, content, response_time_ms, **kwargs
            )
    
    def get_conversation_context(
        self,
        conversation_id: str,
    ) -> List[Dict[str, str]]:
        """Get conversation context for LLM"""
        return self.conversation_memory.get_context_for_llm(conversation_id)
    
    def end_conversation(
        self,
        conversation_id: str,
        outcome: ConversationOutcome,
        notes: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """End a conversation with outcome"""
        success = self.conversation_memory.end_conversation(
            conversation_id, outcome, notes
        )
        
        if success:
            self.audit.log(
                action=AuditAction.UPDATE,
                description="Conversation ended",
                user_id=user_id,
                resource_type="conversation",
                resource_id=conversation_id,
                details={"outcome": outcome.value},
            )
        
        return success
    
    def record_outcome_feedback(
        self,
        conversation_id: str,
        message_id: str,
        helpful: bool,
        feedback: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """Record feedback on a conversation message"""
        success = self.conversation_memory.record_feedback(
            conversation_id, message_id, helpful, feedback
        )
        
        if success:
            self.audit.log(
                action=AuditAction.UPDATE,
                description="Feedback recorded",
                user_id=user_id,
                resource_type="conversation",
                resource_id=conversation_id,
                details={"message_id": message_id, "helpful": helpful},
            )
        
        return success
    
    # Analytics
    def get_user_conversation_history(
        self,
        user_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get conversation history for a user"""
        conversations = self.conversation_memory.get_user_conversations(user_id, limit)
        return [
            self.conversation_memory.get_conversation_summary(c.id)
            for c in conversations
        ]
    
    def get_feedback_analytics(
        self,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get analytics on user feedback"""
        conversations = self.conversation_memory.search_conversations(user_id=user_id)
        
        total_messages = 0
        helpful_count = 0
        not_helpful_count = 0
        
        for conv in conversations:
            for msg in conv.messages:
                if msg.helpful is not None:
                    total_messages += 1
                    if msg.helpful:
                        helpful_count += 1
                    else:
                        not_helpful_count += 1
        
        return {
            "total_rated_messages": total_messages,
            "helpful_count": helpful_count,
            "not_helpful_count": not_helpful_count,
            "helpfulness_rate": helpful_count / total_messages if total_messages > 0 else 0,
        }


_memory_service: Optional[MemoryService] = None


def get_memory_service() -> MemoryService:
    """Get or create memory service singleton"""
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService()
    return _memory_service
