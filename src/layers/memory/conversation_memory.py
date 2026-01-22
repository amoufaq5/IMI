"""
Conversation Memory - Manages conversation history and context

Provides:
- Session-based conversation tracking
- Context window management
- Relevant history retrieval
- Outcome feedback loops
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
import uuid


class MessageRole(str, Enum):
    """Message roles in conversation"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationOutcome(str, Enum):
    """Outcome of a conversation"""
    RESOLVED = "resolved"
    REFERRED = "referred"
    ONGOING = "ongoing"
    ABANDONED = "abandoned"
    ESCALATED = "escalated"


class Message(BaseModel):
    """A single message in conversation"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Metadata
    tokens: Optional[int] = None
    verified: bool = False
    safety_checked: bool = False
    
    # Context used
    knowledge_sources: List[str] = Field(default_factory=list)
    rules_applied: List[str] = Field(default_factory=list)
    
    # User feedback
    helpful: Optional[bool] = None
    feedback: Optional[str] = None


class Conversation(BaseModel):
    """A complete conversation session"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    patient_id: Optional[str] = None
    
    # Messages
    messages: List[Message] = Field(default_factory=list)
    
    # Context
    topic: Optional[str] = None
    medical_context: Dict[str, Any] = Field(default_factory=dict)
    user_role: Optional[str] = None
    
    # Timing
    started_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    
    # Outcome
    outcome: ConversationOutcome = ConversationOutcome.ONGOING
    outcome_notes: Optional[str] = None
    
    # Metrics
    total_tokens: int = 0
    response_times_ms: List[float] = Field(default_factory=list)
    
    def add_message(
        self,
        role: MessageRole,
        content: str,
        **kwargs,
    ) -> Message:
        """Add a message to the conversation"""
        message = Message(role=role, content=content, **kwargs)
        self.messages.append(message)
        self.last_activity = datetime.utcnow()
        if message.tokens:
            self.total_tokens += message.tokens
        return message
    
    def get_history(
        self,
        max_messages: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> List[Message]:
        """Get conversation history with limits"""
        messages = self.messages
        
        if max_messages:
            messages = messages[-max_messages:]
        
        if max_tokens:
            # Take messages from end until token limit
            selected = []
            token_count = 0
            for msg in reversed(messages):
                msg_tokens = msg.tokens or len(msg.content.split()) * 1.3
                if token_count + msg_tokens > max_tokens:
                    break
                selected.insert(0, msg)
                token_count += msg_tokens
            messages = selected
        
        return messages
    
    def get_formatted_history(
        self,
        max_messages: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """Get history formatted for LLM input"""
        messages = self.get_history(max_messages)
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]
    
    def end_conversation(
        self,
        outcome: ConversationOutcome,
        notes: Optional[str] = None,
    ) -> None:
        """End the conversation with outcome"""
        self.ended_at = datetime.utcnow()
        self.outcome = outcome
        self.outcome_notes = notes
    
    @property
    def duration_seconds(self) -> float:
        """Get conversation duration in seconds"""
        end = self.ended_at or datetime.utcnow()
        return (end - self.started_at).total_seconds()
    
    @property
    def message_count(self) -> int:
        """Get total message count"""
        return len(self.messages)
    
    @property
    def average_response_time_ms(self) -> float:
        """Get average response time"""
        if not self.response_times_ms:
            return 0
        return sum(self.response_times_ms) / len(self.response_times_ms)


class ConversationMemory:
    """Manages conversation memory and retrieval"""
    
    MAX_CONTEXT_TOKENS = 4096
    MAX_HISTORY_MESSAGES = 20
    
    def __init__(self):
        self._conversations: Dict[str, Conversation] = {}
        self._user_conversations: Dict[str, List[str]] = {}  # user_id -> [conv_ids]
    
    def create_conversation(
        self,
        user_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        topic: Optional[str] = None,
        user_role: Optional[str] = None,
        medical_context: Optional[Dict[str, Any]] = None,
    ) -> Conversation:
        """Create a new conversation"""
        conv = Conversation(
            user_id=user_id,
            patient_id=patient_id,
            topic=topic,
            user_role=user_role,
            medical_context=medical_context or {},
        )
        self._conversations[conv.id] = conv
        
        if user_id:
            if user_id not in self._user_conversations:
                self._user_conversations[user_id] = []
            self._user_conversations[user_id].append(conv.id)
        
        return conv
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID"""
        return self._conversations.get(conversation_id)
    
    def add_user_message(
        self,
        conversation_id: str,
        content: str,
        **kwargs,
    ) -> Optional[Message]:
        """Add a user message to conversation"""
        conv = self._conversations.get(conversation_id)
        if not conv:
            return None
        return conv.add_message(MessageRole.USER, content, **kwargs)
    
    def add_assistant_message(
        self,
        conversation_id: str,
        content: str,
        response_time_ms: Optional[float] = None,
        **kwargs,
    ) -> Optional[Message]:
        """Add an assistant message to conversation"""
        conv = self._conversations.get(conversation_id)
        if not conv:
            return None
        
        message = conv.add_message(MessageRole.ASSISTANT, content, **kwargs)
        
        if response_time_ms:
            conv.response_times_ms.append(response_time_ms)
        
        return message
    
    def get_context_for_llm(
        self,
        conversation_id: str,
        include_system: bool = True,
    ) -> List[Dict[str, str]]:
        """Get conversation context formatted for LLM"""
        conv = self._conversations.get(conversation_id)
        if not conv:
            return []
        
        history = conv.get_formatted_history(self.MAX_HISTORY_MESSAGES)
        
        if include_system and conv.medical_context:
            # Add medical context as system message
            context_str = self._format_medical_context(conv.medical_context)
            history.insert(0, {"role": "system", "content": context_str})
        
        return history
    
    def _format_medical_context(self, context: Dict[str, Any]) -> str:
        """Format medical context for system prompt"""
        parts = ["Relevant patient/medical context:"]
        
        if "conditions" in context:
            parts.append(f"- Conditions: {', '.join(context['conditions'])}")
        if "medications" in context:
            parts.append(f"- Current medications: {', '.join(context['medications'])}")
        if "allergies" in context:
            parts.append(f"- Allergies: {', '.join(context['allergies'])}")
        if "age" in context:
            parts.append(f"- Age: {context['age']}")
        
        return "\n".join(parts)
    
    def get_user_conversations(
        self,
        user_id: str,
        limit: int = 10,
    ) -> List[Conversation]:
        """Get recent conversations for a user"""
        conv_ids = self._user_conversations.get(user_id, [])
        conversations = [
            self._conversations[cid]
            for cid in conv_ids[-limit:]
            if cid in self._conversations
        ]
        return sorted(conversations, key=lambda c: c.last_activity, reverse=True)
    
    def search_conversations(
        self,
        user_id: Optional[str] = None,
        topic: Optional[str] = None,
        outcome: Optional[ConversationOutcome] = None,
        since: Optional[datetime] = None,
    ) -> List[Conversation]:
        """Search conversations with filters"""
        results = []
        
        for conv in self._conversations.values():
            if user_id and conv.user_id != user_id:
                continue
            if topic and topic.lower() not in (conv.topic or "").lower():
                continue
            if outcome and conv.outcome != outcome:
                continue
            if since and conv.started_at < since:
                continue
            results.append(conv)
        
        return sorted(results, key=lambda c: c.last_activity, reverse=True)
    
    def record_feedback(
        self,
        conversation_id: str,
        message_id: str,
        helpful: bool,
        feedback: Optional[str] = None,
    ) -> bool:
        """Record user feedback on a message"""
        conv = self._conversations.get(conversation_id)
        if not conv:
            return False
        
        for msg in conv.messages:
            if msg.id == message_id:
                msg.helpful = helpful
                msg.feedback = feedback
                return True
        
        return False
    
    def end_conversation(
        self,
        conversation_id: str,
        outcome: ConversationOutcome,
        notes: Optional[str] = None,
    ) -> bool:
        """End a conversation"""
        conv = self._conversations.get(conversation_id)
        if not conv:
            return False
        
        conv.end_conversation(outcome, notes)
        return True
    
    def get_conversation_summary(
        self,
        conversation_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get summary of a conversation"""
        conv = self._conversations.get(conversation_id)
        if not conv:
            return None
        
        return {
            "id": conv.id,
            "topic": conv.topic,
            "message_count": conv.message_count,
            "duration_seconds": conv.duration_seconds,
            "outcome": conv.outcome.value,
            "average_response_time_ms": conv.average_response_time_ms,
            "total_tokens": conv.total_tokens,
            "started_at": conv.started_at.isoformat(),
            "ended_at": conv.ended_at.isoformat() if conv.ended_at else None,
        }
    
    def cleanup_old_conversations(
        self,
        days: int = 30,
    ) -> int:
        """Remove conversations older than specified days"""
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        to_remove = [
            cid for cid, conv in self._conversations.items()
            if conv.last_activity < cutoff
        ]
        
        for cid in to_remove:
            conv = self._conversations.pop(cid)
            if conv.user_id and conv.user_id in self._user_conversations:
                self._user_conversations[conv.user_id].remove(cid)
        
        return len(to_remove)
