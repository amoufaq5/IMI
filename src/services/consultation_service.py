"""
UMI Consultation Service
ASMETHOD protocol implementation and consultation management
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import NotFoundError, ValidationError, MedicalSafetyError
from src.core.logging import get_logger
from src.models.patient import (
    Consultation,
    ConsultationStatus,
    ConsultationOutcome,
    PatientProfile,
)
from src.schemas.consultation import (
    ASMETHODData,
    ConsultationCreate,
    ConsultationMessage,
    ConsultationResponse,
    DiagnosisItem,
    DrugRecommendation,
)

logger = get_logger(__name__)


class DangerSignsDetector:
    """Detects danger signs that require immediate medical attention."""
    
    RED_FLAGS = {
        "chest_pain_breathing": {
            "keywords": ["chest pain", "difficulty breathing", "shortness of breath", "can't breathe"],
            "message": "Chest pain with breathing difficulty - possible cardiac or pulmonary emergency",
            "urgency": "emergency",
        },
        "stroke_signs": {
            "keywords": ["face drooping", "arm weakness", "speech difficulty", "sudden confusion", 
                        "sudden numbness", "sudden severe headache", "vision loss"],
            "message": "Possible stroke symptoms - FAST assessment needed",
            "urgency": "emergency",
        },
        "severe_allergic": {
            "keywords": ["throat swelling", "can't swallow", "tongue swelling", "anaphylaxis",
                        "severe allergic", "hives spreading"],
            "message": "Possible anaphylaxis - emergency treatment needed",
            "urgency": "emergency",
        },
        "severe_bleeding": {
            "keywords": ["won't stop bleeding", "heavy bleeding", "blood won't clot", "vomiting blood",
                        "blood in stool", "black stool"],
            "message": "Severe bleeding - immediate medical attention required",
            "urgency": "emergency",
        },
        "head_injury": {
            "keywords": ["head injury", "lost consciousness", "confusion after fall", "severe headache sudden"],
            "message": "Head injury with concerning symptoms",
            "urgency": "emergency",
        },
        "suicidal": {
            "keywords": ["want to die", "suicidal", "kill myself", "end my life", "self harm"],
            "message": "Mental health crisis - immediate support needed",
            "urgency": "emergency",
        },
        "meningitis": {
            "keywords": ["stiff neck", "severe headache", "sensitivity to light", "rash doesn't fade",
                        "high fever headache"],
            "message": "Possible meningitis symptoms",
            "urgency": "emergency",
        },
    }
    
    AMBER_FLAGS = {
        "prolonged_symptoms": {
            "condition": lambda data: data.get("symptom_duration", "").lower() in ["more than 7 days", "over a week", "2 weeks"],
            "message": "Symptoms persisting beyond expected duration",
            "urgency": "urgent",
        },
        "high_fever": {
            "keywords": ["high fever", "fever over 39", "fever 3 days", "persistent fever"],
            "message": "Persistent or high fever requiring evaluation",
            "urgency": "urgent",
        },
        "vulnerable_population": {
            "condition": lambda data: data.get("age", 100) < 2 or data.get("age", 0) > 65,
            "message": "Vulnerable age group - lower threshold for referral",
            "urgency": "urgent",
        },
    }
    
    @classmethod
    def detect(cls, symptoms_text: str, asmethod_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect danger signs in symptoms.
        
        Returns:
            Dict with detected flags and urgency level
        """
        symptoms_lower = symptoms_text.lower()
        detected_red = []
        detected_amber = []
        
        # Check red flags
        for flag_id, flag_info in cls.RED_FLAGS.items():
            for keyword in flag_info["keywords"]:
                if keyword in symptoms_lower:
                    detected_red.append({
                        "id": flag_id,
                        "message": flag_info["message"],
                        "urgency": flag_info["urgency"],
                    })
                    break
        
        # Check amber flags
        for flag_id, flag_info in cls.AMBER_FLAGS.items():
            if "keywords" in flag_info:
                for keyword in flag_info["keywords"]:
                    if keyword in symptoms_lower:
                        detected_amber.append({
                            "id": flag_id,
                            "message": flag_info["message"],
                            "urgency": flag_info["urgency"],
                        })
                        break
            elif "condition" in flag_info:
                if flag_info["condition"](asmethod_data):
                    detected_amber.append({
                        "id": flag_id,
                        "message": flag_info["message"],
                        "urgency": flag_info["urgency"],
                    })
        
        # Determine overall urgency
        if detected_red:
            urgency = "emergency"
        elif detected_amber:
            urgency = "urgent"
        else:
            urgency = "routine"
        
        return {
            "red_flags": detected_red,
            "amber_flags": detected_amber,
            "urgency": urgency,
            "requires_referral": len(detected_red) > 0,
        }


class ASMETHODEngine:
    """
    ASMETHOD Protocol Engine for structured patient consultation.
    
    A - Age
    S - Self or Someone Else
    M - Medications
    E - Exact Symptoms
    T - Time/Duration
    H - History
    O - Other Symptoms
    D - Danger Signs
    """
    
    PROTOCOL_STEPS = [
        {
            "step": "A",
            "name": "Age",
            "question": "How old is the patient?",
            "field": "age",
            "required": True,
        },
        {
            "step": "S",
            "name": "Self or Other",
            "question": "Is this consultation for yourself or someone else?",
            "field": "self_or_other",
            "required": True,
        },
        {
            "step": "M",
            "name": "Medications",
            "question": "Are you currently taking any medications? Do you have any known allergies?",
            "fields": ["current_medications", "allergies"],
            "required": False,
        },
        {
            "step": "E",
            "name": "Exact Symptoms",
            "question": "Please describe your symptoms in detail. Where exactly is the problem? How severe is it on a scale of 1-10?",
            "fields": ["exact_symptoms", "symptom_location", "symptom_severity"],
            "required": True,
        },
        {
            "step": "T",
            "name": "Time/Duration",
            "question": "When did the symptoms start? Did they come on suddenly or gradually? Are they constant or do they come and go?",
            "fields": ["symptom_duration", "symptom_onset", "symptom_pattern"],
            "required": True,
        },
        {
            "step": "H",
            "name": "History",
            "question": "Do you have any relevant medical history? Have you experienced this before? If so, what helped?",
            "fields": ["medical_history", "previous_episodes", "previous_treatment"],
            "required": False,
        },
        {
            "step": "O",
            "name": "Other Symptoms",
            "question": "Are you experiencing any other symptoms along with the main complaint?",
            "field": "other_symptoms",
            "required": False,
        },
        {
            "step": "D",
            "name": "Danger Signs",
            "question": None,  # This is assessed automatically
            "field": "danger_signs",
            "required": False,
        },
    ]
    
    @classmethod
    def get_current_step(cls, asmethod_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Determine the current step in the protocol based on collected data."""
        for step in cls.PROTOCOL_STEPS:
            if step["step"] == "D":
                continue  # Danger signs are assessed automatically
            
            field = step.get("field") or step.get("fields", [None])[0]
            if field and not asmethod_data.get(field):
                return step
        
        return None  # All steps completed
    
    @classmethod
    def get_next_question(cls, asmethod_data: Dict[str, Any]) -> Optional[str]:
        """Get the next question to ask based on protocol progress."""
        current_step = cls.get_current_step(asmethod_data)
        if current_step:
            return current_step["question"]
        return None
    
    @classmethod
    def is_complete(cls, asmethod_data: Dict[str, Any]) -> bool:
        """Check if all required ASMETHOD data has been collected."""
        for step in cls.PROTOCOL_STEPS:
            if not step["required"]:
                continue
            
            field = step.get("field") or step.get("fields", [None])[0]
            if field and not asmethod_data.get(field):
                return False
        
        return True
    
    @classmethod
    def get_completion_percentage(cls, asmethod_data: Dict[str, Any]) -> float:
        """Calculate protocol completion percentage."""
        total_fields = 0
        completed_fields = 0
        
        for step in cls.PROTOCOL_STEPS:
            if step["step"] == "D":
                continue
            
            fields = [step.get("field")] if step.get("field") else step.get("fields", [])
            for field in fields:
                if field:
                    total_fields += 1
                    if asmethod_data.get(field):
                        completed_fields += 1
        
        return (completed_fields / total_fields * 100) if total_fields > 0 else 0


class ConsultationService:
    """Service for managing patient consultations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.danger_detector = DangerSignsDetector()
        self.asmethod_engine = ASMETHODEngine()
    
    async def create_consultation(
        self,
        user_id: uuid.UUID,
        data: ConsultationCreate,
    ) -> Consultation:
        """Create a new consultation session."""
        
        consultation = Consultation(
            user_id=user_id,
            status=ConsultationStatus.IN_PROGRESS,
            asmethod_data={},
            messages=[],
            started_at=datetime.now(timezone.utc),
        )
        
        # Add initial system message
        system_message = {
            "role": "system",
            "content": "Starting ASMETHOD consultation protocol.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        consultation.messages.append(system_message)
        
        # Add welcome message
        welcome_message = {
            "role": "assistant",
            "content": (
                "Hello! I'm here to help assess your health concern. "
                "I'll ask you a series of questions to better understand your situation. "
                "Please note that this is not a replacement for professional medical advice.\n\n"
                "Let's start: How old is the patient?"
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        consultation.messages.append(welcome_message)
        
        # If initial message provided, process it
        if data.initial_message:
            user_message = {
                "role": "user",
                "content": data.initial_message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            consultation.messages.append(user_message)
        
        self.db.add(consultation)
        await self.db.flush()
        
        logger.info(
            "consultation_created",
            consultation_id=str(consultation.id),
            user_id=str(user_id),
        )
        
        return consultation
    
    async def get_consultation(
        self,
        consultation_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> Consultation:
        """Get a consultation by ID."""
        
        result = await self.db.execute(
            select(Consultation).where(
                Consultation.id == consultation_id,
                Consultation.user_id == user_id,
            )
        )
        consultation = result.scalar_one_or_none()
        
        if not consultation:
            raise NotFoundError("Consultation", consultation_id)
        
        return consultation
    
    async def add_message(
        self,
        consultation_id: uuid.UUID,
        user_id: uuid.UUID,
        message: str,
        asmethod_update: Optional[Dict[str, Any]] = None,
    ) -> Consultation:
        """Add a user message to the consultation and get AI response."""
        
        consultation = await self.get_consultation(consultation_id, user_id)
        
        if consultation.status != ConsultationStatus.IN_PROGRESS:
            raise ValidationError("Consultation is not in progress")
        
        # Add user message
        user_message = {
            "role": "user",
            "content": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        consultation.messages.append(user_message)
        
        # Update ASMETHOD data if provided
        if asmethod_update:
            consultation.asmethod_data.update(asmethod_update)
        
        # Check for danger signs
        all_symptoms = " ".join([
            consultation.asmethod_data.get("exact_symptoms", ""),
            " ".join(consultation.asmethod_data.get("other_symptoms", [])),
            message,
        ])
        
        danger_check = self.danger_detector.detect(all_symptoms, consultation.asmethod_data)
        
        if danger_check["red_flags"]:
            consultation.danger_signs_detected = True
            consultation.asmethod_data["danger_signs"] = [
                f["message"] for f in danger_check["red_flags"]
            ]
            
            # Generate emergency referral response
            emergency_response = self._generate_emergency_response(danger_check)
            consultation.messages.append({
                "role": "assistant",
                "content": emergency_response,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            
            consultation.status = ConsultationStatus.REFERRED
            consultation.outcome = ConsultationOutcome.EMERGENCY_REFERRAL
            consultation.referral_urgency = "emergency"
            consultation.completed_at = datetime.now(timezone.utc)
        else:
            # Continue with ASMETHOD protocol
            # This would normally call the AI service
            next_question = self.asmethod_engine.get_next_question(consultation.asmethod_data)
            
            if next_question:
                consultation.messages.append({
                    "role": "assistant",
                    "content": next_question,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            elif self.asmethod_engine.is_complete(consultation.asmethod_data):
                # Protocol complete - generate assessment
                # This would call the AI service for full analysis
                consultation.messages.append({
                    "role": "assistant",
                    "content": (
                        "Thank you for providing all the information. "
                        "I'm now analyzing your symptoms to provide recommendations. "
                        "Please wait a moment..."
                    ),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
        
        await self.db.flush()
        
        return consultation
    
    async def complete_consultation(
        self,
        consultation_id: uuid.UUID,
        user_id: uuid.UUID,
        outcome: ConsultationOutcome,
        recommendation: str,
        recommended_drugs: Optional[List[Dict[str, Any]]] = None,
        referral_specialty: Optional[str] = None,
        referral_urgency: Optional[str] = None,
    ) -> Consultation:
        """Complete a consultation with final assessment."""
        
        consultation = await self.get_consultation(consultation_id, user_id)
        
        consultation.status = ConsultationStatus.COMPLETED
        consultation.outcome = outcome
        consultation.recommendation = recommendation
        consultation.recommended_drugs = recommended_drugs
        consultation.referral_specialty = referral_specialty
        consultation.referral_urgency = referral_urgency
        consultation.completed_at = datetime.now(timezone.utc)
        
        # Add final message
        consultation.messages.append({
            "role": "assistant",
            "content": recommendation,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        await self.db.flush()
        
        logger.info(
            "consultation_completed",
            consultation_id=str(consultation_id),
            outcome=outcome.value,
        )
        
        return consultation
    
    def _generate_emergency_response(self, danger_check: Dict[str, Any]) -> str:
        """Generate emergency referral response."""
        flags = danger_check["red_flags"]
        messages = [f["message"] for f in flags]
        
        response = (
            "⚠️ **IMPORTANT: Based on your symptoms, you should seek immediate medical attention.**\n\n"
            "The following concerning signs have been detected:\n"
        )
        
        for msg in messages:
            response += f"• {msg}\n"
        
        response += (
            "\n**Please do one of the following immediately:**\n"
            "1. Call emergency services (999 in UK, 998 in UAE)\n"
            "2. Go to the nearest Emergency Room\n"
            "3. If you cannot get to emergency services, call for help\n\n"
            "This consultation has been flagged for urgent review. "
            "Do not delay seeking medical care."
        )
        
        return response
    
    async def get_user_consultations(
        self,
        user_id: uuid.UUID,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        """Get paginated list of user's consultations."""
        
        # Count total
        count_result = await self.db.execute(
            select(Consultation).where(Consultation.user_id == user_id)
        )
        total = len(count_result.scalars().all())
        
        # Get page
        offset = (page - 1) * page_size
        result = await self.db.execute(
            select(Consultation)
            .where(Consultation.user_id == user_id)
            .order_by(Consultation.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        consultations = result.scalars().all()
        
        return {
            "items": consultations,
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": (total + page_size - 1) // page_size,
        }
