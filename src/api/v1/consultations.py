"""
UMI Consultations API
ASMETHOD-based patient consultation endpoints
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_db
from src.core.exceptions import NotFoundError, ValidationError
from src.api.deps import get_current_user
from src.models.user import User
from src.models.patient import ConsultationStatus
from src.services.consultation_service import ConsultationService
from src.schemas.consultation import (
    ConsultationCreate,
    ConsultationResponse,
    ConsultationMessageRequest,
    ConsultationListResponse,
    ConsultationSummary,
    ConsultationFeedback,
    ImageUploadResponse,
    SymptomCheckRequest,
    SymptomCheckResponse,
)

router = APIRouter()


@router.post("", response_model=ConsultationResponse, status_code=status.HTTP_201_CREATED)
async def start_consultation(
    data: ConsultationCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Start a new ASMETHOD consultation session.
    
    The consultation will guide you through the ASMETHOD protocol:
    - **A**ge: Patient age
    - **S**elf or Other: Who is the patient?
    - **M**edications: Current medications and allergies
    - **E**xact Symptoms: Detailed symptom description
    - **T**ime: Duration and onset
    - **H**istory: Relevant medical history
    - **O**ther Symptoms: Associated symptoms
    - **D**anger Signs: Red flags (assessed automatically)
    """
    service = ConsultationService(db)
    
    try:
        consultation = await service.create_consultation(
            user_id=current_user.id,
            data=data,
        )
        return consultation
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )


@router.get("", response_model=ConsultationListResponse)
async def list_consultations(
    page: int = 1,
    page_size: int = 20,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    List user's consultation history.
    
    - **page**: Page number (default: 1)
    - **page_size**: Items per page (default: 20, max: 100)
    """
    page_size = min(page_size, 100)
    
    service = ConsultationService(db)
    result = await service.get_user_consultations(
        user_id=current_user.id,
        page=page,
        page_size=page_size,
    )
    
    return result


@router.get("/{consultation_id}", response_model=ConsultationResponse)
async def get_consultation(
    consultation_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Get a specific consultation by ID."""
    service = ConsultationService(db)
    
    try:
        consultation = await service.get_consultation(
            consultation_id=consultation_id,
            user_id=current_user.id,
        )
        return consultation
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Consultation not found",
        )


@router.post("/{consultation_id}/message", response_model=ConsultationResponse)
async def send_message(
    consultation_id: UUID,
    data: ConsultationMessageRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Send a message in an active consultation.
    
    The AI will analyze your response and either:
    - Ask the next ASMETHOD question
    - Detect danger signs and recommend emergency care
    - Provide assessment and recommendations when complete
    """
    service = ConsultationService(db)
    
    try:
        consultation = await service.add_message(
            consultation_id=consultation_id,
            user_id=current_user.id,
            message=data.message,
            asmethod_update=data.asmethod_update,
        )
        return consultation
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Consultation not found",
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/{consultation_id}/image", response_model=ImageUploadResponse)
async def upload_image(
    consultation_id: UUID,
    file: UploadFile = File(...),
    image_type: str = "photo",
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Upload a medical image for analysis.
    
    Supported types:
    - **photo**: General photo (skin condition, etc.)
    - **xray**: X-ray image
    - **ct**: CT scan
    - **mri**: MRI scan
    - **lab_report**: Lab report document
    
    Supported formats: JPEG, PNG, DICOM, PDF
    """
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "application/dicom", "application/pdf"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: {allowed_types}",
        )
    
    # Validate file size (max 50MB)
    max_size = 50 * 1024 * 1024
    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File too large. Maximum size: 50MB",
        )
    
    # TODO: Implement actual image storage and analysis
    # For now, return mock response
    return {
        "id": "00000000-0000-0000-0000-000000000000",
        "filename": file.filename,
        "file_type": image_type,
        "analysis_status": "pending",
        "analysis_result": None,
    }


@router.post("/{consultation_id}/feedback")
async def submit_feedback(
    consultation_id: UUID,
    data: ConsultationFeedback,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Submit feedback for a completed consultation.
    
    - **rating**: 1-5 star rating
    - **feedback**: Optional text feedback
    - **was_helpful**: Whether the consultation was helpful
    - **followed_recommendation**: Whether you followed the recommendation
    """
    service = ConsultationService(db)
    
    try:
        consultation = await service.get_consultation(
            consultation_id=consultation_id,
            user_id=current_user.id,
        )
        
        consultation.user_rating = data.rating
        consultation.user_feedback = data.feedback
        
        await db.commit()
        
        return {"message": "Feedback submitted successfully"}
    
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Consultation not found",
        )


@router.post("/symptom-check", response_model=SymptomCheckResponse)
async def quick_symptom_check(
    data: SymptomCheckRequest,
    current_user: User = Depends(get_current_user),
) -> dict:
    """
    Quick symptom check without full consultation.
    
    Provides:
    - Possible conditions
    - Urgency level
    - Whether to see a doctor
    
    **Note**: This is not a diagnosis. For detailed assessment,
    start a full consultation.
    """
    # TODO: Implement with AI service
    # For now, return mock response
    return {
        "possible_conditions": [
            {
                "condition": "Common Cold",
                "probability": 0.7,
                "icd_code": "J00",
                "reasoning": "Based on reported symptoms",
            }
        ],
        "urgency_level": "low",
        "recommendation": "Rest and stay hydrated. OTC medications may help with symptoms.",
        "should_see_doctor": False,
        "disclaimer": (
            "This is not a medical diagnosis. If symptoms worsen or persist, "
            "please consult a healthcare professional."
        ),
    }
