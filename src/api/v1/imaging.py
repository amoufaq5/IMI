"""
UMI Medical Imaging API
Medical image analysis endpoints
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_db
from src.api.deps import get_current_user
from src.models.user import User
from src.ai.vision_service import MedicalVisionService, ImageType, BodyRegion

router = APIRouter()


class ImageAnalysisResponse(BaseModel):
    """Response from image analysis."""
    image_type: str
    body_region: Optional[str]
    findings: list
    impression: str
    confidence: float
    recommendations: list
    abnormalities_detected: bool
    urgency: str
    processing_time_ms: float


@router.post("/analyze", response_model=ImageAnalysisResponse)
async def analyze_medical_image(
    file: UploadFile = File(...),
    image_type: str = Form(..., description="Type: xray, ct, mri, ultrasound, dermoscopy, lab_report"),
    body_region: Optional[str] = Form(None, description="Region: head, chest, abdomen, spine, extremity"),
    consultation_id: Optional[str] = Form(None, description="Associated consultation ID"),
    current_user: User = Depends(get_current_user),
) -> dict:
    """
    Analyze a medical image.
    
    Supported image types:
    - **xray**: X-ray images
    - **ct**: CT scan images
    - **mri**: MRI images
    - **ultrasound**: Ultrasound images
    - **dermoscopy**: Skin lesion images
    - **lab_report**: Lab report documents (OCR)
    
    Supported body regions (for xray/ct/mri):
    - **head**, **chest**, **abdomen**, **spine**, **extremity**, **pelvis**
    
    Returns findings, impression, and recommendations.
    """
    # Validate image type
    try:
        img_type = ImageType(image_type.lower())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image type. Supported: {[t.value for t in ImageType]}",
        )
    
    # Validate body region
    region = None
    if body_region:
        try:
            region = BodyRegion(body_region.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid body region. Supported: {[r.value for r in BodyRegion]}",
            )
    
    # Validate file type
    allowed_types = [
        "image/jpeg", "image/png", "image/dicom",
        "application/dicom", "application/pdf",
    ]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: JPEG, PNG, DICOM, PDF",
        )
    
    # Read file
    content = await file.read()
    
    # Validate file size (max 100MB for medical images)
    max_size = 100 * 1024 * 1024
    if len(content) > max_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File too large. Maximum size: 100MB",
        )
    
    # Analyze image
    vision_service = MedicalVisionService()
    
    try:
        result = await vision_service.analyze(
            image_source=content,
            image_type=img_type,
            body_region=region,
        )
        
        return {
            "image_type": result.image_type.value,
            "body_region": result.body_region.value if result.body_region else None,
            "findings": result.findings,
            "impression": result.impression,
            "confidence": result.confidence,
            "recommendations": result.recommendations,
            "abnormalities_detected": result.abnormalities_detected,
            "urgency": result.urgency,
            "processing_time_ms": result.processing_time_ms,
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image analysis failed: {str(e)}",
        )


@router.get("/supported-types")
async def get_supported_types(
    current_user: User = Depends(get_current_user),
) -> dict:
    """Get list of supported image types and body regions."""
    return {
        "image_types": [
            {"value": t.value, "description": _get_type_description(t)}
            for t in ImageType
        ],
        "body_regions": [
            {"value": r.value, "description": r.value.replace("_", " ").title()}
            for r in BodyRegion
        ],
    }


def _get_type_description(image_type: ImageType) -> str:
    """Get description for image type."""
    descriptions = {
        ImageType.XRAY: "X-ray radiograph",
        ImageType.CT: "Computed Tomography scan",
        ImageType.MRI: "Magnetic Resonance Imaging",
        ImageType.ULTRASOUND: "Ultrasound imaging",
        ImageType.MAMMOGRAM: "Mammography",
        ImageType.DERMOSCOPY: "Skin lesion dermoscopy",
        ImageType.FUNDUS: "Eye fundus photography",
        ImageType.LAB_REPORT: "Laboratory report (OCR)",
        ImageType.PHOTO: "General medical photograph",
    }
    return descriptions.get(image_type, image_type.value)
