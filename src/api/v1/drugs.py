"""
UMI Drugs API
Drug information, interactions, and recommendations
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel

from src.api.deps import get_current_user
from src.models.user import User

router = APIRouter()


class DrugInfo(BaseModel):
    """Drug information response."""
    id: str
    name: str
    generic_name: str
    drug_class: str
    description: str
    indications: List[str]
    dosage_forms: List[dict]
    common_side_effects: List[str]
    warnings: List[str]
    contraindications: List[str]
    pregnancy_category: Optional[str] = None


class DrugInteractionResult(BaseModel):
    """Drug interaction check result."""
    drug1: str
    drug2: str
    severity: str
    description: str
    mechanism: Optional[str] = None
    management: Optional[str] = None


class DrugSearchResult(BaseModel):
    """Drug search result."""
    id: str
    name: str
    generic_name: str
    drug_class: str
    is_otc: bool


@router.get("/search", response_model=List[DrugSearchResult])
async def search_drugs(
    q: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(20, ge=1, le=100),
    otc_only: bool = Query(False, description="Only show OTC drugs"),
    current_user: User = Depends(get_current_user),
) -> List[dict]:
    """
    Search for drugs by name.
    
    - **q**: Search query (minimum 2 characters)
    - **limit**: Maximum results (default: 20)
    - **otc_only**: Filter to OTC drugs only
    """
    # TODO: Implement actual drug search from database
    # Mock response for now
    return [
        {
            "id": "drug_1",
            "name": "Paracetamol",
            "generic_name": "Acetaminophen",
            "drug_class": "Analgesic/Antipyretic",
            "is_otc": True,
        },
        {
            "id": "drug_2",
            "name": "Ibuprofen",
            "generic_name": "Ibuprofen",
            "drug_class": "NSAID",
            "is_otc": True,
        },
    ]


@router.get("/{drug_id}", response_model=DrugInfo)
async def get_drug(
    drug_id: str,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Get detailed information about a specific drug."""
    # TODO: Implement actual drug lookup
    # Mock response for now
    return {
        "id": drug_id,
        "name": "Paracetamol",
        "generic_name": "Acetaminophen",
        "drug_class": "Analgesic/Antipyretic",
        "description": "Paracetamol is a commonly used medicine that can help treat pain and reduce a high temperature (fever).",
        "indications": [
            "Mild to moderate pain",
            "Fever",
            "Headache",
            "Muscle aches",
        ],
        "dosage_forms": [
            {"form": "Tablet", "strengths": ["500mg", "1000mg"]},
            {"form": "Liquid", "strengths": ["120mg/5ml", "250mg/5ml"]},
        ],
        "common_side_effects": [
            "Nausea (rare)",
            "Allergic reactions (rare)",
        ],
        "warnings": [
            "Do not exceed recommended dose",
            "Avoid alcohol while taking this medication",
            "Check other medications for paracetamol content",
        ],
        "contraindications": [
            "Severe liver disease",
            "Known hypersensitivity to paracetamol",
        ],
        "pregnancy_category": "B",
    }


@router.post("/interactions", response_model=List[DrugInteractionResult])
async def check_interactions(
    drugs: List[str] = Query(..., min_length=2, description="List of drug names to check"),
    current_user: User = Depends(get_current_user),
) -> List[dict]:
    """
    Check for drug-drug interactions.
    
    Provide a list of drug names to check for interactions between them.
    """
    if len(drugs) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 drugs required for interaction check",
        )
    
    # TODO: Implement actual interaction checking
    # Mock response for now
    interactions = []
    
    # Example interaction
    if "warfarin" in [d.lower() for d in drugs] and "aspirin" in [d.lower() for d in drugs]:
        interactions.append({
            "drug1": "Warfarin",
            "drug2": "Aspirin",
            "severity": "major",
            "description": "Concurrent use may increase the risk of bleeding.",
            "mechanism": "Aspirin inhibits platelet aggregation and may displace warfarin from protein binding sites.",
            "management": "Avoid combination if possible. If necessary, monitor INR closely and watch for signs of bleeding.",
        })
    
    return interactions


@router.get("/{drug_id}/alternatives", response_model=List[DrugSearchResult])
async def get_alternatives(
    drug_id: str,
    current_user: User = Depends(get_current_user),
) -> List[dict]:
    """
    Get alternative drugs for a specific medication.
    
    Returns drugs in the same therapeutic class or with similar indications.
    """
    # TODO: Implement actual alternative lookup
    return [
        {
            "id": "drug_3",
            "name": "Ibuprofen",
            "generic_name": "Ibuprofen",
            "drug_class": "NSAID",
            "is_otc": True,
        },
        {
            "id": "drug_4",
            "name": "Naproxen",
            "generic_name": "Naproxen",
            "drug_class": "NSAID",
            "is_otc": True,
        },
    ]


class OTCRecommendation(BaseModel):
    """OTC drug recommendation."""
    drug_name: str
    generic_name: str
    indication: str
    dosage: str
    frequency: str
    max_duration: str
    warnings: List[str]
    when_to_see_doctor: List[str]


@router.post("/otc-recommendation", response_model=List[OTCRecommendation])
async def get_otc_recommendation(
    symptoms: List[str] = Query(..., description="List of symptoms"),
    age: Optional[int] = Query(None, ge=0, le=120),
    allergies: Optional[List[str]] = Query(None),
    current_medications: Optional[List[str]] = Query(None),
    current_user: User = Depends(get_current_user),
) -> List[dict]:
    """
    Get OTC drug recommendations based on symptoms.
    
    **Important**: This is for informational purposes only. 
    Always read the label and consult a pharmacist if unsure.
    """
    # TODO: Implement actual recommendation logic with AI
    # Mock response for now
    recommendations = []
    
    symptom_lower = [s.lower() for s in symptoms]
    
    if any(s in symptom_lower for s in ["headache", "pain", "ache"]):
        recommendations.append({
            "drug_name": "Paracetamol",
            "generic_name": "Acetaminophen",
            "indication": "Pain relief",
            "dosage": "500mg-1000mg",
            "frequency": "Every 4-6 hours as needed",
            "max_duration": "3 days without medical advice",
            "warnings": [
                "Do not exceed 4g in 24 hours",
                "Check other medications for paracetamol content",
                "Avoid alcohol",
            ],
            "when_to_see_doctor": [
                "Pain persists for more than 3 days",
                "Fever accompanies pain",
                "Symptoms worsen",
            ],
        })
    
    if any(s in symptom_lower for s in ["cold", "runny nose", "congestion"]):
        recommendations.append({
            "drug_name": "Pseudoephedrine",
            "generic_name": "Pseudoephedrine",
            "indication": "Nasal congestion",
            "dosage": "60mg",
            "frequency": "Every 4-6 hours",
            "max_duration": "7 days",
            "warnings": [
                "May cause insomnia",
                "Avoid if you have high blood pressure",
                "Do not take late in the day",
            ],
            "when_to_see_doctor": [
                "Symptoms persist beyond 7 days",
                "Fever develops",
                "Green/yellow discharge",
            ],
        })
    
    return recommendations
