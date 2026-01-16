"""
UMI Health Information API
General health information and medical knowledge endpoints
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from src.api.deps import get_current_user
from src.models.user import User

router = APIRouter()


class DiseaseInfo(BaseModel):
    """Disease information response."""
    id: str
    name: str
    icd_code: str
    description: str
    symptoms: List[str]
    causes: List[str]
    risk_factors: List[str]
    treatment_overview: str
    when_to_see_doctor: List[str]
    prevention: List[str]


class HealthTopic(BaseModel):
    """Health topic response."""
    id: str
    title: str
    category: str
    summary: str
    content: str
    related_topics: List[str]
    sources: List[str]


class MedicalTermDefinition(BaseModel):
    """Medical term definition."""
    term: str
    definition: str
    pronunciation: Optional[str] = None
    related_terms: List[str]
    examples: List[str]


@router.get("/diseases/search", response_model=List[dict])
async def search_diseases(
    q: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
) -> List[dict]:
    """
    Search for diseases/conditions by name or symptoms.
    """
    # TODO: Implement actual disease search
    return [
        {
            "id": "disease_1",
            "name": "Common Cold",
            "icd_code": "J00",
            "category": "Respiratory",
            "brief": "Viral infection of the upper respiratory tract",
        },
        {
            "id": "disease_2",
            "name": "Influenza",
            "icd_code": "J11",
            "category": "Respiratory",
            "brief": "Viral infection causing fever, body aches, and respiratory symptoms",
        },
    ]


@router.get("/diseases/{disease_id}", response_model=DiseaseInfo)
async def get_disease(
    disease_id: str,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Get detailed information about a disease/condition."""
    # TODO: Implement actual disease lookup
    return {
        "id": disease_id,
        "name": "Common Cold",
        "icd_code": "J00",
        "description": (
            "The common cold is a viral infection of your nose and throat (upper respiratory tract). "
            "It's usually harmless, although it might not feel that way. Many types of viruses can cause a common cold."
        ),
        "symptoms": [
            "Runny or stuffy nose",
            "Sore throat",
            "Cough",
            "Congestion",
            "Slight body aches",
            "Sneezing",
            "Low-grade fever",
            "Generally feeling unwell",
        ],
        "causes": [
            "Rhinoviruses (most common)",
            "Coronaviruses",
            "RSV (Respiratory Syncytial Virus)",
            "Parainfluenza viruses",
        ],
        "risk_factors": [
            "Age (children under 6 are at greatest risk)",
            "Weakened immune system",
            "Time of year (fall and winter)",
            "Exposure to infected individuals",
            "Smoking",
        ],
        "treatment_overview": (
            "There's no cure for the common cold. Treatment focuses on relieving symptoms. "
            "Rest, fluids, and over-the-counter medications can help. Most colds resolve within 7-10 days."
        ),
        "when_to_see_doctor": [
            "Symptoms last more than 10 days",
            "Symptoms are severe or unusual",
            "Fever higher than 38.5°C (101.3°F)",
            "Fever lasting more than 3 days",
            "Shortness of breath or wheezing",
        ],
        "prevention": [
            "Wash hands frequently",
            "Avoid touching face",
            "Stay away from sick people",
            "Disinfect frequently touched surfaces",
            "Maintain healthy lifestyle",
        ],
    }


@router.get("/topics", response_model=List[dict])
async def list_health_topics(
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
) -> List[dict]:
    """
    List available health topics.
    
    Categories include:
    - general_health
    - nutrition
    - exercise
    - mental_health
    - preventive_care
    - chronic_conditions
    """
    # TODO: Implement actual topic listing
    return [
        {
            "id": "topic_1",
            "title": "Understanding Blood Pressure",
            "category": "general_health",
            "summary": "Learn about blood pressure, what the numbers mean, and how to maintain healthy levels.",
        },
        {
            "id": "topic_2",
            "title": "Healthy Sleep Habits",
            "category": "general_health",
            "summary": "Tips for improving sleep quality and understanding sleep disorders.",
        },
        {
            "id": "topic_3",
            "title": "Managing Stress",
            "category": "mental_health",
            "summary": "Effective strategies for managing stress and maintaining mental well-being.",
        },
    ]


@router.get("/topics/{topic_id}", response_model=HealthTopic)
async def get_health_topic(
    topic_id: str,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Get detailed health topic information."""
    # TODO: Implement actual topic lookup
    return {
        "id": topic_id,
        "title": "Understanding Blood Pressure",
        "category": "general_health",
        "summary": "Learn about blood pressure, what the numbers mean, and how to maintain healthy levels.",
        "content": """
# Understanding Blood Pressure

Blood pressure is the force of blood pushing against the walls of your arteries as your heart pumps blood.

## What Do the Numbers Mean?

Blood pressure is recorded as two numbers:
- **Systolic pressure** (top number): The pressure when your heart beats
- **Diastolic pressure** (bottom number): The pressure when your heart rests between beats

## Blood Pressure Categories

| Category | Systolic | Diastolic |
|----------|----------|-----------|
| Normal | Less than 120 | Less than 80 |
| Elevated | 120-129 | Less than 80 |
| High (Stage 1) | 130-139 | 80-89 |
| High (Stage 2) | 140 or higher | 90 or higher |

## Tips for Healthy Blood Pressure

1. Maintain a healthy weight
2. Exercise regularly
3. Eat a balanced diet low in sodium
4. Limit alcohol consumption
5. Don't smoke
6. Manage stress
7. Take medications as prescribed
        """,
        "related_topics": ["Heart Health", "Healthy Diet", "Exercise Guidelines"],
        "sources": ["NHS", "British Heart Foundation", "NICE Guidelines"],
    }


@router.get("/glossary", response_model=List[MedicalTermDefinition])
async def search_medical_terms(
    q: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
) -> List[dict]:
    """
    Search medical terminology glossary.
    """
    # TODO: Implement actual glossary search
    return [
        {
            "term": "Hypertension",
            "definition": "A condition in which the force of blood against artery walls is too high. Also known as high blood pressure.",
            "pronunciation": "hy-per-TEN-shun",
            "related_terms": ["Blood pressure", "Cardiovascular disease", "Systolic", "Diastolic"],
            "examples": [
                "The patient was diagnosed with hypertension and prescribed medication.",
                "Hypertension is a major risk factor for heart disease and stroke.",
            ],
        },
    ]


@router.get("/ask")
async def ask_health_question(
    question: str = Query(..., min_length=10, max_length=500, description="Health question"),
    current_user: User = Depends(get_current_user),
) -> dict:
    """
    Ask a general health question.
    
    Get AI-powered answers to health-related questions.
    
    **Note**: This is for informational purposes only and does not
    constitute medical advice. For specific health concerns, please
    consult a healthcare professional.
    """
    # TODO: Implement with AI service
    return {
        "question": question,
        "answer": (
            "Thank you for your question. Based on general medical knowledge, "
            "I can provide the following information:\n\n"
            "[AI-generated response would appear here]\n\n"
            "**Disclaimer**: This information is for educational purposes only "
            "and should not replace professional medical advice. If you have "
            "specific health concerns, please consult a healthcare provider."
        ),
        "sources": ["Medical literature", "Clinical guidelines"],
        "confidence": 0.85,
        "related_topics": ["Related Topic 1", "Related Topic 2"],
    }
