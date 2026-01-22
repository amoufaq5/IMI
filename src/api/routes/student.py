"""Student API endpoints"""
from fastapi import APIRouter, Depends
from typing import Optional, List
from pydantic import BaseModel, Field

from src.domains.student import StudentService, ExamType, QuestionDifficulty
from src.core.security.authentication import get_current_user, UserContext

router = APIRouter()

student_service = StudentService()


class QuestionRequest(BaseModel):
    """USMLE question request"""
    question: str
    options: Optional[List[str]] = None
    exam_type: ExamType = ExamType.USMLE_STEP1


class ConceptRequest(BaseModel):
    """Concept explanation request"""
    concept: str
    depth: str = "intermediate"
    include_clinical: bool = True


class PracticeRequest(BaseModel):
    """Practice questions request"""
    topic: str
    count: int = 5
    difficulty: QuestionDifficulty = QuestionDifficulty.MEDIUM
    exam_type: ExamType = ExamType.USMLE_STEP1


class EssayReviewRequest(BaseModel):
    """Essay review request"""
    essay_text: str
    essay_type: str = "research_paper"


@router.post("/answer-question")
async def answer_question(
    request: QuestionRequest,
    user: UserContext = Depends(get_current_user),
):
    """Answer a USMLE-style question with detailed explanation"""
    return await student_service.answer_question(
        question=request.question,
        options=request.options,
        exam_type=request.exam_type,
        user_id=user.user_id if user else None,
    )


@router.post("/explain-concept")
async def explain_concept(
    request: ConceptRequest,
    user: UserContext = Depends(get_current_user),
):
    """Explain a medical concept"""
    return await student_service.explain_concept(
        concept=request.concept,
        depth=request.depth,
        include_clinical=request.include_clinical,
        user_id=user.user_id if user else None,
    )


@router.post("/generate-practice")
async def generate_practice_questions(
    request: PracticeRequest,
    user: UserContext = Depends(get_current_user),
):
    """Generate practice questions on a topic"""
    questions = await student_service.generate_practice_questions(
        topic=request.topic,
        count=request.count,
        difficulty=request.difficulty,
        exam_type=request.exam_type,
        user_id=user.user_id if user else None,
    )
    return {"questions": questions}


@router.post("/review-essay")
async def review_essay(
    request: EssayReviewRequest,
    user: UserContext = Depends(get_current_user),
):
    """Review and provide feedback on medical writing"""
    return await student_service.review_essay(
        essay_text=request.essay_text,
        essay_type=request.essay_type,
        user_id=user.user_id if user else None,
    )


@router.post("/start-session")
async def start_study_session(
    exam_type: ExamType = ExamType.USMLE_STEP1,
    user: UserContext = Depends(get_current_user),
):
    """Start a new study session"""
    session = student_service.start_study_session(
        user_id=user.user_id if user else "anonymous",
        exam_type=exam_type,
    )
    return {"session_id": session.session_id, "exam_type": exam_type.value}


@router.get("/recommendations")
async def get_study_recommendations(
    exam_type: ExamType = ExamType.USMLE_STEP1,
    user: UserContext = Depends(get_current_user),
):
    """Get personalized study recommendations"""
    return student_service.get_study_recommendations(
        user_id=user.user_id if user else "anonymous",
        exam_type=exam_type,
    )
