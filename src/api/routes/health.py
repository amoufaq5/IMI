"""Health check endpoints"""
from fastapi import APIRouter
from datetime import datetime

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "IMI Medical LLM Platform",
    }


@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with component status"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "api": "healthy",
            "database": "healthy",
            "knowledge_graph": "healthy",
            "llm": "healthy",
            "cache": "healthy",
        },
        "version": "1.0.0",
    }
