"""
UMI API v1 Router
Main router aggregating all v1 endpoints
"""

from fastapi import APIRouter

from src.api.v1 import auth, users, consultations, pharma, drugs, health, imaging

api_router = APIRouter()

# Include all route modules
api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["Authentication"],
)

api_router.include_router(
    users.router,
    prefix="/users",
    tags=["Users"],
)

api_router.include_router(
    consultations.router,
    prefix="/consultations",
    tags=["Consultations"],
)

api_router.include_router(
    pharma.router,
    prefix="/pharma",
    tags=["Pharmaceutical"],
)

api_router.include_router(
    drugs.router,
    prefix="/drugs",
    tags=["Drugs"],
)

api_router.include_router(
    health.router,
    prefix="/health",
    tags=["Health Information"],
)

api_router.include_router(
    imaging.router,
    prefix="/imaging",
    tags=["Medical Imaging"],
)
