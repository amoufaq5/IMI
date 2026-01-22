"""
IMI API - Main FastAPI Application

Medical LLM Platform API with HIPAA compliance
"""
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time

from src.core.config import settings
from src.core.security.audit import get_audit_logger, AuditAction

from .routes import patient, doctor, student, researcher, pharma, hospital, general, health

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting IMI Medical LLM Platform...")
    
    # Initialize services
    audit = get_audit_logger()
    audit.log(
        action=AuditAction.SYSTEM_START,
        description="IMI Platform started",
        user_id="system",
    )
    
    yield
    
    # Cleanup
    audit.log(
        action=AuditAction.SYSTEM_STOP,
        description="IMI Platform stopped",
        user_id="system",
    )
    logger.info("IMI Platform shutdown complete")


app = FastAPI(
    title="IMI - Intelligent Medical Interface",
    description="""
    Medical LLM Platform providing AI-powered healthcare assistance.
    
    ## Features
    - Patient symptom assessment and triage
    - Clinical decision support for doctors
    - Medical education for students
    - Research assistance
    - Pharmaceutical QA/regulatory support
    - Hospital operations management
    
    ## Security
    - HIPAA compliant
    - End-to-end encryption for PHI
    - Comprehensive audit logging
    - Role-based access control
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins if hasattr(settings, 'cors_origins') else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    audit = get_audit_logger()
    audit.log(
        action=AuditAction.ERROR,
        description=f"Unhandled exception: {str(exc)}",
        user_id="system",
        severity_override="error",
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
        },
    )


# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(patient.router, prefix="/api/v1/patient", tags=["Patient"])
app.include_router(doctor.router, prefix="/api/v1/doctor", tags=["Doctor"])
app.include_router(student.router, prefix="/api/v1/student", tags=["Student"])
app.include_router(researcher.router, prefix="/api/v1/researcher", tags=["Researcher"])
app.include_router(pharma.router, prefix="/api/v1/pharma", tags=["Pharmaceutical"])
app.include_router(hospital.router, prefix="/api/v1/hospital", tags=["Hospital"])
app.include_router(general.router, prefix="/api/v1/general", tags=["General"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "IMI - Intelligent Medical Interface",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs",
    }
