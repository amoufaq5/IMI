"""
UMI - Universal Medical Intelligence
Main Application Entry Point
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from src.core.config import settings
from src.core.database import init_db, close_db
from src.core.logging import setup_logging, get_logger, log_request
from src.core.exceptions import UMIException
from src.api.v1.router import api_router

# Setup logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan manager."""
    # Startup
    logger.info("starting_application", app_name=settings.app_name, env=settings.app_env)
    
    # Initialize database
    await init_db()
    logger.info("database_initialized")
    
    # Initialize RAG collections (if enabled)
    if settings.feature_pharma_enabled or settings.feature_imaging_enabled:
        try:
            from src.ai.rag_service import RAGService
            rag = RAGService()
            await rag.initialize_collections()
            logger.info("rag_collections_initialized")
        except Exception as e:
            logger.warning("rag_initialization_failed", error=str(e))
    
    yield
    
    # Shutdown
    logger.info("shutting_down_application")
    await close_db()
    logger.info("database_connections_closed")


# Create FastAPI application
app = FastAPI(
    title="UMI - Universal Medical Intelligence",
    description="""
    ## Universal Medical Intelligence API
    
    UMI is a comprehensive medical AI platform serving:
    
    - **Patients**: Symptom analysis, OTC recommendations, doctor referrals
    - **Pharmaceutical Companies**: QA/QC documentation, compliance tracking
    - **Hospitals**: ER triage, patient profiling
    - **Researchers**: Literature review, paper assistance
    - **Students**: Medical education, exam preparation
    
    ### Key Features
    
    - 🏥 **ASMETHOD Consultation**: Structured patient assessment protocol
    - 💊 **Drug Information**: Interactions, dosing, contraindications
    - 📋 **QA/QC Documentation**: AI-generated pharmaceutical documents
    - 🔬 **Medical Knowledge**: Evidence-based health information
    - 🖼️ **Medical Imaging**: CT, X-ray, MRI analysis (coming soon)
    
    ### Authentication
    
    All endpoints require JWT authentication. Obtain tokens via `/api/v1/auth/login`.
    
    ### Rate Limits
    
    - Free tier: 100 requests/minute
    - Premium: 500 requests/minute
    - Enterprise: Custom limits
    """,
    version="0.1.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(UMIException)
async def umi_exception_handler(request: Request, exc: UMIException) -> JSONResponse:
    """Handle custom UMI exceptions."""
    logger.warning(
        "umi_exception",
        code=exc.code,
        message=exc.message,
        path=request.url.path,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle request validation errors."""
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": {"errors": errors},
            }
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger.error(
        "unhandled_exception",
        error=str(exc),
        path=request.url.path,
        exc_info=True,
    )
    
    # Don't expose internal errors in production
    message = str(exc) if settings.debug else "An unexpected error occurred"
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": message,
            }
        },
    )


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests."""
    import time
    
    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000
    
    # Skip health check logging
    if request.url.path not in ["/health", "/ready"]:
        log_request(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
    
    return response


# Include API router
app.include_router(api_router, prefix=settings.api_prefix)


# Health check endpoints
@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    """Basic health check endpoint."""
    return {"status": "healthy", "service": "umi"}


@app.get("/ready", tags=["Health"])
async def readiness_check() -> dict:
    """
    Readiness check endpoint.
    
    Verifies that all dependencies are available.
    """
    checks = {
        "database": "unknown",
        "cache": "unknown",
        "ai_service": "unknown",
    }
    
    # Check database
    try:
        from src.core.database import get_db_context
        async with get_db_context() as db:
            await db.execute("SELECT 1")
        checks["database"] = "healthy"
    except Exception as e:
        checks["database"] = f"unhealthy: {str(e)}"
    
    # Check cache (Redis)
    try:
        import redis.asyncio as redis
        r = redis.from_url(settings.redis_url)
        await r.ping()
        await r.close()
        checks["cache"] = "healthy"
    except Exception as e:
        checks["cache"] = f"unhealthy: {str(e)}"
    
    # Check AI service
    try:
        from src.ai.llm_service import LLMService
        llm = LLMService()
        checks["ai_service"] = "healthy"
    except Exception as e:
        checks["ai_service"] = f"unhealthy: {str(e)}"
    
    all_healthy = all(v == "healthy" for v in checks.values())
    
    return {
        "status": "ready" if all_healthy else "degraded",
        "checks": checks,
    }


@app.get("/", tags=["Root"])
async def root() -> dict:
    """Root endpoint with API information."""
    return {
        "name": "UMI - Universal Medical Intelligence",
        "version": "0.1.0",
        "description": "Medical AI platform for patients, pharma, and healthcare providers",
        "documentation": "/docs" if settings.debug else "Contact support for API documentation",
        "health": "/health",
        "api": settings.api_prefix,
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        workers=settings.workers if not settings.reload else 1,
    )
