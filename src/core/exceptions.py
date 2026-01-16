"""
UMI Custom Exceptions
Centralized exception definitions for the application
"""

from typing import Any, Dict, Optional


class UMIException(Exception):
    """Base exception for all UMI errors."""

    def __init__(
        self,
        message: str,
        code: str = "UMI_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API response."""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }


# Authentication & Authorization Errors
class AuthenticationError(UMIException):
    """Authentication failed."""

    def __init__(self, message: str = "Authentication failed", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            code="AUTH_FAILED",
            status_code=401,
            details=details,
        )


class AuthorizationError(UMIException):
    """User not authorized for this action."""

    def __init__(self, message: str = "Not authorized", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            code="NOT_AUTHORIZED",
            status_code=403,
            details=details,
        )


class TokenExpiredError(UMIException):
    """JWT token has expired."""

    def __init__(self, message: str = "Token has expired"):
        super().__init__(
            message=message,
            code="TOKEN_EXPIRED",
            status_code=401,
        )


# Resource Errors
class NotFoundError(UMIException):
    """Resource not found."""

    def __init__(self, resource: str, identifier: Any = None):
        message = f"{resource} not found"
        if identifier:
            message = f"{resource} with id '{identifier}' not found"
        super().__init__(
            message=message,
            code="NOT_FOUND",
            status_code=404,
            details={"resource": resource, "identifier": str(identifier) if identifier else None},
        )


class AlreadyExistsError(UMIException):
    """Resource already exists."""

    def __init__(self, resource: str, identifier: Any = None):
        message = f"{resource} already exists"
        if identifier:
            message = f"{resource} with id '{identifier}' already exists"
        super().__init__(
            message=message,
            code="ALREADY_EXISTS",
            status_code=409,
            details={"resource": resource, "identifier": str(identifier) if identifier else None},
        )


# Validation Errors
class ValidationError(UMIException):
    """Input validation failed."""

    def __init__(self, message: str, field: Optional[str] = None, details: Optional[Dict] = None):
        error_details = details or {}
        if field:
            error_details["field"] = field
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            status_code=422,
            details=error_details,
        )


# AI/ML Errors
class AIModelError(UMIException):
    """AI model inference error."""

    def __init__(self, message: str = "AI model error", model: Optional[str] = None):
        super().__init__(
            message=message,
            code="AI_MODEL_ERROR",
            status_code=500,
            details={"model": model} if model else {},
        )


class AITimeoutError(UMIException):
    """AI model inference timeout."""

    def __init__(self, message: str = "AI model timeout", timeout_seconds: Optional[float] = None):
        super().__init__(
            message=message,
            code="AI_TIMEOUT",
            status_code=504,
            details={"timeout_seconds": timeout_seconds} if timeout_seconds else {},
        )


class RAGError(UMIException):
    """RAG retrieval error."""

    def __init__(self, message: str = "Failed to retrieve relevant documents"):
        super().__init__(
            message=message,
            code="RAG_ERROR",
            status_code=500,
        )


# Medical/Clinical Errors
class MedicalSafetyError(UMIException):
    """Medical safety check failed - requires human review."""

    def __init__(self, message: str, danger_signs: Optional[list] = None):
        super().__init__(
            message=message,
            code="MEDICAL_SAFETY",
            status_code=200,  # Not an error, but a safety redirect
            details={"danger_signs": danger_signs or [], "requires_referral": True},
        )


class DrugInteractionWarning(UMIException):
    """Drug interaction detected."""

    def __init__(self, drugs: list, severity: str, description: str):
        super().__init__(
            message=f"Drug interaction detected: {description}",
            code="DRUG_INTERACTION",
            status_code=200,  # Warning, not error
            details={
                "drugs": drugs,
                "severity": severity,
                "description": description,
            },
        )


# External Service Errors
class ExternalServiceError(UMIException):
    """External service (API) error."""

    def __init__(self, service: str, message: str = "External service error"):
        super().__init__(
            message=f"{service}: {message}",
            code="EXTERNAL_SERVICE_ERROR",
            status_code=502,
            details={"service": service},
        )


class RateLimitError(UMIException):
    """Rate limit exceeded."""

    def __init__(self, retry_after: Optional[int] = None):
        super().__init__(
            message="Rate limit exceeded",
            code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            details={"retry_after": retry_after} if retry_after else {},
        )


# Database Errors
class DatabaseError(UMIException):
    """Database operation error."""

    def __init__(self, message: str = "Database error"):
        super().__init__(
            message=message,
            code="DATABASE_ERROR",
            status_code=500,
        )


# File/Storage Errors
class FileUploadError(UMIException):
    """File upload failed."""

    def __init__(self, message: str = "File upload failed", filename: Optional[str] = None):
        super().__init__(
            message=message,
            code="FILE_UPLOAD_ERROR",
            status_code=400,
            details={"filename": filename} if filename else {},
        )


class UnsupportedFileTypeError(UMIException):
    """Unsupported file type."""

    def __init__(self, file_type: str, supported_types: list):
        super().__init__(
            message=f"Unsupported file type: {file_type}",
            code="UNSUPPORTED_FILE_TYPE",
            status_code=400,
            details={"file_type": file_type, "supported_types": supported_types},
        )
