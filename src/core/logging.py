"""
UMI Logging Configuration
Structured logging with JSON format support
"""

import logging
import sys
from typing import Any, Dict

import structlog

from src.core.config import settings


def setup_logging() -> None:
    """Configure structured logging for the application."""

    # Set log level
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
            if settings.log_format == "json"
            else structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Suppress noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Configured logger instance
    """
    return structlog.get_logger(name)


class LogContext:
    """Context manager for adding temporary log context."""

    def __init__(self, **kwargs: Any):
        self.context = kwargs

    def __enter__(self) -> None:
        structlog.contextvars.bind_contextvars(**self.context)

    def __exit__(self, *args: Any) -> None:
        structlog.contextvars.unbind_contextvars(*self.context.keys())


def log_request(
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    user_id: str | None = None,
    extra: Dict[str, Any] | None = None,
) -> None:
    """
    Log HTTP request details.
    
    Args:
        method: HTTP method
        path: Request path
        status_code: Response status code
        duration_ms: Request duration in milliseconds
        user_id: Optional user ID
        extra: Additional context
    """
    logger = get_logger("http")
    
    log_data = {
        "method": method,
        "path": path,
        "status_code": status_code,
        "duration_ms": round(duration_ms, 2),
    }
    
    if user_id:
        log_data["user_id"] = user_id
    
    if extra:
        log_data.update(extra)
    
    if status_code >= 500:
        logger.error("request_error", **log_data)
    elif status_code >= 400:
        logger.warning("request_warning", **log_data)
    else:
        logger.info("request", **log_data)


def log_ai_request(
    model: str,
    tokens_in: int,
    tokens_out: int,
    duration_ms: float,
    user_id: str | None = None,
    consultation_id: str | None = None,
) -> None:
    """
    Log AI model request details.
    
    Args:
        model: Model name
        tokens_in: Input tokens
        tokens_out: Output tokens
        duration_ms: Request duration
        user_id: Optional user ID
        consultation_id: Optional consultation ID
    """
    logger = get_logger("ai")
    
    logger.info(
        "ai_request",
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        duration_ms=round(duration_ms, 2),
        user_id=user_id,
        consultation_id=consultation_id,
    )
