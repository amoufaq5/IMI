"""
UMI Celery Worker
Async task processing for AI inference, document generation, and background jobs
"""

from celery import Celery
from celery.signals import worker_ready

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)

# Create Celery app
celery_app = Celery(
    "umi",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minutes max
    task_soft_time_limit=540,  # 9 minutes soft limit
    worker_prefetch_multiplier=1,  # One task at a time for GPU tasks
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    result_expires=3600,  # Results expire after 1 hour
)

# Task queues
celery_app.conf.task_routes = {
    "src.worker.tasks.ai.*": {"queue": "ai"},
    "src.worker.tasks.documents.*": {"queue": "documents"},
    "src.worker.tasks.notifications.*": {"queue": "notifications"},
}


@worker_ready.connect
def on_worker_ready(**kwargs):
    """Called when worker is ready."""
    logger.info("celery_worker_ready")


# Import tasks to register them
from src.worker import tasks  # noqa
