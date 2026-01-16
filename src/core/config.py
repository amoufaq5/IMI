"""
UMI Configuration Management
Centralized configuration using Pydantic Settings
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "UMI"
    app_env: str = "development"
    debug: bool = False
    api_version: str = "v1"
    api_prefix: str = "/api/v1"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://umi_user:umi_password@localhost:5432/umi_db"
    )
    database_pool_size: int = 20
    database_max_overflow: int = 10

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_password: Optional[str] = None
    redis_ssl: bool = False

    # Security
    secret_key: str = Field(default="change-me-in-production")
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            import json
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [origin.strip() for origin in v.split(",")]
        return v

    # AI/ML Configuration
    openai_api_key: Optional[str] = None
    hf_token: Optional[str] = None
    hf_model_cache: str = "/app/models"

    # LLM Settings
    llm_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.7
    llm_use_gpu: bool = True
    llm_quantization: str = "4bit"

    # Vision Model
    vision_model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf"
    vision_max_image_size: int = 1024

    # Vector Database
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_collection_prefix: str = "umi_"

    # Embedding Model
    embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO"
    embedding_dimension: int = 768

    # External APIs
    pubmed_api_key: Optional[str] = None
    pubmed_email: Optional[str] = None
    drugbank_api_key: Optional[str] = None
    openfda_api_key: Optional[str] = None

    # Storage
    storage_type: str = "minio"
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "umi-files"
    minio_secure: bool = False

    # Celery
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # Monitoring
    sentry_dsn: Optional[str] = None
    prometheus_enabled: bool = True
    prometheus_port: int = 9090

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: Optional[str] = None

    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60

    # Feature Flags
    feature_imaging_enabled: bool = False
    feature_pharma_enabled: bool = True
    feature_research_enabled: bool = False
    feature_hospital_enabled: bool = False

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
