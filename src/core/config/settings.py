"""Application settings with environment variable support"""
from functools import lru_cache
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application
    app_name: str = Field(default="IMI", description="Application name")
    app_env: str = Field(default="development", description="Environment")
    debug: bool = Field(default=False, description="Debug mode")
    secret_key: str = Field(default="change-me-in-production", description="Secret key")
    api_version: str = Field(default="v1", description="API version")
    
    # Database - PostgreSQL
    database_url: str = Field(
        default="postgresql+asyncpg://user:password@localhost:5432/imi_db",
        description="PostgreSQL connection URL"
    )
    database_pool_size: int = Field(default=20, description="Connection pool size")
    database_max_overflow: int = Field(default=10, description="Max overflow connections")
    
    # Neo4j Knowledge Graph
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j URI")
    neo4j_user: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="password", description="Neo4j password")
    
    # Redis Cache
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis URL")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    
    # Vector Database
    chromadb_host: str = Field(default="localhost", description="ChromaDB host")
    chromadb_port: int = Field(default=8001, description="ChromaDB port")
    chromadb_collection: str = Field(default="medical_embeddings", description="Collection name")
    
    # LLM Configuration
    llm_model_path: str = Field(default="/models/meditron-7b", description="LLM model path")
    llm_device: str = Field(default="cuda", description="Device for LLM")
    llm_max_length: int = Field(default=4096, description="Max sequence length")
    llm_temperature: float = Field(default=0.7, description="Sampling temperature")
    llm_top_p: float = Field(default=0.9, description="Top-p sampling")
    
    # Verifier Model
    verifier_model_path: str = Field(default="/models/medical-verifier", description="Verifier path")
    verifier_threshold: float = Field(default=0.85, description="Verification threshold")
    
    # HIPAA Compliance
    encryption_key: str = Field(default="change-this-32-byte-key-now!!!!", description="Encryption key")
    encryption_algorithm: str = Field(default="AES-256-GCM", description="Encryption algorithm")
    audit_log_retention_days: int = Field(default=2555, description="Audit log retention (7 years)")
    phi_anonymization_enabled: bool = Field(default=True, description="Enable PHI anonymization")
    
    # JWT Authentication
    jwt_secret_key: str = Field(default="jwt-secret-change-me", description="JWT secret")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Access token expiry")
    refresh_token_expire_days: int = Field(default=7, description="Refresh token expiry")
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, description="Rate limit requests")
    rate_limit_window_seconds: int = Field(default=60, description="Rate limit window")
    
    # Logging
    log_level: str = Field(default="INFO", description="Log level")
    log_format: str = Field(default="json", description="Log format")
    audit_log_path: str = Field(default="/var/log/imi/audit.log", description="Audit log path")
    
    # External APIs
    pubmed_api_key: Optional[str] = Field(default=None, description="PubMed API key")
    fda_api_key: Optional[str] = Field(default=None, description="FDA API key")
    
    # Medical Imaging
    dicom_storage_path: str = Field(default="/data/dicom", description="DICOM storage path")
    max_image_size_mb: int = Field(default=100, description="Max image size in MB")
    
    # Monitoring
    prometheus_enabled: bool = Field(default=True, description="Enable Prometheus")
    prometheus_port: int = Field(default=9090, description="Prometheus port")
    opentelemetry_enabled: bool = Field(default=True, description="Enable OpenTelemetry")
    otlp_endpoint: str = Field(default="http://localhost:4317", description="OTLP endpoint")
    
    @field_validator("encryption_key")
    @classmethod
    def validate_encryption_key(cls, v: str) -> str:
        if len(v) < 32:
            raise ValueError("Encryption key must be at least 32 characters")
        return v
    
    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        return self.app_env.lower() == "development"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


settings = get_settings()
