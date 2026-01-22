"""HIPAA-compliant audit logging service"""
import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from pathlib import Path

import structlog
from pydantic import BaseModel, Field

from src.core.config import settings


class AuditAction(str, Enum):
    """Audit action types"""
    # Authentication
    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"
    PASSWORD_CHANGE = "password_change"
    TOKEN_REFRESH = "token_refresh"
    
    # Data Access
    VIEW = "view"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    EXPORT = "export"
    PRINT = "print"
    
    # PHI Access
    PHI_ACCESS = "phi_access"
    PHI_MODIFY = "phi_modify"
    PHI_EXPORT = "phi_export"
    
    # Medical Actions
    DIAGNOSIS_REQUEST = "diagnosis_request"
    DIAGNOSIS_RESULT = "diagnosis_result"
    PRESCRIPTION_CREATE = "prescription_create"
    LAB_ORDER = "lab_order"
    IMAGING_VIEW = "imaging_view"
    
    # System Actions
    SYSTEM_CONFIG_CHANGE = "system_config_change"
    USER_MANAGEMENT = "user_management"
    PERMISSION_CHANGE = "permission_change"
    
    # AI/LLM Actions
    LLM_QUERY = "llm_query"
    LLM_RESPONSE = "llm_response"
    VERIFICATION_CHECK = "verification_check"
    KNOWLEDGE_GRAPH_QUERY = "knowledge_graph_query"
    RULE_ENGINE_EVALUATION = "rule_engine_evaluation"


class AuditSeverity(str, Enum):
    """Audit log severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditLogEntry(BaseModel):
    """Audit log entry model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Actor information
    user_id: Optional[str] = None
    user_role: Optional[str] = None
    user_email: Optional[str] = None
    organization_id: Optional[str] = None
    
    # Action details
    action: AuditAction
    severity: AuditSeverity = AuditSeverity.INFO
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    
    # Request context
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Action details
    description: str
    details: Dict[str, Any] = Field(default_factory=dict)
    
    # Outcome
    success: bool = True
    error_message: Optional[str] = None
    
    # PHI indicator
    contains_phi: bool = False
    phi_types: List[str] = Field(default_factory=list)
    
    def to_json(self) -> str:
        """Serialize to JSON for storage"""
        return self.model_dump_json()


class AuditLogger:
    """HIPAA-compliant audit logger"""
    
    def __init__(self):
        self.logger = structlog.get_logger("audit")
        self._setup_file_handler()
    
    def _setup_file_handler(self) -> None:
        """Setup file handler for audit logs"""
        log_path = Path(settings.audit_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def log(
        self,
        action: AuditAction,
        description: str,
        user_id: Optional[str] = None,
        user_role: Optional[str] = None,
        user_email: Optional[str] = None,
        organization_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        severity: AuditSeverity = AuditSeverity.INFO,
        contains_phi: bool = False,
        phi_types: Optional[List[str]] = None,
    ) -> AuditLogEntry:
        """Create and store an audit log entry"""
        entry = AuditLogEntry(
            user_id=user_id,
            user_role=user_role,
            user_email=user_email,
            organization_id=organization_id,
            action=action,
            severity=severity,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            user_agent=user_agent,
            request_id=request_id,
            session_id=session_id,
            description=description,
            details=details or {},
            success=success,
            error_message=error_message,
            contains_phi=contains_phi,
            phi_types=phi_types or [],
        )
        
        # Log to structured logger
        log_data = entry.model_dump(mode="json")
        
        if severity == AuditSeverity.CRITICAL:
            self.logger.critical("audit_event", **log_data)
        elif severity == AuditSeverity.ERROR:
            self.logger.error("audit_event", **log_data)
        elif severity == AuditSeverity.WARNING:
            self.logger.warning("audit_event", **log_data)
        else:
            self.logger.info("audit_event", **log_data)
        
        # Write to audit file
        self._write_to_file(entry)
        
        return entry
    
    def _write_to_file(self, entry: AuditLogEntry) -> None:
        """Write audit entry to file"""
        try:
            log_path = Path(settings.audit_log_path)
            with open(log_path, "a") as f:
                f.write(entry.to_json() + "\n")
        except Exception as e:
            self.logger.error("audit_file_write_error", error=str(e))
    
    def log_phi_access(
        self,
        user_id: str,
        user_role: str,
        patient_id: str,
        phi_types: List[str],
        action: AuditAction,
        description: str,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditLogEntry:
        """Log PHI access - required for HIPAA compliance"""
        return self.log(
            action=action,
            description=description,
            user_id=user_id,
            user_role=user_role,
            resource_type="patient",
            resource_id=patient_id,
            ip_address=ip_address,
            details=details,
            contains_phi=True,
            phi_types=phi_types,
            severity=AuditSeverity.INFO,
        )
    
    def log_authentication(
        self,
        action: AuditAction,
        user_id: Optional[str],
        user_email: Optional[str],
        ip_address: Optional[str],
        success: bool,
        error_message: Optional[str] = None,
    ) -> AuditLogEntry:
        """Log authentication events"""
        severity = AuditSeverity.INFO if success else AuditSeverity.WARNING
        description = f"Authentication: {action.value}"
        
        return self.log(
            action=action,
            description=description,
            user_id=user_id,
            user_email=user_email,
            ip_address=ip_address,
            success=success,
            error_message=error_message,
            severity=severity,
        )
    
    def log_llm_interaction(
        self,
        user_id: str,
        user_role: str,
        query: str,
        response: str,
        model_name: str,
        verification_passed: bool,
        latency_ms: float,
        contains_phi: bool = False,
    ) -> AuditLogEntry:
        """Log LLM query and response"""
        return self.log(
            action=AuditAction.LLM_QUERY,
            description="LLM interaction",
            user_id=user_id,
            user_role=user_role,
            details={
                "query_length": len(query),
                "response_length": len(response),
                "model": model_name,
                "verification_passed": verification_passed,
                "latency_ms": latency_ms,
            },
            contains_phi=contains_phi,
        )


# Singleton instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get or create audit logger singleton"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
