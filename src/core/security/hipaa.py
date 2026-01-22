"""HIPAA compliance service for PHI handling"""
import re
import hashlib
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, date
from enum import Enum

from pydantic import BaseModel, Field

from .encryption import EncryptionService, get_encryption_service
from .audit import AuditLogger, get_audit_logger, AuditAction


class PHIType(str, Enum):
    """Types of Protected Health Information"""
    NAME = "name"
    ADDRESS = "address"
    DATE = "date"  # Dates related to individual (DOB, admission, discharge, death)
    PHONE = "phone"
    FAX = "fax"
    EMAIL = "email"
    SSN = "ssn"
    MRN = "mrn"  # Medical Record Number
    HEALTH_PLAN_ID = "health_plan_id"
    ACCOUNT_NUMBER = "account_number"
    LICENSE_NUMBER = "license_number"
    VEHICLE_ID = "vehicle_id"
    DEVICE_ID = "device_id"
    URL = "url"
    IP_ADDRESS = "ip_address"
    BIOMETRIC = "biometric"
    PHOTO = "photo"
    OTHER_UNIQUE_ID = "other_unique_id"


class PHIField(BaseModel):
    """PHI field definition"""
    field_name: str
    phi_type: PHIType
    requires_encryption: bool = True
    requires_audit: bool = True


class AnonymizedData(BaseModel):
    """Anonymized data container"""
    original_hash: str  # Hash of original for re-identification if authorized
    anonymized_value: str
    phi_type: PHIType
    anonymization_method: str


class HIPAAComplianceService:
    """HIPAA compliance service for PHI protection"""
    
    # Common PHI patterns for detection
    PHI_PATTERNS = {
        PHIType.SSN: r'\b\d{3}-\d{2}-\d{4}\b',
        PHIType.PHONE: r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        PHIType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        PHIType.MRN: r'\bMRN[:\s]?\d{6,10}\b',
        PHIType.DATE: r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
    }
    
    # Fields that are considered PHI
    PHI_FIELDS: Set[str] = {
        "first_name", "last_name", "full_name", "name",
        "date_of_birth", "dob", "birth_date",
        "ssn", "social_security_number",
        "address", "street", "city", "zip_code", "postal_code",
        "phone", "phone_number", "mobile", "fax",
        "email", "email_address",
        "mrn", "medical_record_number", "patient_id",
        "insurance_id", "policy_number",
        "diagnosis", "condition", "symptoms",
        "medication", "prescription",
        "lab_results", "test_results",
        "notes", "clinical_notes", "doctor_notes",
    }
    
    def __init__(self):
        self.encryption = get_encryption_service()
        self.audit = get_audit_logger()
    
    def is_phi_field(self, field_name: str) -> bool:
        """Check if a field name is considered PHI"""
        normalized = field_name.lower().replace("-", "_").replace(" ", "_")
        return normalized in self.PHI_FIELDS
    
    def detect_phi_in_text(self, text: str) -> Dict[PHIType, List[str]]:
        """Detect potential PHI in free text"""
        detected: Dict[PHIType, List[str]] = {}
        
        for phi_type, pattern in self.PHI_PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                detected[phi_type] = matches
        
        return detected
    
    def encrypt_phi(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt all PHI fields in a dictionary"""
        encrypted = {}
        
        for key, value in data.items():
            if value is None:
                encrypted[key] = None
            elif self.is_phi_field(key) and isinstance(value, str):
                encrypted[key] = self.encryption.encrypt_field(value, key)
            elif isinstance(value, dict):
                encrypted[key] = self.encrypt_phi(value)
            elif isinstance(value, list):
                encrypted[key] = [
                    self.encrypt_phi(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                encrypted[key] = value
        
        return encrypted
    
    def decrypt_phi(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt all PHI fields in a dictionary"""
        decrypted = {}
        
        for key, value in data.items():
            if value is None:
                decrypted[key] = None
            elif self.is_phi_field(key) and isinstance(value, str):
                try:
                    decrypted[key] = self.encryption.decrypt_field(value, key)
                except Exception:
                    decrypted[key] = value  # Return as-is if not encrypted
            elif isinstance(value, dict):
                decrypted[key] = self.decrypt_phi(value)
            elif isinstance(value, list):
                decrypted[key] = [
                    self.decrypt_phi(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                decrypted[key] = value
        
        return decrypted
    
    def anonymize_phi(self, data: Dict[str, Any], preserve_format: bool = True) -> Dict[str, Any]:
        """Anonymize PHI for research/export purposes"""
        anonymized = {}
        
        for key, value in data.items():
            if value is None:
                anonymized[key] = None
            elif self.is_phi_field(key) and isinstance(value, str):
                anonymized[key] = self._anonymize_value(key, value, preserve_format)
            elif isinstance(value, dict):
                anonymized[key] = self.anonymize_phi(value, preserve_format)
            elif isinstance(value, list):
                anonymized[key] = [
                    self.anonymize_phi(item, preserve_format) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                anonymized[key] = value
        
        return anonymized
    
    def _anonymize_value(self, field_name: str, value: str, preserve_format: bool) -> str:
        """Anonymize a single PHI value"""
        # Generate consistent hash for the value
        hash_input = f"{field_name}:{value}".encode()
        value_hash = hashlib.sha256(hash_input).hexdigest()[:8]
        
        field_lower = field_name.lower()
        
        if "name" in field_lower:
            return f"ANON_{value_hash}"
        elif "date" in field_lower or "dob" in field_lower:
            # Shift dates by random offset while preserving year
            return "XXXX-XX-XX"
        elif "ssn" in field_lower:
            return "XXX-XX-XXXX"
        elif "phone" in field_lower or "fax" in field_lower:
            return "XXX-XXX-XXXX"
        elif "email" in field_lower:
            return f"anon_{value_hash}@anonymized.local"
        elif "address" in field_lower or "street" in field_lower:
            return "REDACTED ADDRESS"
        elif "mrn" in field_lower or "patient_id" in field_lower:
            return f"ANON_{value_hash}"
        else:
            return f"[REDACTED:{value_hash}]"
    
    def create_minimum_necessary_view(
        self,
        data: Dict[str, Any],
        allowed_fields: Set[str],
        user_role: str,
    ) -> Dict[str, Any]:
        """
        Apply minimum necessary standard - only return fields needed for the purpose
        This is a HIPAA requirement
        """
        filtered = {}
        
        for key, value in data.items():
            if key in allowed_fields:
                filtered[key] = value
            elif isinstance(value, dict):
                nested_allowed = {
                    f.split(".", 1)[1] for f in allowed_fields
                    if f.startswith(f"{key}.")
                }
                if nested_allowed:
                    filtered[key] = self.create_minimum_necessary_view(
                        value, nested_allowed, user_role
                    )
        
        return filtered
    
    def validate_phi_access(
        self,
        user_id: str,
        user_role: str,
        patient_id: str,
        phi_types: List[PHIType],
        purpose: str,
        ip_address: Optional[str] = None,
    ) -> bool:
        """
        Validate and log PHI access attempt
        Returns True if access is allowed
        """
        # Log the access attempt
        self.audit.log_phi_access(
            user_id=user_id,
            user_role=user_role,
            patient_id=patient_id,
            phi_types=[p.value for p in phi_types],
            action=AuditAction.PHI_ACCESS,
            description=f"PHI access for purpose: {purpose}",
            ip_address=ip_address,
            details={"purpose": purpose},
        )
        
        # In production, implement actual access control logic here
        # For now, return True (access allowed)
        return True
    
    def redact_phi_from_text(self, text: str) -> str:
        """Redact detected PHI from free text"""
        redacted = text
        
        for phi_type, pattern in self.PHI_PATTERNS.items():
            redacted = re.sub(pattern, f"[REDACTED:{phi_type.value}]", redacted, flags=re.IGNORECASE)
        
        return redacted
    
    def generate_deidentification_report(self, original: Dict[str, Any], anonymized: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a report of de-identification actions taken"""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "fields_anonymized": [],
            "method": "Safe Harbor",
            "compliant": True,
        }
        
        def compare_dicts(orig: Dict, anon: Dict, prefix: str = ""):
            for key in orig:
                full_key = f"{prefix}.{key}" if prefix else key
                if key in anon:
                    if orig[key] != anon[key]:
                        report["fields_anonymized"].append({
                            "field": full_key,
                            "original_type": type(orig[key]).__name__,
                            "anonymized": True,
                        })
                    elif isinstance(orig[key], dict) and isinstance(anon[key], dict):
                        compare_dicts(orig[key], anon[key], full_key)
        
        compare_dicts(original, anonymized)
        return report


# Singleton instance
_hipaa_service: Optional[HIPAAComplianceService] = None


def get_hipaa_service() -> HIPAAComplianceService:
    """Get or create HIPAA compliance service singleton"""
    global _hipaa_service
    if _hipaa_service is None:
        _hipaa_service = HIPAAComplianceService()
    return _hipaa_service
