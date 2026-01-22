"""Role-based access control (RBAC) authorization service"""
from functools import wraps
from typing import Callable, List, Optional, Set
from enum import Enum

from fastapi import HTTPException, status, Depends

from .authentication import UserContext, UserRole, get_current_user


class Permission(str, Enum):
    """System permissions"""
    # Patient permissions
    VIEW_OWN_RECORDS = "view_own_records"
    UPDATE_OWN_PROFILE = "update_own_profile"
    REQUEST_DIAGNOSIS = "request_diagnosis"
    VIEW_OTC_RECOMMENDATIONS = "view_otc_recommendations"
    
    # Doctor permissions
    VIEW_PATIENT_RECORDS = "view_patient_records"
    CREATE_DIAGNOSIS = "create_diagnosis"
    PRESCRIBE_MEDICATION = "prescribe_medication"
    ORDER_TESTS = "order_tests"
    VIEW_CLINICAL_GUIDELINES = "view_clinical_guidelines"
    
    # Student permissions
    ACCESS_STUDY_MATERIALS = "access_study_materials"
    TAKE_PRACTICE_EXAMS = "take_practice_exams"
    VIEW_RESEARCH_PAPERS = "view_research_papers"
    
    # Researcher permissions
    ACCESS_RESEARCH_DATA = "access_research_data"
    CREATE_RESEARCH_PROJECT = "create_research_project"
    MANAGE_PATENTS = "manage_patents"
    EXPORT_ANONYMIZED_DATA = "export_anonymized_data"
    
    # Pharma permissions
    VIEW_DRUG_DATA = "view_drug_data"
    MANAGE_QA_DOCUMENTS = "manage_qa_documents"
    MANAGE_QC_RECORDS = "manage_qc_records"
    VIEW_SALES_DATA = "view_sales_data"
    MANAGE_REGULATORY_DOCS = "manage_regulatory_docs"
    MANAGE_FACILITY_RECORDS = "manage_facility_records"
    
    # Hospital permissions
    MANAGE_ER_QUEUE = "manage_er_queue"
    VIEW_INSURANCE_DATA = "view_insurance_data"
    MANAGE_PATIENT_BOOKING = "manage_patient_booking"
    VIEW_HOSPITAL_ANALYTICS = "view_hospital_analytics"
    
    # Admin permissions
    MANAGE_USERS = "manage_users"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    MANAGE_SYSTEM_CONFIG = "manage_system_config"
    MANAGE_KNOWLEDGE_GRAPH = "manage_knowledge_graph"


# Role to permissions mapping
ROLE_PERMISSIONS: dict[UserRole, Set[Permission]] = {
    UserRole.ADMIN: set(Permission),  # All permissions
    
    UserRole.PATIENT: {
        Permission.VIEW_OWN_RECORDS,
        Permission.UPDATE_OWN_PROFILE,
        Permission.REQUEST_DIAGNOSIS,
        Permission.VIEW_OTC_RECOMMENDATIONS,
    },
    
    UserRole.DOCTOR: {
        Permission.VIEW_PATIENT_RECORDS,
        Permission.CREATE_DIAGNOSIS,
        Permission.PRESCRIBE_MEDICATION,
        Permission.ORDER_TESTS,
        Permission.VIEW_CLINICAL_GUIDELINES,
        Permission.VIEW_OWN_RECORDS,
        Permission.UPDATE_OWN_PROFILE,
    },
    
    UserRole.STUDENT: {
        Permission.ACCESS_STUDY_MATERIALS,
        Permission.TAKE_PRACTICE_EXAMS,
        Permission.VIEW_RESEARCH_PAPERS,
        Permission.VIEW_OWN_RECORDS,
        Permission.UPDATE_OWN_PROFILE,
    },
    
    UserRole.RESEARCHER: {
        Permission.ACCESS_RESEARCH_DATA,
        Permission.CREATE_RESEARCH_PROJECT,
        Permission.MANAGE_PATENTS,
        Permission.EXPORT_ANONYMIZED_DATA,
        Permission.VIEW_RESEARCH_PAPERS,
        Permission.VIEW_OWN_RECORDS,
        Permission.UPDATE_OWN_PROFILE,
    },
    
    UserRole.PHARMA_ADMIN: {
        Permission.VIEW_DRUG_DATA,
        Permission.MANAGE_QA_DOCUMENTS,
        Permission.MANAGE_QC_RECORDS,
        Permission.VIEW_SALES_DATA,
        Permission.MANAGE_REGULATORY_DOCS,
        Permission.MANAGE_FACILITY_RECORDS,
        Permission.MANAGE_USERS,
        Permission.VIEW_OWN_RECORDS,
        Permission.UPDATE_OWN_PROFILE,
    },
    
    UserRole.PHARMA_USER: {
        Permission.VIEW_DRUG_DATA,
        Permission.MANAGE_QA_DOCUMENTS,
        Permission.MANAGE_QC_RECORDS,
        Permission.VIEW_SALES_DATA,
        Permission.VIEW_OWN_RECORDS,
        Permission.UPDATE_OWN_PROFILE,
    },
    
    UserRole.HOSPITAL_ADMIN: {
        Permission.MANAGE_ER_QUEUE,
        Permission.VIEW_INSURANCE_DATA,
        Permission.MANAGE_PATIENT_BOOKING,
        Permission.VIEW_HOSPITAL_ANALYTICS,
        Permission.VIEW_PATIENT_RECORDS,
        Permission.MANAGE_USERS,
        Permission.VIEW_OWN_RECORDS,
        Permission.UPDATE_OWN_PROFILE,
    },
    
    UserRole.HOSPITAL_STAFF: {
        Permission.MANAGE_ER_QUEUE,
        Permission.VIEW_INSURANCE_DATA,
        Permission.MANAGE_PATIENT_BOOKING,
        Permission.VIEW_PATIENT_RECORDS,
        Permission.VIEW_OWN_RECORDS,
        Permission.UPDATE_OWN_PROFILE,
    },
    
    UserRole.GENERAL: {
        Permission.VIEW_OWN_RECORDS,
        Permission.UPDATE_OWN_PROFILE,
    },
}


class AuthorizationService:
    """Authorization service for RBAC"""
    
    def __init__(self):
        self.role_permissions = ROLE_PERMISSIONS
    
    def get_role_permissions(self, role: UserRole) -> Set[Permission]:
        """Get all permissions for a role"""
        return self.role_permissions.get(role, set())
    
    def has_permission(self, user: UserContext, permission: Permission) -> bool:
        """Check if user has a specific permission"""
        # Admin has all permissions
        if user.role == UserRole.ADMIN:
            return True
        
        # Check role-based permissions
        role_perms = self.get_role_permissions(user.role)
        if permission in role_perms:
            return True
        
        # Check explicit user permissions
        return permission.value in user.permissions
    
    def has_any_permission(self, user: UserContext, permissions: List[Permission]) -> bool:
        """Check if user has any of the specified permissions"""
        return any(self.has_permission(user, p) for p in permissions)
    
    def has_all_permissions(self, user: UserContext, permissions: List[Permission]) -> bool:
        """Check if user has all specified permissions"""
        return all(self.has_permission(user, p) for p in permissions)
    
    def check_permission(self, user: UserContext, permission: Permission) -> None:
        """Check permission and raise exception if not authorized"""
        if not self.has_permission(user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: {permission.value}"
            )
    
    def check_role(self, user: UserContext, allowed_roles: List[UserRole]) -> None:
        """Check if user has one of the allowed roles"""
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role not authorized. Required: {[r.value for r in allowed_roles]}"
            )


# Singleton instance
_authz_service: Optional[AuthorizationService] = None


def get_authz_service() -> AuthorizationService:
    """Get or create authorization service singleton"""
    global _authz_service
    if _authz_service is None:
        _authz_service = AuthorizationService()
    return _authz_service


def require_permission(permission: Permission):
    """Decorator to require specific permission"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, user: UserContext = Depends(get_current_user), **kwargs):
            authz = get_authz_service()
            authz.check_permission(user, permission)
            return await func(*args, user=user, **kwargs)
        return wrapper
    return decorator


def require_role(allowed_roles: List[UserRole]):
    """Decorator to require specific roles"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, user: UserContext = Depends(get_current_user), **kwargs):
            authz = get_authz_service()
            authz.check_role(user, allowed_roles)
            return await func(*args, user=user, **kwargs)
        return wrapper
    return decorator


class PermissionChecker:
    """FastAPI dependency for permission checking"""
    
    def __init__(self, permission: Permission):
        self.permission = permission
    
    async def __call__(self, user: UserContext = Depends(get_current_user)) -> UserContext:
        authz = get_authz_service()
        authz.check_permission(user, self.permission)
        return user


class RoleChecker:
    """FastAPI dependency for role checking"""
    
    def __init__(self, allowed_roles: List[UserRole]):
        self.allowed_roles = allowed_roles
    
    async def __call__(self, user: UserContext = Depends(get_current_user)) -> UserContext:
        authz = get_authz_service()
        authz.check_role(user, self.allowed_roles)
        return user
