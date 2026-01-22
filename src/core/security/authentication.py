"""JWT-based authentication service"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from enum import Enum
import uuid

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field

from src.core.config import settings


class UserRole(str, Enum):
    """User roles for RBAC"""
    ADMIN = "admin"
    DOCTOR = "doctor"
    PATIENT = "patient"
    STUDENT = "student"
    RESEARCHER = "researcher"
    PHARMA_ADMIN = "pharma_admin"
    PHARMA_USER = "pharma_user"
    HOSPITAL_ADMIN = "hospital_admin"
    HOSPITAL_STAFF = "hospital_staff"
    GENERAL = "general"


class TokenType(str, Enum):
    """Token types"""
    ACCESS = "access"
    REFRESH = "refresh"


class TokenData(BaseModel):
    """JWT token payload"""
    sub: str  # User ID
    email: Optional[str] = None
    role: UserRole
    permissions: list[str] = Field(default_factory=list)
    organization_id: Optional[str] = None
    token_type: TokenType
    exp: datetime
    iat: datetime
    jti: str  # JWT ID for revocation


class UserContext(BaseModel):
    """Current user context from token"""
    user_id: str
    email: Optional[str] = None
    role: UserRole
    permissions: list[str] = Field(default_factory=list)
    organization_id: Optional[str] = None
    token_id: str
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission"""
        return permission in self.permissions or self.role == UserRole.ADMIN
    
    def has_any_permission(self, permissions: list[str]) -> bool:
        """Check if user has any of the specified permissions"""
        return any(self.has_permission(p) for p in permissions)


class AuthenticationService:
    """JWT authentication service"""
    
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
        self.secret_key = settings.jwt_secret_key
        self.algorithm = settings.jwt_algorithm
        self.access_token_expire = timedelta(minutes=settings.access_token_expire_minutes)
        self.refresh_token_expire = timedelta(days=settings.refresh_token_expire_days)
        self._revoked_tokens: set[str] = set()  # In production, use Redis
    
    def hash_password(self, password: str) -> str:
        """Hash password using Argon2"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(
        self,
        user_id: str,
        role: UserRole,
        email: Optional[str] = None,
        permissions: Optional[list[str]] = None,
        organization_id: Optional[str] = None,
    ) -> str:
        """Create JWT access token"""
        now = datetime.utcnow()
        token_data = TokenData(
            sub=user_id,
            email=email,
            role=role,
            permissions=permissions or [],
            organization_id=organization_id,
            token_type=TokenType.ACCESS,
            exp=now + self.access_token_expire,
            iat=now,
            jti=str(uuid.uuid4()),
        )
        return jwt.encode(token_data.model_dump(mode="json"), self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(
        self,
        user_id: str,
        role: UserRole,
    ) -> str:
        """Create JWT refresh token"""
        now = datetime.utcnow()
        token_data = TokenData(
            sub=user_id,
            role=role,
            permissions=[],
            token_type=TokenType.REFRESH,
            exp=now + self.refresh_token_expire,
            iat=now,
            jti=str(uuid.uuid4()),
        )
        return jwt.encode(token_data.model_dump(mode="json"), self.secret_key, algorithm=self.algorithm)
    
    def decode_token(self, token: str) -> TokenData:
        """Decode and validate JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            token_data = TokenData(**payload)
            
            # Check if token is revoked
            if token_data.jti in self._revoked_tokens:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )
            
            return token_data
        except JWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}"
            )
    
    def revoke_token(self, token_id: str) -> None:
        """Revoke a token by its JTI"""
        self._revoked_tokens.add(token_id)
    
    def get_user_context(self, token: str) -> UserContext:
        """Extract user context from token"""
        token_data = self.decode_token(token)
        
        if token_data.token_type != TokenType.ACCESS:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        return UserContext(
            user_id=token_data.sub,
            email=token_data.email,
            role=token_data.role,
            permissions=token_data.permissions,
            organization_id=token_data.organization_id,
            token_id=token_data.jti,
        )


# Singleton instance
_auth_service: Optional[AuthenticationService] = None


def get_auth_service() -> AuthenticationService:
    """Get or create authentication service singleton"""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthenticationService()
    return _auth_service


# FastAPI dependency
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> UserContext:
    """FastAPI dependency to get current authenticated user"""
    auth_service = get_auth_service()
    return auth_service.get_user_context(credentials.credentials)


async def get_optional_user(
    request: Request,
) -> Optional[UserContext]:
    """FastAPI dependency for optional authentication"""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    
    token = auth_header.split(" ")[1]
    try:
        auth_service = get_auth_service()
        return auth_service.get_user_context(token)
    except HTTPException:
        return None
