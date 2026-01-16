"""
UMI User Schemas
Pydantic models for user-related API requests and responses
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, field_validator

from src.models.user import UserRole, SubscriptionTier


class UserBase(BaseModel):
    """Base user schema."""
    email: EmailStr


class UserCreate(UserBase):
    """Schema for user registration."""
    password: str = Field(..., min_length=8, max_length=100)
    confirm_password: str = Field(..., min_length=8, max_length=100)
    role: UserRole = UserRole.GENERAL_USER
    
    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, v, info):
        if "password" in info.data and v != info.data["password"]:
            raise ValueError("Passwords do not match")
        return v
    
    @field_validator("password")
    @classmethod
    def password_strength(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserUpdate(BaseModel):
    """Schema for updating user."""
    email: Optional[EmailStr] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


class UserProfileBase(BaseModel):
    """Base profile schema."""
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    date_of_birth: Optional[datetime] = None
    gender: Optional[str] = Field(None, max_length=20)
    phone: Optional[str] = Field(None, max_length=20)
    country: Optional[str] = Field(None, max_length=100)
    city: Optional[str] = Field(None, max_length=100)
    language: str = "en"
    timezone: str = "UTC"


class UserProfileCreate(UserProfileBase):
    """Schema for creating user profile."""
    pass


class UserProfileUpdate(UserProfileBase):
    """Schema for updating user profile."""
    pass


class UserProfileResponse(UserProfileBase):
    """Schema for profile response."""
    id: UUID
    user_id: UUID
    full_name: str
    avatar_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class UserResponse(UserBase):
    """Schema for user response."""
    id: UUID
    role: UserRole
    subscription_tier: SubscriptionTier
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    profile: Optional[UserProfileResponse] = None

    class Config:
        from_attributes = True


class UserListResponse(BaseModel):
    """Schema for paginated user list."""
    items: List[UserResponse]
    total: int
    page: int
    page_size: int
    pages: int


class TokenResponse(BaseModel):
    """Schema for authentication token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class LoginRequest(BaseModel):
    """Schema for login request."""
    email: EmailStr
    password: str


class RefreshTokenRequest(BaseModel):
    """Schema for token refresh request."""
    refresh_token: str


class PasswordChangeRequest(BaseModel):
    """Schema for password change."""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=100)
    confirm_password: str
    
    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, v, info):
        if "new_password" in info.data and v != info.data["new_password"]:
            raise ValueError("Passwords do not match")
        return v


class PasswordResetRequest(BaseModel):
    """Schema for password reset request."""
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Schema for password reset confirmation."""
    token: str
    new_password: str = Field(..., min_length=8, max_length=100)
    confirm_password: str
    
    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, v, info):
        if "new_password" in info.data and v != info.data["new_password"]:
            raise ValueError("Passwords do not match")
        return v
