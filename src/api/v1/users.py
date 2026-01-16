"""
UMI Users API
User profile and account management
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_db
from src.core.security import get_password_hash, verify_password
from src.api.deps import get_current_user
from src.models.user import User, UserProfile
from src.schemas.user import (
    UserResponse,
    UserUpdate,
    UserProfileCreate,
    UserProfileUpdate,
    UserProfileResponse,
    PasswordChangeRequest,
)

router = APIRouter()


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user),
) -> User:
    """Get current user information."""
    return current_user


@router.put("/me", response_model=UserResponse)
async def update_current_user(
    data: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Update current user information."""
    update_data = data.model_dump(exclude_unset=True)
    
    # Prevent role escalation for non-superusers
    if "role" in update_data and not current_user.is_superuser:
        del update_data["role"]
    
    for field, value in update_data.items():
        setattr(current_user, field, value)
    
    await db.commit()
    await db.refresh(current_user)
    
    return current_user


@router.get("/me/profile", response_model=UserProfileResponse)
async def get_current_user_profile(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserProfile:
    """Get current user's profile."""
    if not current_user.profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Profile not found",
        )
    return current_user.profile


@router.put("/me/profile", response_model=UserProfileResponse)
async def update_current_user_profile(
    data: UserProfileUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserProfile:
    """Update current user's profile."""
    profile = current_user.profile
    
    if not profile:
        # Create profile if doesn't exist
        profile = UserProfile(user_id=current_user.id)
        db.add(profile)
    
    update_data = data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(profile, field, value)
    
    await db.commit()
    await db.refresh(profile)
    
    return profile


@router.post("/me/change-password")
async def change_password(
    data: PasswordChangeRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Change current user's password."""
    # Verify current password
    if not verify_password(data.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )
    
    # Update password
    current_user.password_hash = get_password_hash(data.new_password)
    await db.commit()
    
    return {"message": "Password changed successfully"}
