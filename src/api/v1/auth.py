"""
UMI Authentication API
User registration, login, and token management
"""

from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_db
from src.core.security import (
    create_access_token,
    create_refresh_token,
    get_password_hash,
    verify_password,
    verify_refresh_token,
    TokenError,
)
from src.core.logging import get_logger
from src.models.user import User, UserProfile
from src.schemas.user import (
    UserCreate,
    UserResponse,
    TokenResponse,
    LoginRequest,
    RefreshTokenRequest,
)

logger = get_logger(__name__)
router = APIRouter()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    data: UserCreate,
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Register a new user account.
    
    - **email**: Valid email address (must be unique)
    - **password**: Minimum 8 characters with uppercase, lowercase, and digit
    - **role**: User role (default: general_user)
    """
    # Check if email already exists
    result = await db.execute(
        select(User).where(User.email == data.email)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )
    
    # Create user
    user = User(
        email=data.email,
        password_hash=get_password_hash(data.password),
        role=data.role,
        is_active=True,
        is_verified=False,
    )
    
    db.add(user)
    await db.flush()
    
    # Create empty profile
    profile = UserProfile(user_id=user.id)
    db.add(profile)
    
    await db.commit()
    await db.refresh(user)
    
    logger.info("user_registered", user_id=str(user.id), email=user.email)
    
    return user


@router.post("/login", response_model=TokenResponse)
async def login(
    data: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Authenticate user and return access tokens.
    
    - **email**: Registered email address
    - **password**: User password
    """
    # Find user
    result = await db.execute(
        select(User).where(User.email == data.email)
    )
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled",
        )
    
    # Update last login
    user.last_login = datetime.now(timezone.utc)
    await db.commit()
    
    # Generate tokens
    access_token = create_access_token(
        subject=str(user.id),
        additional_claims={"role": user.role.value},
    )
    refresh_token = create_refresh_token(subject=str(user.id))
    
    logger.info("user_login", user_id=str(user.id))
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": 1800,  # 30 minutes
    }


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    data: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Refresh access token using refresh token.
    
    - **refresh_token**: Valid refresh token from login
    """
    try:
        payload = verify_refresh_token(data.refresh_token)
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )
        
        # Verify user still exists and is active
        result = await db.execute(
            select(User).where(User.id == UUID(user_id))
        )
        user = result.scalar_one_or_none()
        
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive",
            )
        
        # Generate new tokens
        access_token = create_access_token(
            subject=str(user.id),
            additional_claims={"role": user.role.value},
        )
        new_refresh_token = create_refresh_token(subject=str(user.id))
        
        return {
            "access_token": access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer",
            "expires_in": 1800,
        }
    
    except TokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        )


@router.post("/logout")
async def logout() -> dict:
    """
    Logout user (client should discard tokens).
    
    Note: In a production system, you would also invalidate
    the refresh token in a token blacklist.
    """
    return {"message": "Successfully logged out"}
