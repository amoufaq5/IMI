"""
UMI API Dependencies
Common dependencies for API routes
"""

from typing import AsyncGenerator, Optional
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_db
from src.core.security import verify_access_token, TokenError
from src.models.user import User, UserRole

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Get current authenticated user from JWT token.
    
    Raises:
        HTTPException: If token is invalid or user not found
    """
    try:
        payload = verify_access_token(credentials.credentials)
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )
        
        result = await db.execute(
            select(User).where(User.id == UUID(user_id))
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is disabled",
            )
        
        return user
    
    except TokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        )


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Get current active user."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )
    return current_user


async def get_current_superuser(
    current_user: User = Depends(get_current_user),
) -> User:
    """Get current superuser."""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    return current_user


def require_roles(*roles: UserRole):
    """
    Dependency factory for role-based access control.
    
    Usage:
        @router.get("/admin", dependencies=[Depends(require_roles(UserRole.SYSTEM_ADMIN))])
    """
    async def role_checker(
        current_user: User = Depends(get_current_user),
    ) -> User:
        if current_user.role not in roles and not current_user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required roles: {[r.value for r in roles]}",
            )
        return current_user
    
    return role_checker


class RateLimiter:
    """Simple rate limiter dependency."""
    
    def __init__(self, requests: int = 100, window: int = 60):
        self.requests = requests
        self.window = window
        self._cache = {}  # In production, use Redis
    
    async def __call__(
        self,
        current_user: User = Depends(get_current_user),
    ) -> None:
        """Check rate limit for user."""
        # Simple implementation - in production use Redis
        user_key = str(current_user.id)
        
        # For now, just pass through
        # In production, implement proper rate limiting with Redis
        pass


# Default rate limiter
rate_limiter = RateLimiter()
