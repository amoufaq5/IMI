"""
UMI Security Module
Authentication, authorization, and encryption utilities
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from src.core.config import settings

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class SecurityError(Exception):
    """Base security exception."""
    pass


class TokenError(SecurityError):
    """Token-related errors."""
    pass


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)


def create_access_token(
    subject: str | Any,
    expires_delta: Optional[timedelta] = None,
    additional_claims: Optional[dict] = None,
) -> str:
    """
    Create JWT access token.
    
    Args:
        subject: Token subject (usually user ID)
        expires_delta: Custom expiration time
        additional_claims: Extra claims to include in token
    
    Returns:
        Encoded JWT token
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.jwt_access_token_expire_minutes
        )

    to_encode = {
        "sub": str(subject),
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "access",
    }

    if additional_claims:
        to_encode.update(additional_claims)

    return jwt.encode(
        to_encode,
        settings.secret_key,
        algorithm=settings.jwt_algorithm,
    )


def create_refresh_token(
    subject: str | Any,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create JWT refresh token.
    
    Args:
        subject: Token subject (usually user ID)
        expires_delta: Custom expiration time
    
    Returns:
        Encoded JWT refresh token
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            days=settings.jwt_refresh_token_expire_days
        )

    to_encode = {
        "sub": str(subject),
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "refresh",
    }

    return jwt.encode(
        to_encode,
        settings.secret_key,
        algorithm=settings.jwt_algorithm,
    )


def decode_token(token: str) -> dict:
    """
    Decode and validate JWT token.
    
    Args:
        token: JWT token string
    
    Returns:
        Decoded token payload
    
    Raises:
        TokenError: If token is invalid or expired
    """
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        return payload
    except JWTError as e:
        raise TokenError(f"Invalid token: {str(e)}")


def verify_access_token(token: str) -> dict:
    """
    Verify access token and return payload.
    
    Args:
        token: JWT access token
    
    Returns:
        Token payload with user info
    
    Raises:
        TokenError: If token is invalid or not an access token
    """
    payload = decode_token(token)
    
    if payload.get("type") != "access":
        raise TokenError("Invalid token type")
    
    return payload


def verify_refresh_token(token: str) -> dict:
    """
    Verify refresh token and return payload.
    
    Args:
        token: JWT refresh token
    
    Returns:
        Token payload with user info
    
    Raises:
        TokenError: If token is invalid or not a refresh token
    """
    payload = decode_token(token)
    
    if payload.get("type") != "refresh":
        raise TokenError("Invalid token type")
    
    return payload


class FieldEncryption:
    """
    Field-level encryption for sensitive data.
    Uses Fernet symmetric encryption.
    """

    def __init__(self, key: Optional[str] = None):
        from cryptography.fernet import Fernet
        
        if key:
            self._fernet = Fernet(key.encode())
        else:
            # Derive key from secret_key
            import base64
            import hashlib
            
            key_bytes = hashlib.sha256(settings.secret_key.encode()).digest()
            self._fernet = Fernet(base64.urlsafe_b64encode(key_bytes))

    def encrypt(self, data: str) -> str:
        """Encrypt string data."""
        return self._fernet.encrypt(data.encode()).decode()

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt encrypted string."""
        return self._fernet.decrypt(encrypted_data.encode()).decode()


# Singleton instance for field encryption
field_encryption = FieldEncryption()
