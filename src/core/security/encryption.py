"""HIPAA-compliant encryption service using AES-256-GCM"""
import base64
import hashlib
import secrets
from typing import Optional, Tuple
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

from src.core.config import settings


class EncryptionService:
    """AES-256-GCM encryption for PHI data - HIPAA compliant"""
    
    NONCE_SIZE = 12  # 96 bits for GCM
    KEY_SIZE = 32    # 256 bits
    TAG_SIZE = 16    # 128 bits authentication tag
    SALT_SIZE = 16   # 128 bits salt
    
    def __init__(self, master_key: Optional[str] = None):
        """Initialize with master encryption key"""
        self._master_key = (master_key or settings.encryption_key).encode()
        self._validate_key()
    
    def _validate_key(self) -> None:
        """Validate master key meets requirements"""
        if len(self._master_key) < 32:
            raise ValueError("Master key must be at least 32 bytes for AES-256")
    
    def _derive_key(self, salt: bytes) -> bytes:
        """Derive encryption key from master key using PBKDF2"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self.KEY_SIZE,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        return kdf.derive(self._master_key)
    
    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt plaintext using AES-256-GCM
        
        Returns: Base64 encoded string containing salt + nonce + ciphertext + tag
        """
        if not plaintext:
            return ""
        
        # Generate random salt and nonce
        salt = secrets.token_bytes(self.SALT_SIZE)
        nonce = secrets.token_bytes(self.NONCE_SIZE)
        
        # Derive key from master key
        key = self._derive_key(salt)
        
        # Encrypt using AES-GCM
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext.encode('utf-8'), None)
        
        # Combine: salt + nonce + ciphertext (includes tag)
        combined = salt + nonce + ciphertext
        
        return base64.b64encode(combined).decode('utf-8')
    
    def decrypt(self, encrypted_data: str) -> str:
        """
        Decrypt AES-256-GCM encrypted data
        
        Args:
            encrypted_data: Base64 encoded string from encrypt()
            
        Returns: Decrypted plaintext
        """
        if not encrypted_data:
            return ""
        
        # Decode from base64
        combined = base64.b64decode(encrypted_data.encode('utf-8'))
        
        # Extract components
        salt = combined[:self.SALT_SIZE]
        nonce = combined[self.SALT_SIZE:self.SALT_SIZE + self.NONCE_SIZE]
        ciphertext = combined[self.SALT_SIZE + self.NONCE_SIZE:]
        
        # Derive key from master key
        key = self._derive_key(salt)
        
        # Decrypt using AES-GCM
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        
        return plaintext.decode('utf-8')
    
    def hash_identifier(self, identifier: str) -> str:
        """
        Create a deterministic hash for identifiers (for lookups)
        Uses SHA-256 with application salt
        """
        salted = f"{settings.secret_key}:{identifier}".encode()
        return hashlib.sha256(salted).hexdigest()
    
    def encrypt_field(self, value: str, field_name: str) -> str:
        """
        Encrypt a specific field with additional context
        Adds field name as associated data for integrity
        """
        if not value:
            return ""
        
        salt = secrets.token_bytes(self.SALT_SIZE)
        nonce = secrets.token_bytes(self.NONCE_SIZE)
        key = self._derive_key(salt)
        
        aesgcm = AESGCM(key)
        # Use field name as associated authenticated data
        ciphertext = aesgcm.encrypt(nonce, value.encode('utf-8'), field_name.encode('utf-8'))
        
        combined = salt + nonce + ciphertext
        return base64.b64encode(combined).decode('utf-8')
    
    def decrypt_field(self, encrypted_data: str, field_name: str) -> str:
        """Decrypt a field encrypted with encrypt_field"""
        if not encrypted_data:
            return ""
        
        combined = base64.b64decode(encrypted_data.encode('utf-8'))
        
        salt = combined[:self.SALT_SIZE]
        nonce = combined[self.SALT_SIZE:self.SALT_SIZE + self.NONCE_SIZE]
        ciphertext = combined[self.SALT_SIZE + self.NONCE_SIZE:]
        
        key = self._derive_key(salt)
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, field_name.encode('utf-8'))
        
        return plaintext.decode('utf-8')
    
    @staticmethod
    def generate_key() -> str:
        """Generate a new random encryption key"""
        return secrets.token_urlsafe(32)


# Singleton instance
_encryption_service: Optional[EncryptionService] = None


def get_encryption_service() -> EncryptionService:
    """Get or create encryption service singleton"""
    global _encryption_service
    if _encryption_service is None:
        _encryption_service = EncryptionService()
    return _encryption_service
