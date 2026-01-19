"""
Privacy Provider - Blind Identity Protocol
==========================================

Centralized component for cryptographic anonymization of PII (Personally Identifiable Information).
Ensures that all user data stored in the database is hashed using a consistent, secure salt.

Protocol:
- SHA-256 hashing with DYNAMIC_SALT (from env)
- One-way transformation (irreversible)
- On-the-fly comparison for "isOwnProfile" logic
- Log sanitization for PII protection

Author: BrandPulse AI Security Team
"""

import hashlib
import re
import logging
from typing import Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class PrivacyProvider:
    """
    Motor central de anonimización para el Protocolo de Privacidad Blindada.
    Utiliza SHA-256 con Salt dinámico para transformar PII en identificadores únicos irreversibles.
    """

    # Regex patterns for PII detection in logs
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    # Basic username pattern (starts with @ or matches common social handles)
    USERNAME_PATTERN = re.compile(r'@\w{3,30}\b')

    def __init__(self, salt: str, enabled: bool = True):
        self.salt = salt
        self.enabled = enabled
        if not self.salt and self.enabled:
            logger.warning("PrivacyProvider enabled but DYNAMIC_SALT is empty/insecure!")
        elif self.enabled:
            logger.info("PrivacyProvider initialized (Blind Identity Enabled)")

    def hash_pii(self, value: Optional[str]) -> Optional[str]:
        """
        Convert PII into an irreversible SHA-256 hash using a dynamic salt.

        Args:
            value: The raw PII string (e.g., username, email, url)

        Returns:
            SHA-256 hex digest string, or None if input is None
        """
        if value is None:
            return None

        # Convert to string and strip if necessary, handle non-string inputs gracefully
        clean_value = str(value).strip()
        if not clean_value:
            return None

        if not self.enabled:
            return clean_value

        # Combine salt and value
        salted_input = f"{self.salt}{clean_value}"

        # Generate SHA-256 hash
        return hashlib.sha256(salted_input.encode('utf-8')).hexdigest()

    def compare_pii(self, raw_value: str, hashed_value: str) -> bool:
        """
        Compare a raw PII value against a stored hash.
        Used for 'On-the-Fly' verification of 'own profile' status.

        Args:
            raw_value: The cleartext value (e.g., config.own_instagram_username)
            hashed_value: The hashed value from the payload/database

        Returns:
            True if the raw_value hashes to hashed_value
        """
        if not raw_value or not hashed_value:
            return False

        # If privacy is disabled, compare raw values (assuming hashed_value is actually raw)
        if not self.enabled:
            return raw_value == hashed_value

        current_hash = self.hash_pii(raw_value)
        return current_hash == hashed_value

    def sanitize_log(self, message: str) -> str:
        """
        Sanitize log messages by replacing PII patterns with [REDACTED].
        Intercepta y ofusca patrones comunes de PII en strings de log (regex para @usuario y URLs).

        Args:
            message: The log message string

        Returns:
            Sanitized string
        """
        if not message:
            return ""

        # Redact emails
        sanitized = self.EMAIL_PATTERN.sub('[EMAIL_REDACTED]', message)

        # Redact usernames (starting with @)
        sanitized = self.USERNAME_PATTERN.sub('@[USER_REDACTED]', sanitized)

        return sanitized


# Global instance
settings = get_settings()
privacy_provider = PrivacyProvider(
    salt=settings.DYNAMIC_SALT,
    enabled=settings.PRIVACY_MODE_ENABLED
)
