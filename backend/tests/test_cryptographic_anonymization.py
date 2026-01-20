import pytest
from app.core.privacy import PrivacyProvider

def test_hashing_consistency():
    """Test that hashing is deterministic with the same salt."""
    provider = PrivacyProvider(salt="test-salt", enabled=True)
    pii = "user@example.com"

    hash1 = provider.hash_pii(pii)
    hash2 = provider.hash_pii(pii)

    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 hex digest length
    assert hash1 != pii

def test_salt_impact():
    """Test that different salts produce different hashes."""
    provider1 = PrivacyProvider(salt="salt-1", enabled=True)
    provider2 = PrivacyProvider(salt="salt-2", enabled=True)
    pii = "same-user"

    hash1 = provider1.hash_pii(pii)
    hash2 = provider2.hash_pii(pii)

    assert hash1 != hash2

def test_disabled_mode():
    """Test that PrivacyProvider returns raw value when disabled."""
    provider = PrivacyProvider(salt="test-salt", enabled=False)
    pii = "raw-data"

    result = provider.hash_pii(pii)
    assert result == pii

def test_compare_pii():
    """Test on-the-fly comparison logic."""
    provider = PrivacyProvider(salt="test-salt", enabled=True)
    raw_value = "my-secret-username"
    hashed_value = provider.hash_pii(raw_value)

    # Correct comparison
    assert provider.compare_pii(raw_value, hashed_value) is True

    # Incorrect comparison
    assert provider.compare_pii("wrong-username", hashed_value) is False

def test_sanitize_log():
    """Test log sanitization of emails and usernames."""
    provider = PrivacyProvider(salt="test-salt", enabled=True)

    # Test Email
    log_msg = "Error processing user test.user@example.com in module X"
    sanitized = provider.sanitize_log(log_msg)
    assert "test.user@example.com" not in sanitized
    assert "[EMAIL_REDACTED]" in sanitized

    # Test Username (@handle)
    log_msg = "Ingesting profile @awesome_brand for analysis"
    sanitized = provider.sanitize_log(log_msg)
    assert "@awesome_brand" not in sanitized
    assert "@[USER_REDACTED]" in sanitized

    # Test Mixed
    log_msg = "User @brand_x (contact: support@brandx.com) failed"
    sanitized = provider.sanitize_log(log_msg)
    assert "@[USER_REDACTED]" in sanitized
    assert "[EMAIL_REDACTED]" in sanitized

def test_hash_pii_none_handling():
    """Test handling of None and empty strings."""
    provider = PrivacyProvider(salt="test-salt", enabled=True)

    assert provider.hash_pii(None) is None
    assert provider.hash_pii("") is None
    assert provider.hash_pii("   ") is None
