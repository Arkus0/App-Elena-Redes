
import os
import pytest
from pydantic import ValidationError

# CRITICAL: We must ensure DEBUG=True is set BEFORE importing Settings
# otherwise the module-level instantiation of 'settings' in config.py
# will fail validation and crash the test collection.
os.environ["DEBUG"] = "true"

from app.core.config import Settings

def test_settings_security_validation():
    """
    Test that the application refuses to start (raises ValidationError)
    if DEBUG=False but secrets are still at their insecure default values.
    """
    # 1. Test SECURE configuration (should pass)
    try:
        Settings(
            DEBUG=False,
            SECRET_KEY="secure-random-key-that-is-long-enough-for-production",
            DYNAMIC_SALT="another-secure-random-salt-for-production-hashing",
        )
    except ValidationError as e:
        pytest.fail(f"Secure configuration failed validation: {e}")

    # 2. Test INSECURE configuration (should fail)
    # This matches the default values in config.py
    insecure_default_secret = "your-secret-key-change-in-production-min-32-chars"
    insecure_default_salt = "dev-dynamic-salt-change-in-prod-v1"

    with pytest.raises(ValidationError) as excinfo:
        Settings(
            DEBUG=False,
            SECRET_KEY=insecure_default_secret,
            DYNAMIC_SALT=insecure_default_salt,
        )

    # We check for the specific error message
    error_str = str(excinfo.value)
    assert "CRITICAL SECURITY ERROR" in error_str
    assert "must be changed in production" in error_str

def test_settings_dev_mode_allowed():
    """
    Test that default insecure secrets ARE allowed when DEBUG=True (dev mode).
    """
    try:
        # We can pass default insecure values explicitly, or let them default
        Settings(
            DEBUG=True,
            SECRET_KEY="your-secret-key-change-in-production-min-32-chars",
            DYNAMIC_SALT="dev-dynamic-salt-change-in-prod-v1",
        )
    except ValidationError as e:
        pytest.fail(f"Dev configuration failed validation: {e}")
