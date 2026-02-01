import pytest
from pydantic import ValidationError
import os
import sys

# Ensure backend is in path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# CRITICAL: We must set DEBUG=true in the environment BEFORE importing config
# because config.py instantiates 'settings' at module level.
# If DEBUG defaults to False (production), and secrets are default,
# the module import itself would fail with ValidationError.
os.environ["DEBUG"] = "true"

from app.core.config import Settings, DEFAULT_SECRET_KEY, DEFAULT_DYNAMIC_SALT

def test_insecure_default_config_raises_error():
    """
    Test that starting the app with DEBUG=False and default secrets raises an error.
    """
    # Verify that it raises ValidationError when we force DEBUG=False
    # Note: We must explicitly pass the values to override any potential env vars
    with pytest.raises(ValidationError, match="Production configuration error"):
        Settings(
            DEBUG=False,
            SECRET_KEY=DEFAULT_SECRET_KEY,
            DYNAMIC_SALT=DEFAULT_DYNAMIC_SALT
        )

def test_secure_config_passes():
    """
    Test that starting the app with DEBUG=False and CHANGED secrets passes.
    """
    secure_secret = "a" * 32  # 32 chars
    secure_salt = "random-salt-value-for-production"

    settings = Settings(
        DEBUG=False,
        SECRET_KEY=secure_secret,
        DYNAMIC_SALT=secure_salt
    )
    assert settings.SECRET_KEY == secure_secret

def test_insecure_config_allowed_in_debug():
    """
    Test that default secrets ARE allowed when DEBUG=True.
    """
    settings = Settings(
        DEBUG=True,
        SECRET_KEY=DEFAULT_SECRET_KEY
    )
    assert settings.SECRET_KEY == DEFAULT_SECRET_KEY

def test_defaults_are_correct():
    """
    Test that we are testing against the actual defaults defined in the module.
    """
    assert DEFAULT_SECRET_KEY == "your-secret-key-change-in-production-min-32-chars"
    assert DEFAULT_DYNAMIC_SALT == "dev-dynamic-salt-change-in-prod-v1"
