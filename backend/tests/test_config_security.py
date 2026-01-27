import pytest
import os
from pydantic import ValidationError
from app.core.config import Settings

def test_production_security_check_defaults():
    """
    Test that the application refuses to start in production (DEBUG=False)
    if the default secrets are still in use.
    """
    # Simulate production environment with default secrets
    # We pass these explicitly to override any potential environment variables
    env_vars = {
        "DEBUG": False,
        "SECRET_KEY": "your-secret-key-change-in-production-min-32-chars",
        "DYNAMIC_SALT": "dev-dynamic-salt-change-in-prod-v1"
    }

    # Expect a validation error because defaults are unsafe for production
    with pytest.raises(ValidationError) as excinfo:
        Settings(**env_vars)

    assert "Production security check failed" in str(excinfo.value)
    assert "default SECRET_KEY" in str(excinfo.value) or "default DYNAMIC_SALT" in str(excinfo.value)

def test_production_security_check_safe_secrets():
    """
    Test that the application starts correctly in production
    if the secrets have been changed.
    """
    env_vars = {
        "DEBUG": False,
        "SECRET_KEY": "a-very-secure-random-production-key-that-is-long-enough",
        "DYNAMIC_SALT": "a-very-secure-random-production-salt-value"
    }

    try:
        settings = Settings(**env_vars)
        assert settings.DEBUG is False
        assert settings.SECRET_KEY != "your-secret-key-change-in-production-min-32-chars"
    except ValidationError:
        pytest.fail("Settings raised ValidationError with secure secrets in production")

def test_dev_mode_allows_defaults():
    """
    Test that the application allows default secrets in debug mode.
    """
    env_vars = {
        "DEBUG": True,
        "SECRET_KEY": "your-secret-key-change-in-production-min-32-chars",
        "DYNAMIC_SALT": "dev-dynamic-salt-change-in-prod-v1"
    }

    try:
        settings = Settings(**env_vars)
        assert settings.DEBUG is True
    except ValidationError:
        pytest.fail("Settings raised ValidationError in DEBUG mode")
