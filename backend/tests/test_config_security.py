import pytest
from pydantic import ValidationError
from app.core.config import Settings
import os

# We need to make sure environment variables don't interfere with our tests
# Since BaseSettings reads from env, we'll try to instantiate with explicit values

def test_production_security_enforcement():
    """
    Critical Security Test:
    Ensures that the application refuses to start in production (DEBUG=False)
    if the default insecure secrets are still in use.
    """

    # These are the defaults in config.py
    insecure_defaults = {
        "DEBUG": False,
        "SECRET_KEY": "your-secret-key-change-in-production-min-32-chars",
        "DYNAMIC_SALT": "dev-dynamic-salt-change-in-prod-v1"
    }

    # Should raise ValidationError because we are in prod with default secrets
    with pytest.raises(ValidationError):
        Settings(**insecure_defaults)

def test_development_mode_allows_defaults():
    """
    Verifies that development mode (DEBUG=True) allows default secrets
    for easier developer onboarding.
    """
    dev_config = {
        "DEBUG": True,
        "SECRET_KEY": "your-secret-key-change-in-production-min-32-chars",
        "DYNAMIC_SALT": "dev-dynamic-salt-change-in-prod-v1"
    }

    # Should NOT raise
    try:
        Settings(**dev_config)
    except ValidationError:
        pytest.fail("Development mode should allow default secrets")

def test_production_valid_secrets():
    """
    Verifies that production mode accepts changed secrets.
    """
    secure_config = {
        "DEBUG": False,
        "SECRET_KEY": "a-very-secure-random-key-that-is-changed-for-prod-and-is-long-enough",
        "DYNAMIC_SALT": "a-very-secure-salt-that-is-changed-for-prod"
    }

    # Should NOT raise
    try:
        Settings(**secure_config)
    except ValidationError:
        pytest.fail("Production mode should accept secure secrets")
