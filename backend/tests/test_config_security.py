import pytest
from pydantic import ValidationError
from app.core.config import Settings

def test_security_validation_in_production():
    """
    Test that the application refuses to start in production (DEBUG=False)
    if insecure default secrets are used.
    """
    # Case 1: DEBUG=False with default secrets -> Should Fail
    # Note: We pass the values explicitly to ensure we are testing the logic,
    # regardless of what environment variables might be set in the system.
    with pytest.raises(ValidationError, match="insecure default secrets"):
        Settings(
            DEBUG=False,
            SECRET_KEY="your-secret-key-change-in-production-min-32-chars",
            DYNAMIC_SALT="dev-dynamic-salt-change-in-prod-v1"
        )

def test_security_validation_dev_allowed():
    """
    Test that development mode (DEBUG=True) allows default secrets.
    """
    # Case 2: DEBUG=True with default secrets -> Should Pass
    try:
        settings = Settings(
            DEBUG=True,
            SECRET_KEY="your-secret-key-change-in-production-min-32-chars",
            DYNAMIC_SALT="dev-dynamic-salt-change-in-prod-v1"
        )
        assert settings.DEBUG is True
    except ValidationError:
        pytest.fail("Settings failed validation in DEBUG=True mode with default secrets")

def test_security_validation_prod_secure():
    """
    Test that production mode works with secure secrets.
    """
    # Case 3: DEBUG=False with changed secrets -> Should Pass
    try:
        settings = Settings(
            DEBUG=False,
            SECRET_KEY="secure-production-key-that-is-very-long-and-random",
            DYNAMIC_SALT="secure-production-salt-random-string"
        )
        assert settings.DEBUG is False
    except ValidationError:
        pytest.fail("Settings failed validation in DEBUG=False mode with secure secrets")
