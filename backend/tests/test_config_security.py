
import pytest
from pydantic import ValidationError
import os
from importlib import reload
import app.core.config

def test_security_safe_defaults_in_prod():
    """
    Test that the application refuses to start if DEBUG=False
    but the secrets are still set to their default insecure values.
    """
    # Force environment variables
    os.environ["DEBUG"] = "false"
    # These match the default values in config.py
    os.environ["SECRET_KEY"] = "your-secret-key-change-in-production-min-32-chars"
    os.environ["DYNAMIC_SALT"] = "dev-dynamic-salt-change-in-prod-v1"

    # Reload config to pick up env vars (though pydantic reads env vars,
    # the 'settings' object is cached, so we might need to clear cache or re-instantiate)

    # We directly verify the validation logic by instantiating Settings
    # Since we want to test that it RAISES an error, we use pytest.raises
    with pytest.raises(ValidationError) as excinfo:
        app.core.config.Settings()

    assert "SECRET_KEY must be changed in production" in str(excinfo.value) or \
           "DYNAMIC_SALT must be changed in production" in str(excinfo.value)

def test_security_allows_defaults_in_debug():
    """
    Test that insecure defaults are allowed when DEBUG=True
    """
    os.environ["DEBUG"] = "true"
    os.environ["SECRET_KEY"] = "your-secret-key-change-in-production-min-32-chars"
    os.environ["DYNAMIC_SALT"] = "dev-dynamic-salt-change-in-prod-v1"

    # Should NOT raise error
    try:
        app.core.config.Settings()
    except ValidationError:
        pytest.fail("Should not raise ValidationError in DEBUG mode")

def test_security_valid_prod_config():
    """
    Test that valid production config works
    """
    os.environ["DEBUG"] = "false"
    os.environ["SECRET_KEY"] = "super-secure-key-that-is-very-long-and-random-12345"
    os.environ["DYNAMIC_SALT"] = "super-secure-salt-random-value-98765"

    # Should NOT raise error
    try:
        app.core.config.Settings()
    except ValidationError:
        pytest.fail("Should not raise ValidationError with valid production config")
