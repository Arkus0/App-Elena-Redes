import pytest
from pydantic import ValidationError
from app.core.config import Settings

def test_insecure_default_config_raises_error():
    """
    Test that the application refuses to start if DEBUG=False
    and insecure default secrets are used.
    """
    with pytest.raises(ValidationError) as excinfo:
        Settings(
            DEBUG=False,
            SECRET_KEY="your-secret-key-change-in-production-min-32-chars",
            DYNAMIC_SALT="dev-dynamic-salt-change-in-prod-v1"
        )
    assert "CRITICAL SECURITY ERROR" in str(excinfo.value)
    assert "default SECRET_KEY" in str(excinfo.value)

def test_insecure_salt_raises_error():
    """Test that insecure salt raises error even if secret key is changed."""
    with pytest.raises(ValidationError) as excinfo:
        Settings(
            DEBUG=False,
            SECRET_KEY="secure-key-that-is-long-enough-and-complex",
            DYNAMIC_SALT="dev-dynamic-salt-change-in-prod-v1"
        )
    assert "CRITICAL SECURITY ERROR" in str(excinfo.value)
    assert "default DYNAMIC_SALT" in str(excinfo.value)

def test_production_config_with_secure_secrets_passes():
    """Test that secure secrets allow startup in production mode."""
    settings = Settings(
        DEBUG=False,
        SECRET_KEY="secure-key-that-is-long-enough-and-complex",
        DYNAMIC_SALT="secure-salt-for-production-environment"
    )
    assert settings.DEBUG is False
    assert settings.SECRET_KEY != "your-secret-key-change-in-production-min-32-chars"

def test_debug_mode_allows_default_secrets():
    """Test that DEBUG=True allows default secrets (for dev convenience)."""
    settings = Settings(
        DEBUG=True,
        SECRET_KEY="your-secret-key-change-in-production-min-32-chars",
        DYNAMIC_SALT="dev-dynamic-salt-change-in-prod-v1"
    )
    assert settings.DEBUG is True
