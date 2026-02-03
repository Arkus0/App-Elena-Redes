import os
import pytest
from pydantic import ValidationError
from app.core.config import Settings

# Note: Pydantic BaseSettings reads from os.environ.
# We use monkeypatch to safely set environment variables for each test.

def test_production_mode_insecure_secret_key(monkeypatch):
    """Fail if DEBUG=False and default SECRET_KEY is used"""
    monkeypatch.setenv("DEBUG", "False")
    # Explicitly set the default insecure key
    monkeypatch.setenv("SECRET_KEY", "your-secret-key-change-in-production-min-32-chars")
    monkeypatch.setenv("DYNAMIC_SALT", "secure-salt") # Set other to secure to isolate test

    with pytest.raises(ValidationError) as excinfo:
        Settings()

    # We expect our validator to add a message about insecure config
    assert "default insecure" in str(excinfo.value).lower()

def test_production_mode_insecure_dynamic_salt(monkeypatch):
    """Fail if DEBUG=False and default DYNAMIC_SALT is used"""
    monkeypatch.setenv("DEBUG", "False")
    monkeypatch.setenv("SECRET_KEY", "secure-key-is-long-enough-1234567890")
    monkeypatch.setenv("DYNAMIC_SALT", "dev-dynamic-salt-change-in-prod-v1")

    with pytest.raises(ValidationError) as excinfo:
        Settings()

    assert "default insecure" in str(excinfo.value).lower()

def test_development_mode_allows_insecure_defaults(monkeypatch):
    """Allow default secrets if DEBUG=True"""
    monkeypatch.setenv("DEBUG", "True")
    monkeypatch.setenv("SECRET_KEY", "your-secret-key-change-in-production-min-32-chars")
    monkeypatch.setenv("DYNAMIC_SALT", "dev-dynamic-salt-change-in-prod-v1")

    try:
        settings = Settings()
        assert settings.DEBUG is True
    except ValidationError:
        pytest.fail("Should not raise ValidationError in DEBUG mode")

def test_production_mode_secure_config(monkeypatch):
    """Allow if DEBUG=False and secrets are changed"""
    monkeypatch.setenv("DEBUG", "False")
    monkeypatch.setenv("SECRET_KEY", "secure-key-that-is-long-enough-for-production")
    monkeypatch.setenv("DYNAMIC_SALT", "secure-salt-value-v1")

    try:
        settings = Settings()
        assert settings.DEBUG is False
    except ValidationError:
        pytest.fail("Should not raise ValidationError with secure config")
