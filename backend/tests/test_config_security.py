import pytest
import os
from pydantic import ValidationError
from app.core.config import Settings

class TestConfigSecurity:
    def test_secure_settings_validation_defaults(self):
        """
        Test that the application refuses to start (raises ValidationError)
        if DEBUG is False and insecure default secrets are used.
        """
        # We need to ensure we are not picking up the "DEBUG=true" from conftest/env
        # We can pass arguments directly to Settings to override env vars.

        # Case 1: DEBUG=False with default secrets -> Should Fail
        # We must verify the exact strings match what is in config.py
        default_secret = "your-secret-key-change-in-production-min-32-chars"
        default_salt = "dev-dynamic-salt-change-in-prod-v1"

        print("Testing Insecure Defaults...")
        with pytest.raises(ValidationError) as excinfo:
            Settings(
                DEBUG=False,
                SECRET_KEY=default_secret,
                DYNAMIC_SALT=default_salt
            )

        error_msg = str(excinfo.value)
        assert "SECRET_KEY must be changed in production" in error_msg or \
               "DYNAMIC_SALT must be changed in production" in error_msg

    def test_secure_settings_validation_debug_mode(self):
        """Test that default secrets are allowed in DEBUG mode"""
        default_secret = "your-secret-key-change-in-production-min-32-chars"
        default_salt = "dev-dynamic-salt-change-in-prod-v1"

        try:
            Settings(
                DEBUG=True,
                SECRET_KEY=default_secret,
                DYNAMIC_SALT=default_salt
            )
        except ValidationError as e:
            pytest.fail(f"Should not raise ValidationError when DEBUG=True: {e}")

    def test_secure_settings_validation_custom_secrets(self):
        """Test that custom secrets are allowed in Production mode"""
        try:
            Settings(
                DEBUG=False,
                SECRET_KEY="correct-horse-battery-staple-secure-key",
                DYNAMIC_SALT="another-secure-salt-value-12345"
            )
        except ValidationError as e:
            pytest.fail(f"Should not raise ValidationError when secure secrets are provided: {e}")
