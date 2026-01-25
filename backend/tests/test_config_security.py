import pytest
from pydantic import ValidationError
from app.core.config import Settings

def test_settings_security_validation_in_production():
    """
    Verify that starting the app in production (DEBUG=False)
    with insecure default secrets raises a validation error.
    """
    # These are the insecure defaults from the code
    insecure_secret = "your-secret-key-change-in-production-min-32-chars"
    insecure_salt = "dev-dynamic-salt-change-in-prod-v1"

    # 1. Test: DEBUG=False with default SECRET_KEY should FAIL
    with pytest.raises(ValidationError, match="SECRET_KEY must be changed in production"):
        Settings(
            DEBUG=False,
            SECRET_KEY=insecure_secret,
            DYNAMIC_SALT="some-secure-salt-v1"
        )

    # 2. Test: DEBUG=False with default DYNAMIC_SALT should FAIL
    with pytest.raises(ValidationError, match="DYNAMIC_SALT must be changed in production"):
        Settings(
            DEBUG=False,
            SECRET_KEY="some-secure-secret-key-that-is-long-enough",
            DYNAMIC_SALT=insecure_salt
        )

    # 3. Test: DEBUG=True with defaults should PASS
    try:
        Settings(
            DEBUG=True,
            SECRET_KEY=insecure_secret,
            DYNAMIC_SALT=insecure_salt
        )
    except ValidationError:
        pytest.fail("Settings should pass validation when DEBUG=True")

    # 4. Test: DEBUG=False with changed secrets should PASS
    try:
        Settings(
            DEBUG=False,
            SECRET_KEY="secure-production-key-that-is-very-long-and-random",
            DYNAMIC_SALT="secure-production-salt-v1"
        )
    except ValidationError:
        pytest.fail("Settings should pass validation with secure secrets in production")
