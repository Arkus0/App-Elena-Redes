import pytest
from pydantic import ValidationError
from app.core.config import Settings

def test_insecure_defaults_prohibited():
    """Test that starting with DEBUG=False and default secrets raises an error"""
    with pytest.raises(ValidationError) as excinfo:
        Settings(
            DEBUG=False,
            # Implicitly using default SECRET_KEY and DYNAMIC_SALT
        )
    # Check that the error message mentions the specific problem
    errors = excinfo.value.errors()
    assert any("SECRET_KEY must be changed in production" in str(e) for e in errors) or \
           any("DYNAMIC_SALT must be changed in production" in str(e) for e in errors) or \
           any("must be changed in production" in str(e['msg']) for e in errors)

def test_insecure_defaults_allowed_in_debug():
    """Test that starting with DEBUG=True allows default secrets"""
    try:
        Settings(
            DEBUG=True,
            # Implicitly using default SECRET_KEY and DYNAMIC_SALT
        )
    except ValidationError:
        pytest.fail("Should not raise ValidationError when DEBUG=True")

def test_secure_values_allowed():
    """Test that starting with DEBUG=False and changed secrets is allowed"""
    try:
        Settings(
            DEBUG=False,
            SECRET_KEY="secure-production-key-that-is-very-long-and-secure",
            DYNAMIC_SALT="secure-production-salt-that-is-random"
        )
    except ValidationError:
        pytest.fail("Should not raise ValidationError with secure secrets")
