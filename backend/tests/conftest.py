import os
import pytest

# Global setup for all tests
# Ensure that by default, tests run in a "secure" configuration environment
# or at least in a mode that doesn't trigger the production security check immediately
# upon import.

# We set DEBUG=True so that importing app.core.config (which instantiates Settings)
# does not raise ValidationError due to default secrets.
os.environ["DEBUG"] = "true"

# We also set a compliant secret key just in case some tests run with DEBUG=False
# and we want them to pass validation, OR if we want to test with DEBUG=True but proper keys.
# However, setting this might interfere with the specific test case that wants to check
# for insecure defaults.
#
# BUT: The Settings class loads from env vars. If we set os.environ["SECRET_KEY"],
# pydantic will use it.
#
# For `test_config_security.py`, we want to test the behavior when secrets are NOT set (or are defaults).
# So we should probably NOT set SECRET_KEY here globally, or if we do, we need to unset it
# in that specific test.
#
# Since the security check only fires when DEBUG=False, setting DEBUG=True is sufficient
# to prevent import-time crashes for general tests.
#
# So I will ONLY set DEBUG=true.

@pytest.fixture(autouse=True)
def set_test_env():
    # This fixture runs before each test, but os.environ set at module level
    # happens at import time.
    # The 'conftest.py' module level code runs before tests are collected (mostly).
    pass
