import os
import pytest

# Set DEBUG=true for testing to bypass security checks on default secrets
# This must be done before any app modules are imported
os.environ["DEBUG"] = "true"
