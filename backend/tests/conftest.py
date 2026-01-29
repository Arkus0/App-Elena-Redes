import os

# Force DEBUG=True for tests to avoid security validation errors during import
# This allows tests to import app.core.config without crashing.
if os.environ.get("DEBUG") is None:
    os.environ["DEBUG"] = "true"
