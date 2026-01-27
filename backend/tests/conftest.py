import os
import sys
from pathlib import Path

# Add backend to python path to ensure imports work
backend_path = str(Path(__file__).parent.parent)
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

# Set DEBUG=True by default for tests to bypass production security checks
# This must be done before importing any app modules that instantiate settings
if os.environ.get("DEBUG") is None:
    os.environ["DEBUG"] = "True"
