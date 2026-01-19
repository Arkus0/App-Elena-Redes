
import sys
import os

# Add backend to path so imports work
sys.path.append(os.path.join(os.getcwd(), "backend"))

try:
    from backend.app.services.content_generator import ContentGenerator
    print("ContentGenerator imported successfully.")
except ImportError as e:
    print(f"Failed to import ContentGenerator: {e}")
    sys.exit(1)
except NameError as e:
    print(f"NameError in ContentGenerator: {e}")
    sys.exit(1)

try:
    from backend.app.api.abtest import perform_thompson_sampling
    print("perform_thompson_sampling imported successfully from api.")
except ImportError as e:
    print(f"Failed to import from abtest api: {e}")
    # This might fail if we removed it from api but import it from service now
    # Check service instead
    try:
        from backend.app.services.abtest_service import perform_thompson_sampling
        print("perform_thompson_sampling imported successfully from service.")
    except ImportError as e2:
        print(f"Failed to import from service: {e2}")
        sys.exit(1)

print("Imports verification passed.")
