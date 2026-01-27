
print("1. Testing Imports...")
from app.main import app
from app.api.abtest import router as abtest_router
from app.services.content_generator import ContentGenerator
from app.services.abtest_service import perform_thompson_sampling
print("✅ Imports Successful")

print("2. Testing Pydantic Schemas...")
from app.api.abtest import OnlineFeedbackResponse
test_obj = OnlineFeedbackResponse(
    post_id=1,
    online_learning_enabled=True,
    samples_processed=10,
    total_samples=100,
    current_mae=0.5,
    improvement_detected=False,
    trigger_full_retrain=False,
    update_time_ms=12.5,
    message="Test OK"
)
print("✅ Schema Instantiation Successful")

print("3. Testing App Startup & Routing...")
from fastapi.testclient import TestClient
client = TestClient(app)
# Just check if app mounts without crashing
print("✅ App initialized in TestClient successfully")
