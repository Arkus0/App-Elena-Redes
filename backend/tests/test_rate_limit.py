
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import time

# Helper to mock package structure
def mock_package(name):
    m = MagicMock()
    sys.modules[name] = m
    return m

# Mock dependencies heavily
mock_package('numpy')
mock_package('pandas')
mock_package('xgboost')
mock_package('river')
mock_package('river.forest')
mock_package('river.drift')
mock_package('shap')
mock_package('cv2')
mock_package('librosa')
mock_package('av')
mock_package('joblib')
mock_package('nltk')
mock_package('sentence_transformers')
mock_package('faster_whisper')
mock_package('easyocr')
mock_package('apify_client')

sklearn = mock_package('sklearn')
mock_package('sklearn.model_selection')
mock_package('sklearn.metrics')
mock_package('sklearn.preprocessing')
mock_package('sklearn.decomposition')
mock_package('sklearn.multioutput')

# Explicitly mock app services
mock_ml_service = MagicMock()
mock_ml_service.get_ml_predictor = MagicMock(return_value=MagicMock())
mock_ml_service.FeatureExtractor = MagicMock()
sys.modules['app.services.ml_service'] = mock_ml_service

mock_apify = MagicMock()
sys.modules['app.services.apify_service'] = mock_apify

sys.modules['app.services.content_processor'] = MagicMock()
sys.modules['app.services.user_config_service'] = MagicMock()
sys.modules['ml.light_processors'] = MagicMock()

from app.core.database import get_db

try:
    from app.main import app
    from app.api.auth import login_limiter
except ImportError as e:
    print(f"ImportError during app import: {e}")
    sys.exit(1)

@pytest.fixture
def client():
    async def mock_get_db():
        mock_session = AsyncMock()
        # Mock result for db.execute
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        yield mock_session
    app.dependency_overrides[get_db] = mock_get_db
    return TestClient(app)

@pytest.fixture(autouse=True)
def reset_limiter():
    login_limiter.history.clear()
    yield

def test_rate_limit_login(client):
    """Test rate limiting on login endpoint"""
    # 5 requests should be fine
    for i in range(5):
        response = client.post("/api/v1/auth/login", json={"email": "test@example.com", "password": "password"})
        # We expect 401 (invalid creds) or 422 (validation), NOT 429
        assert response.status_code != 429, f"Request {i+1} failed with 429"

    # 6th request should fail
    response = client.post("/api/v1/auth/login", json={"email": "test@example.com", "password": "password"})
    assert response.status_code == 429
    assert "Too many requests" in response.json()["detail"]

def test_rate_limit_reset(client):
    """Test that rate limit resets after window"""

    # Mock time.time in the limiter module
    with patch("app.core.limiter.time.time") as mock_time:
        mock_time.return_value = 1000.0

        # 5 requests at t=1000
        for _ in range(5):
            client.post("/api/v1/auth/login", json={"email": "reset@example.com", "password": "password"})

        # 6th fails
        response = client.post("/api/v1/auth/login", json={"email": "reset@example.com", "password": "password"})
        assert response.status_code == 429

        # Move time forward by 61 seconds (window is 60)
        mock_time.return_value = 1061.0

        # Should succeed now (history is cleared/filtered)
        response = client.post("/api/v1/auth/login", json={"email": "reset@example.com", "password": "password"})
        assert response.status_code != 429

if __name__ == "__main__":
    pytest.main([__file__])
