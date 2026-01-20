
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import types

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
except ImportError as e:
    print(f"ImportError during app import: {e}")
    sys.exit(1)

@pytest.fixture
def client():
    async def mock_get_db():
        yield MagicMock()
    app.dependency_overrides[get_db] = mock_get_db
    return TestClient(app)

def test_ingest_missing_auth(client):
    """Test that requests with businessId but no API key are rejected"""
    payload = {
        "source": "attacker",
        "businessId": 123,
        "userId": 456,
        "content": {
            "platform": "instagram",
            "contentType": "post",
            "contentId": "vuln_test_123",
            "contentUrl": "http://example.com/post",
            "author": { "username": "victim_user" },
            "metrics": { "likes": 100 },
            "extractedAt": "2024-01-01T00:00:00",
            "sourceUrl": "http://example.com",
            "extractionMethod": "dom_scraping"
        }
    }

    response = client.post("/api/ingest/raw", json=payload)
    assert response.status_code == 401
    assert "API Key required" in response.json()["detail"]

def test_ingest_invalid_key(client):
    """Test that requests with mismatching API key are rejected"""
    payload = {
        "source": "attacker",
        "businessId": 123,
        "userId": 456,
        "content": {
            "platform": "instagram",
            "contentType": "post",
            "contentId": "vuln_test_123",
            "contentUrl": "http://example.com/post",
            "author": { "username": "victim_user" },
            "metrics": { "likes": 100 },
            "extractedAt": "2024-01-01T00:00:00",
            "sourceUrl": "http://example.com",
            "extractionMethod": "dom_scraping"
        }
    }

    # Mock get_business_by_api_key to return a business with different ID
    with patch("app.api.ingest.get_business_by_api_key", new_callable=AsyncMock) as mock_get_biz:
        mock_biz = MagicMock()
        mock_biz.id = 999 # Mismatch
        mock_get_biz.return_value = mock_biz

        response = client.post(
            "/api/ingest/raw",
            json=payload,
            headers={"X-Extension-API-Key": "wrong_key"}
        )
        assert response.status_code == 401
        assert "Invalid API Key" in response.json()["detail"]

def test_ingest_success(client):
    """Test that requests with valid API key are accepted"""
    payload = {
        "source": "valid_user",
        "businessId": 123,
        "userId": 456,
        "content": {
            "platform": "instagram",
            "contentType": "post",
            "contentId": "valid_test_123",
            "contentUrl": "http://example.com/post",
            "author": { "username": "valid_user" },
            "metrics": { "likes": 100 },
            "extractedAt": "2024-01-01T00:00:00",
            "sourceUrl": "http://example.com",
            "extractionMethod": "dom_scraping"
        }
    }

    # Mock get_business_by_api_key to return correct business
    with patch("app.api.ingest.get_business_by_api_key", new_callable=AsyncMock) as mock_get_biz:
        mock_biz = MagicMock()
        mock_biz.id = 123 # Match
        mock_get_biz.return_value = mock_biz

        response = client.post(
            "/api/ingest/raw",
            json=payload,
            headers={"X-Extension-API-Key": "valid_key"}
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

def test_ingest_anonymous(client):
    """Test that requests WITHOUT businessId are allowed (anonymous)"""
    payload = {
        "source": "anonymous",
        # No businessId
        "content": {
            "platform": "instagram",
            "contentType": "post",
            "contentId": "anon_test_123",
            "contentUrl": "http://example.com/post",
            "author": { "username": "anon_user" },
            "metrics": { "likes": 100 },
            "extractedAt": "2024-01-01T00:00:00",
            "sourceUrl": "http://example.com",
            "extractionMethod": "dom_scraping"
        }
    }

    response = client.post("/api/ingest/raw", json=payload)
    assert response.status_code == 200
    assert response.json()["success"] is True

if __name__ == "__main__":
    # Allow running as script
    pytest.main([__file__])
