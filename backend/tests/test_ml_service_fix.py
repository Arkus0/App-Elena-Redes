
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Add backend to path as well
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.ml_service import MLPredictor, FeatureExtractor

class TestMLPredictorFix:
    def test_manual_feature_columns_duplicates(self):
        """
        Verify that MANUAL_FEATURE_COLUMNS has no overlaps with MULTIMODAL_FEATURE_COLUMNS.
        This explicitly checks for the bug that caused XGBoost crashes.
        """
        predictor = MLPredictor()

        from app.services.ml_service import MULTIMODAL_FEATURE_COLUMNS

        manual_features = set(predictor.MANUAL_FEATURE_COLUMNS)
        multimodal_features = set(MULTIMODAL_FEATURE_COLUMNS)

        intersection = manual_features.intersection(multimodal_features)
        assert len(intersection) == 0, f"Found duplicates: {intersection}"

    def test_train_and_predict_no_crash(self):
        """
        Test that training and prediction do not crash due to feature name conflicts.
        """
        predictor = MLPredictor()

        # Create mock data
        mock_data = []
        for i in range(10):
            mock_data.append({
                "caption": f"Test caption {i}",
                "likes": 100 + i,
                "comments": 10 + i,
                "saves": 5 + i,
                "shares": 2 + i,
                "views": 1000 + i * 10,
                "business_type": "otros",
                "is_reel": 1,
                "media_type": "reel"
            })

        # Try to train (this triggers feature extraction and model fitting)
        # We need to mock FeatureExtractor.extract_features if we don't want to run full extraction
        # But we want to test the full pipeline feature names.
        # However, we might not have all ML models (sentence-transformers) downloaded or available.
        # FeatureExtractor will fill zeros if models are missing, which is fine for this test.

        try:
            predictor.train(mock_data, retrain=True)
        except Exception as e:
            pytest.fail(f"Training failed: {e}")

        # Try prediction
        try:
            result = predictor.predict_engagement(mock_data[0])
            assert "score" in result
        except Exception as e:
            pytest.fail(f"Prediction failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
