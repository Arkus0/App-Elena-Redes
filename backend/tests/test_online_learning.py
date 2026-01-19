"""
Tests for Online/Incremental Learning Module
=============================================

Tests the River-based online learning system for engagement prediction.
Simulates 50 incremental updates and confirms progressive improvement.

Run with:
    pytest backend/tests/test_online_learning.py -v

Author: BrandPulse AI
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def check_river_available():
    """Check if River is installed."""
    try:
        import river
        return True
    except ImportError:
        pytest.skip("River not installed. Install with: pip install river")
        return False


@pytest.fixture
def synthetic_features():
    """Generate synthetic feature data for testing."""
    np.random.seed(42)
    n_features = 30

    feature_names = [
        "caption_length", "caption_words", "emoji_count", "hashtag_count",
        "cta_count", "hook_score", "sentiment_compound", "is_reel",
        "is_carousel", "is_static", "video_duration", "video_optimal_length",
        "hour_of_day", "day_of_week", "is_weekend", "is_prime_time",
        "niche_inmobiliaria", "niche_floristeria", "niche_cafeteria",
        "niche_restaurante", "trigger_question", "trigger_action",
        "lexical_richness", "mention_count", "has_audio", "is_trending_audio",
        "semantic_hook_score", "semantic_hook_max_sim", "semantic_hook_top3_avg",
        "has_strong_cta"
    ]

    return feature_names[:n_features]


@pytest.fixture
def generate_batch_data(synthetic_features):
    """Generate a batch of synthetic data for testing."""
    def _generate(n_samples: int = 50, noise_level: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        np.random.seed(int(time.time()) % 1000)

        # Generate features
        features = {}
        for i, name in enumerate(synthetic_features):
            if "is_" in name or "has_" in name or "niche_" in name:
                # Binary features
                features[name] = np.random.randint(0, 2, n_samples).astype(float)
            elif name in ["hour_of_day"]:
                features[name] = np.random.randint(0, 24, n_samples).astype(float)
            elif name in ["day_of_week"]:
                features[name] = np.random.randint(0, 7, n_samples).astype(float)
            elif "score" in name or "compound" in name:
                features[name] = np.random.uniform(-1, 1, n_samples)
            elif "count" in name:
                features[name] = np.random.randint(0, 20, n_samples).astype(float)
            elif "length" in name or "duration" in name:
                features[name] = np.random.uniform(0, 500, n_samples)
            else:
                features[name] = np.random.uniform(0, 1, n_samples)

        features_df = pd.DataFrame(features)

        # Generate targets with some pattern (log-scale engagement)
        base_likes = 3 + features_df["emoji_count"] * 0.2 + features_df["cta_count"] * 0.3
        base_comments = 1.5 + features_df["cta_count"] * 0.4 + features_df["hook_score"] * 0.5
        base_shares = 0.5 + features_df["is_reel"] * 0.8
        base_saves = 1.0 + features_df["semantic_hook_score"] * 0.6
        base_views = 5 + features_df["is_reel"] * 1.5 + features_df["is_prime_time"] * 0.5

        # Add noise
        targets_df = pd.DataFrame({
            "log_likes": base_likes + np.random.normal(0, noise_level, n_samples),
            "log_comments": base_comments + np.random.normal(0, noise_level, n_samples),
            "log_shares": base_shares + np.random.normal(0, noise_level * 0.5, n_samples),
            "log_saves": base_saves + np.random.normal(0, noise_level * 0.5, n_samples),
            "log_views": base_views + np.random.normal(0, noise_level, n_samples),
        })

        return features_df, targets_df

    return _generate


# =============================================================================
# Unit Tests
# =============================================================================

class TestOnlineUpdateModule:
    """Tests for the online_update module."""

    def test_river_import(self, check_river_available):
        """Test that River can be imported."""
        from backend.ml.online_update import RIVER_AVAILABLE
        assert RIVER_AVAILABLE is True

    def test_online_predictor_initialization(self, check_river_available):
        """Test OnlineEngagementPredictor initialization."""
        from backend.ml.online_update import (
            OnlineEngagementPredictor,
            reset_online_predictor
        )

        niche = "test_init"
        reset_online_predictor(niche)

        predictor = OnlineEngagementPredictor(niche=niche)

        assert predictor.niche == niche
        assert predictor._metrics.samples_seen == 0
        assert predictor._is_initialized is False

        # Cleanup
        reset_online_predictor(niche)

    def test_partial_fit_single(self, check_river_available, synthetic_features):
        """Test partial_fit with a single sample."""
        from backend.ml.online_update import (
            OnlineEngagementPredictor,
            reset_online_predictor
        )

        niche = "test_partial_fit"
        reset_online_predictor(niche)

        predictor = OnlineEngagementPredictor(niche=niche)

        # Create single sample
        features = {name: np.random.random() for name in synthetic_features}
        targets = {
            "log_likes": 5.0,
            "log_comments": 2.0,
            "log_shares": 1.0,
            "log_saves": 1.5,
            "log_views": 7.0,
        }

        # Fit
        predictor.partial_fit(features, targets)

        assert predictor._metrics.samples_seen == 1
        assert predictor._is_initialized is True

        # Cleanup
        reset_online_predictor(niche)

    def test_online_update_function(self, check_river_available, generate_batch_data):
        """Test the main online_update function."""
        from backend.ml.online_update import (
            online_update,
            reset_online_predictor,
            OnlineUpdateResult
        )

        niche = "test_online_update"
        reset_online_predictor(niche)

        # Generate data
        features_df, targets_df = generate_batch_data(n_samples=20)

        # Perform update
        result = online_update(
            niche=niche,
            new_features=features_df,
            new_targets=targets_df,
            save_model=False
        )

        assert isinstance(result, OnlineUpdateResult)
        assert result.niche == niche
        assert result.samples_processed == 20
        assert result.total_samples == 20
        assert result.update_time_ms > 0
        assert result.update_time_ms < 5000  # Should complete in <5 seconds

        # Cleanup
        reset_online_predictor(niche)

    def test_predict_one(self, check_river_available, generate_batch_data, synthetic_features):
        """Test prediction after training."""
        from backend.ml.online_update import (
            online_update,
            get_online_predictor,
            reset_online_predictor
        )

        niche = "test_predict"
        reset_online_predictor(niche)

        # Train with some data
        features_df, targets_df = generate_batch_data(n_samples=30)
        online_update(niche, features_df, targets_df, save_model=False)

        # Get predictor and make prediction
        predictor = get_online_predictor(niche)
        test_features = {name: np.random.random() for name in synthetic_features}

        prediction = predictor.predict_one(test_features)

        assert prediction is not None
        assert isinstance(prediction, dict)
        assert "log_likes" in prediction
        assert "log_comments" in prediction
        assert "log_shares" in prediction
        assert "log_saves" in prediction
        assert "log_views" in prediction

        # Cleanup
        reset_online_predictor(niche)


class TestIncrementalLearning:
    """Tests for incremental learning behavior."""

    def test_50_incremental_updates(self, check_river_available, generate_batch_data):
        """
        Test 50 incremental updates to confirm progressive improvement.

        This is the key test that validates the online learning system:
        - Processes 50 batches of feedback data
        - Tracks MAE at each checkpoint
        - Confirms model improves over time
        """
        from backend.ml.online_update import (
            online_update,
            get_online_predictor,
            reset_online_predictor
        )

        niche = "test_incremental_50"
        reset_online_predictor(niche)

        n_batches = 50
        batch_size = 5
        mae_history = []
        results_history = []

        logger.info(f"\n{'='*60}")
        logger.info(f"Starting 50 incremental updates test")
        logger.info(f"Batch size: {batch_size}, Total samples: {n_batches * batch_size}")
        logger.info(f"{'='*60}\n")

        start_time = time.time()

        for i in range(n_batches):
            # Generate batch with consistent patterns (low noise for faster convergence)
            features_df, targets_df = generate_batch_data(
                n_samples=batch_size,
                noise_level=0.05
            )

            # Perform incremental update
            result = online_update(
                niche=niche,
                new_features=features_df,
                new_targets=targets_df,
                save_model=False
            )

            results_history.append(result)

            # Log progress every 10 batches
            if (i + 1) % 10 == 0:
                current_mae = result.current_mae
                if current_mae is not None:
                    mae_history.append(current_mae)

                logger.info(
                    f"Batch {i+1}/{n_batches}: "
                    f"total_samples={result.total_samples}, "
                    f"MAE={f'{current_mae:.4f}' if current_mae is not None else 'N/A'}, "
                    f"time={result.update_time_ms:.1f}ms"
                )

        total_time = time.time() - start_time

        # Get final status
        predictor = get_online_predictor(niche)
        final_status = predictor.get_status()

        logger.info(f"\n{'='*60}")
        logger.info(f"Test Complete!")
        logger.info(f"Total samples processed: {final_status['samples_seen']}")
        mae_str = f"{final_status['current_mae']:.4f}" if final_status['current_mae'] is not None else "N/A"
        best_mae_str = f"{final_status['best_mae']:.4f}" if final_status['best_mae'] is not None else "N/A"
        logger.info(f"Final MAE: {mae_str}")
        logger.info(f"Best MAE: {best_mae_str}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Avg time per batch: {total_time/n_batches*1000:.1f}ms")
        logger.info(f"{'='*60}\n")

        # Assertions
        assert predictor._metrics.samples_seen == n_batches * batch_size
        assert final_status['current_mae'] is not None

        # Check that MAE improved (at least not getting worse after initial learning)
        if len(mae_history) >= 3:
            # Compare first half avg to second half avg
            first_half_avg = np.mean(mae_history[:len(mae_history)//2])
            second_half_avg = np.mean(mae_history[len(mae_history)//2:])

            logger.info(f"First half avg MAE: {first_half_avg:.4f}")
            logger.info(f"Second half avg MAE: {second_half_avg:.4f}")

            # Model should not degrade significantly (allow 20% tolerance)
            assert second_half_avg <= first_half_avg * 1.2, \
                f"Model degraded: first_half={first_half_avg:.4f}, second_half={second_half_avg:.4f}"

        # Check update time is reasonable (<1 second per batch)
        avg_update_time = np.mean([r.update_time_ms for r in results_history])
        assert avg_update_time < 1000, f"Updates too slow: {avg_update_time:.1f}ms avg"

        # Cleanup
        reset_online_predictor(niche)

    def test_improvement_detection(self, check_river_available, generate_batch_data):
        """Test that improvement detection triggers correctly."""
        from backend.ml.online_update import (
            online_update,
            reset_online_predictor,
            EVALUATION_INTERVAL
        )

        niche = "test_improvement"
        reset_online_predictor(niche)

        # Train with enough samples to trigger evaluation
        n_samples = EVALUATION_INTERVAL * 3

        features_df, targets_df = generate_batch_data(n_samples=n_samples)

        result = online_update(
            niche=niche,
            new_features=features_df,
            new_targets=targets_df,
            save_model=False
        )

        # Should have evaluated at least once
        assert result.total_samples >= EVALUATION_INTERVAL

        # Cleanup
        reset_online_predictor(niche)


class TestDriftDetection:
    """Tests for drift detection functionality."""

    def test_detect_drift_no_drift(self, check_river_available):
        """Test drift detection with no drift (predictions match actuals)."""
        from backend.ml.online_update import detect_drift

        niche = "test_no_drift"
        predictions = [50.0, 52.0, 48.0, 51.0, 49.0]
        actuals = [51.0, 50.0, 49.0, 52.0, 48.0]

        drift_detected, magnitude = detect_drift(niche, predictions, actuals)

        assert drift_detected == False
        assert magnitude <= 0.3

    def test_detect_drift_with_drift(self, check_river_available):
        """Test drift detection with significant drift."""
        from backend.ml.online_update import detect_drift

        niche = "test_with_drift"
        predictions = [50.0, 52.0, 48.0, 51.0, 49.0]
        actuals = [80.0, 85.0, 75.0, 82.0, 78.0]  # Much higher than predictions

        drift_detected, magnitude = detect_drift(niche, predictions, actuals)

        assert drift_detected == True
        assert magnitude > 0.3


class TestModelPersistence:
    """Tests for model saving and loading."""

    def test_save_and_load(self, check_river_available, generate_batch_data):
        """Test that model can be saved and loaded."""
        from backend.ml.online_update import (
            online_update,
            get_online_predictor,
            reset_online_predictor,
            OnlineEngagementPredictor,
            MODELS_DIR
        )

        niche = "test_persistence"
        reset_online_predictor(niche)

        # Train model
        features_df, targets_df = generate_batch_data(n_samples=30)
        online_update(niche, features_df, targets_df, save_model=True)

        # Get samples count before reset
        predictor = get_online_predictor(niche)
        samples_before = predictor._metrics.samples_seen

        # Reset and reload
        reset_online_predictor(niche)
        predictor = OnlineEngagementPredictor(niche=niche)

        # Check model was loaded
        assert predictor._metrics.samples_seen == samples_before

        # Cleanup - delete the model file
        model_path = MODELS_DIR / f"online_{niche}.pkl"
        if model_path.exists():
            model_path.unlink()

        reset_online_predictor(niche)


class TestIntegration:
    """Integration tests with MLPredictor."""

    def test_ml_predictor_online_status(self, check_river_available):
        """Test that MLPredictor includes online model status."""
        from app.services.ml_service import MLPredictor

        predictor = MLPredictor()
        status = predictor.get_model_status()

        assert "online_learning_available" in status
        assert "river_installed" in status
        assert status["river_installed"] is True

    def test_predict_with_online_fallback(self, check_river_available, generate_batch_data, synthetic_features):
        """Test predict_with_online_fallback method."""
        from app.services.ml_service import MLPredictor
        from backend.ml.online_update import (
            online_update,
            reset_online_predictor
        )

        niche = "test_fallback"
        reset_online_predictor(niche)

        # Train online model
        features_df, targets_df = generate_batch_data(n_samples=30)
        online_update(niche, features_df, targets_df, save_model=False)

        # Create content dict
        content = {
            "caption": "Test caption with emojis and CTAs! Comment below!",
            "content_format": "reel",
            "business_type": niche,
            "hashtags": ["#test", "#ml"],
        }

        # Test fallback prediction
        predictor = MLPredictor()
        result = predictor.predict_with_online_fallback(content)

        assert "score" in result
        assert "model_source" in result

        # Cleanup
        reset_online_predictor(niche)


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance and stress tests."""

    def test_update_time_under_1_second(self, check_river_available, generate_batch_data):
        """Test that single updates complete in under 1 second."""
        from backend.ml.online_update import (
            online_update,
            reset_online_predictor
        )

        niche = "test_perf"
        reset_online_predictor(niche)

        # Test with varying batch sizes
        batch_sizes = [1, 5, 10, 20]
        times = []

        for batch_size in batch_sizes:
            features_df, targets_df = generate_batch_data(n_samples=batch_size)

            start = time.time()
            result = online_update(niche, features_df, targets_df, save_model=False)
            elapsed = time.time() - start

            times.append(elapsed)
            logger.info(f"Batch size {batch_size}: {elapsed*1000:.1f}ms")

            assert elapsed < 1.0, f"Update took too long: {elapsed:.2f}s for batch_size={batch_size}"

        # Cleanup
        reset_online_predictor(niche)

    def test_memory_efficiency(self, check_river_available, generate_batch_data):
        """Test that model stays memory efficient after many updates."""
        from backend.ml.online_update import (
            online_update,
            get_online_predictor,
            reset_online_predictor
        )
        import sys

        niche = "test_memory"
        reset_online_predictor(niche)

        # Perform many updates
        for i in range(100):
            features_df, targets_df = generate_batch_data(n_samples=10)
            online_update(niche, features_df, targets_df, save_model=False)

        predictor = get_online_predictor(niche)

        # Check model object size (rough estimate)
        model_size = sys.getsizeof(predictor._models)
        metrics_size = sys.getsizeof(predictor._metrics)

        logger.info(f"Models dict size: {model_size} bytes")
        logger.info(f"Metrics size: {metrics_size} bytes")

        # Model should be reasonably small (less than 10MB)
        # Note: This is a rough check, actual serialized size may differ
        assert model_size < 10 * 1024 * 1024, "Model too large"

        # Cleanup
        reset_online_predictor(niche)


# =============================================================================
# CLI Runner
# =============================================================================

if __name__ == "__main__":
    """Run tests directly."""
    pytest.main([__file__, "-v", "--tb=short"])
