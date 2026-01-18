#!/usr/bin/env python3
"""
Tests for ML Model Evaluation with Granular Metrics and Drift Detection
========================================================================

Verifies:
1. Global metrics computation (MAE, R2, RMSE per target)
2. Granular evaluation by format (Reel, Image, etc.)
3. Time-based segmentation (hour, day)
4. RPI (Relative Performance Index) calculation
5. Drift detection with KS test and MAE baseline
6. Calibration analysis
7. Insight generation
8. Result storage and retrieval

Author: BrandPulse AI
"""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch
import sys
import os

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.ml.evaluate_model import (
    # Core functions
    evaluate,
    compute_mae,
    compute_rmse,
    compute_r2,
    compute_mape,
    compute_weighted_rpi,
    # Granular evaluation
    evaluate_global_metrics,
    evaluate_by_format,
    evaluate_by_time,
    evaluate_rpi,
    # Drift detection
    compute_drift_score,
    detect_drift_ks_test,
    detect_drift_mae,
    # Calibration
    analyze_calibration,
    # Insights
    generate_insights,
    # Storage
    save_evaluation_results,
    load_evaluation_history,
    get_baseline_mae,
    # Online evaluation
    should_evaluate_online,
    reset_online_counter,
    # Data classes
    EvaluationResult,
    DriftResult,
    CalibrationResult,
    MetricResult,
    # Constants
    TARGET_COLUMNS,
    DEFAULT_RPI_WEIGHTS,
    DRIFT_ALERT_THRESHOLD,
    ONLINE_EVAL_INTERVAL,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_data():
    """Generate sample test data with subgroups."""
    np.random.seed(42)
    n_samples = 200
    n_targets = 5

    # Ground truth
    y_test = np.random.uniform(2, 8, (n_samples, n_targets))

    # Predictions with controlled error
    y_pred = y_test + np.random.randn(n_samples, n_targets) * 0.3

    # Metadata with subgroups
    metadata = pd.DataFrame({
        "content_format": np.random.choice(["reel", "carousel", "static_image"], n_samples, p=[0.5, 0.3, 0.2]),
        "hour_of_day": np.random.randint(0, 24, n_samples),
        "day_of_week": np.random.randint(0, 7, n_samples),
        "is_reel": [1 if f == "reel" else 0 for f in np.random.choice(["reel", "other"], n_samples)],
        "is_carousel": [1 if f == "carousel" else 0 for f in np.random.choice(["carousel", "other"], n_samples)],
    })

    return y_test, y_pred, metadata


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    class MockModel:
        def __init__(self, predictions):
            self._predictions = predictions

        def predict(self, X):
            n = len(X)
            return self._predictions[:n]

    return MockModel


@pytest.fixture
def sample_drift_data():
    """Generate data with drift for testing."""
    np.random.seed(42)
    n_samples = 100

    # Baseline predictions (historical)
    baseline = np.random.uniform(3, 7, n_samples)

    # Current predictions (with drift - shifted distribution)
    current_no_drift = baseline + np.random.randn(n_samples) * 0.2
    current_with_drift = baseline + 1.5 + np.random.randn(n_samples) * 0.5  # Shifted mean

    # Actuals
    actuals = baseline + np.random.randn(n_samples) * 0.1

    return {
        "baseline": baseline,
        "current_no_drift": current_no_drift,
        "current_with_drift": current_with_drift,
        "actuals": actuals,
    }


# =============================================================================
# Test: Basic Metrics
# =============================================================================

class TestBasicMetrics:
    """Test basic metric computation functions."""

    def test_compute_mae(self):
        """Test MAE computation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 5.3])

        mae = compute_mae(y_true, y_pred)

        assert mae > 0
        assert mae < 0.5  # Should be small for this data
        assert isinstance(mae, float)

    def test_compute_rmse(self):
        """Test RMSE computation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 5.3])

        rmse = compute_rmse(y_true, y_pred)

        assert rmse > 0
        assert rmse >= compute_mae(y_true, y_pred)  # RMSE >= MAE always
        assert isinstance(rmse, float)

    def test_compute_r2(self):
        """Test R-squared computation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.0])

        r2 = compute_r2(y_true, y_pred)

        assert r2 > 0.9  # Good predictions should have high R2
        assert r2 <= 1.0
        assert isinstance(r2, float)

    def test_compute_r2_poor_predictions(self):
        """Test R2 with poor predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([5, 4, 3, 2, 1])  # Completely wrong

        r2 = compute_r2(y_true, y_pred)

        assert r2 < 0  # Negative R2 for very poor predictions

    def test_compute_mape(self):
        """Test MAPE computation."""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 390, 510])

        mape = compute_mape(y_true, y_pred)

        assert mape > 0
        assert mape < 10  # Should be ~5% error
        assert isinstance(mape, float)


# =============================================================================
# Test: Global Metrics (Multi-output)
# =============================================================================

class TestGlobalMetrics:
    """Test global metrics computation for multi-output."""

    def test_evaluate_global_metrics(self, sample_data):
        """Test global metrics evaluation."""
        y_test, y_pred, _ = sample_data

        metrics = evaluate_global_metrics(y_test, y_pred, TARGET_COLUMNS)

        # Check structure
        assert "_aggregate" in metrics
        assert "log_likes" in metrics
        assert "log_comments" in metrics

        # Check metric keys
        for target, target_metrics in metrics.items():
            if target == "_aggregate":
                assert "mae" in target_metrics
                assert "rmse" in target_metrics
                assert "r2" in target_metrics
            else:
                assert "mae" in target_metrics
                assert "rmse" in target_metrics
                assert "r2" in target_metrics
                assert "mape" in target_metrics

        # Check values are reasonable
        assert metrics["_aggregate"]["mae"] < 1.0
        assert metrics["_aggregate"]["r2"] > 0.5

    def test_evaluate_global_metrics_1d_input(self):
        """Test with 1D input arrays."""
        y_test = np.random.uniform(2, 8, 100)
        y_pred = y_test + np.random.randn(100) * 0.3

        metrics = evaluate_global_metrics(y_test, y_pred, ["single_target"])

        assert "single_target" in metrics
        assert "_aggregate" in metrics

    def test_evaluate_global_metrics_dataframe_input(self, sample_data):
        """Test with DataFrame input."""
        y_test, y_pred, _ = sample_data

        y_test_df = pd.DataFrame(y_test, columns=TARGET_COLUMNS)
        y_pred_df = pd.DataFrame(y_pred, columns=TARGET_COLUMNS)

        metrics = evaluate_global_metrics(y_test_df, y_pred_df, TARGET_COLUMNS)

        assert "_aggregate" in metrics
        assert metrics["_aggregate"]["mae"] < 1.0


# =============================================================================
# Test: Granular Evaluation by Format
# =============================================================================

class TestFormatEvaluation:
    """Test evaluation segmented by content format."""

    def test_evaluate_by_format(self, sample_data):
        """Test format-segmented evaluation."""
        y_test, y_pred, metadata = sample_data

        metrics = evaluate_by_format(y_test, y_pred, metadata)

        # Check that formats are present
        assert len(metrics) > 0

        # Check structure for each format
        for fmt, fmt_metrics in metrics.items():
            if not fmt.startswith("_"):
                assert "mae" in fmt_metrics
                assert "n_samples" in fmt_metrics
                assert fmt_metrics["n_samples"] >= 5  # Minimum threshold

    def test_evaluate_by_format_with_flags(self, sample_data):
        """Test format evaluation using is_reel/is_carousel flags."""
        y_test, y_pred, _ = sample_data
        n = len(y_test)

        # Create metadata with only flags (no content_format)
        metadata = pd.DataFrame({
            "is_reel": np.random.randint(0, 2, n),
            "is_carousel": np.random.randint(0, 2, n),
        })

        metrics = evaluate_by_format(y_test, y_pred, metadata)

        # Should still produce format metrics
        assert len(metrics) > 0

    def test_reel_vs_static_difference(self, sample_data):
        """Verify metrics can differ between formats."""
        y_test, _, metadata = sample_data

        # Create predictions with different error levels for different formats
        y_pred = y_test.copy()
        reel_mask = metadata["content_format"] == "reel"
        y_pred[reel_mask] += np.random.randn(reel_mask.sum(), y_test.shape[1]) * 0.5  # More error for reels

        metrics = evaluate_by_format(y_test, y_pred, metadata)

        # Reels should have higher MAE
        if "reel" in metrics and "static_image" in metrics:
            # This test verifies segmentation works, actual difference may vary
            assert metrics["reel"]["mae"] != metrics["static_image"]["mae"]


# =============================================================================
# Test: Time-based Evaluation
# =============================================================================

class TestTimeEvaluation:
    """Test evaluation segmented by time."""

    def test_evaluate_by_time(self, sample_data):
        """Test time-segmented evaluation."""
        y_test, y_pred, metadata = sample_data

        metrics = evaluate_by_time(y_test, y_pred, metadata)

        # Check hour segments
        hour_segments = [k for k in metrics.keys() if k.startswith("hour_")]
        assert len(hour_segments) > 0

        # Check day segments
        day_segments = [k for k in metrics.keys() if k.startswith("day_")]
        assert len(day_segments) > 0

    def test_prime_time_evaluation(self, sample_data):
        """Test prime time segment (18-21h weekdays)."""
        y_test, y_pred, _ = sample_data
        n = len(y_test)

        # Create metadata with prime time samples
        metadata = pd.DataFrame({
            "hour_of_day": np.random.choice([19, 20, 21], n),  # All prime time hours
            "day_of_week": np.random.choice([0, 1, 2, 3, 4], n),  # All weekdays
        })

        metrics = evaluate_by_time(y_test, y_pred, metadata)

        assert "prime_time" in metrics
        assert "n_samples" in metrics["prime_time"]

    def test_weekend_vs_weekday(self, sample_data):
        """Test weekend vs weekday segmentation."""
        y_test, y_pred, _ = sample_data
        n = len(y_test)

        metadata = pd.DataFrame({
            "hour_of_day": np.random.randint(0, 24, n),
            "day_of_week": np.random.choice([0, 1, 2, 3, 4, 5, 6], n),  # All days
        })

        metrics = evaluate_by_time(y_test, y_pred, metadata)

        # Should have both weekday and weekend
        assert "day_weekday" in metrics or "day_weekend" in metrics


# =============================================================================
# Test: RPI Evaluation
# =============================================================================

class TestRPIEvaluation:
    """Test Relative Performance Index evaluation."""

    def test_compute_weighted_rpi(self):
        """Test weighted RPI computation."""
        predictions = {
            "log_likes": np.array([5.0, 6.0, 7.0]),
            "log_comments": np.array([2.0, 3.0, 4.0]),
            "log_shares": np.array([1.0, 2.0, 3.0]),
        }

        rpi = compute_weighted_rpi(predictions, DEFAULT_RPI_WEIGHTS)

        assert len(rpi) == 3
        assert np.all(rpi >= 0)
        assert np.all(rpi <= 100)

    def test_evaluate_rpi(self, sample_data):
        """Test RPI metrics evaluation."""
        y_test, y_pred, _ = sample_data

        metrics = evaluate_rpi(y_test, y_pred, TARGET_COLUMNS)

        assert "rpi_mae" in metrics
        assert "rpi_correlation" in metrics
        assert "rpi_rmse" in metrics

        # Correlation should be positive for correlated data
        assert metrics["rpi_correlation"] > 0

    def test_rpi_with_custom_weights(self, sample_data):
        """Test RPI with custom weights."""
        y_test, y_pred, _ = sample_data

        custom_weights = {
            "log_likes": 0.5,
            "log_comments": 5.0,  # Prioritize comments
            "log_shares": 1.0,
            "log_saves": 1.0,
            "log_views": 0.1,
        }

        metrics = evaluate_rpi(y_test, y_pred, TARGET_COLUMNS, custom_weights)

        assert "rpi_mae" in metrics
        # Results should be different from default weights


# =============================================================================
# Test: Drift Detection
# =============================================================================

class TestDriftDetection:
    """Test drift detection functionality."""

    def test_detect_drift_mae_no_drift(self, sample_drift_data):
        """Test MAE-based drift detection with no drift."""
        baseline_mae = 0.2
        current_mae = 0.22  # Slight increase

        detected, relative_increase = detect_drift_mae(current_mae, baseline_mae)

        assert not detected  # 10% increase is within tolerance
        assert relative_increase < 0.2

    def test_detect_drift_mae_with_drift(self, sample_drift_data):
        """Test MAE-based drift detection with drift."""
        baseline_mae = 0.2
        current_mae = 0.35  # 75% increase

        detected, relative_increase = detect_drift_mae(current_mae, baseline_mae)

        assert detected
        assert relative_increase > 0.5

    def test_detect_drift_ks_test(self, sample_drift_data):
        """Test KS test for distribution drift."""
        baseline = sample_drift_data["baseline"]
        current_with_drift = sample_drift_data["current_with_drift"]

        ks_stat, p_value = detect_drift_ks_test(current_with_drift, baseline)

        if ks_stat is not None:  # scipy available
            assert ks_stat > 0.3  # Significant drift
            assert p_value < 0.05  # Statistically significant

    def test_compute_drift_score_no_drift(self, sample_drift_data):
        """Test drift score computation without drift."""
        actuals = sample_drift_data["actuals"]
        predictions = sample_drift_data["current_no_drift"]

        result = compute_drift_score(
            y_true=actuals,
            y_pred=predictions,
            baseline_mae=0.25
        )

        assert isinstance(result, DriftResult)
        assert result.drift_score < DRIFT_ALERT_THRESHOLD
        assert not result.drift_detected

    def test_compute_drift_score_with_drift(self, sample_drift_data):
        """Test drift score computation with drift."""
        actuals = sample_drift_data["actuals"]
        predictions = sample_drift_data["current_with_drift"]

        result = compute_drift_score(
            y_true=actuals,
            y_pred=predictions,
            baseline_predictions=sample_drift_data["baseline"],
            baseline_mae=0.15  # Low baseline MAE
        )

        assert isinstance(result, DriftResult)
        assert result.drift_score > 0.3
        # Drift should be detected with both KS and MAE signals
        assert result.current_mae > result.baseline_mae

    def test_drift_alert_message(self, sample_drift_data):
        """Test drift alert message generation."""
        actuals = sample_drift_data["actuals"]
        predictions = sample_drift_data["current_with_drift"]

        result = compute_drift_score(
            y_true=actuals,
            y_pred=predictions,
            baseline_mae=0.1  # Very low baseline
        )

        if result.drift_detected:
            assert result.alert_message is not None
            assert len(result.alert_message) > 0


# =============================================================================
# Test: Calibration Analysis
# =============================================================================

class TestCalibration:
    """Test calibration analysis."""

    def test_analyze_calibration_well_calibrated(self):
        """Test calibration with well-calibrated predictions."""
        np.random.seed(42)
        y_true = np.random.uniform(0, 100, 200)
        y_pred = y_true + np.random.randn(200) * 5  # Small noise

        result = analyze_calibration(y_true, y_pred)

        assert isinstance(result, CalibrationResult)
        assert result.calibration_error < 15  # Reasonable error
        assert len(result.expected_percentiles) > 0
        assert len(result.observed_percentiles) > 0

    def test_analyze_calibration_poorly_calibrated(self):
        """Test calibration with poorly calibrated predictions."""
        np.random.seed(42)
        y_true = np.random.uniform(0, 100, 200)
        # Systematically biased predictions
        y_pred = y_true * 1.5 + 20  # Overestimate

        result = analyze_calibration(y_true, y_pred)

        # Should detect poor calibration
        assert result.calibration_error > 5  # Higher error

    def test_calibration_percentiles(self):
        """Test that percentiles are correctly computed."""
        np.random.seed(42)
        y_true = np.random.uniform(0, 100, 500)
        y_pred = y_true + np.random.randn(500) * 2

        result = analyze_calibration(y_true, y_pred, n_bins=10)

        # Should have ~10 bins
        assert len(result.expected_percentiles) <= 10
        # Percentiles should be in valid range
        assert all(0 <= p <= 100 for p in result.expected_percentiles)
        assert all(0 <= p <= 100 for p in result.observed_percentiles)


# =============================================================================
# Test: Insight Generation
# =============================================================================

class TestInsightGeneration:
    """Test insight generation."""

    def test_generate_insights_stable_model(self):
        """Test insights for a stable model."""
        global_metrics = {
            "_aggregate": {"mae": 0.2, "r2": 0.9, "rmse": 0.25},
            "log_likes": {"mae": 0.15, "r2": 0.92, "rmse": 0.2},
            "log_comments": {"mae": 0.25, "r2": 0.85, "rmse": 0.3},
        }
        format_metrics = {
            "reel": {"mae": 0.2, "n_samples": 50},
            "static_image": {"mae": 0.22, "n_samples": 30},
        }
        time_metrics = {
            "prime_time": {"mae": 0.3, "n_samples": 20},
        }
        drift = DriftResult(drift_detected=False, drift_score=0.1)

        insights = generate_insights(global_metrics, format_metrics, time_metrics, drift)

        assert len(insights) > 0
        # Should indicate stability
        assert any("estable" in i.lower() for i in insights)

    def test_generate_insights_with_drift(self):
        """Test insights when drift is detected."""
        global_metrics = {
            "_aggregate": {"mae": 0.5, "r2": 0.7, "rmse": 0.6},
        }
        format_metrics = {}
        time_metrics = {}
        drift = DriftResult(
            drift_detected=True,
            drift_score=0.45,
            alert_message="MAE degraded 50% vs baseline"
        )

        insights = generate_insights(global_metrics, format_metrics, time_metrics, drift)

        # Should contain alert
        assert any("ALERTA" in i or "Drift" in i for i in insights)

    def test_generate_insights_high_mae_target(self):
        """Test insights for high MAE on specific target."""
        global_metrics = {
            "_aggregate": {"mae": 0.4, "r2": 0.8, "rmse": 0.45},
            "log_likes": {"mae": 0.2, "r2": 0.9, "rmse": 0.25},
            "log_comments": {"mae": 0.7, "r2": 0.6, "rmse": 0.8},  # High MAE
        }
        format_metrics = {}
        time_metrics = {}
        drift = DriftResult(drift_detected=False, drift_score=0.1)

        insights = generate_insights(global_metrics, format_metrics, time_metrics, drift)

        # Should mention comments issue
        assert any("comments" in i.lower() or "mae" in i.lower() for i in insights)

    def test_generate_insights_format_issues(self):
        """Test insights for format-specific issues."""
        global_metrics = {"_aggregate": {"mae": 0.3, "r2": 0.85}}
        format_metrics = {
            "reel": {"mae": 0.8, "n_samples": 50},  # High MAE for reels
            "static_image": {"mae": 0.2, "n_samples": 30},
        }
        time_metrics = {}
        drift = DriftResult(drift_detected=False, drift_score=0.1)

        insights = generate_insights(global_metrics, format_metrics, time_metrics, drift)

        # Should mention reel issues
        assert any("reel" in i.lower() for i in insights)


# =============================================================================
# Test: Online Evaluation Counter
# =============================================================================

class TestOnlineEvaluation:
    """Test online evaluation trigger mechanism."""

    def test_should_evaluate_online_initial(self):
        """Test online evaluation counter from start."""
        reset_online_counter("test_niche")

        # Add samples one by one
        for i in range(ONLINE_EVAL_INTERVAL - 1):
            result = should_evaluate_online("test_niche", 1)
            assert not result  # Should not trigger yet

        # This one should trigger
        result = should_evaluate_online("test_niche", 1)
        assert result

    def test_should_evaluate_online_batch(self):
        """Test online evaluation with batch updates."""
        reset_online_counter("batch_niche")

        # Add batch of samples
        result = should_evaluate_online("batch_niche", ONLINE_EVAL_INTERVAL)
        assert result  # Should trigger immediately

    def test_reset_online_counter(self):
        """Test counter reset."""
        should_evaluate_online("reset_niche", 30)  # Add some samples
        reset_online_counter("reset_niche")

        # After reset, need full interval again
        for i in range(ONLINE_EVAL_INTERVAL - 1):
            result = should_evaluate_online("reset_niche", 1)
            assert not result


# =============================================================================
# Test: Full Evaluation Function
# =============================================================================

class TestFullEvaluation:
    """Test the main evaluate() function."""

    def test_evaluate_basic(self, sample_data, mock_model):
        """Test basic evaluation."""
        y_test, y_pred, metadata = sample_data
        X_test = np.random.randn(len(y_test), 50)  # Dummy features

        model = mock_model(y_pred)

        result = evaluate(
            niche="test_niche",
            model=model,
            X_test=X_test,
            y_test=y_test,
            metadata=metadata,
            save_results=False
        )

        assert isinstance(result, EvaluationResult)
        assert result.niche == "test_niche"
        assert result.n_samples == len(y_test)
        assert "_aggregate" in result.global_metrics
        assert len(result.insights) > 0

    def test_evaluate_with_drift_baseline(self, sample_data, mock_model):
        """Test evaluation with drift baseline."""
        y_test, y_pred, metadata = sample_data
        X_test = np.random.randn(len(y_test), 50)

        model = mock_model(y_pred)

        result = evaluate(
            niche="drift_niche",
            model=model,
            X_test=X_test,
            y_test=y_test,
            baseline_mae=0.1,  # Low baseline
            save_results=False
        )

        assert result.drift is not None
        assert result.drift.baseline_mae == 0.1

    def test_evaluate_returns_dict(self, sample_data, mock_model):
        """Test that to_dict() works."""
        y_test, y_pred, metadata = sample_data
        X_test = np.random.randn(len(y_test), 50)

        model = mock_model(y_pred)

        result = evaluate(
            niche="dict_niche",
            model=model,
            X_test=X_test,
            y_test=y_test,
            save_results=False
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "niche" in result_dict
        assert "global_metrics" in result_dict
        assert "drift" in result_dict
        assert "insights" in result_dict


# =============================================================================
# Test: Storage Functions
# =============================================================================

class TestStorage:
    """Test result storage functions."""

    def test_save_and_load_evaluation(self, sample_data, mock_model, tmp_path):
        """Test saving and loading evaluation results."""
        y_test, y_pred, metadata = sample_data
        X_test = np.random.randn(len(y_test), 50)

        model = mock_model(y_pred)

        # Patch LOGS_DIR to use temp directory
        with patch('backend.ml.evaluate_model.LOGS_DIR', tmp_path):
            result = evaluate(
                niche="storage_test",
                model=model,
                X_test=X_test,
                y_test=y_test,
                save_results=True
            )

            # Verify file was created
            assert (tmp_path / "eval_storage_test.json").exists()

            # Load history
            history = load_evaluation_history("storage_test")
            assert len(history) == 1
            assert history[0]["niche"] == "storage_test"

    def test_get_baseline_mae(self, tmp_path):
        """Test baseline MAE retrieval."""
        # Create fake history
        history_data = {
            "niche": "baseline_test",
            "history": [
                {"global_metrics": {"_aggregate": {"mae": 0.3}}},
                {"global_metrics": {"_aggregate": {"mae": 0.25}}},
                {"global_metrics": {"_aggregate": {"mae": 0.28}}},
            ]
        }

        with patch('backend.ml.evaluate_model.LOGS_DIR', tmp_path):
            with open(tmp_path / "eval_baseline_test.json", "w") as f:
                json.dump(history_data, f)

            baseline = get_baseline_mae("baseline_test", lookback=3)

            assert baseline is not None
            assert 0.25 <= baseline <= 0.3  # Should be average

    def test_get_baseline_mae_no_history(self, tmp_path):
        """Test baseline MAE when no history exists."""
        with patch('backend.ml.evaluate_model.LOGS_DIR', tmp_path):
            baseline = get_baseline_mae("nonexistent_niche")
            assert baseline is None


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_metadata(self, sample_data, mock_model):
        """Test evaluation with no metadata."""
        y_test, y_pred, _ = sample_data
        X_test = np.random.randn(len(y_test), 50)

        model = mock_model(y_pred)

        result = evaluate(
            niche="no_metadata",
            model=model,
            X_test=X_test,
            y_test=y_test,
            metadata=None,
            save_results=False
        )

        # Should still work with limited granularity
        assert result is not None
        assert "_aggregate" in result.global_metrics

    def test_small_sample_size(self, mock_model):
        """Test evaluation with very few samples."""
        n = 10  # Small sample
        y_test = np.random.uniform(2, 8, (n, 5))
        y_pred = y_test + np.random.randn(n, 5) * 0.3
        X_test = np.random.randn(n, 50)

        model = mock_model(y_pred)

        result = evaluate(
            niche="small_sample",
            model=model,
            X_test=X_test,
            y_test=y_test,
            save_results=False
        )

        # Should handle small samples gracefully
        assert result is not None
        # Calibration might be skipped for small samples
        assert result.calibration is None  # Below 50 samples

    def test_single_target(self, mock_model):
        """Test evaluation with single target."""
        n = 100
        y_test = np.random.uniform(2, 8, n)
        y_pred = y_test + np.random.randn(n) * 0.3
        X_test = np.random.randn(n, 50)

        model = mock_model(y_pred.reshape(-1, 1))

        result = evaluate(
            niche="single_target",
            model=model,
            X_test=X_test,
            y_test=y_test,
            target_columns=["single_metric"],
            save_results=False
        )

        assert result is not None
        assert "single_metric" in result.global_metrics


# =============================================================================
# Test: Integration
# =============================================================================

class TestIntegration:
    """Integration tests for the evaluation pipeline."""

    def test_full_pipeline_simulation(self, sample_data, mock_model):
        """Simulate full evaluation pipeline."""
        y_test, y_pred, metadata = sample_data
        X_test = np.random.randn(len(y_test), 50)

        model = mock_model(y_pred)

        # Run evaluation
        result = evaluate(
            niche="integration_test",
            model=model,
            X_test=X_test,
            y_test=y_test,
            metadata=metadata,
            baseline_mae=0.25,
            evaluation_type="full",
            save_results=False
        )

        # Verify all components
        assert result.niche == "integration_test"
        assert result.evaluation_type == "full"
        assert result.n_samples == len(y_test)

        # Global metrics
        assert "_aggregate" in result.global_metrics
        agg = result.global_metrics["_aggregate"]
        assert all(k in agg for k in ["mae", "rmse", "r2"])

        # Granular metrics
        assert len(result.format_metrics) > 0
        assert len(result.time_metrics) >= 0

        # Drift
        assert result.drift is not None
        assert isinstance(result.drift.drift_score, float)
        assert 0 <= result.drift.drift_score <= 1

        # Calibration (for enough samples)
        assert result.calibration is not None

        # Insights
        assert len(result.insights) > 0

        # Serialization
        result_dict = result.to_dict()
        assert json.dumps(result_dict)  # Should be JSON serializable


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
