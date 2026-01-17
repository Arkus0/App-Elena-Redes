"""
Unit Tests for MetricNormalizer - Relative Performance Index (RPI)

These tests verify that:
1. Weighted engagement is calculated correctly
2. Rolling baseline uses median of last N posts (excluding current)
3. RPI = current / baseline with log transformation
4. Cold start fallback uses niche or global baseline
5. DataFrame normalization works end-to-end
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Direct import of normalization module without loading full services package
import importlib.util

backend_path = Path(__file__).parent.parent
normalization_path = backend_path / "app" / "services" / "normalization.py"

spec = importlib.util.spec_from_file_location("normalization", normalization_path)
_normalization = importlib.util.module_from_spec(spec)
sys.modules["normalization"] = _normalization
spec.loader.exec_module(_normalization)

# Import classes/functions for easier access in tests
MetricNormalizer = _normalization.MetricNormalizer
NormalizationConfig = _normalization.NormalizationConfig
NormalizationStrategy = _normalization.NormalizationStrategy
NicheBaseline = _normalization.NicheBaseline
normalize_for_training = _normalization.normalize_for_training
create_rpi_from_raw_metrics = _normalization.create_rpi_from_raw_metrics


class TestWeightedEngagement:
    """Tests for weighted engagement calculation"""

    def test_basic_weighted_engagement(self):
        """Verify weighted engagement formula: likes*1 + comments*3 + saves*5 + shares*4"""
        # Using module-level imports

        normalizer = MetricNormalizer()

        post = {
            "likes_count": 100,      # 100 * 1 = 100
            "comments_count": 10,    # 10 * 3 = 30
            "saves_count": 5,        # 5 * 5 = 25
            "shares_count": 2,       # 2 * 4 = 8
        }
        # Total: 100 + 30 + 25 + 8 = 163

        engagement = normalizer.calculate_weighted_engagement(post)
        assert engagement == 163.0, f"Expected 163, got {engagement}"

    def test_weighted_engagement_with_views(self):
        """Verify views have low weight (0.1)"""
        # Using module-level imports

        normalizer = MetricNormalizer()

        post = {
            "likes": 0,
            "comments": 0,
            "saves": 0,
            "shares": 0,
            "views": 10000,  # 10000 * 0.1 = 1000
        }

        engagement = normalizer.calculate_weighted_engagement(post)
        assert engagement == 1000.0, f"Expected 1000, got {engagement}"

    def test_weighted_engagement_alternative_field_names(self):
        """Verify support for alternative field names (likeCount, etc.)"""
        # Using module-level imports

        normalizer = MetricNormalizer()

        post = {
            "likeCount": 500,     # likes * 1 = 500
            "commentCount": 50,   # comments * 3 = 150
        }

        engagement = normalizer.calculate_weighted_engagement(post)
        assert engagement == 650.0, f"Expected 650, got {engagement}"

    def test_weighted_engagement_with_none_values(self):
        """Verify None values are treated as 0"""
        # Using module-level imports

        normalizer = MetricNormalizer()

        post = {
            "likes_count": 100,
            "comments_count": None,
            "saves_count": 10,
        }

        engagement = normalizer.calculate_weighted_engagement(post)
        assert engagement == 150.0, f"Expected 150 (100 + 0 + 50), got {engagement}"


class TestRPICalculation:
    """Tests for Relative Performance Index calculation"""

    def test_rpi_basic_calculation(self):
        """Verify RPI = current / baseline"""
        # Using module-level imports

        config = NormalizationConfig(apply_log_transform=False)
        normalizer = MetricNormalizer(config)

        rpi = normalizer.calculate_rpi(1500, 1000)
        assert rpi == 1.5, f"Expected RPI=1.5, got {rpi}"

    def test_rpi_with_log_transform(self):
        """Verify log1p transformation: log1p(1.5) ≈ 0.916"""
        # Using module-level imports

        normalizer = MetricNormalizer()  # log transform is default

        rpi = normalizer.calculate_rpi(1500, 1000)
        expected = np.log1p(1.5)  # ~0.916

        assert abs(rpi - expected) < 0.001, f"Expected ~{expected:.3f}, got {rpi:.3f}"

    def test_rpi_clipping_high_outliers(self):
        """Verify extreme viral posts are clipped to max (100x)"""
        # Using module-level imports

        config = NormalizationConfig(apply_log_transform=False, clip_rpi_max=100.0)
        normalizer = MetricNormalizer(config)

        # Post with 200x normal engagement
        rpi = normalizer.calculate_rpi(200000, 1000)
        assert rpi == 100.0, f"Expected clipped RPI=100, got {rpi}"

    def test_rpi_clipping_low_outliers(self):
        """Verify very low engagement is clipped to min (0.01)"""
        # Using module-level imports

        config = NormalizationConfig(apply_log_transform=False, clip_rpi_min=0.01)
        normalizer = MetricNormalizer(config)

        # Post with 0.001x normal engagement
        rpi = normalizer.calculate_rpi(1, 1000)
        assert rpi == 0.01, f"Expected clipped RPI=0.01, got {rpi}"

    def test_rpi_zero_baseline_handling(self):
        """Verify division by zero is handled"""
        # Using module-level imports

        config = NormalizationConfig(apply_log_transform=False)
        normalizer = MetricNormalizer(config)

        rpi = normalizer.calculate_rpi(1000, 0)
        # Should use baseline=1.0 as fallback
        assert rpi <= 100.0, f"RPI should be clipped, got {rpi}"


class TestRollingBaseline:
    """Tests for rolling baseline calculation"""

    def test_rolling_baseline_median(self):
        """Verify baseline uses median of last N posts (excluding current)"""
        # Using module-level imports

        config = NormalizationConfig(window_size=5, min_posts_for_baseline=3)
        normalizer = MetricNormalizer(config)

        # Create DataFrame with known pattern
        df = pd.DataFrame({
            "author_id": ["author_1"] * 10,
            "weighted_engagement": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            "posted_at": pd.date_range("2024-01-01", periods=10),
        })

        # For post at index 7 (engagement=800):
        # Last 5 posts are indices 2-6: [300, 400, 500, 600, 700]
        # Median = 500

        df_normalized = normalizer.normalize_dataframe(
            df,
            author_col="author_id",
            niche_col=None,
            date_col="posted_at"
        )

        baseline_at_7 = df_normalized.loc[7, "baseline_engagement"]
        assert baseline_at_7 == 500.0, f"Expected baseline=500, got {baseline_at_7}"

    def test_rolling_baseline_excludes_current_post(self):
        """Verify current post is NOT included in baseline calculation"""
        # Using module-level imports

        config = NormalizationConfig(window_size=20, min_posts_for_baseline=5)
        normalizer = MetricNormalizer(config)

        # Post with huge engagement at the end
        df = pd.DataFrame({
            "author_id": ["author_1"] * 10,
            "weighted_engagement": [100, 100, 100, 100, 100, 100, 100, 100, 100, 10000],
            "posted_at": pd.date_range("2024-01-01", periods=10),
        })

        df_normalized = normalizer.normalize_dataframe(
            df,
            author_col="author_id",
            niche_col=None,
            date_col="posted_at"
        )

        # Baseline for last post should be ~100 (not including 10000)
        baseline_last = df_normalized.loc[9, "baseline_engagement"]
        assert baseline_last == 100.0, f"Expected baseline=100 (excluding current), got {baseline_last}"

    def test_rolling_baseline_window_size(self):
        """Verify only last N posts are used"""
        # Using module-level imports

        config = NormalizationConfig(window_size=3, min_posts_for_baseline=3)
        normalizer = MetricNormalizer(config)

        # Old posts have low engagement, recent posts have high
        df = pd.DataFrame({
            "author_id": ["author_1"] * 10,
            "weighted_engagement": [10, 20, 30, 40, 50, 600, 700, 800, 900, 1000],
            "posted_at": pd.date_range("2024-01-01", periods=10),
        })

        df_normalized = normalizer.normalize_dataframe(
            df,
            author_col="author_id",
            niche_col=None,
            date_col="posted_at"
        )

        # For post at index 9 (engagement=1000):
        # Last 3 posts (excluding current) are indices 6-8: [700, 800, 900]
        # Median = 800

        baseline_last = df_normalized.loc[9, "baseline_engagement"]
        assert baseline_last == 800.0, f"Expected baseline=800, got {baseline_last}"


class TestColdStart:
    """Tests for cold start handling"""

    def test_cold_start_uses_niche_baseline(self):
        """Verify accounts with < min_posts use niche median"""
        # Using module-level imports

        config = NormalizationConfig(min_posts_for_baseline=5)
        normalizer = MetricNormalizer(config)

        # Create DataFrame with established and new authors
        df = pd.DataFrame({
            "author_id": (
                ["established"] * 10 +
                ["new_author"] * 3  # Less than min_posts
            ),
            "niche": ["food"] * 13,
            "weighted_engagement": (
                [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400] +
                [2000, 2100, 2200]  # New author
            ),
            "posted_at": pd.date_range("2024-01-01", periods=13),
        })

        df_normalized = normalizer.normalize_dataframe(
            df,
            author_col="author_id",
            niche_col="niche",
            date_col="posted_at"
        )

        # New author posts should use niche baseline (not personal)
        new_author_mask = df_normalized["author_id"] == "new_author"
        baseline_sources = df_normalized.loc[new_author_mask, "baseline_source"].unique()

        assert "personal" not in baseline_sources, "New author should not have personal baseline"
        assert "niche" in baseline_sources or "global" in baseline_sources

    def test_cold_start_rate_calculated(self):
        """Verify cold start rate is reported correctly"""
        # Using module-level imports

        config = NormalizationConfig(min_posts_for_baseline=5)
        normalizer = MetricNormalizer(config)

        # 50% authors have < min_posts
        df = pd.DataFrame({
            "author_id": (
                ["established"] * 10 +
                ["new_1"] * 2 +
                ["new_2"] * 3
            ),
            "niche": ["food"] * 15,
            "weighted_engagement": [100] * 15,
            "posted_at": pd.date_range("2024-01-01", periods=15),
        })

        df_normalized = normalizer.normalize_dataframe(
            df,
            author_col="author_id",
            niche_col="niche",
            date_col="posted_at"
        )

        stats = normalizer.get_stats(df_normalized)
        cold_start_rate = stats["cold_start_rate"]

        # new_1 (2 posts) + new_2 (3 posts) = 5 posts with cold start
        # established: first 4 posts cold start, last 6 personal
        # Total cold start: 5 + 4 = 9 out of 15
        assert cold_start_rate > 0.5, f"Expected cold start rate > 50%, got {cold_start_rate:.1%}"


class TestNicheBaselines:
    """Tests for niche baseline fitting"""

    def test_fit_niche_baselines(self):
        """Verify niche baselines are calculated correctly"""
        # Using module-level imports

        normalizer = MetricNormalizer()

        df = pd.DataFrame({
            "niche": ["food"] * 100 + ["beauty"] * 100,
            "weighted_engagement": (
                list(np.random.normal(1000, 100, 100)) +  # Food niche
                list(np.random.normal(500, 50, 100))       # Beauty niche
            ),
        })

        baselines = normalizer.fit_niche_baselines(df, niche_col="niche")

        assert "food" in baselines, "Should have food niche baseline"
        assert "beauty" in baselines, "Should have beauty niche baseline"

        # Food has higher median (~1000) than beauty (~500)
        assert baselines["food"].median_engagement > baselines["beauty"].median_engagement

    def test_fallback_hierarchy(self):
        """Verify fallback: niche -> global -> default"""
        # Using module-level imports

        normalizer = MetricNormalizer()

        # Fit with some niches
        df = pd.DataFrame({
            "niche": ["food"] * 50,
            "weighted_engagement": [1000] * 50,
        })
        normalizer.fit_niche_baselines(df, niche_col="niche")

        # Test fallback hierarchy
        food_baseline = normalizer.get_fallback_baseline("food")
        unknown_baseline = normalizer.get_fallback_baseline("unknown_niche")
        none_baseline = normalizer.get_fallback_baseline(None)

        assert food_baseline == 1000.0, f"Food baseline should be 1000, got {food_baseline}"
        assert unknown_baseline == 1000.0, f"Unknown should fall back to global (1000)"
        assert none_baseline == 1000.0, f"None should fall back to global (1000)"


class TestDataFrameNormalization:
    """Tests for full DataFrame normalization"""

    def test_normalize_adds_required_columns(self):
        """Verify all required columns are added"""
        # Using module-level imports

        normalizer = MetricNormalizer()

        df = pd.DataFrame({
            "author_id": ["a"] * 10,
            "likes_count": [100] * 10,
            "comments_count": [10] * 10,
            "posted_at": pd.date_range("2024-01-01", periods=10),
        })

        df_normalized = normalizer.normalize_dataframe(df, niche_col=None)

        required_cols = [
            "weighted_engagement",
            "baseline_engagement",
            "baseline_source",
            "rpi_raw",
            "rpi_score"
        ]

        for col in required_cols:
            assert col in df_normalized.columns, f"Missing column: {col}"

    def test_normalize_preserves_original_data(self):
        """Verify original columns are not modified"""
        # Using module-level imports

        normalizer = MetricNormalizer()

        df = pd.DataFrame({
            "author_id": ["a"] * 5,
            "likes_count": [100, 200, 300, 400, 500],
            "posted_at": pd.date_range("2024-01-01", periods=5),
        })

        df_normalized = normalizer.normalize_dataframe(df, niche_col=None)

        # Original likes should be unchanged
        assert list(df_normalized["likes_count"]) == [100, 200, 300, 400, 500]

    def test_normalize_sorts_by_date(self):
        """Verify DataFrame is sorted chronologically"""
        # Using module-level imports

        normalizer = MetricNormalizer()

        # Create unsorted DataFrame
        df = pd.DataFrame({
            "author_id": ["a"] * 5,
            "likes_count": [500, 100, 300, 200, 400],
            "posted_at": [
                datetime(2024, 1, 5),
                datetime(2024, 1, 1),
                datetime(2024, 1, 3),
                datetime(2024, 1, 2),
                datetime(2024, 1, 4),
            ],
        })

        df_normalized = normalizer.normalize_dataframe(df, niche_col=None)

        # Should be sorted by date
        dates = df_normalized["posted_at"].tolist()
        assert dates == sorted(dates), "DataFrame should be sorted by date"


class TestSinglePostNormalization:
    """Tests for real-time single post normalization"""

    def test_normalize_single_post_with_history(self):
        """Verify single post normalization with sufficient history"""
        # Using module-level imports

        config = NormalizationConfig(
            min_posts_for_baseline=3,
            apply_log_transform=False
        )
        normalizer = MetricNormalizer(config)

        post = {"likes_count": 200, "comments_count": 20}

        history = [
            {"likes_count": 100, "comments_count": 10},  # engagement = 130
            {"likes_count": 100, "comments_count": 10},  # engagement = 130
            {"likes_count": 100, "comments_count": 10},  # engagement = 130
            {"likes_count": 100, "comments_count": 10},  # engagement = 130
            {"likes_count": 100, "comments_count": 10},  # engagement = 130
        ]

        result = normalizer.normalize_single_post(post, history)

        # Current engagement: 200 + 60 = 260
        # Baseline: median of [130, 130, 130, 130, 130] = 130
        # RPI = 260 / 130 = 2.0

        assert result["baseline_source"] == "personal"
        assert abs(result["rpi_raw"] - 2.0) < 0.01, f"Expected RPI=2.0, got {result['rpi_raw']}"

    def test_normalize_single_post_cold_start(self):
        """Verify single post normalization with insufficient history"""
        # Using module-level imports

        config = NormalizationConfig(min_posts_for_baseline=5)
        normalizer = MetricNormalizer(config)

        # Fit a niche baseline
        df = pd.DataFrame({
            "niche": ["food"] * 50,
            "weighted_engagement": [500] * 50,
        })
        normalizer.fit_niche_baselines(df, niche_col="niche")

        post = {"likes_count": 100}
        history = [{"likes_count": 100}] * 2  # Only 2 posts, less than min

        result = normalizer.normalize_single_post(post, history, niche="food")

        assert result["baseline_source"] in ["niche", "global"]
        assert result["baseline_engagement"] == 500.0


class TestInverseTransform:
    """Tests for RPI to absolute engagement conversion"""

    def test_inverse_transform_no_log(self):
        """Verify inverse transform without log"""
        # Using module-level imports

        config = NormalizationConfig(apply_log_transform=False)
        normalizer = MetricNormalizer(config)

        # RPI = 1.5, baseline = 1000 -> engagement = 1500
        engagement = normalizer.inverse_transform(1.5, 1000)
        assert engagement == 1500.0

    def test_inverse_transform_with_log(self):
        """Verify inverse transform with log1p"""
        # Using module-level imports

        normalizer = MetricNormalizer()  # log transform enabled

        # If RPI_score = log1p(1.5) ≈ 0.916
        # Then inverse: expm1(0.916) ≈ 1.5
        # engagement = 1.5 * 1000 = 1500

        rpi_score = np.log1p(1.5)
        engagement = normalizer.inverse_transform(rpi_score, 1000)

        assert abs(engagement - 1500) < 1, f"Expected ~1500, got {engagement}"


class TestUtilityFunctions:
    """Tests for utility functions"""

    def test_normalize_for_training(self):
        """Verify convenience function for training data preparation"""
        # Using module-level imports

        df = pd.DataFrame({
            "author_id": ["a"] * 20,
            "niche": ["food"] * 20,
            "likes_count": list(range(100, 2100, 100)),
            "posted_at": pd.date_range("2024-01-01", periods=20),
        })

        df_normalized, normalizer = normalize_for_training(df)

        assert "rpi_score" in df_normalized.columns
        assert normalizer is not None

    def test_create_rpi_from_raw_metrics(self):
        """Verify quick RPI calculation function"""
        # Using module-level imports

        rpi = create_rpi_from_raw_metrics(
            likes=1000,
            comments=100,
            saves=50,
            shares=25,
            author_baseline=1000,
            apply_log=False
        )

        # Weighted: 1000 + 300 + 250 + 100 = 1650
        # RPI = 1650 / 1000 = 1.65

        assert abs(rpi - 1.65) < 0.01, f"Expected RPI=1.65, got {rpi}"


class TestMultipleAuthors:
    """Tests for proper handling of multiple authors"""

    def test_separate_baselines_per_author(self):
        """Verify each author has their own baseline"""
        # Using module-level imports

        config = NormalizationConfig(min_posts_for_baseline=3)
        normalizer = MetricNormalizer(config)

        df = pd.DataFrame({
            "author_id": (
                ["low_engagement"] * 10 +
                ["high_engagement"] * 10
            ),
            "weighted_engagement": (
                [100] * 10 +  # Low engagement author
                [10000] * 10  # High engagement author
            ),
            "posted_at": pd.date_range("2024-01-01", periods=20),
        })

        df_normalized = normalizer.normalize_dataframe(df, niche_col=None)

        # Get baselines for each author's last post
        low_baseline = df_normalized[
            df_normalized["author_id"] == "low_engagement"
        ].iloc[-1]["baseline_engagement"]

        high_baseline = df_normalized[
            df_normalized["author_id"] == "high_engagement"
        ].iloc[-1]["baseline_engagement"]

        assert low_baseline == 100, f"Low engagement baseline should be 100, got {low_baseline}"
        assert high_baseline == 10000, f"High engagement baseline should be 10000, got {high_baseline}"

        # Both RPI scores should be ~1.0 (normal performance)
        low_rpi = df_normalized[
            df_normalized["author_id"] == "low_engagement"
        ].iloc[-1]["rpi_raw"]

        high_rpi = df_normalized[
            df_normalized["author_id"] == "high_engagement"
        ].iloc[-1]["rpi_raw"]

        assert abs(low_rpi - 1.0) < 0.01, f"Low author RPI should be ~1.0, got {low_rpi}"
        assert abs(high_rpi - 1.0) < 0.01, f"High author RPI should be ~1.0, got {high_rpi}"


# Pytest fixtures
@pytest.fixture
def normalizer():
    """Default normalizer instance"""
    from app.services.normalization import MetricNormalizer
    return MetricNormalizer()


@pytest.fixture
def sample_df():
    """Sample DataFrame for testing"""
    return pd.DataFrame({
        "author_id": ["author_1"] * 20,
        "niche": ["food"] * 20,
        "likes_count": list(range(100, 2100, 100)),
        "comments_count": [10] * 20,
        "posted_at": pd.date_range("2024-01-01", periods=20),
    })


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
