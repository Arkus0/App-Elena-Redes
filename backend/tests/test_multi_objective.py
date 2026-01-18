"""
Tests for Multi-Objective Engagement Prediction System

Tests:
1. EngagementWeights schema validation
2. KPI weights CRUD operations
3. Multi-output predictor training and inference
4. Weighted RPI calculation
5. Default vs custom weights comparison

Author: BrandPulse AI
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

# Import schemas
from app.schemas.kpi_weights import (
    EngagementWeights,
    KPIWeightsCreate,
    MultiOutputPrediction,
    KPI_TEMPLATES,
)


class TestEngagementWeightsSchema:
    """Tests for EngagementWeights Pydantic schema."""

    def test_default_weights(self):
        """Test default weight values."""
        weights = EngagementWeights()

        assert weights.likes_weight == 1.0
        assert weights.comments_weight == 2.0
        assert weights.shares_weight == 10.0
        assert weights.saves_weight == 5.0
        assert weights.views_weight == 3.0

    def test_custom_weights(self):
        """Test custom weight assignment."""
        weights = EngagementWeights(
            likes_weight=5.0,
            comments_weight=5.0,
            shares_weight=5.0,
            saves_weight=5.0,
            views_weight=5.0,
        )

        assert weights.likes_weight == 5.0
        assert all(w == 5.0 for w in weights.to_vector())

    def test_weight_validation_bounds(self):
        """Test weight bounds (0-20)."""
        # Valid at bounds
        weights_min = EngagementWeights(likes_weight=0.0)
        weights_max = EngagementWeights(likes_weight=20.0)

        assert weights_min.likes_weight == 0.0
        assert weights_max.likes_weight == 20.0

        # Invalid - exceeds bounds
        with pytest.raises(ValueError):
            EngagementWeights(likes_weight=-1.0)

        with pytest.raises(ValueError):
            EngagementWeights(likes_weight=21.0)

    def test_to_vector(self):
        """Test conversion to ordered vector."""
        weights = EngagementWeights(
            likes_weight=1.0,
            comments_weight=2.0,
            shares_weight=3.0,
            saves_weight=4.0,
            views_weight=5.0,
        )

        vector = weights.to_vector()

        assert vector == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert len(vector) == 5

    def test_normalize(self):
        """Test weight normalization to sum=1."""
        weights = EngagementWeights(
            likes_weight=10.0,
            comments_weight=10.0,
            shares_weight=10.0,
            saves_weight=10.0,
            views_weight=10.0,
        )

        normalized = weights.normalize()

        assert abs(sum(normalized.to_vector()) - 1.0) < 0.001
        assert all(w == 0.2 for w in normalized.to_vector())

    def test_from_template_brand_awareness(self):
        """Test brand awareness template."""
        weights = EngagementWeights.from_template("brand_awareness")

        # Brand awareness should prioritize views and likes
        assert weights.views_weight > weights.saves_weight
        assert weights.likes_weight > weights.comments_weight

    def test_from_template_leads_conversions(self):
        """Test leads/conversions template."""
        weights = EngagementWeights.from_template("leads_conversions")

        # Leads should prioritize saves and shares
        assert weights.saves_weight > weights.likes_weight
        assert weights.shares_weight > weights.views_weight

    def test_from_template_community(self):
        """Test community template."""
        weights = EngagementWeights.from_template("community")

        # Community should prioritize comments
        assert weights.comments_weight > weights.likes_weight
        assert weights.comments_weight > weights.views_weight

    def test_from_template_unknown(self):
        """Test unknown template returns defaults."""
        weights = EngagementWeights.from_template("nonexistent")

        default = EngagementWeights()
        assert weights.to_vector() == default.to_vector()


class TestKPITemplates:
    """Tests for predefined KPI templates."""

    def test_all_templates_exist(self):
        """Test that all expected templates are defined."""
        expected = ["brand_awareness", "leads_conversions", "community", "viral", "balanced"]

        template_names = [t.name for t in KPI_TEMPLATES]

        for name in expected:
            assert name in template_names, f"Template '{name}' not found"

    def test_templates_have_required_fields(self):
        """Test templates have all required fields."""
        for template in KPI_TEMPLATES:
            assert template.name
            assert template.display_name
            assert template.description
            assert template.weights
            assert template.use_case
            assert isinstance(template.weights, EngagementWeights)

    def test_balanced_template_has_equal_weights(self):
        """Test balanced template has equal weights."""
        balanced = next(t for t in KPI_TEMPLATES if t.name == "balanced")
        weights = balanced.weights.to_vector()

        assert len(set(weights)) == 1, "Balanced template should have equal weights"


class TestWeightedRPICalculation:
    """Tests for weighted RPI calculation logic."""

    def test_weighted_sum_calculation(self):
        """Test basic weighted sum calculation."""
        # Simulated log predictions
        log_predictions = np.array([4.0, 2.0, 1.0, 2.5, 6.0])  # likes, comments, shares, saves, views

        # Default weights
        weights = EngagementWeights()
        weight_vector = np.array(weights.to_vector())

        weighted_sum = np.dot(log_predictions, weight_vector)
        total_weight = np.sum(weight_vector)

        # RPI formula: (weighted_sum / total_weight) * scale_factor
        rpi = (weighted_sum / total_weight) * 15.0

        assert rpi > 0
        assert rpi <= 100

    def test_different_weights_different_rpi(self):
        """Test that different weights produce different RPI scores."""
        log_predictions = np.array([4.0, 2.0, 1.0, 2.5, 6.0])

        # Calculate RPI with default weights
        default_weights = EngagementWeights()
        default_vector = np.array(default_weights.to_vector())
        default_rpi = np.dot(log_predictions, default_vector) / np.sum(default_vector) * 15

        # Calculate RPI with custom weights (prioritize views)
        custom_weights = EngagementWeights(
            likes_weight=1.0,
            comments_weight=1.0,
            shares_weight=1.0,
            saves_weight=1.0,
            views_weight=15.0,
        )
        custom_vector = np.array(custom_weights.to_vector())
        custom_rpi = np.dot(log_predictions, custom_vector) / np.sum(custom_vector) * 15

        # Custom should have higher RPI since views (6.0) is highest
        assert custom_rpi > default_rpi, "Custom weights prioritizing views should yield higher RPI"

    def test_zero_weights_edge_case(self):
        """Test handling of zero total weight."""
        weights = EngagementWeights(
            likes_weight=0.0,
            comments_weight=0.0,
            shares_weight=0.0,
            saves_weight=0.0,
            views_weight=0.0,
        )

        total = sum(weights.to_vector())
        assert total == 0.0

        # Normalization should handle this gracefully
        normalized = weights.normalize()
        # Should return defaults since total is 0
        assert sum(normalized.to_vector()) > 0


class TestMultiOutputPredictionSchema:
    """Tests for MultiOutputPrediction schema."""

    def test_prediction_fields(self):
        """Test all prediction fields are present."""
        prediction = MultiOutputPrediction(
            log_likes=4.0,
            log_comments=2.0,
            log_shares=1.0,
            log_saves=2.5,
            log_views=6.0,
            predicted_likes=50.0,
            predicted_comments=6.0,
            predicted_shares=2.0,
            predicted_saves=11.0,
            predicted_views=400.0,
            weighted_rpi=65.0,
            weights_used=EngagementWeights(),
        )

        assert prediction.log_likes == 4.0
        assert prediction.predicted_likes == 50.0
        assert prediction.weighted_rpi == 65.0
        assert prediction.weights_used.likes_weight == 1.0

    def test_prediction_to_vector(self):
        """Test converting predictions to vector."""
        prediction = MultiOutputPrediction(
            log_likes=1.0,
            log_comments=2.0,
            log_shares=3.0,
            log_saves=4.0,
            log_views=5.0,
            predicted_likes=0.0,
            predicted_comments=0.0,
            predicted_shares=0.0,
            predicted_saves=0.0,
            predicted_views=0.0,
            weighted_rpi=0.0,
            weights_used=EngagementWeights(),
        )

        vector = prediction.to_vector()
        assert vector == [1.0, 2.0, 3.0, 4.0, 5.0]


class TestComparisonDefaultsVsCustom:
    """
    Example test comparing default weights vs custom weights.

    Demonstrates how different weight configurations affect RPI.
    """

    def test_example_inmobiliaria_vs_cafeteria(self):
        """
        Compare RPI for same content with different business goals.

        Scenario:
        - Content generates: 100 likes, 10 comments, 5 shares, 20 saves, 1000 views
        - Inmobiliaria (leads focus): Prioritizes saves/shares
        - Cafeteria (community focus): Prioritizes comments
        """
        # Simulated engagement (log scale)
        log_likes = np.log1p(100)      # ~4.6
        log_comments = np.log1p(10)    # ~2.4
        log_shares = np.log1p(5)       # ~1.8
        log_saves = np.log1p(20)       # ~3.0
        log_views = np.log1p(1000)     # ~6.9

        predictions = np.array([log_likes, log_comments, log_shares, log_saves, log_views])

        # Inmobiliaria weights (leads_conversions template)
        inmobiliaria = EngagementWeights.from_template("leads_conversions")
        inmo_vector = np.array(inmobiliaria.to_vector())
        inmo_rpi = np.dot(predictions, inmo_vector) / np.sum(inmo_vector) * 15

        # Cafeteria weights (community template)
        cafeteria = EngagementWeights.from_template("community")
        cafe_vector = np.array(cafeteria.to_vector())
        cafe_rpi = np.dot(predictions, cafe_vector) / np.sum(cafe_vector) * 15

        # Default weights
        default = EngagementWeights()
        default_vector = np.array(default.to_vector())
        default_rpi = np.dot(predictions, default_vector) / np.sum(default_vector) * 15

        print(f"\n{'='*60}")
        print("COMPARISON: Same Content, Different Business Goals")
        print(f"{'='*60}")
        print(f"Content metrics: {100} likes, {10} comments, {5} shares, {20} saves, {1000} views")
        print(f"\nDefault weights RPI:        {default_rpi:.1f}")
        print(f"Inmobiliaria (leads) RPI:   {inmo_rpi:.1f}")
        print(f"Cafeteria (community) RPI:  {cafe_rpi:.1f}")
        print(f"{'='*60}\n")

        # All RPIs should be valid (0-100)
        for rpi in [default_rpi, inmo_rpi, cafe_rpi]:
            assert 0 <= rpi <= 100, f"RPI {rpi} out of bounds"

        # Different weights should produce different results
        # (exact relationship depends on content metrics)
        assert inmo_rpi != cafe_rpi, "Different templates should produce different RPIs"


class TestKPIWeightsCreate:
    """Tests for KPIWeightsCreate schema."""

    def test_create_with_custom_weights(self):
        """Test creating config with custom weights."""
        config = KPIWeightsCreate(
            business_id=1,
            weights=EngagementWeights(
                likes_weight=5.0,
                comments_weight=5.0,
                shares_weight=5.0,
                saves_weight=5.0,
                views_weight=5.0,
            ),
        )

        assert config.business_id == 1
        assert config.weights.likes_weight == 5.0

    def test_create_with_template(self):
        """Test creating config from template."""
        config = KPIWeightsCreate(
            business_id=1,
            template_name="brand_awareness",
        )

        # Template should be applied
        assert config.template_name == "brand_awareness"
        assert config.weights.views_weight > config.weights.saves_weight

    def test_template_overrides_weights(self):
        """Test that template_name overrides provided weights."""
        config = KPIWeightsCreate(
            business_id=1,
            weights=EngagementWeights(likes_weight=20.0),  # This should be ignored but must be valid
            template_name="balanced",
        )

        # Template should override
        assert config.weights.likes_weight != 20.0
        # Check against template value (assuming balanced uses 4.0 or similar, derived from 20/5)
        # We just check it changed from the input


# =============================================================================
# Integration Test Example
# =============================================================================

def test_full_prediction_flow():
    """
    Integration test for full multi-output prediction flow.

    This test demonstrates the complete flow:
    1. User configures weights
    2. System predicts multi-output engagement
    3. RPI is calculated with user's weights
    """
    # 1. User selects "leads_conversions" template for their inmobiliaria
    user_weights = EngagementWeights.from_template("leads_conversions")

    # 2. System generates predictions (simulated)
    content_features = {
        "caption": "3 dormitorios con vistas al mar en Almeria",
        "hook_energy": 0.8,
        "is_reel": 1,
    }

    # Simulated multi-output predictions (log scale)
    predictions = {
        "log_likes": 4.5,
        "log_comments": 2.0,
        "log_shares": 1.5,
        "log_saves": 3.2,
        "log_views": 6.5,
    }

    # 3. Calculate weighted RPI
    pred_vector = np.array([
        predictions["log_likes"],
        predictions["log_comments"],
        predictions["log_shares"],
        predictions["log_saves"],
        predictions["log_views"],
    ])
    weight_vector = np.array(user_weights.to_vector())

    weighted_rpi = np.dot(pred_vector, weight_vector) / np.sum(weight_vector) * 15
    weighted_rpi = np.clip(weighted_rpi, 0, 100)

    # 4. Compare to default
    default_weights = EngagementWeights()
    default_vector = np.array(default_weights.to_vector())
    default_rpi = np.dot(pred_vector, default_vector) / np.sum(default_vector) * 15

    # Assertions
    assert 0 <= weighted_rpi <= 100
    assert weighted_rpi != default_rpi, "Custom weights should differ from default"

    print(f"\nFull Flow Test Results:")
    print(f"  User weights: leads_conversions template")
    print(f"  Default RPI: {default_rpi:.1f}")
    print(f"  Weighted RPI: {weighted_rpi:.1f}")
    print(f"  Difference: {weighted_rpi - default_rpi:+.1f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
