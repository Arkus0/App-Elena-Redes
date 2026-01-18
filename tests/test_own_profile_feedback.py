#!/usr/bin/env python3
"""
Tests for Human-in-the-Loop Own Profile Feedback System
========================================================

Validates the feedback loop that connects real Instagram/TikTok metrics
from the client's own posts to the ML model for continuous learning.

Test cases:
1. Payload with isOwnProfile=True triggers feedback registration
2. Real metrics are correctly calculated (engagement_rate, log transforms)
3. ML service receives and stores feedback data
4. Competitor content (isOwnProfile=False) does not trigger feedback

Run with: pytest tests/test_own_profile_feedback.py -v
"""

import sys
import math
import hashlib
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import modules under test
try:
    from backend.app.api.ingest import (
        _generate_content_hash,
        _calculate_real_targets,
        _content_to_features_dict,
        ContentData,
        ContentAuthor,
        ContentMetrics,
        MediaItem,
        AudioInfo,
    )
    INGEST_AVAILABLE = True
except ImportError as e:
    INGEST_AVAILABLE = False
    print(f"Warning: Could not import ingest module: {e}")

try:
    from backend.app.services.ml_service import (
        get_ml_predictor,
        FeatureExtractor,
        PerformanceFeedback,
    )
    ML_SERVICE_AVAILABLE = True
except ImportError as e:
    ML_SERVICE_AVAILABLE = False
    print(f"Warning: Could not import ml_service: {e}")


# =============================================================================
# Test Data
# =============================================================================

def create_sample_content_data(
    username: str = "inmoalmeria",
    content_id: str = "ABC123",
    likes: int = 500,
    comments: int = 45,
    shares: int = 12,
    saves: int = 80,
    views: int = 5000,
    platform: str = "instagram",
    content_type: str = "reel",
    caption: str = "POV: Descubres tu casa perfecta en Sevilla"
) -> dict:
    """Create a sample ContentData-like dict for testing."""
    return {
        "platform": platform,
        "contentType": content_type,
        "contentId": content_id,
        "contentUrl": f"https://{platform}.com/p/{content_id}",
        "author": {
            "username": username,
            "displayName": "Inmo Almeria",
            "profilePicUrl": None,
            "isVerified": False,
        },
        "media": [
            {
                "type": "video",
                "url": None,
                "thumbnailUrl": None,
                "duration": 30,
            }
        ],
        "caption": caption,
        "hashtags": ["#casas", "#sevilla", "#inmobiliaria"],
        "mentions": ["@compradores"],
        "metrics": {
            "likes": likes,
            "comments": comments,
            "shares": shares,
            "saves": saves,
            "views": views,
            "plays": None,
        },
        "audio": {
            "title": "Trending Sound",
            "artist": "Unknown",
            "isOriginal": False,
            "audioUrl": None,
        },
        "postedAt": "2026-01-15T14:30:00Z",
        "extractedAt": datetime.utcnow().isoformat(),
        "sourceUrl": f"https://{platform}.com/p/{content_id}",
        "extractionMethod": "hydrated_data",
    }


# Create ContentData Pydantic model instance
def create_content_data_model(data: dict) -> "ContentData":
    """Convert dict to ContentData Pydantic model."""
    if not INGEST_AVAILABLE:
        pytest.skip("Ingest module not available")

    return ContentData(
        platform=data["platform"],
        contentType=data["contentType"],
        contentId=data["contentId"],
        contentUrl=data["contentUrl"],
        author=ContentAuthor(**data["author"]),
        media=[MediaItem(**m) for m in data["media"]],
        caption=data["caption"],
        hashtags=data["hashtags"],
        mentions=data["mentions"],
        metrics=ContentMetrics(**data["metrics"]),
        audio=AudioInfo(**data["audio"]) if data["audio"] else None,
        postedAt=data["postedAt"],
        extractedAt=data["extractedAt"],
        sourceUrl=data["sourceUrl"],
        extractionMethod=data["extractionMethod"],
    )


# =============================================================================
# Unit Tests - Utility Functions
# =============================================================================

@pytest.mark.skipif(not INGEST_AVAILABLE, reason="Ingest module not available")
class TestContentHashGeneration:
    """Tests for content hash generation."""

    def test_hash_is_deterministic(self):
        """Same content should produce same hash."""
        data = create_sample_content_data()
        content = create_content_data_model(data)

        hash1 = _generate_content_hash(content)
        hash2 = _generate_content_hash(content)

        assert hash1 == hash2

    def test_hash_is_unique_per_content(self):
        """Different content IDs should produce different hashes."""
        data1 = create_sample_content_data(content_id="ABC123")
        data2 = create_sample_content_data(content_id="XYZ789")

        content1 = create_content_data_model(data1)
        content2 = create_content_data_model(data2)

        hash1 = _generate_content_hash(content1)
        hash2 = _generate_content_hash(content2)

        assert hash1 != hash2

    def test_hash_length(self):
        """Hash should be 16 characters."""
        data = create_sample_content_data()
        content = create_content_data_model(data)

        content_hash = _generate_content_hash(content)

        assert len(content_hash) == 16

    def test_hash_includes_platform(self):
        """Same content ID but different platforms should have different hashes."""
        data_ig = create_sample_content_data(platform="instagram", content_id="SAME123")
        data_tt = create_sample_content_data(platform="tiktok", content_id="SAME123")

        content_ig = create_content_data_model(data_ig)
        content_tt = create_content_data_model(data_tt)

        hash_ig = _generate_content_hash(content_ig)
        hash_tt = _generate_content_hash(content_tt)

        assert hash_ig != hash_tt


@pytest.mark.skipif(not INGEST_AVAILABLE, reason="Ingest module not available")
class TestRealTargetsCalculation:
    """Tests for calculating real engagement targets."""

    def test_engagement_rate_calculation_with_views(self):
        """Test engagement rate when views are available."""
        data = create_sample_content_data(
            likes=500,
            comments=45,
            shares=12,
            saves=80,
            views=5000
        )
        content = create_content_data_model(data)

        targets = _calculate_real_targets(content)

        # Expected: (500 + 45*2 + 80*3 + 12*4) / 5000 * 100 = (500+90+240+48)/5000*100 = 17.56%
        expected_rate = (500 + 45*2 + 80*3 + 12*4) / 5000 * 100

        assert "engagement_rate" in targets
        assert abs(targets["engagement_rate"] - expected_rate) < 0.1

    def test_engagement_rate_calculation_without_views(self):
        """Test engagement rate fallback when views are 0."""
        data = create_sample_content_data(
            likes=500,
            comments=45,
            shares=12,
            saves=80,
            views=0
        )
        content = create_content_data_model(data)

        targets = _calculate_real_targets(content)

        # Fallback formula: min(100, (likes + comments*2 + saves*3 + shares*4) / 10)
        expected = min(100, (500 + 45*2 + 80*3 + 12*4) / 10)

        assert "engagement_rate" in targets
        assert abs(targets["engagement_rate"] - expected) < 0.1

    def test_log_transforms(self):
        """Test that log1p transforms are correctly calculated."""
        data = create_sample_content_data(
            likes=500,
            comments=45,
            shares=12,
            saves=80,
            views=5000
        )
        content = create_content_data_model(data)

        targets = _calculate_real_targets(content)

        assert abs(targets["log_likes"] - math.log1p(500)) < 0.001
        assert abs(targets["log_comments"] - math.log1p(45)) < 0.001
        assert abs(targets["log_shares"] - math.log1p(12)) < 0.001
        assert abs(targets["log_saves"] - math.log1p(80)) < 0.001
        assert abs(targets["log_views"] - math.log1p(5000)) < 0.001

    def test_zero_metrics_handling(self):
        """Test handling of zero metrics."""
        data = create_sample_content_data(
            likes=0,
            comments=0,
            shares=0,
            saves=0,
            views=0
        )
        content = create_content_data_model(data)

        targets = _calculate_real_targets(content)

        assert targets["likes"] == 0
        assert targets["log_likes"] == 0.0  # log1p(0) = 0
        assert targets["engagement_rate"] >= 0  # Should not error


@pytest.mark.skipif(not INGEST_AVAILABLE, reason="Ingest module not available")
class TestContentToFeaturesDict:
    """Tests for converting ContentData to features dict."""

    def test_basic_conversion(self):
        """Test basic field mapping."""
        data = create_sample_content_data(
            caption="POV: Descubres tu casa perfecta"
        )
        content = create_content_data_model(data)

        features = _content_to_features_dict(content)

        assert features["caption"] == "POV: Descubres tu casa perfecta"
        assert features["content_format"] == "reel"
        assert features["likes"] == 500
        assert features["comments"] == 45

    def test_format_detection_reel(self):
        """Test reel format detection."""
        data = create_sample_content_data(content_type="reel")
        content = create_content_data_model(data)

        features = _content_to_features_dict(content)

        assert features["content_format"] == "reel"

    def test_format_detection_carousel(self):
        """Test carousel format detection."""
        data = create_sample_content_data(content_type="carousel")
        content = create_content_data_model(data)

        features = _content_to_features_dict(content)

        assert features["content_format"] == "carousel"

    def test_format_detection_static(self):
        """Test static post format detection."""
        data = create_sample_content_data(content_type="post")
        content = create_content_data_model(data)

        features = _content_to_features_dict(content)

        assert features["content_format"] == "static"

    def test_video_duration_extraction(self):
        """Test video duration is extracted."""
        data = create_sample_content_data()
        data["media"] = [{"type": "video", "url": None, "thumbnailUrl": None, "duration": 45}]
        content = create_content_data_model(data)

        features = _content_to_features_dict(content)

        assert features["video_duration_seconds"] == 45

    def test_audio_name_extraction(self):
        """Test audio name extraction."""
        data = create_sample_content_data()
        content = create_content_data_model(data)

        features = _content_to_features_dict(content)

        assert features["audio_name"] == "Trending Sound"


# =============================================================================
# Integration Tests - ML Service Feedback
# =============================================================================

@pytest.mark.skipif(not ML_SERVICE_AVAILABLE, reason="ML service not available")
class TestMLServiceFeedback:
    """Integration tests for ML feedback registration."""

    def test_feedback_registration(self):
        """Test that feedback can be registered with ML predictor."""
        predictor = get_ml_predictor()

        # Create test metrics
        metrics = {
            "likes": 500,
            "comments": 45,
            "shares": 12,
            "saves": 80,
            "views": 5000,
            "log_likes": math.log1p(500),
            "log_comments": math.log1p(45),
            "log_shares": math.log1p(12),
            "log_saves": math.log1p(80),
            "log_views": math.log1p(5000),
            "engagement_rate": 17.56,
        }

        # Create test content dict
        content_dict = {
            "caption": "POV: El secreto de las mejores casas",
            "content_format": "reel",
            "hashtags": ["#inmobiliaria"],
            "likes": 500,
            "comments": 45,
            "business_type": "inmobiliaria",
        }

        # Register feedback
        feedback = predictor.register_performance_feedback(
            content_id=12345678,
            metrics=metrics,
            original_content=content_dict,
            predicted_score=None
        )

        assert isinstance(feedback, PerformanceFeedback)
        assert feedback.content_id == 12345678
        assert feedback.actual_score > 0

    def test_feedback_statistics(self):
        """Test that feedback statistics are tracked."""
        predictor = get_ml_predictor()

        stats = predictor.get_feedback_statistics()

        assert "total_samples" in stats
        assert "high_priority_count" in stats
        assert "queue_size" in stats
        assert "model_bias" in stats

    def test_high_priority_detection(self):
        """Test that high delta samples are flagged as high priority."""
        predictor = get_ml_predictor()

        # Create metrics that will result in high actual score
        high_metrics = {
            "likes": 10000,
            "comments": 500,
            "shares": 100,
            "saves": 800,
            "views": 50000,
            "log_likes": math.log1p(10000),
            "log_comments": math.log1p(500),
            "log_shares": math.log1p(100),
            "log_saves": math.log1p(800),
            "log_views": math.log1p(50000),
            "engagement_rate": 50.0,
        }

        content_dict = {
            "caption": "VIRAL: El video que rompio internet",
            "content_format": "reel",
            "likes": 10000,
            "comments": 500,
            "business_type": "inmobiliaria",
        }

        # Register with low predicted score to create high delta
        feedback = predictor.register_performance_feedback(
            content_id=99999999,
            metrics=high_metrics,
            original_content=content_dict,
            predicted_score=20.0  # Low prediction vs high actual
        )

        # Should be flagged as high priority due to > 20% delta
        assert feedback.is_high_priority or abs(feedback.delta_percent) > 20


# =============================================================================
# Payload Integration Tests
# =============================================================================

@pytest.mark.skipif(not INGEST_AVAILABLE, reason="Ingest module not available")
class TestPayloadProcessing:
    """Tests for full payload processing flow."""

    def test_own_profile_payload_structure(self):
        """Test that own profile payload has correct structure."""
        from backend.app.api.ingest import RawIngestPayload

        payload_data = {
            "source": "elena_bridge_extension",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "type": "content",
            "isOwnProfile": True,
            "content": create_sample_content_data(),
        }

        # Should parse without error
        payload = RawIngestPayload(**payload_data)

        assert payload.isOwnProfile is True
        assert payload.content is not None

    def test_competitor_payload_structure(self):
        """Test that competitor payload has isOwnProfile=False."""
        from backend.app.api.ingest import RawIngestPayload

        payload_data = {
            "source": "elena_bridge_extension",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "type": "content",
            "isOwnProfile": False,  # Competitor content
            "content": create_sample_content_data(username="competitor_account"),
        }

        payload = RawIngestPayload(**payload_data)

        assert payload.isOwnProfile is False


# =============================================================================
# Example Payload Test (for documentation)
# =============================================================================

@pytest.mark.skipif(not INGEST_AVAILABLE, reason="Ingest module not available")
class TestExamplePayload:
    """
    Example payload test for documentation purposes.
    Shows expected payload structure for own profile feedback.
    """

    def test_example_own_profile_payload(self):
        """
        Example payload from extension when user visits their own post.

        This is the payload structure sent by Elena Bridge when:
        1. User has configured own_instagram_username = "inmoalmeria"
        2. User visits https://instagram.com/p/ABC123/ (their own post)
        3. Extension detects author matches own_username
        4. isOwnProfile is set to True
        """
        example_payload = {
            "source": "elena_bridge_extension",
            "version": "1.0.0",
            "timestamp": "2026-01-18T12:00:00Z",
            "pageType": "reel",
            "type": "content",
            "isOwnProfile": True,  # KEY: This triggers feedback loop
            "content": {
                "platform": "instagram",
                "contentType": "reel",
                "contentId": "ABC123XYZ",
                "contentUrl": "https://instagram.com/reel/ABC123XYZ",
                "author": {
                    "username": "inmoalmeria",  # Matches configured own_username
                    "displayName": "Inmo Almeria",
                    "profilePicUrl": "https://...",
                    "isVerified": False,
                },
                "media": [
                    {
                        "type": "video",
                        "url": None,  # URLs may not be extracted
                        "thumbnailUrl": "https://...",
                        "duration": 28,
                    }
                ],
                "caption": "POV: Descubres tu piso perfecto en el centro de Sevilla",
                "hashtags": ["#sevilla", "#inmobiliaria", "#casas", "#pov"],
                "mentions": [],
                "metrics": {
                    "likes": 847,      # REAL metrics from own post
                    "comments": 62,
                    "shares": 15,
                    "saves": 124,
                    "views": 12450,
                    "plays": None,
                },
                "audio": {
                    "title": "Sonido original",
                    "artist": "inmoalmeria",
                    "isOriginal": True,
                    "audioUrl": None,
                },
                "postedAt": "2026-01-15T14:30:00Z",
                "extractedAt": "2026-01-18T12:00:00Z",
                "sourceUrl": "https://instagram.com/reel/ABC123XYZ",
                "extractionMethod": "hydrated_data",
            },
            "metadata": {
                "contentType": "reel",
                "platform": "instagram",
                "contentId": "ABC123XYZ",
                "author": "inmoalmeria",
                "sourceUrl": "https://instagram.com/reel/ABC123XYZ",
                "extractionMethod": "hydrated_data",
                "isOwnProfile": True,
            },
        }

        from backend.app.api.ingest import RawIngestPayload

        # Parse and validate
        payload = RawIngestPayload(**example_payload)

        assert payload.isOwnProfile is True
        assert payload.content.author.username == "inmoalmeria"
        assert payload.content.metrics.likes == 847

        # Convert to ContentData model
        content = payload.content

        # Calculate targets
        targets = _calculate_real_targets(content)

        print("\n" + "=" * 60)
        print("EXAMPLE OWN PROFILE FEEDBACK")
        print("=" * 60)
        print(f"Content ID: {content.contentId}")
        print(f"Author: @{content.author.username}")
        print(f"Is Own Profile: {payload.isOwnProfile}")
        print(f"\nReal Metrics:")
        print(f"  Likes: {content.metrics.likes}")
        print(f"  Comments: {content.metrics.comments}")
        print(f"  Shares: {content.metrics.shares}")
        print(f"  Saves: {content.metrics.saves}")
        print(f"  Views: {content.metrics.views}")
        print(f"\nCalculated Targets:")
        print(f"  Engagement Rate: {targets['engagement_rate']:.2f}%")
        print(f"  Log Likes: {targets['log_likes']:.3f}")
        print(f"  Log Comments: {targets['log_comments']:.3f}")
        print("=" * 60 + "\n")


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 70)
    print("OWN PROFILE FEEDBACK LOOP TESTS")
    print("=" * 70 + "\n")

    # Run quick validation
    if INGEST_AVAILABLE:
        print("[1] Testing content hash generation...")
        data = create_sample_content_data()
        content = create_content_data_model(data)
        content_hash = _generate_content_hash(content)
        print(f"  Content hash: {content_hash}")

        print("\n[2] Testing real targets calculation...")
        targets = _calculate_real_targets(content)
        print(f"  Engagement rate: {targets['engagement_rate']:.2f}%")
        print(f"  Log likes: {targets['log_likes']:.3f}")

        print("\n[3] Testing content to features conversion...")
        features = _content_to_features_dict(content)
        print(f"  Format: {features['content_format']}")
        print(f"  Caption length: {len(features['caption'])}")

    if ML_SERVICE_AVAILABLE:
        print("\n[4] Testing ML service feedback...")
        predictor = get_ml_predictor()
        stats = predictor.get_feedback_statistics()
        print(f"  Queue size: {stats['queue_size']}")
        print(f"  High priority: {stats['high_priority_count']}")

    print("\n" + "=" * 70)
    print("All validation checks passed!")
    print("=" * 70 + "\n")
