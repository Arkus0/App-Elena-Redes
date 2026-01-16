"""
Integration tests for the Survivor Bias Fix - Balanced Sampling Pipeline

These tests verify that:
1. BalancedScraper correctly collects both top and bottom posts
2. Preprocessing calculates engagement_ratio correctly
3. is_viral labeling works with Top 20% = 1, Bottom 20% = 0
4. Dataset balancing ensures minimum 30% negative cases
5. ApifyService integrates balanced sampling correctly
"""

import pytest
import numpy as np
from datetime import datetime, timedelta


class TestBalancedScraper:
    """Tests for the BalancedScraper class"""

    def test_balanced_sampling_collects_top_and_bottom(self):
        """Verify that balanced sampling returns both top and bottom posts"""
        from app.services.scraper.balanced_scraper import (
            BalancedScraper,
            SamplingStrategy,
            SamplingConfig
        )

        # Create scraper with balanced strategy
        config = SamplingConfig(
            strategy=SamplingStrategy.BALANCED,
            top_n=5,
            bottom_n=3
        )
        scraper = BalancedScraper(strategy=SamplingStrategy.BALANCED, config=config)

        # Generate test posts with varying engagement
        posts = [
            {"platform_id": f"post_{i}", "likes": 1000 - i * 50, "comments": 100 - i * 5}
            for i in range(20)
        ]

        # Apply balanced sampling
        result, metadata = scraper.apply_balanced_sampling(posts, follower_count=10000)

        # Verify we got both top and bottom posts
        assert metadata["top_posts_count"] > 0, "Should have top posts"
        assert metadata["bottom_posts_count"] > 0, "Should have bottom posts"
        assert metadata["strategy"] == "balanced"

        # Verify total count matches config
        expected_count = min(config.top_n + config.bottom_n, len(posts))
        assert len(result) <= expected_count + 2  # Allow small margin

    def test_minimum_5_flop_posts(self):
        """Verify that at least 5 flop posts are collected (requirement)"""
        from app.services.scraper.balanced_scraper import SamplingConfig, SamplingStrategy

        # Config should enforce minimum 5 bottom posts
        config = SamplingConfig(strategy=SamplingStrategy.BALANCED, bottom_n=3)

        # Should be corrected to minimum of 5
        assert config.bottom_n >= 5, "bottom_n should be at least 5"

    def test_get_flop_posts_returns_lowest_engagement(self):
        """Verify get_flop_posts returns posts with lowest engagement"""
        from app.services.scraper.balanced_scraper import BalancedScraper, SamplingStrategy

        scraper = BalancedScraper(strategy=SamplingStrategy.BALANCED)

        # Create posts with known engagement scores
        posts = [
            {"platform_id": "high", "likes": 10000, "comments": 500},
            {"platform_id": "medium", "likes": 1000, "comments": 50},
            {"platform_id": "low", "likes": 10, "comments": 1},
            {"platform_id": "very_low", "likes": 5, "comments": 0},
        ]

        flops = scraper.get_flop_posts(posts, n=2, follower_count=10000)

        # Verify we got the lowest engagement posts
        assert len(flops) == 2
        flop_ids = [p["platform_id"] for p in flops]
        assert "very_low" in flop_ids, "Should include very_low engagement post"
        assert "low" in flop_ids, "Should include low engagement post"


class TestPreprocessing:
    """Tests for the preprocessing module"""

    def test_calculate_engagement_ratio(self):
        """Verify engagement ratio calculation: Engagement / Followers * 100"""
        from app.services.scraper.preprocessing import calculate_engagement_ratio

        post = {
            "likes_count": 100,
            "comments_count": 10,    # weighted 3x = 30
            "saves_count": 5,        # weighted 5x = 25
            "shares_count": 2,       # weighted 4x = 8
        }
        # Total weighted: 100 + 30 + 25 + 8 = 163

        ratio = calculate_engagement_ratio(post, follower_count=10000)

        # Expected: (163 / 10000) * 100 = 1.63%
        assert 1.5 <= ratio <= 1.8, f"Expected ~1.63, got {ratio}"

    def test_engagement_ratio_with_zero_followers(self):
        """Verify handling of zero followers"""
        from app.services.scraper.preprocessing import calculate_engagement_ratio

        post = {"likes_count": 100, "comments_count": 10}
        ratio = calculate_engagement_ratio(post, follower_count=0)

        assert ratio == 0.0, "Should return 0 for zero followers"

    def test_label_viral_status_top_20_percent(self):
        """Verify is_viral=1 for Top 20% of posts"""
        from app.services.scraper.preprocessing import label_viral_status, LabelingConfig

        # Create 10 posts with engagement ratios
        posts = [
            {"engagement_ratio": 10.0 - i}
            for i in range(10)
        ]
        # Engagement ratios: 10, 9, 8, 7, 6, 5, 4, 3, 2, 1

        config = LabelingConfig(top_percentile=0.20, bottom_percentile=0.20)
        labeled = label_viral_status(posts, config)

        # Top 20% = top 2 posts (engagement_ratio 10, 9)
        viral_posts = [p for p in labeled if p.get("is_viral") == 1]
        assert len(viral_posts) == 2, f"Expected 2 viral posts, got {len(viral_posts)}"

        # Verify they have highest engagement
        viral_ratios = [p["engagement_ratio"] for p in viral_posts]
        assert 10.0 in viral_ratios and 9.0 in viral_ratios

    def test_label_viral_status_bottom_20_percent(self):
        """Verify is_viral=0 for Bottom 20% of posts"""
        from app.services.scraper.preprocessing import label_viral_status, LabelingConfig

        # Create 10 posts
        posts = [
            {"engagement_ratio": 10.0 - i}
            for i in range(10)
        ]

        config = LabelingConfig(top_percentile=0.20, bottom_percentile=0.20)
        labeled = label_viral_status(posts, config)

        # Bottom 20% = bottom 2 posts (engagement_ratio 2, 1)
        flop_posts = [p for p in labeled if p.get("is_viral") == 0]
        assert len(flop_posts) == 2, f"Expected 2 flop posts, got {len(flop_posts)}"

        # Verify they have lowest engagement
        flop_ratios = [p["engagement_ratio"] for p in flop_posts]
        assert 1.0 in flop_ratios and 2.0 in flop_ratios

    def test_label_viral_status_middle_excluded(self):
        """Verify Middle 60% has is_viral=None (excluded from binary training)"""
        from app.services.scraper.preprocessing import label_viral_status, LabelingConfig

        posts = [{"engagement_ratio": 10.0 - i} for i in range(10)]

        config = LabelingConfig(top_percentile=0.20, bottom_percentile=0.20)
        labeled = label_viral_status(posts, config)

        # Middle 60% = 6 posts
        middle_posts = [p for p in labeled if p.get("is_viral") is None]
        assert len(middle_posts) == 6, f"Expected 6 middle posts, got {len(middle_posts)}"


class TestDatasetBalancing:
    """Tests for the dataset balancing function"""

    def test_balance_ensures_minimum_30_percent_negative(self):
        """Verify dataset has at least 30% negative cases after balancing"""
        from app.services.scraper.preprocessing import balance_dataset, BalancingConfig

        # Create imbalanced dataset: 90% positive, 10% negative
        posts = []
        for i in range(90):
            posts.append({"id": f"viral_{i}", "is_viral": 1})
        for i in range(10):
            posts.append({"id": f"flop_{i}", "is_viral": 0})

        config = BalancingConfig(min_negative_ratio=0.30)
        balanced, metadata = balance_dataset(posts, config)

        # Calculate final ratio
        negative_count = sum(1 for p in balanced if p.get("is_viral") == 0)
        total_count = len(balanced)
        negative_ratio = negative_count / total_count

        assert negative_ratio >= 0.30, \
            f"Negative ratio {negative_ratio:.1%} should be >= 30%"

    def test_balance_does_not_change_already_balanced_dataset(self):
        """Verify balanced datasets are not modified unnecessarily"""
        from app.services.scraper.preprocessing import balance_dataset, BalancingConfig

        # Create already balanced dataset: 50% positive, 50% negative
        posts = []
        for i in range(50):
            posts.append({"id": f"viral_{i}", "is_viral": 1})
        for i in range(50):
            posts.append({"id": f"flop_{i}", "is_viral": 0})

        config = BalancingConfig(min_negative_ratio=0.30)
        balanced, metadata = balance_dataset(posts, config)

        assert metadata["action"] == "none", "Should not modify already balanced dataset"

    def test_balance_metadata_includes_statistics(self):
        """Verify balancing returns useful metadata"""
        from app.services.scraper.preprocessing import balance_dataset, BalancingConfig

        posts = [
            {"id": f"viral_{i}", "is_viral": 1} for i in range(80)
        ] + [
            {"id": f"flop_{i}", "is_viral": 0} for i in range(20)
        ]

        config = BalancingConfig(min_negative_ratio=0.30)
        balanced, metadata = balance_dataset(posts, config)

        assert "original_positive" in metadata
        assert "original_negative" in metadata
        assert "final_ratio" in metadata
        assert "strategy" in metadata


class TestFullPipeline:
    """Integration tests for the complete pipeline"""

    def test_preprocessor_full_pipeline(self):
        """Test the full preprocessing pipeline"""
        from app.services.scraper.preprocessing import DataPreprocessor

        # Create realistic posts
        posts = []
        for i in range(30):
            posts.append({
                "platform_id": f"post_{i}",
                "likes_count": max(10, 1000 - i * 30),
                "comments_count": max(1, 100 - i * 3),
                "saves_count": max(0, 50 - i * 2),
                "shares_count": max(0, 20 - i),
            })

        preprocessor = DataPreprocessor()
        result, stats = preprocessor.preprocess_for_training(posts, follower_count=10000)

        # Verify pipeline ran successfully
        assert stats["input_count"] == 30
        assert stats["viral_count"] > 0, "Should have viral posts"
        assert stats["flop_count"] > 0, "Should have flop posts"
        assert stats["balancing"]["final_ratio"] >= 0.30, \
            "Should have at least 30% negative ratio"

    def test_apify_service_uses_balanced_sampling(self):
        """Verify ApifyService integrates balanced sampling"""
        from app.services.apify_service import ApifyService
        from app.services.scraper.balanced_scraper import SamplingStrategy

        service = ApifyService(
            sampling_strategy=SamplingStrategy.BALANCED,
            top_n=5,
            bottom_n=3
        )

        # Verify scraper is configured correctly
        assert service.sampling_strategy == SamplingStrategy.BALANCED
        assert service.balanced_scraper is not None

    def test_mock_data_includes_flop_posts(self):
        """Verify mock data includes both viral and flop posts"""
        from app.services.apify_service import ApifyService
        from app.services.scraper.balanced_scraper import SamplingStrategy

        service = ApifyService(sampling_strategy=SamplingStrategy.BALANCED)

        # Get mock Instagram data
        mock_data = service._get_mock_instagram_data("test_user")
        posts = mock_data["posts"]

        # Check for flop posts
        flop_posts = [p for p in posts if p.get("_is_flop")]
        viral_posts = [p for p in posts if not p.get("_is_flop")]

        assert len(flop_posts) >= 5, \
            f"Should have at least 5 flop posts, got {len(flop_posts)}"
        assert len(viral_posts) >= 5, \
            f"Should have at least 5 viral posts, got {len(viral_posts)}"


class TestFlopsCharacteristics:
    """Tests to verify flop posts have correct characteristics"""

    def test_flop_posts_have_low_engagement_scores(self):
        """Verify flop posts have significantly lower engagement scores"""
        from app.services.apify_service import ApifyService

        service = ApifyService()
        mock_data = service._get_mock_instagram_data("test_user")
        posts = mock_data["posts"]

        flop_posts = [p for p in posts if p.get("_is_flop")]
        non_flop_posts = [p for p in posts if not p.get("_is_flop")]

        avg_flop_score = np.mean([p.get("engagement_score", 0) for p in flop_posts])
        avg_viral_score = np.mean([p.get("engagement_score", 0) for p in non_flop_posts])

        assert avg_flop_score < avg_viral_score, \
            f"Flop avg ({avg_flop_score:.1f}) should be < viral avg ({avg_viral_score:.1f})"

        # Flops should have significantly lower scores
        assert avg_flop_score < 20, \
            f"Flop average score ({avg_flop_score:.1f}) should be < 20"

    def test_flop_posts_have_failure_characteristics(self):
        """Verify flop posts demonstrate typical failure patterns"""
        from app.services.apify_service import ApifyService

        service = ApifyService()
        mock_data = service._get_mock_instagram_data("test_user")
        posts = mock_data["posts"]

        flop_posts = [p for p in posts if p.get("_is_flop")]

        # Check for common flop characteristics
        has_short_caption = any(len(p.get("caption", "")) < 50 for p in flop_posts)
        has_few_hashtags = any(len(p.get("hashtags", [])) < 3 for p in flop_posts)
        has_no_question = any("?" not in p.get("caption", "") for p in flop_posts)

        assert has_short_caption or has_few_hashtags or has_no_question, \
            "Flop posts should demonstrate at least one failure characteristic"


# Pytest fixtures for common test data
@pytest.fixture
def sample_posts():
    """Generate sample posts for testing"""
    return [
        {
            "platform_id": f"post_{i}",
            "likes_count": max(10, 1000 - i * 30),
            "comments_count": max(1, 100 - i * 3),
            "saves_count": max(0, 50 - i * 2),
            "shares_count": max(0, 20 - i),
            "caption": f"Sample caption {i}" + ("?" if i % 3 == 0 else ""),
            "hashtags": [f"tag{j}" for j in range(i % 5 + 1)],
        }
        for i in range(30)
    ]


@pytest.fixture
def imbalanced_dataset():
    """Generate imbalanced dataset (90% positive, 10% negative)"""
    return [
        {"id": f"viral_{i}", "is_viral": 1} for i in range(90)
    ] + [
        {"id": f"flop_{i}", "is_viral": 0} for i in range(10)
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
