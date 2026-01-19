import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.growth_prediction_engine import GrowthPredictionEngine, GrowthPredictionConfig
from app.services.pattern_extractor import PatternExtractor
from app.models.pattern import PatternType
from app.models.scraped_post import ScrapedPost, ContentFormat

class TestGrowthPredictionEngineWeightedLoss:
    def test_calculate_sample_weights(self):
        config = GrowthPredictionConfig(
            loss_penalty_factor=2.5,
            negative_score_threshold=1.0
        )
        engine = GrowthPredictionEngine(config=config)

        # Test data: 3 samples
        # 1. Score 0.5 (Negative, < 1.0) -> Should have weight 1 + 2.5 = 3.5
        # 2. Score 0.9 (Negative, < 1.0) -> Should have weight 3.5
        # 3. Score 1.5 (Positive, >= 1.0) -> Should have weight 1.0
        y = np.array([0.5, 0.9, 1.5], dtype=np.float32)

        weights = engine._calculate_sample_weights(y)

        assert len(weights) == 3
        assert weights[0] == 3.5
        assert weights[1] == 3.5
        assert weights[2] == 1.0

    def test_calculate_sample_weights_custom_config(self):
        config = GrowthPredictionConfig(
            loss_penalty_factor=5.0,
            negative_score_threshold=0.5
        )
        engine = GrowthPredictionEngine(config=config)

        # Test data
        y = np.array([0.2, 0.6, 2.0], dtype=np.float32)

        weights = engine._calculate_sample_weights(y)

        assert weights[0] == 6.0  # 1 + 5.0
        assert weights[1] == 1.0  # 0.6 > 0.5
        assert weights[2] == 1.0

class TestPatternExtractorAntiPatterns:
    @pytest.mark.asyncio
    async def test_extract_anti_patterns_low_data(self):
        """Test that it skips if not enough posts"""
        extractor = PatternExtractor()

        # Mock DB session
        db = AsyncMock(spec=AsyncSession)

        # Mock count result <= 50
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 40
        db.execute.return_value = mock_count_result

        patterns = await extractor.extract_anti_patterns(db, 1, "test_niche")

        assert len(patterns) == 0
        # Should execute count query but not the posts query
        assert db.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_extract_anti_patterns_success(self):
        """Test successful extraction"""
        extractor = PatternExtractor()
        extractor.ai_service = AsyncMock()

        # Mock DB session
        db = AsyncMock(spec=AsyncSession)

        # 1. Mock count result > 50
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 100

        # 2. Mock posts result
        mock_posts_result = MagicMock()
        mock_post = ScrapedPost(
            id=1,
            caption="Test caption",
            content_format=ContentFormat.REEL,
            engagement_score=5.0,
            hashtags=[]
        )
        mock_posts_result.scalars.return_value.all.return_value = [mock_post] * 20

        # Setup side_effect for db.execute
        # First call: count, Second call: posts, Third call: competitor (not needed here based on logic?)
        # Let's verify extract_anti_patterns logic:
        # 1. Count query
        # 2. Posts query
        # 3. No competitor query, it iterates posts.
        db.execute.side_effect = [mock_count_result, mock_posts_result]

        # Mock AI response
        extractor.ai_service.analyze_anti_patterns.return_value = [
            {
                "pattern_name": "Test Anti-Pattern",
                "description": "Desc",
                "avoid_strategy": "Avoid this"
            }
        ]

        patterns = await extractor.extract_anti_patterns(db, 1, "test_niche")

        assert len(patterns) == 1
        assert patterns[0].pattern_type == PatternType.NEGATIVE_SIGNAL
        assert patterns[0].pattern_name == "Test Anti-Pattern"
        assert patterns[0].business_id == 1

        # Verify db.add was called
        assert db.add.called
        assert db.commit.called

    @pytest.mark.asyncio
    async def test_extract_anti_patterns_none_caption(self):
        """Test robustness with None caption"""
        extractor = PatternExtractor()
        extractor.ai_service = AsyncMock()
        db = AsyncMock(spec=AsyncSession)

        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 100

        mock_post = ScrapedPost(
            id=1,
            caption=None,  # Should not crash
            content_format=ContentFormat.REEL,
            engagement_score=5.0,
            hashtags=[]
        )
        mock_posts_result = MagicMock()
        mock_posts_result.scalars.return_value.all.return_value = [mock_post]

        db.execute.side_effect = [mock_count_result, mock_posts_result]

        extractor.ai_service.analyze_anti_patterns.return_value = []

        await extractor.extract_anti_patterns(db, 1, "test_niche")

        # Should call analyze_anti_patterns without error
        assert extractor.ai_service.analyze_anti_patterns.called
        call_args = extractor.ai_service.analyze_anti_patterns.call_args
        posts_data = call_args[0][0]
        assert posts_data[0]["caption"] is None
        # Note: The fix is inside AIService._build_anti_pattern_prompt, but we are mocking analyze_anti_patterns here.
        # Ideally we should verify AIService handles it, but that would require testing AIService directly.
        # Since I modified AIService directly, I should check if I can test AIService.

from app.services.ai_service import AIService

class TestAIServiceRobustness:
    def test_build_anti_pattern_prompt_none_caption(self):
        service = AIService()
        posts = [
            {"caption": None, "type": "reel", "engagement_score": 1.0},
            {"caption": "Good caption", "type": "image", "engagement_score": 2.0}
        ]

        prompt = service._build_anti_pattern_prompt(posts, "test_niche")

        # Should execute without error
        assert "Good caption" in prompt
        # Ensure json.dumps inside didn't crash
