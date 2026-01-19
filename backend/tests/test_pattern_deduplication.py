"""
Tests for Pattern Extractor Semantic Deduplication
==================================================

Verifies:
1. Intra-batch deduplication
2. Semantic comparison against existing DB patterns
3. Check-Merge-Insert logic (EMA score, timestamps, examples)

Run with: pytest backend/tests/test_pattern_deduplication.py -v
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import pytest
import numpy as np
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.pattern_extractor import PatternExtractor
from app.models.pattern import ExtractedPattern, PatternType

# Mock the embedding extractor module
sys.modules["ml.features_embeddings"] = MagicMock()
from ml.features_embeddings import get_embedding_extractor

@pytest.fixture
def mock_db():
    m = AsyncMock()
    m.add = MagicMock() # db.add is synchronous
    return m

@pytest.fixture
def pattern_extractor():
    return PatternExtractor()

@pytest.fixture
def mock_embedding_extractor():
    mock_extractor = MagicMock()
    # Setup mock to return deterministic embeddings
    # We will use simple 2D vectors for easy similarity calculation
    # Sim(v1, v2) = dot(v1, v2) / (|v1|*|v2|)
    # [1, 0] and [1, 0] -> 1.0
    # [1, 0] and [0.99, 0.01] -> ~0.99
    # [1, 0] and [0, 1] -> 0.0

    def get_raw_embedding(text):
        if "pattern_A" in text:
            return np.array([1.0, 0.0])
        elif "pattern_A_duplicate" in text:
            # Very similar to A: cosine ~ 0.99
            return np.array([0.99, 0.14]) # normalized roughly
        elif "pattern_B" in text:
             # Orthogonal to A
            return np.array([0.0, 1.0])
        else:
            return np.array([0.5, 0.5])

    def transform(raw):
        # Pass through for test
        return raw

    mock_extractor.get_raw_embedding.side_effect = get_raw_embedding
    mock_extractor.transform.side_effect = transform
    mock_extractor.uses_reduction = True

    with patch("app.services.pattern_extractor.get_embedding_extractor", return_value=mock_extractor):
        with patch("app.services.pattern_extractor.EMBEDDINGS_AVAILABLE", True):
            yield mock_extractor

class TestPatternDeduplication:

    def test_compute_similarity(self, pattern_extractor):
        """Test cosine similarity calculation."""
        v1 = [1.0, 0.0]
        v2 = [1.0, 0.0]
        assert pattern_extractor._compute_similarity(v1, v2) > 0.99

        v3 = [0.0, 1.0]
        assert pattern_extractor._compute_similarity(v1, v3) < 0.01

        v4 = [0.707, 0.707] # 45 degrees
        assert 0.70 < pattern_extractor._compute_similarity(v1, v4) < 0.72

    def test_intra_batch_deduplication(self, pattern_extractor, mock_embedding_extractor):
        """Test that duplicates within a batch are merged."""

        # Create patterns
        p1 = ExtractedPattern(description="This is pattern_A description", examples=["ex1"])
        p2 = ExtractedPattern(description="This is pattern_A_duplicate description", examples=["ex2"]) # Similar to p1
        p3 = ExtractedPattern(description="This is pattern_B description", examples=["ex3"]) # Distinct

        patterns = [p1, p2, p3]

        # Run dedup
        unique = pattern_extractor._deduplicate_batch(patterns)

        # Should have p1 (merged) and p3. p2 should be gone.
        assert len(unique) == 2
        assert p1 in unique
        assert p3 in unique
        assert p2 not in unique

        # Check merging of examples
        # p1 should now have ex1 + ex2
        assert "ex1" in p1.examples
        assert "ex2" in p1.examples
        assert p1.usage_count == 2 # 1 (default) + 1 (merged) if logic implemented

    @pytest.mark.asyncio
    async def test_check_merge_flow(self, pattern_extractor, mock_db, mock_embedding_extractor):
        """Test deduplication against existing DB patterns."""

        # Mock DB setup
        # Existing pattern in DB (pattern_A)
        existing_pattern = ExtractedPattern(
            id=1,
            description="Existing pattern_A",
            embedding=[1.0, 0.0],
            avg_engagement_score=50.0,
            usage_count=5,
            examples=["old_ex"],
            last_active_at=datetime(2020, 1, 1)
        )

        # Mock DB select result
        mock_result = MagicMock()
        mock_result.scalars().return_value.all.return_value = [existing_pattern]
        mock_db.execute.return_value = mock_result

        # New pattern from AI (similar to A)
        new_pattern = ExtractedPattern(
            description="New pattern_A_duplicate",
            avg_engagement_score=80.0,
            examples=["new_ex"]
        )

        # We need to bypass the extract_patterns... method's initial AI call part
        # and just test the saving logic. But the logic is inside extract_patterns_from_competitor.
        # So we'll have to mock ai_service and everything.

        # Or better, we can invoke the dedup logic if we extracted it to a method.
        # Since I didn't extract the "Save" block to a separate method, I have to test `extract_patterns_from_competitor`.

        # Mock AI Service
        pattern_extractor.ai_service.analyze_posts_for_patterns = AsyncMock(return_value={
            "hook_patterns": [{
                "pattern_name": "Test Hook",
                "description": "New pattern_A_duplicate", # This will trigger the similarity match
                "avg_engagement": 80.0,
                "examples": ["new_ex"]
            }]
        })

        # Mock Competitor and Posts queries
        mock_post = MagicMock()
        mock_post.id = 100
        mock_post.engagement_score = 80

        mock_competitor = MagicMock()
        mock_competitor.id = 1
        mock_competitor.platform.value = "instagram"

        # Setup DB side effects for queries
        # 1. ScrapedPost query
        # 2. Competitor query
        # 3. Existing Patterns query

        async def side_effect(query):
            query_str = str(query)
            if "scraped_posts" in query_str:
                m = MagicMock()
                m.scalars().all.return_value = [mock_post]
                return m
            elif "competitors" in query_str:
                m = MagicMock()
                m.scalar_one_or_none.return_value = mock_competitor
                return m
            elif "extracted_patterns" in query_str:
                m = MagicMock()
                m.scalars().all.return_value = [existing_pattern]
                return m
            return MagicMock()

        mock_db.execute.side_effect = side_effect

        # Run
        result = await pattern_extractor.extract_patterns_from_competitor(
            mock_db, competitor_id=1, business_id=99, business_type="test"
        )

        # Assertions

        # 1. Should NOT insert new pattern (patterns_to_add should be empty or handled internally)
        # result contains updated patterns too.

        # 2. Existing pattern should be updated
        # EMA Score: (50 * 0.3) + (80 * 0.7) = 15 + 56 = 71.0
        assert existing_pattern.avg_engagement_score == 71.0

        # Usage Count: 5 + 1 = 6
        assert existing_pattern.usage_count == 6

        # Examples: ["old_ex", "new_ex"]
        assert "new_ex" in existing_pattern.examples

        # Timestamp updated
        assert existing_pattern.last_active_at > datetime(2021, 1, 1)

    @pytest.mark.asyncio
    async def test_insert_new_flow(self, pattern_extractor, mock_db, mock_embedding_extractor):
        """Test inserting a completely new pattern."""

        # No existing patterns
        async def side_effect(query):
            query_str = str(query)
            if "scraped_posts" in query_str:
                m = MagicMock()
                m.scalars().all.return_value = [MagicMock(id=1, engagement_score=50)]
                return m
            elif "competitors" in query_str:
                m = MagicMock()
                m.scalar_one_or_none.return_value = MagicMock(platform=MagicMock(value="instagram"))
                return m
            elif "extracted_patterns" in query_str:
                m = MagicMock()
                m.scalars().all.return_value = [] # Empty
                return m
            return MagicMock()

        mock_db.execute.side_effect = side_effect

        pattern_extractor.ai_service.analyze_posts_for_patterns = AsyncMock(return_value={
            "hook_patterns": [{
                "pattern_name": "Unique Hook",
                "description": "This is pattern_B description", # Orthogonal/Unique
                "avg_engagement": 60.0,
                "examples": ["ex_B"]
            }]
        })

        result = await pattern_extractor.extract_patterns_from_competitor(
            mock_db, competitor_id=1, business_id=99, business_type="test"
        )

        # Should call db.add
        mock_db.add.assert_called()

        # Verify the object added
        added_pattern = mock_db.add.call_args[0][0]
        assert added_pattern.pattern_name == "Unique Hook"
        assert added_pattern.usage_count == 1
        assert added_pattern.last_active_at is not None
