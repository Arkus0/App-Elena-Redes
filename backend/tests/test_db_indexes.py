import pytest
from app.models.business import Business
from app.models.pattern import ExtractedPattern
from app.models.content import GeneratedContent

def test_business_user_id_index():
    """Verify Business.user_id is indexed for fast lookups by user."""
    # SQLAlchemy columns store index status in .index attribute (if explicitly set)
    # or .primary_key implying an index
    assert Business.user_id.index is True, "Business.user_id should be indexed"

def test_pattern_business_id_index():
    """Verify ExtractedPattern.business_id is indexed for fast lookups by business."""
    assert ExtractedPattern.business_id.index is True, "ExtractedPattern.business_id should be indexed"

def test_content_variation_of_index():
    """Verify GeneratedContent.variation_of is indexed for fast A/B test variant lookups."""
    assert GeneratedContent.variation_of.index is True, "GeneratedContent.variation_of should be indexed"
