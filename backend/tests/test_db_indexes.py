from app.models.competitor import Competitor
from app.models.scraped_post import ScrapedPost
import pytest

def test_competitor_handle_index():
    """Verify that Competitor.handle is indexed for fast lookups."""
    # SQLAlchemy Column object has an 'index' attribute which is True if index=True is set
    assert Competitor.handle.index is True, "Competitor.handle should be indexed"

def test_scraped_post_platform_id_index():
    """Verify that ScrapedPost.platform_post_id is indexed for fast lookups."""
    assert ScrapedPost.platform_post_id.index is True, "ScrapedPost.platform_post_id should be indexed"
