import pytest
from sqlalchemy import create_engine, inspect
from app.models.scraped_post import ScrapedPost
from app.core.database import Base

def test_scraped_post_indexes():
    """Verify that ScrapedPost table has the expected composite indexes."""
    # Create in-memory SQLite DB
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    # Inspect indexes
    inspector = inspect(engine)
    indexes = inspector.get_indexes("scraped_posts")

    # Extract index names and columns
    index_map = {idx['name']: idx['column_names'] for idx in indexes}

    print(f"\nFound indexes: {index_map}")

    # Verify composite indexes exist
    assert 'ix_scraped_posts_competitor_engagement' in index_map, \
        "Missing index: ix_scraped_posts_competitor_engagement"
    assert index_map['ix_scraped_posts_competitor_engagement'] == ['competitor_id', 'engagement_score'], \
        "Incorrect columns for ix_scraped_posts_competitor_engagement"

    assert 'ix_scraped_posts_competitor_platform_id' in index_map, \
        "Missing index: ix_scraped_posts_competitor_platform_id"
    assert index_map['ix_scraped_posts_competitor_platform_id'] == ['competitor_id', 'platform_post_id'], \
        "Incorrect columns for ix_scraped_posts_competitor_platform_id"
