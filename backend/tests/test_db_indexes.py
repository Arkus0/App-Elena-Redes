"""
Database Index Verification Tests
=================================
Verifies that critical database indexes are present, especially those
optimized for performance (like platform_post_id).
"""

import pytest
from sqlalchemy import create_engine, inspect
from app.core.database import Base
from app.models.scraped_post import ScrapedPost
# Import other models to ensure they are registered with Base
from app.models.competitor import Competitor
from app.models.business import Business
from app.models.user import User

def test_scraped_post_platform_post_id_index():
    """
    Verify that an index exists on scraped_posts.platform_post_id.
    This index is critical for deduplication performance.
    """
    # Create an in-memory SQLite database
    engine = create_engine("sqlite:///:memory:")

    # Create all tables defined in Base
    Base.metadata.create_all(engine)

    # Inspect the database
    inspector = inspect(engine)

    # Get indexes for scraped_posts table
    indexes = inspector.get_indexes("scraped_posts")

    # Check if platform_post_id is indexed
    platform_id_indexed = False
    for index in indexes:
        # Check if this index covers 'platform_post_id'
        if "platform_post_id" in index["column_names"]:
            platform_id_indexed = True
            break

    assert platform_id_indexed, "Index on scraped_posts.platform_post_id is missing!"

    # Optionally verify index name if expected (SQLAlchemy auto-naming might vary)
    # print([i['name'] for i in indexes])

if __name__ == "__main__":
    pytest.main([__file__])
