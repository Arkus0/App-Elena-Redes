import pytest
from sqlalchemy import create_engine, inspect
from app.core.database import Base
from app.models.scraped_post import ScrapedPost  # Import to register model

def test_scraped_posts_indexes():
    """
    Verify that ScrapedPost model has the correct indexes,
    specifically the composite index for competitor filtering optimization.
    """
    # Use in-memory SQLite for fast schema verification
    engine = create_engine("sqlite:///:memory:")

    # Create all tables defined in Base
    Base.metadata.create_all(engine)

    # Inspect the created schema
    inspector = inspect(engine)
    indexes = inspector.get_indexes("scraped_posts")

    # Print indexes for debugging if test fails
    print(f"\nFound indexes on scraped_posts: {[idx['name'] for idx in indexes]}")

    # Check for composite index
    composite_index = next(
        (idx for idx in indexes if idx["name"] == "ix_scraped_posts_competitor_engagement"),
        None
    )

    assert composite_index is not None, "Composite index 'ix_scraped_posts_competitor_engagement' not found"

    # Verify columns
    # Note: SQLite might not preserve order in index definition introspection in some versions,
    # but usually it does.
    expected_columns = ["competitor_id", "engagement_score"]
    assert composite_index["column_names"] == expected_columns, \
        f"Index columns mismatch. Expected {expected_columns}, got {composite_index['column_names']}"

    print("\n✅ Composite index verified successfully.")
