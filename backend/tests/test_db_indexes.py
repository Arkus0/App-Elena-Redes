"""
Test Database Indexes
=====================

Verifies that critical database indexes are correctly defined in the models.
"""

import pytest
from sqlalchemy import create_engine, inspect
from app.core.database import Base
from app.models.competitor import Competitor

def test_competitor_handle_index_exists():
    """Verify that an index exists on the Competitor.handle column."""
    # Create in-memory SQLite database
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    # Inspect the database
    inspector = inspect(engine)
    indexes = inspector.get_indexes("competitors")

    # Look for an index on the 'handle' column
    handle_index = None
    for index in indexes:
        if "handle" in index["column_names"]:
            handle_index = index
            break

    assert handle_index is not None, "Index on 'handle' column in 'competitors' table is missing"
    # SQLite might return 0/1 for boolean columns in inspection
    is_unique = handle_index["unique"]
    assert not is_unique, f"Index on 'handle' should not be unique (scoped by business logic). Got unique={is_unique}"

    print(f"\nFound index on Competitor.handle: {handle_index['name']}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
