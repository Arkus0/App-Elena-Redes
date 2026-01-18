## 2024-05-23 - Database Indexing Strategy
**Learning:** Adding indexes to Foreign Keys in SQLAlchemy models is critical for query performance, especially for filtering large tables like `scraped_posts`.
**Action:** Always check `ForeignKey` definitions and consider adding `index=True` if the column is frequently used in `WHERE` clauses.
