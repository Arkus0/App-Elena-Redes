## 2024-05-23 - Database Indexing Strategy
**Learning:** Adding indexes to Foreign Keys in SQLAlchemy models is critical for query performance, especially for filtering large tables like `scraped_posts`.
**Action:** Always check `ForeignKey` definitions and consider adding `index=True` if the column is frequently used in `WHERE` clauses.

## 2024-05-24 - Missing Root Entity Indexes
**Learning:** Core root entities like `Business` (linked to `User`) often miss indexes on the ownership column (`user_id`), causing full table scans on every dashboard load.
**Action:** Always verify that the primary "ownership" foreign key (e.g., `user_id`, `business_id`) is explicitly indexed in the model definition.
