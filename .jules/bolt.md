## 2024-05-23 - Database Indexing Strategy
**Learning:** Adding indexes to Foreign Keys in SQLAlchemy models is critical for query performance, especially for filtering large tables like `scraped_posts`.
**Action:** Always check `ForeignKey` definitions and consider adding `index=True` if the column is frequently used in `WHERE` clauses.

## 2024-05-24 - High Cardinality String Lookups
**Learning:** String identifiers like `platform_post_id` used for deduplication/existence checks MUST be indexed. A 100k row table showed ~1400x speedup (13ms -> 0.009ms) with an index.
**Action:** Audit all `unique` or "lookup" columns in models, not just Foreign Keys, for `index=True`.
