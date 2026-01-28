## 2024-05-23 - Database Indexing Strategy
**Learning:** Adding indexes to Foreign Keys in SQLAlchemy models is critical for query performance, especially for filtering large tables like `scraped_posts`.
**Action:** Always check `ForeignKey` definitions and consider adding `index=True` if the column is frequently used in `WHERE` clauses.

## 2024-05-25 - Blind Ingestion Anti-Pattern
**Learning:** Found critical anti-pattern in `scrape_single_competitor`: iterating and inserting scraped posts without checking if they already exist. This leads to massive data duplication and database bloat.
**Action:** When ingesting data from external sources, ALWAYS implement an existence check (or upsert logic) using a unique identifier (like `platform_post_id`). Ensure that identifier is indexed!
