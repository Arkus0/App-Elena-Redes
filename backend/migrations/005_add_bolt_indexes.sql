-- 005_add_bolt_indexes.sql
-- Bolt Optimization: Add indexes for faster lookups on frequently filtered columns

-- Index for Competitor lookups by handle (used in existence checks and growth prediction)
CREATE INDEX IF NOT EXISTS ix_competitors_handle ON competitors (handle);

-- Index for ScrapedPost existence checks by platform ID (used in ingestion)
CREATE INDEX IF NOT EXISTS ix_scraped_posts_platform_post_id ON scraped_posts (platform_post_id);
