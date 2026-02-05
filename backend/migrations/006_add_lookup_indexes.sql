-- Migration: Add composite indexes for ScrapedPost lookups and ranking
-- Description: Optimizes ingestion existence checks and "Top Posts" queries
-- Author: Bolt
-- Date: 2024-05-25

-- Add composite index for ingestion lookup (competitor_id + platform_post_id)
-- This optimizes: WHERE competitor_id = ? AND platform_post_id = ?
CREATE INDEX IF NOT EXISTS ix_scraped_posts_comp_platform ON scraped_posts (competitor_id, platform_post_id);

-- Add composite index for ranking (competitor_id + engagement_score DESC)
-- This optimizes: WHERE competitor_id = ? ORDER BY engagement_score DESC
-- Note: SQLite ignores DESC in index definition unless compiled with proper support,
-- but it's still valid syntax and helps on other DBs like Postgres.
CREATE INDEX IF NOT EXISTS ix_scraped_posts_comp_ranking ON scraped_posts (competitor_id, engagement_score);
