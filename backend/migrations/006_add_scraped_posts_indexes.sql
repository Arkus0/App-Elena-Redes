-- Migration: Add composite indexes for ScrapedPost
-- Description: Adds composite indexes to optimize frequent filtering and sorting queries
-- Author: Bolt
-- Date: 2024-05-24

-- Add composite index for filtering by competitor and sorting by engagement (Top Posts query)
CREATE INDEX IF NOT EXISTS ix_scraped_posts_competitor_engagement ON scraped_posts (competitor_id, engagement_score);

-- Add composite index for filtering by competitor and platform_post_id (Ingest existence check)
CREATE INDEX IF NOT EXISTS ix_scraped_posts_competitor_platform_id ON scraped_posts (competitor_id, platform_post_id);
