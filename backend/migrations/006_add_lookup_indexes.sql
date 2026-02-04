-- Migration: Add lookup indexes for performance optimization
-- Description: Adds indexes to Competitor.handle and ScrapedPost.platform_post_id to speed up lookups and deduplication
-- Author: Bolt
-- Date: 2024-05-24

-- Add index to Competitor.handle for faster lookup by username
CREATE INDEX IF NOT EXISTS ix_competitors_handle ON competitors (handle);

-- Add index to ScrapedPost.platform_post_id for faster deduplication checks
CREATE INDEX IF NOT EXISTS ix_scraped_posts_platform_post_id ON scraped_posts (platform_post_id);
