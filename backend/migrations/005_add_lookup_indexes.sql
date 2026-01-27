-- Migration: Add lookup indexes for deduplication and search
-- Description: Adds indexes to platform_post_id in scraped_posts and handle in competitors
-- Author: Bolt
-- Date: 2024-05-24

-- Add index to ScrapedPost.platform_post_id for deduplication checks
CREATE INDEX IF NOT EXISTS ix_scraped_posts_platform_post_id ON scraped_posts (platform_post_id);

-- Add index to Competitor.handle for faster lookups
CREATE INDEX IF NOT EXISTS ix_competitors_handle ON competitors (handle);
