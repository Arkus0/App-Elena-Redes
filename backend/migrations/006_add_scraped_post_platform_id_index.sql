-- Migration: Add database index to ScrapedPost.platform_post_id
-- Description: Adds index to platform_post_id in scraped_posts table for faster deduplication
-- Author: Bolt
-- Date: 2024-05-24

-- Add index to ScrapedPost.platform_post_id
CREATE INDEX IF NOT EXISTS ix_scraped_posts_platform_post_id ON scraped_posts (platform_post_id);
