-- Migration: Add index to platform_post_id in scraped_posts
-- Description: Adds index to platform_post_id to optimize deduplication lookups during ingestion
-- Author: Bolt
-- Date: 2024-05-24

CREATE INDEX IF NOT EXISTS ix_scraped_posts_platform_post_id ON scraped_posts (platform_post_id);
