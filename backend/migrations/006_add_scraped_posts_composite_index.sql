-- Migration: Add composite index for competitor posts filtering
-- Description: Adds composite index on (competitor_id, engagement_score DESC) to optimize "top posts" queries
-- Author: Bolt
-- Date: 2024-05-24

CREATE INDEX IF NOT EXISTS ix_scraped_posts_competitor_engagement ON scraped_posts (competitor_id, engagement_score DESC);
