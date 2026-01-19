-- Migration: Add database indexes for performance optimization
-- Description: Adds indexes to foreign key columns in scraped_posts, competitors, and generated_content tables
-- Author: Bolt
-- Date: 2024-05-23

-- Add index to ScrapedPost.competitor_id
CREATE INDEX IF NOT EXISTS ix_scraped_posts_competitor_id ON scraped_posts (competitor_id);

-- Add index to Competitor.business_id
CREATE INDEX IF NOT EXISTS ix_competitors_business_id ON competitors (business_id);

-- Add index to GeneratedContent.business_id
CREATE INDEX IF NOT EXISTS ix_generated_content_business_id ON generated_content (business_id);

-- Add index to GeneratedContent.calendar_id
CREATE INDEX IF NOT EXISTS ix_generated_content_calendar_id ON generated_content (calendar_id);

-- Add index to ContentCalendar.business_id
CREATE INDEX IF NOT EXISTS ix_content_calendars_business_id ON content_calendars (business_id);
