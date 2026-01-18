-- Migration: 002_add_competitor_activity_fields
-- Description: Add activity tracking fields to competitors table
-- Purpose: Filter inactive competitors to ensure extension can capture recent content
-- Date: 2026-01-18

-- Add activity tracking columns to competitors table
-- These fields help identify inactive accounts that won't provide useful content

-- last_post_date: Date of the most recent post from this competitor
ALTER TABLE competitors ADD COLUMN last_post_date TIMESTAMP NULL;

-- days_since_last_post: Cached value for efficient querying
ALTER TABLE competitors ADD COLUMN days_since_last_post INTEGER NULL;

-- activity_status: Human-readable status (active, moderately_active, inactive, dormant, unknown)
ALTER TABLE competitors ADD COLUMN activity_status VARCHAR(20) DEFAULT 'unknown';

-- activity_score: Numeric score 0-100 combining recency and frequency
ALTER TABLE competitors ADD COLUMN activity_score INTEGER DEFAULT 0;

-- posting_frequency: Average posts per month
ALTER TABLE competitors ADD COLUMN posting_frequency INTEGER NULL;

-- Create index for efficient filtering by activity
CREATE INDEX IF NOT EXISTS idx_competitors_activity_status ON competitors(activity_status);
CREATE INDEX IF NOT EXISTS idx_competitors_activity_score ON competitors(activity_score);
CREATE INDEX IF NOT EXISTS idx_competitors_days_since_last_post ON competitors(days_since_last_post);
