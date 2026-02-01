-- Migration: Add index to Competitor.handle
-- Description: Adds index to handle column in competitors table for faster lookup
-- Author: Bolt
-- Date: 2024-05-24

CREATE INDEX IF NOT EXISTS ix_competitors_handle ON competitors (handle);
