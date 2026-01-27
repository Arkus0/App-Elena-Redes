-- Migration: Add index to Competitor.handle
-- Description: Adds index to the handle column in competitors table for faster lookups
-- Author: Bolt
-- Date: 2024-05-24

CREATE INDEX IF NOT EXISTS ix_competitors_handle ON competitors (handle);
