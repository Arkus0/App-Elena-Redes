-- Migration: Add missing indexes to Foreign Keys
-- Description: Adds indexes to user_id in businesses, business_id in extracted_patterns, and variation_of in generated_content
-- Author: Bolt
-- Date: 2024-05-24

-- Add index to Business.user_id
CREATE INDEX IF NOT EXISTS ix_businesses_user_id ON businesses (user_id);

-- Add index to ExtractedPattern.business_id
CREATE INDEX IF NOT EXISTS ix_extracted_patterns_business_id ON extracted_patterns (business_id);

-- Add index to GeneratedContent.variation_of
CREATE INDEX IF NOT EXISTS ix_generated_content_variation_of ON generated_content (variation_of);
