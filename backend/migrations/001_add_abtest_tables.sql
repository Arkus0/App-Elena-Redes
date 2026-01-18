-- Migration: Add A/B Test Logging Tables
-- Version: 001
-- Date: 2026-01-18
-- Description: Adds tables for A/B test logging and prediction audit trails

-- =============================================================================
-- Table: ab_test_logs
-- =============================================================================
-- Tracks predictions vs actual results for published content.
-- Used for model performance monitoring and identifying high-delta samples.

CREATE TABLE IF NOT EXISTS ab_test_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- References
    post_id INTEGER NOT NULL REFERENCES generated_content(id) ON DELETE CASCADE,
    business_id INTEGER NOT NULL REFERENCES businesses(id) ON DELETE CASCADE,

    -- Prediction data
    predicted_rpi REAL NOT NULL,
    predicted_engagement_score REAL,
    confidence_score REAL,

    -- Actual performance data
    actual_engagement REAL,
    actual_likes INTEGER,
    actual_comments INTEGER,
    actual_saves INTEGER,
    actual_shares INTEGER,
    actual_views INTEGER,
    actual_reach INTEGER,

    -- Calculated metrics
    delta_percent REAL,  -- (actual - predicted) / predicted * 100
    is_high_priority BOOLEAN DEFAULT 0,  -- True if delta > 20%

    -- Content metadata
    content_format VARCHAR(50),
    platform VARCHAR(50),

    -- Timestamps
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    published_date TIMESTAMP,
    metrics_collected_at TIMESTAMP,

    -- Status: pending, published, collected, analyzed
    status VARCHAR(20) DEFAULT 'pending'
);

-- Indexes for ab_test_logs
CREATE INDEX IF NOT EXISTS idx_abtest_post_id ON ab_test_logs(post_id);
CREATE INDEX IF NOT EXISTS idx_abtest_business_id ON ab_test_logs(business_id);
CREATE INDEX IF NOT EXISTS idx_abtest_high_priority ON ab_test_logs(is_high_priority);
CREATE INDEX IF NOT EXISTS idx_abtest_status ON ab_test_logs(status);


-- =============================================================================
-- Table: prediction_logs
-- =============================================================================
-- Anonymized prediction audit log for governance and GDPR compliance.
-- Usernames are SHA-256 hashed for privacy protection.

CREATE TABLE IF NOT EXISTS prediction_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Anonymized identifiers (SHA-256 hashed)
    user_hash VARCHAR(64),
    business_hash VARCHAR(64),
    session_hash VARCHAR(64),

    -- Prediction details
    prediction_type VARCHAR(50) NOT NULL,  -- rpi_score, format_recommendation, engagement
    predicted_score REAL NOT NULL,
    confidence REAL,

    -- Anonymized feature summary (no PII)
    features_summary JSON,

    -- SHAP explanation (Top 5 factors)
    shap_explanation TEXT,
    top_factors JSON,

    -- Request context (anonymized)
    platform VARCHAR(50),
    content_format VARCHAR(50),
    business_type VARCHAR(50),

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Model version for reproducibility
    model_version VARCHAR(50)
);

-- Indexes for prediction_logs
CREATE INDEX IF NOT EXISTS idx_predlog_user_hash ON prediction_logs(user_hash);
CREATE INDEX IF NOT EXISTS idx_predlog_business_hash ON prediction_logs(business_hash);
CREATE INDEX IF NOT EXISTS idx_predlog_created_at ON prediction_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_predlog_prediction_type ON prediction_logs(prediction_type);


-- =============================================================================
-- Add GDPR consent columns to businesses table (if not exists)
-- =============================================================================
-- Note: SQLite doesn't support IF NOT EXISTS for ALTER TABLE ADD COLUMN
-- Run this manually if the columns don't exist:

-- ALTER TABLE businesses ADD COLUMN gdpr_consent BOOLEAN DEFAULT 0;
-- ALTER TABLE businesses ADD COLUMN data_processing_consent BOOLEAN DEFAULT 0;
-- ALTER TABLE businesses ADD COLUMN marketing_consent BOOLEAN DEFAULT 0;
-- ALTER TABLE businesses ADD COLUMN consent_date TIMESTAMP;


-- =============================================================================
-- PostgreSQL version (if using PostgreSQL instead of SQLite)
-- =============================================================================
-- Uncomment below for PostgreSQL:

/*
CREATE TABLE IF NOT EXISTS ab_test_logs (
    id SERIAL PRIMARY KEY,
    post_id INTEGER NOT NULL REFERENCES generated_content(id) ON DELETE CASCADE,
    business_id INTEGER NOT NULL REFERENCES businesses(id) ON DELETE CASCADE,
    predicted_rpi DOUBLE PRECISION NOT NULL,
    predicted_engagement_score DOUBLE PRECISION,
    confidence_score DOUBLE PRECISION,
    actual_engagement DOUBLE PRECISION,
    actual_likes INTEGER,
    actual_comments INTEGER,
    actual_saves INTEGER,
    actual_shares INTEGER,
    actual_views INTEGER,
    actual_reach INTEGER,
    delta_percent DOUBLE PRECISION,
    is_high_priority BOOLEAN DEFAULT FALSE,
    content_format VARCHAR(50),
    platform VARCHAR(50),
    predicted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    published_date TIMESTAMP WITH TIME ZONE,
    metrics_collected_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS prediction_logs (
    id SERIAL PRIMARY KEY,
    user_hash VARCHAR(64),
    business_hash VARCHAR(64),
    session_hash VARCHAR(64),
    prediction_type VARCHAR(50) NOT NULL,
    predicted_score DOUBLE PRECISION NOT NULL,
    confidence DOUBLE PRECISION,
    features_summary JSONB,
    shap_explanation TEXT,
    top_factors JSONB,
    platform VARCHAR(50),
    content_format VARCHAR(50),
    business_type VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    model_version VARCHAR(50)
);

-- Add GDPR columns to businesses
ALTER TABLE businesses ADD COLUMN IF NOT EXISTS gdpr_consent BOOLEAN DEFAULT FALSE;
ALTER TABLE businesses ADD COLUMN IF NOT EXISTS data_processing_consent BOOLEAN DEFAULT FALSE;
ALTER TABLE businesses ADD COLUMN IF NOT EXISTS marketing_consent BOOLEAN DEFAULT FALSE;
ALTER TABLE businesses ADD COLUMN IF NOT EXISTS consent_date TIMESTAMP WITH TIME ZONE;
*/
