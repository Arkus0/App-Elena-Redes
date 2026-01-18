-- Migration: Add Multi-Objective KPI Weights Tables
-- Date: 2025-01-18
-- Description: Adds tables for configurable engagement weights and multi-output model metrics

-- =============================================================================
-- Table: user_kpi_weights
-- Stores user/business-specific engagement weights for multi-objective predictions
-- =============================================================================

CREATE TABLE IF NOT EXISTS user_kpi_weights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    business_id INTEGER NOT NULL,
    niche VARCHAR(100),

    -- Engagement metric weights (0.0 to 20.0)
    likes_weight REAL NOT NULL DEFAULT 1.0,
    comments_weight REAL NOT NULL DEFAULT 2.0,
    shares_weight REAL NOT NULL DEFAULT 10.0,
    saves_weight REAL NOT NULL DEFAULT 5.0,
    views_weight REAL NOT NULL DEFAULT 3.0,

    -- Template info
    template_name VARCHAR(50),
    description VARCHAR(500),

    -- Status
    is_active BOOLEAN NOT NULL DEFAULT 1,

    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Additional config (JSON)
    extra_config TEXT,

    -- Foreign keys
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (business_id) REFERENCES businesses(id) ON DELETE CASCADE
);

-- Indexes for faster lookups
CREATE INDEX IF NOT EXISTS idx_kpi_weights_user_id ON user_kpi_weights(user_id);
CREATE INDEX IF NOT EXISTS idx_kpi_weights_business_id ON user_kpi_weights(business_id);
CREATE INDEX IF NOT EXISTS idx_kpi_weights_niche ON user_kpi_weights(niche);
CREATE INDEX IF NOT EXISTS idx_kpi_weights_active ON user_kpi_weights(is_active);

-- Unique constraint: one active config per user+business+niche
CREATE UNIQUE INDEX IF NOT EXISTS uq_user_business_niche_active
ON user_kpi_weights(user_id, business_id, niche)
WHERE is_active = 1;

-- =============================================================================
-- Table: multi_output_model_metrics
-- Stores training metrics for multi-output ML models
-- =============================================================================

CREATE TABLE IF NOT EXISTS multi_output_model_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    niche VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,

    -- Per-metric training metrics
    likes_rmse REAL,
    likes_r2 REAL,
    comments_rmse REAL,
    comments_r2 REAL,
    shares_rmse REAL,
    shares_r2 REAL,
    saves_rmse REAL,
    saves_r2 REAL,
    views_rmse REAL,
    views_r2 REAL,

    -- Overall metrics
    combined_rmse REAL,
    combined_r2 REAL,

    -- Training info
    training_samples INTEGER,
    feature_count INTEGER,
    is_multi_output BOOLEAN DEFAULT 1,

    -- Timestamp
    trained_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Full metrics (JSON)
    full_metrics TEXT
);

CREATE INDEX IF NOT EXISTS idx_model_metrics_niche ON multi_output_model_metrics(niche);
CREATE INDEX IF NOT EXISTS idx_model_metrics_version ON multi_output_model_metrics(model_version);

-- =============================================================================
-- Trigger: Auto-update updated_at on user_kpi_weights
-- =============================================================================

CREATE TRIGGER IF NOT EXISTS update_kpi_weights_timestamp
AFTER UPDATE ON user_kpi_weights
FOR EACH ROW
BEGIN
    UPDATE user_kpi_weights SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
END;

-- =============================================================================
-- Default data: Insert default templates as reference (optional)
-- =============================================================================

-- Note: Templates are defined in code, not in DB, for easier maintenance
-- This table only stores user-customized weights

-- Migration complete
