"""
A/B Test Logging Models
=======================

Database models for tracking predictions vs actual performance.
Enables continuous model improvement through human-in-the-loop feedback.

Tables:
- ABTestLog: Tracks predicted RPI vs actual engagement for published content
- PredictionLog: Anonymized prediction audit log with hashed usernames

GDPR Compliance:
- Usernames are SHA-256 hashed for anonymization
- Timestamps allow for data retention policy enforcement
"""

import hashlib
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship

from app.core.database import Base


class ABTestLog(Base):
    """
    A/B Test Log - Tracks predictions vs actual results.

    Used for:
    - Model performance monitoring
    - Identifying high-delta samples for retraining
    - A/B test result analysis

    Fields:
    - post_id: Reference to GeneratedContent
    - predicted_rpi: RPI score predicted at generation time
    - actual_engagement: Actual engagement metrics after publication
    - published_date: When the content was published
    """
    __tablename__ = "ab_test_logs"

    id = Column(Integer, primary_key=True, index=True)

    # Reference to generated content
    post_id = Column(Integer, ForeignKey("generated_content.id"), nullable=False, index=True)

    # Business context
    business_id = Column(Integer, ForeignKey("businesses.id"), nullable=False, index=True)

    # Prediction data
    predicted_rpi = Column(Float, nullable=False)
    predicted_engagement_score = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)

    # Actual performance data
    actual_engagement = Column(Float, nullable=True)
    actual_likes = Column(Integer, nullable=True)
    actual_comments = Column(Integer, nullable=True)
    actual_saves = Column(Integer, nullable=True)
    actual_shares = Column(Integer, nullable=True)
    actual_views = Column(Integer, nullable=True)
    actual_reach = Column(Integer, nullable=True)

    # Calculated metrics
    delta_percent = Column(Float, nullable=True)  # (actual - predicted) / predicted * 100
    is_high_priority = Column(Boolean, default=False)  # True if delta > 20%

    # Content metadata
    content_format = Column(String(50), nullable=True)
    platform = Column(String(50), nullable=True)

    # Timestamps
    predicted_at = Column(DateTime, default=datetime.utcnow)
    published_date = Column(DateTime, nullable=True)
    metrics_collected_at = Column(DateTime, nullable=True)

    # Status
    status = Column(String(20), default="pending")  # pending, published, collected, analyzed

    # Relationships
    post = relationship("GeneratedContent", back_populates="ab_test_logs")

    def calculate_delta(self) -> Optional[float]:
        """Calculate delta between predicted and actual engagement."""
        if self.actual_engagement is not None and self.predicted_rpi > 0:
            self.delta_percent = ((self.actual_engagement - self.predicted_rpi) / self.predicted_rpi) * 100
            self.is_high_priority = abs(self.delta_percent) > 20.0
            return self.delta_percent
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "post_id": self.post_id,
            "business_id": self.business_id,
            "predicted_rpi": self.predicted_rpi,
            "actual_engagement": self.actual_engagement,
            "delta_percent": self.delta_percent,
            "is_high_priority": self.is_high_priority,
            "content_format": self.content_format,
            "platform": self.platform,
            "published_date": self.published_date.isoformat() if self.published_date else None,
            "status": self.status,
        }


class PredictionLog(Base):
    """
    Prediction Audit Log - Anonymized logging for governance.

    Stores all predictions with:
    - Anonymized features (hashed usernames)
    - Timestamp for audit trail
    - Prediction scores and explanations

    GDPR Compliance:
    - All potentially identifying information is hashed
    - Supports data retention policies
    - Enables right-to-erasure through hash-based lookup
    """
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)

    # Anonymized identifiers (SHA-256 hashed)
    user_hash = Column(String(64), nullable=True, index=True)  # Hashed user ID
    business_hash = Column(String(64), nullable=True, index=True)  # Hashed business ID
    session_hash = Column(String(64), nullable=True)  # Hashed session ID

    # Prediction details
    prediction_type = Column(String(50), nullable=False)  # "rpi_score", "format_recommendation", "engagement"
    predicted_score = Column(Float, nullable=False)
    confidence = Column(Float, nullable=True)

    # Anonymized feature summary (no PII)
    features_summary = Column(JSON, nullable=True)  # Aggregated feature stats, no raw content

    # SHAP explanation (Top 5 factors)
    shap_explanation = Column(Text, nullable=True)
    top_factors = Column(JSON, nullable=True)

    # Request context (anonymized)
    platform = Column(String(50), nullable=True)
    content_format = Column(String(50), nullable=True)
    business_type = Column(String(50), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Model version for reproducibility
    model_version = Column(String(50), nullable=True)

    @staticmethod
    def hash_identifier(identifier: str) -> str:
        """
        Hash an identifier for anonymization.

        Uses SHA-256 with a salt for security.
        """
        if not identifier:
            return ""
        # Add salt for security (in production, use a secure secret)
        salt = "brandpulse_ai_gdpr_salt_2024"
        salted = f"{salt}:{identifier}"
        return hashlib.sha256(salted.encode()).hexdigest()

    @classmethod
    def create_anonymized(
        cls,
        user_id: Optional[int],
        business_id: Optional[int],
        prediction_type: str,
        predicted_score: float,
        confidence: Optional[float] = None,
        features_summary: Optional[dict] = None,
        shap_explanation: Optional[str] = None,
        top_factors: Optional[list] = None,
        platform: Optional[str] = None,
        content_format: Optional[str] = None,
        business_type: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> "PredictionLog":
        """
        Create a new prediction log with anonymized identifiers.
        """
        return cls(
            user_hash=cls.hash_identifier(str(user_id)) if user_id else None,
            business_hash=cls.hash_identifier(str(business_id)) if business_id else None,
            prediction_type=prediction_type,
            predicted_score=predicted_score,
            confidence=confidence,
            features_summary=features_summary,
            shap_explanation=shap_explanation,
            top_factors=top_factors,
            platform=platform,
            content_format=content_format,
            business_type=business_type,
            model_version=model_version,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses (anonymized)."""
        return {
            "id": self.id,
            "prediction_type": self.prediction_type,
            "predicted_score": self.predicted_score,
            "confidence": self.confidence,
            "shap_explanation": self.shap_explanation,
            "top_factors": self.top_factors,
            "platform": self.platform,
            "content_format": self.content_format,
            "business_type": self.business_type,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "model_version": self.model_version,
        }


class ABTestExperiment(Base):
    """
    A/B Test Experiment - Parent container for bandit tests.
    """
    __tablename__ = "ab_test_experiments"

    id = Column(Integer, primary_key=True, index=True)
    business_id = Column(Integer, ForeignKey("businesses.id"), nullable=False, index=True)
    test_name = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    variants = relationship("ABTestVariant", back_populates="experiment", cascade="all, delete-orphan")


class ABTestVariant(Base):
    """
    A/B Test Variant - Specific option in the bandit (e.g. 'reel', 'carousel').
    Stores Alpha/Beta parameters for Thompson Sampling.
    """
    __tablename__ = "ab_test_variants"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("ab_test_experiments.id"), nullable=False, index=True)
    variant_name = Column(String(50), nullable=False)
    alpha_param = Column(Integer, default=1)  # Success count
    beta_param = Column(Integer, default=1)   # Failure count

    # Relationships
    experiment = relationship("ABTestExperiment", back_populates="variants")
