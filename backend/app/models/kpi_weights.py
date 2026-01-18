"""
User KPI Weights Model - Multi-Objective Engagement Configuration

SQLAlchemy model for storing user/business-specific engagement weights.
Enables personalized RPI calculations based on business goals.

Author: BrandPulse AI
"""

from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, ForeignKey, JSON, UniqueConstraint
from sqlalchemy.orm import relationship
from app.core.database import Base


class UserKPIWeights(Base):
    """
    Stores configurable engagement weights per user and business.

    Each user can have multiple weight configurations:
    - One per business they manage
    - Optional niche-specific overrides

    The weights determine how multi-output predictions are combined
    into a single RPI score.
    """
    __tablename__ = "user_kpi_weights"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    business_id = Column(Integer, ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False, index=True)
    niche = Column(String(100), nullable=True, index=True, comment="Optional niche for niche-specific weights")

    # Individual metric weights (0.0 to 20.0)
    likes_weight = Column(Float, nullable=False, default=1.0, comment="Weight for likes (0-20)")
    comments_weight = Column(Float, nullable=False, default=2.0, comment="Weight for comments (0-20)")
    shares_weight = Column(Float, nullable=False, default=10.0, comment="Weight for shares (0-20)")
    saves_weight = Column(Float, nullable=False, default=5.0, comment="Weight for saves (0-20)")
    views_weight = Column(Float, nullable=False, default=3.0, comment="Weight for views/Reels (0-20)")

    # Template name if created from a template
    template_name = Column(String(50), nullable=True, comment="Template name if applicable")

    # User description/notes
    description = Column(String(500), nullable=True, comment="User notes about this configuration")

    # Status
    is_active = Column(Boolean, default=True, comment="Whether this config is currently active")

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Additional config stored as JSON (for extensibility)
    extra_config = Column(JSON, nullable=True, comment="Additional configuration options")

    # Unique constraint: one active config per user+business+niche combination
    __table_args__ = (
        UniqueConstraint(
            'user_id', 'business_id', 'niche', 'is_active',
            name='uq_user_business_niche_active'
        ),
    )

    def to_weights_dict(self) -> dict:
        """Convert to dictionary format for the EngagementWeights schema."""
        return {
            "likes_weight": self.likes_weight,
            "comments_weight": self.comments_weight,
            "shares_weight": self.shares_weight,
            "saves_weight": self.saves_weight,
            "views_weight": self.views_weight,
        }

    def to_vector(self) -> list:
        """Convert weights to ordered vector [likes, comments, shares, saves, views]."""
        return [
            self.likes_weight,
            self.comments_weight,
            self.shares_weight,
            self.saves_weight,
            self.views_weight,
        ]

    def normalize(self) -> dict:
        """Return normalized weights that sum to 1.0."""
        weights = self.to_vector()
        total = sum(weights)
        if total == 0:
            return {
                "likes_weight": 0.2,
                "comments_weight": 0.2,
                "shares_weight": 0.2,
                "saves_weight": 0.2,
                "views_weight": 0.2,
            }
        return {
            "likes_weight": self.likes_weight / total,
            "comments_weight": self.comments_weight / total,
            "shares_weight": self.shares_weight / total,
            "saves_weight": self.saves_weight / total,
            "views_weight": self.views_weight / total,
        }

    @classmethod
    def get_defaults(cls) -> "UserKPIWeights":
        """Return a new instance with default weights."""
        return cls(
            likes_weight=1.0,
            comments_weight=2.0,
            shares_weight=10.0,
            saves_weight=5.0,
            views_weight=3.0,
        )

    def __repr__(self):
        return (
            f"<UserKPIWeights(id={self.id}, user_id={self.user_id}, "
            f"business_id={self.business_id}, template={self.template_name})>"
        )


class MultiOutputModelMetrics(Base):
    """
    Stores training metrics for multi-output models.

    Tracks per-metric performance (likes, comments, etc.) separately
    to enable model quality monitoring.
    """
    __tablename__ = "multi_output_model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    niche = Column(String(100), nullable=False, index=True)
    model_version = Column(String(50), nullable=False)

    # Per-metric training metrics
    likes_rmse = Column(Float, nullable=True)
    likes_r2 = Column(Float, nullable=True)
    comments_rmse = Column(Float, nullable=True)
    comments_r2 = Column(Float, nullable=True)
    shares_rmse = Column(Float, nullable=True)
    shares_r2 = Column(Float, nullable=True)
    saves_rmse = Column(Float, nullable=True)
    saves_r2 = Column(Float, nullable=True)
    views_rmse = Column(Float, nullable=True)
    views_r2 = Column(Float, nullable=True)

    # Overall metrics
    combined_rmse = Column(Float, nullable=True)
    combined_r2 = Column(Float, nullable=True)

    # Training info
    training_samples = Column(Integer, nullable=True)
    feature_count = Column(Integer, nullable=True)
    is_multi_output = Column(Boolean, default=True)

    # Timestamp
    trained_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Full metrics stored as JSON
    full_metrics = Column(JSON, nullable=True)

    def __repr__(self):
        return f"<MultiOutputModelMetrics(id={self.id}, niche={self.niche}, version={self.model_version})>"
