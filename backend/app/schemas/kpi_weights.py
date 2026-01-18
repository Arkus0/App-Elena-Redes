"""
KPI Weights Schemas - Multi-Objective Engagement Prediction

Pydantic schemas for configurable engagement weights per user/business.
Allows customization of RPI calculation based on business KPIs:
- Brand Awareness: Higher weight on likes/views
- Leads/Conversions: Higher weight on saves/shares
- Community Building: Higher weight on comments

Author: BrandPulse AI
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator


class EngagementWeights(BaseModel):
    """
    Configurable weights for multi-objective engagement prediction.

    Each weight determines how much a particular metric contributes
    to the final RPI score. Higher weights = more importance.

    Default weights are based on industry standards:
    - shares (10.0): Highest value - indicates viral potential
    - saves (5.0): High value - indicates content worth revisiting
    - views (3.0): Good reach indicator, especially for Reels
    - comments (2.0): Engagement quality indicator
    - likes (1.0): Basic engagement, easy to give
    """
    likes_weight: float = Field(default=1.0, ge=0.0, le=20.0, description="Weight for likes metric")
    comments_weight: float = Field(default=2.0, ge=0.0, le=20.0, description="Weight for comments metric")
    shares_weight: float = Field(default=10.0, ge=0.0, le=20.0, description="Weight for shares metric")
    saves_weight: float = Field(default=5.0, ge=0.0, le=20.0, description="Weight for saves metric")
    views_weight: float = Field(default=3.0, ge=0.0, le=20.0, description="Weight for views metric (Reels)")

    def to_vector(self) -> List[float]:
        """Convert weights to ordered vector [likes, comments, shares, saves, views]."""
        return [
            self.likes_weight,
            self.comments_weight,
            self.shares_weight,
            self.saves_weight,
            self.views_weight,
        ]

    def normalize(self) -> "EngagementWeights":
        """Return normalized weights that sum to 1.0."""
        total = sum(self.to_vector())
        if total == 0:
            return EngagementWeights()  # Return defaults
        return EngagementWeights(
            likes_weight=self.likes_weight / total,
            comments_weight=self.comments_weight / total,
            shares_weight=self.shares_weight / total,
            saves_weight=self.saves_weight / total,
            views_weight=self.views_weight / total,
        )

    @classmethod
    def from_template(cls, template_name: str) -> "EngagementWeights":
        """
        Create weights from a predefined template.

        Templates:
        - "brand_awareness": Prioritizes likes and views
        - "leads_conversions": Prioritizes saves and shares
        - "community": Prioritizes comments
        - "viral": Prioritizes shares
        - "balanced": Equal emphasis
        """
        templates = {
            "brand_awareness": cls(
                likes_weight=5.0,
                comments_weight=2.0,
                shares_weight=4.0,
                saves_weight=2.0,
                views_weight=8.0,
            ),
            "leads_conversions": cls(
                likes_weight=1.0,
                comments_weight=3.0,
                shares_weight=10.0,
                saves_weight=12.0,
                views_weight=2.0,
            ),
            "community": cls(
                likes_weight=2.0,
                comments_weight=10.0,
                shares_weight=5.0,
                saves_weight=4.0,
                views_weight=3.0,
            ),
            "viral": cls(
                likes_weight=2.0,
                comments_weight=3.0,
                shares_weight=15.0,
                saves_weight=4.0,
                views_weight=6.0,
            ),
            "balanced": cls(
                likes_weight=4.0,
                comments_weight=4.0,
                shares_weight=4.0,
                saves_weight=4.0,
                views_weight=4.0,
            ),
        }
        return templates.get(template_name, cls())


class KPIWeightsCreate(BaseModel):
    """Request schema for creating/updating KPI weights."""
    business_id: int = Field(..., description="Business ID to associate weights with")
    niche: Optional[str] = Field(None, description="Optional niche for niche-specific weights")
    weights: EngagementWeights = Field(default_factory=EngagementWeights)
    template_name: Optional[str] = Field(
        None,
        description="Apply a template instead of custom weights",
        pattern="^(brand_awareness|leads_conversions|community|viral|balanced)$"
    )
    description: Optional[str] = Field(None, max_length=500, description="User description of this config")

    @model_validator(mode='after')
    def apply_template_if_specified(self):
        """Apply template weights if template_name is provided."""
        if self.template_name:
            self.weights = EngagementWeights.from_template(self.template_name)
        return self


class KPIWeightsResponse(BaseModel):
    """Response schema for KPI weights."""
    id: int
    user_id: int
    business_id: int
    niche: Optional[str] = None
    weights: EngagementWeights
    template_name: Optional[str] = None
    description: Optional[str] = None
    is_active: bool = True
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class KPIWeightsPreview(BaseModel):
    """Preview of how weights affect RPI calculation."""
    weights: EngagementWeights
    sample_prediction: Dict[str, float] = Field(
        default_factory=dict,
        description="Sample multi-output prediction"
    )
    weighted_rpi: float = Field(..., description="Resulting weighted RPI score")
    explanation: str = Field(..., description="Human-readable explanation")
    comparison_to_default: Dict[str, Any] = Field(
        default_factory=dict,
        description="Comparison to default weights"
    )


# =============================================================================
# Multi-Output Prediction Schemas
# =============================================================================

class MultiOutputPrediction(BaseModel):
    """
    Multi-objective prediction output.

    Instead of a single engagement score, predicts individual metrics
    that can be weighted according to user preferences.
    """
    # Individual metric predictions (log-scale for skew normalization)
    log_likes: float = Field(..., description="Predicted log(likes+1)")
    log_comments: float = Field(..., description="Predicted log(comments+1)")
    log_shares: float = Field(..., description="Predicted log(shares+1)")
    log_saves: float = Field(..., description="Predicted log(saves+1)")
    log_views: float = Field(..., description="Predicted log(views+1)")

    # Raw predictions (exp-transformed)
    predicted_likes: float = Field(..., description="Predicted likes count")
    predicted_comments: float = Field(..., description="Predicted comments count")
    predicted_shares: float = Field(..., description="Predicted shares count")
    predicted_saves: float = Field(..., description="Predicted saves count")
    predicted_views: float = Field(..., description="Predicted views count")

    # Weighted RPI calculation
    weighted_rpi: float = Field(..., description="Final weighted RPI (0-100)")
    weights_used: EngagementWeights = Field(..., description="Weights applied")

    # Relative to author baseline
    relative_to_baseline: Optional[float] = Field(
        None,
        description="Performance relative to author's average (1.0 = average)"
    )

    def to_vector(self) -> List[float]:
        """Return log predictions as vector [likes, comments, shares, saves, views]."""
        return [
            self.log_likes,
            self.log_comments,
            self.log_shares,
            self.log_saves,
            self.log_views,
        ]


class MultiOutputPredictionRequest(BaseModel):
    """Request for multi-output prediction."""
    # Content features
    caption: Optional[str] = Field(None, description="Post caption text")
    hashtags: Optional[List[str]] = Field(None, description="Hashtags")
    content_format: str = Field(default="reel", description="reel, carousel, static")

    # Video features (for Reels)
    video_duration_seconds: Optional[float] = Field(None, ge=0, le=300)
    hook_energy: Optional[float] = Field(None, ge=0, le=1)
    retention_energy: Optional[float] = Field(None, ge=0, le=1)
    face_in_hook: Optional[bool] = Field(None)
    tempo: Optional[float] = Field(None, ge=0, le=200)

    # Timing
    posted_at: Optional[datetime] = Field(None)

    # Business context
    business_id: Optional[int] = Field(None)
    business_type: Optional[str] = Field(None)

    # Optional: override weights for this prediction
    custom_weights: Optional[EngagementWeights] = Field(None)

    # Author baseline for relative scoring
    author_avg_likes: Optional[float] = Field(None)
    author_avg_comments: Optional[float] = Field(None)
    author_avg_shares: Optional[float] = Field(None)
    author_avg_saves: Optional[float] = Field(None)
    author_avg_views: Optional[float] = Field(None)


class MultiOutputPredictionResponse(BaseModel):
    """Full response for multi-output prediction with SHAP explanations."""
    prediction: MultiOutputPrediction

    # Per-metric SHAP explanations
    shap_likes: Optional[Dict[str, float]] = Field(
        None, description="Top SHAP factors for likes prediction"
    )
    shap_comments: Optional[Dict[str, float]] = Field(
        None, description="Top SHAP factors for comments prediction"
    )
    shap_shares: Optional[Dict[str, float]] = Field(
        None, description="Top SHAP factors for shares prediction"
    )
    shap_saves: Optional[Dict[str, float]] = Field(
        None, description="Top SHAP factors for saves prediction"
    )
    shap_views: Optional[Dict[str, float]] = Field(
        None, description="Top SHAP factors for views prediction"
    )

    # Combined explanation
    explanation: str = Field(..., description="Human-readable explanation")

    # Confidence
    confidence_interval: Dict[str, tuple] = Field(
        default_factory=dict,
        description="95% CI for each metric prediction"
    )

    # Model metadata
    model_version: str = Field(default="multi-output-v1.0")
    is_multi_output: bool = Field(default=True)


# =============================================================================
# Template Schemas
# =============================================================================

class KPITemplate(BaseModel):
    """Predefined KPI weight template."""
    name: str
    display_name: str
    description: str
    weights: EngagementWeights
    use_case: str
    icon: str = "chart"


class KPITemplatesResponse(BaseModel):
    """List of available KPI templates."""
    templates: List[KPITemplate] = Field(default_factory=list)


# Define available templates
KPI_TEMPLATES = [
    KPITemplate(
        name="brand_awareness",
        display_name="Brand Awareness",
        description="Maximiza alcance y visibilidad. Ideal para negocios nuevos o lanzamientos.",
        weights=EngagementWeights.from_template("brand_awareness"),
        use_case="Negocios nuevos, lanzamientos, campañas de reconocimiento",
        icon="eye"
    ),
    KPITemplate(
        name="leads_conversions",
        display_name="Leads y Conversiones",
        description="Prioriza guardados y compartidos que indican intención de compra.",
        weights=EngagementWeights.from_template("leads_conversions"),
        use_case="Inmobiliarias, servicios profesionales, e-commerce",
        icon="target"
    ),
    KPITemplate(
        name="community",
        display_name="Comunidad",
        description="Fomenta conversaciones y relaciones con seguidores.",
        weights=EngagementWeights.from_template("community"),
        use_case="Restaurantes, cafeterías, negocios locales",
        icon="users"
    ),
    KPITemplate(
        name="viral",
        display_name="Viral",
        description="Optimiza para máxima difusión y compartidos.",
        weights=EngagementWeights.from_template("viral"),
        use_case="Entretenimiento, contenido educativo, tendencias",
        icon="share"
    ),
    KPITemplate(
        name="balanced",
        display_name="Equilibrado",
        description="Peso igual en todas las métricas.",
        weights=EngagementWeights.from_template("balanced"),
        use_case="Uso general, A/B testing",
        icon="balance"
    ),
]
