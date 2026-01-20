"""
ML Service Schemas - Pydantic models for ML predictions
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class FeatureImpact(BaseModel):
    """Single feature's impact on prediction"""
    feature: str
    impact: float


class ShapExplanation(BaseModel):
    """SHAP-based explanation for predictions"""
    top_positive_factors: List[FeatureImpact] = []
    top_negative_factors: List[FeatureImpact] = []
    explanation_text: str = ""
    note: Optional[str] = None


class EngagementPredictionML(BaseModel):
    """ML-based engagement prediction response"""
    score: float = Field(..., ge=0, le=100, description="Engagement score 0-100")
    confidence: float = Field(..., ge=0, le=100, description="Prediction confidence")
    explanation: ShapExplanation
    feature_importance: List[FeatureImpact] = []
    multi_output_breakdown: Optional[Dict[str, float]] = None


class FormatAlternative(BaseModel):
    """Alternative format recommendation"""
    format: str
    score: float


class FormatRecommendation(BaseModel):
    """ML-based format recommendation"""
    recommended_format: str
    confidence: float
    alternatives: List[FormatAlternative] = []
    explanation: Dict[str, Any] = {}


class TriggerSuggestion(BaseModel):
    """Individual trigger suggestion"""
    trigger_type: str
    impact: str  # high, medium, low
    examples: List[str] = []
    reason: str


class TriggerSuggestions(BaseModel):
    """Trigger suggestions response"""
    current_triggers: Dict[str, int] = {}
    suggestions: List[TriggerSuggestion] = []
    improvement_potential: float = 0


class MLPredictionRequest(BaseModel):
    """Request for ML prediction"""
    caption: str = ""
    hashtags: List[str] = []
    content_format: str = "reel"
    business_type: str = "otros"
    video_duration_seconds: int = 0
    audio_name: str = ""
    posted_at: Optional[str] = None
    whisper_transcript: Optional[str] = None
    easyocr_text: Optional[str] = None
    visual_description: Optional[str] = None


class MLFullPrediction(BaseModel):
    """Complete ML prediction with all components"""
    engagement_prediction: EngagementPredictionML
    format_recommendation: FormatRecommendation
    trigger_suggestions: TriggerSuggestions
    optimization_suggestions: List[str] = []
    ml_summary: str = ""
    weighted_rpi: Optional[Dict[str, Any]] = None


class HybridContentRequest(BaseModel):
    """Request for hybrid ML/LLM content generation"""
    platform: str = "instagram"
    content_format: str = "reel"
    goal: str = "engagement"
    topic: Optional[str] = None
    include_ml_optimization: bool = True


class HybridContentResponse(BaseModel):
    """Response from hybrid ML/LLM content generation"""
    ml_prediction: MLFullPrediction
    generated_content: Dict[str, Any]
    optimized: bool = True
    optimization_applied: List[str] = []


class ModelTrainingRequest(BaseModel):
    """Request to train/retrain the ML model"""
    use_synthetic_data: bool = True
    sample_count: int = 500


class ModelTrainingResponse(BaseModel):
    """Response from model training"""
    success: bool
    message: str
    metrics: Dict[str, Any] = {}


class ModelStatusResponse(BaseModel):
    """ML model status response"""
    is_trained: bool
    last_trained: Optional[str] = None
    model_version: str = "1.0"
    feature_count: int = 0
    training_samples: int = 0
