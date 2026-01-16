"""
ML API Endpoints - Hybrid ML/LLM Architecture
Fast ML predictions for engagement, format, and triggers
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any
import logging

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.schemas.ml import (
    MLPredictionRequest,
    MLFullPrediction,
    EngagementPredictionML,
    FormatRecommendation,
    TriggerSuggestions,
    ModelTrainingRequest,
    ModelTrainingResponse,
    ModelStatusResponse,
)
from app.services.ml_service import (
    get_ml_predictor,
    train_initial_model,
    SyntheticDataGenerator,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/status", response_model=ModelStatusResponse)
async def get_model_status(
    current_user: User = Depends(get_current_user)
):
    """Get the current status of the ML model"""
    predictor = get_ml_predictor()

    return ModelStatusResponse(
        is_trained=predictor.is_trained,
        last_trained=None,  # Would need to track this in the model
        model_version="1.0",
        feature_count=len(predictor.FEATURE_COLUMNS),
        training_samples=500 if predictor.is_trained else 0,
    )


@router.post("/train", response_model=ModelTrainingResponse)
async def train_model(
    request: ModelTrainingRequest,
    current_user: User = Depends(get_current_user)
):
    """Train or retrain the ML model"""
    try:
        predictor = get_ml_predictor()

        if request.use_synthetic_data:
            logger.info(f"Training with {request.sample_count} synthetic samples")
            training_data = SyntheticDataGenerator.generate_dataset(request.sample_count)
            predictor.train(training_data, retrain=True)

        return ModelTrainingResponse(
            success=True,
            message=f"Model trained successfully with {request.sample_count} samples",
            metrics={
                "samples": request.sample_count,
                "features": len(predictor.FEATURE_COLUMNS),
            }
        )
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=MLFullPrediction)
async def get_full_prediction(
    request: MLPredictionRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Get complete ML prediction for content
    Returns engagement score, format recommendation, and trigger suggestions
    This is called BEFORE LLM generation for the hybrid approach
    """
    predictor = get_ml_predictor()

    # Ensure model is trained
    if not predictor.is_trained:
        logger.info("Model not trained, training with synthetic data...")
        train_initial_model()

    # Convert request to content dict
    content = {
        "caption": request.caption,
        "hashtags": request.hashtags,
        "content_format": request.content_format,
        "type": request.content_format,
        "business_type": request.business_type,
        "video_duration_seconds": request.video_duration_seconds,
        "audio_name": request.audio_name,
        "posted_at": request.posted_at,
    }

    # Get full prediction
    prediction = predictor.get_full_prediction(content)

    return prediction


@router.post("/predict/engagement", response_model=EngagementPredictionML)
async def predict_engagement(
    request: MLPredictionRequest,
    current_user: User = Depends(get_current_user)
):
    """Get only engagement score prediction"""
    predictor = get_ml_predictor()

    if not predictor.is_trained:
        train_initial_model()

    content = {
        "caption": request.caption,
        "hashtags": request.hashtags,
        "content_format": request.content_format,
        "business_type": request.business_type,
    }

    return predictor.predict_engagement(content)


@router.post("/predict/format", response_model=FormatRecommendation)
async def recommend_format(
    request: MLPredictionRequest,
    current_user: User = Depends(get_current_user)
):
    """Get format recommendation"""
    predictor = get_ml_predictor()

    if not predictor.is_trained:
        train_initial_model()

    content = {
        "caption": request.caption,
        "hashtags": request.hashtags,
        "content_format": request.content_format,
        "business_type": request.business_type,
    }

    return predictor.recommend_format(content)


@router.post("/predict/triggers", response_model=TriggerSuggestions)
async def suggest_triggers(
    request: MLPredictionRequest,
    current_user: User = Depends(get_current_user)
):
    """Get trigger suggestions to improve engagement"""
    predictor = get_ml_predictor()

    if not predictor.is_trained:
        train_initial_model()

    content = {
        "caption": request.caption,
        "hashtags": request.hashtags,
        "content_format": request.content_format,
        "business_type": request.business_type,
    }

    return predictor.suggest_triggers(content)


@router.post("/analyze-draft")
async def analyze_draft_content(
    content: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """
    Analyze draft content before publishing
    Returns ML predictions + suggestions for improvement
    """
    predictor = get_ml_predictor()

    if not predictor.is_trained:
        train_initial_model()

    prediction = predictor.get_full_prediction(content)

    # Add improvement roadmap
    score = prediction["engagement_prediction"]["score"]

    improvement_roadmap = []

    if score < 50:
        improvement_roadmap.append({
            "priority": "high",
            "area": "engagement",
            "action": "Añade más elementos de engagement (hooks, CTAs, emojis)",
            "potential_gain": "+20-30 puntos"
        })

    if score < 70:
        suggestions = prediction.get("optimization_suggestions", [])
        for i, suggestion in enumerate(suggestions[:3]):
            improvement_roadmap.append({
                "priority": "medium" if i > 0 else "high",
                "area": "optimization",
                "action": suggestion,
                "potential_gain": "+5-10 puntos"
            })

    return {
        **prediction,
        "improvement_roadmap": improvement_roadmap,
        "ready_to_publish": score >= 60,
    }
