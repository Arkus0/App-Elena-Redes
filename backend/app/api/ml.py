"""
ML API Endpoints - Hybrid ML/LLM Architecture
Fast ML predictions for engagement, format, and triggers
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
import json

from sqlalchemy import select
from app.core.database import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.business import Business
from app.services.user_config_service import user_config_service
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

# Evaluation logs directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"

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
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
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

    # Get user business for config
    result = await db.execute(select(Business.id).where(Business.user_id == current_user.id))
    business_id = result.scalars().first()

    # Load user config
    config = None
    if business_id:
        config = await user_config_service.get_pipeline_config(db, current_user.id, business_id)

    embedding_precision = config.embedding_precision if config else "low"
    kpi_weights = config.kpi_weights if config else None

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
        "whisper_transcript": request.whisper_transcript,
        "easyocr_text": request.easyocr_text,
        "visual_description": request.visual_description,
    }

    # Get full prediction
    prediction = predictor.get_full_prediction(
        content,
        embedding_precision=embedding_precision,
        kpi_weights=kpi_weights,
        business_id=business_id
    )

    return prediction


@router.post("/predict/engagement", response_model=EngagementPredictionML)
async def predict_engagement(
    request: MLPredictionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get only engagement score prediction"""
    predictor = get_ml_predictor()

    if not predictor.is_trained:
        train_initial_model()

    # Get user business for config
    result = await db.execute(select(Business.id).where(Business.user_id == current_user.id))
    business_id = result.scalars().first()

    # Load user config
    config = None
    if business_id:
        config = await user_config_service.get_pipeline_config(db, current_user.id, business_id)

    embedding_precision = config.embedding_precision if config else "low"
    kpi_weights = config.kpi_weights if config else None

    content = {
        "caption": request.caption,
        "hashtags": request.hashtags,
        "content_format": request.content_format,
        "business_type": request.business_type,
        "whisper_transcript": request.whisper_transcript,
        "easyocr_text": request.easyocr_text,
        "visual_description": request.visual_description,
    }

    return predictor.predict_engagement(
        content,
        embedding_precision=embedding_precision,
        kpi_weights=kpi_weights,
        business_id=business_id
    )


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


@router.get("/health")
async def get_model_health(
    niche: Optional[str] = Query(None, description="Business niche to get health for"),
    current_user: User = Depends(get_current_user)
):
    """
    Get model health metrics including drift detection and granular evaluation.

    Returns latest evaluation results with:
    - Global metrics per target (MAE, R2, RMSE)
    - Drift detection status and score
    - Recent insights and alerts
    - Format/time segmented metrics (if available)
    """
    try:
        # Find evaluation files
        eval_files = list(LOGS_DIR.glob("eval_*.json"))

        if not eval_files:
            return {
                "status": "no_evaluations",
                "message": "No evaluation data available yet. Run training to generate evaluations.",
                "niches": [],
            }

        health_data = {
            "status": "healthy",
            "niches": {},
            "alerts": [],
            "last_updated": None,
        }

        for eval_file in eval_files:
            try:
                with open(eval_file, "r") as f:
                    data = json.load(f)

                niche_name = data.get("niche", eval_file.stem.replace("eval_", ""))

                # Skip if specific niche requested and this isn't it
                if niche and niche != niche_name:
                    continue

                latest = data.get("latest", {})

                # Extract key metrics
                global_metrics = latest.get("global_metrics", {})
                aggregate = global_metrics.get("_aggregate", {})
                drift_data = latest.get("drift", {})
                insights = latest.get("insights", [])

                niche_health = {
                    "niche": niche_name,
                    "last_evaluated": latest.get("timestamp"),
                    "evaluation_type": latest.get("evaluation_type", "unknown"),
                    "n_samples": latest.get("n_samples", 0),
                    "metrics": {
                        "aggregate_mae": aggregate.get("mae"),
                        "aggregate_r2": aggregate.get("r2"),
                        "aggregate_rmse": aggregate.get("rmse"),
                    },
                    "per_target": {
                        k: v for k, v in global_metrics.items()
                        if k != "_aggregate"
                    },
                    "drift": {
                        "detected": drift_data.get("detected", False),
                        "score": drift_data.get("score", 0),
                        "alert": drift_data.get("alert"),
                    },
                    "format_metrics": latest.get("format_metrics", {}),
                    "time_metrics": latest.get("time_metrics", {}),
                    "rpi_metrics": latest.get("rpi_metrics", {}),
                    "calibration": latest.get("calibration"),
                    "insights": insights[:5],  # Top 5 insights
                }

                health_data["niches"][niche_name] = niche_health

                # Check for alerts
                if drift_data.get("detected"):
                    health_data["status"] = "warning"
                    health_data["alerts"].append({
                        "niche": niche_name,
                        "type": "drift",
                        "message": f"Drift detected for {niche_name}: score={drift_data.get('score', 0):.3f}",
                        "severity": "high" if drift_data.get("score", 0) > 0.5 else "medium",
                    })

                # Track most recent update
                if latest.get("timestamp"):
                    if health_data["last_updated"] is None or latest["timestamp"] > health_data["last_updated"]:
                        health_data["last_updated"] = latest["timestamp"]

            except Exception as e:
                logger.warning(f"Error reading evaluation file {eval_file}: {e}")
                continue

        # Add evaluation history summary if specific niche requested
        if niche and niche in health_data["niches"]:
            try:
                eval_file = LOGS_DIR / f"eval_{niche}.json"
                if eval_file.exists():
                    with open(eval_file, "r") as f:
                        data = json.load(f)
                    history = data.get("history", [])

                    # Get MAE trend from last 10 evaluations
                    mae_history = []
                    for h in history[-10:]:
                        agg = h.get("global_metrics", {}).get("_aggregate", {})
                        if agg.get("mae"):
                            mae_history.append({
                                "timestamp": h.get("timestamp"),
                                "mae": agg["mae"],
                                "drift_score": h.get("drift", {}).get("score", 0),
                            })

                    health_data["niches"][niche]["history"] = mae_history

            except Exception as e:
                logger.warning(f"Error loading history for {niche}: {e}")

        return health_data

    except Exception as e:
        logger.error(f"Error getting model health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/summary")
async def get_model_health_summary(
    current_user: User = Depends(get_current_user)
):
    """
    Get a quick summary of model health across all niches.
    Optimized for dashboard display.
    """
    try:
        eval_files = list(LOGS_DIR.glob("eval_*.json"))

        if not eval_files:
            return {
                "total_niches": 0,
                "healthy_count": 0,
                "warning_count": 0,
                "overall_status": "no_data",
                "avg_mae": None,
                "avg_drift_score": None,
                "last_updated": None,
                "alerts": [],
            }

        healthy_count = 0
        warning_count = 0
        mae_values = []
        drift_scores = []
        alerts = []
        last_updated = None

        for eval_file in eval_files:
            try:
                with open(eval_file, "r") as f:
                    data = json.load(f)

                niche_name = data.get("niche", eval_file.stem.replace("eval_", ""))
                latest = data.get("latest", {})

                # Get metrics
                aggregate = latest.get("global_metrics", {}).get("_aggregate", {})
                drift_data = latest.get("drift", {})

                if aggregate.get("mae"):
                    mae_values.append(aggregate["mae"])

                drift_score = drift_data.get("score", 0)
                drift_scores.append(drift_score)

                if drift_data.get("detected"):
                    warning_count += 1
                    alerts.append({
                        "niche": niche_name,
                        "message": f"Drift en {niche_name}",
                        "score": drift_score,
                    })
                else:
                    healthy_count += 1

                # Track last update
                ts = latest.get("timestamp")
                if ts and (last_updated is None or ts > last_updated):
                    last_updated = ts

            except Exception:
                continue

        total_niches = healthy_count + warning_count

        return {
            "total_niches": total_niches,
            "healthy_count": healthy_count,
            "warning_count": warning_count,
            "overall_status": "warning" if warning_count > 0 else "healthy" if total_niches > 0 else "no_data",
            "avg_mae": round(sum(mae_values) / len(mae_values), 4) if mae_values else None,
            "avg_drift_score": round(sum(drift_scores) / len(drift_scores), 4) if drift_scores else None,
            "last_updated": last_updated,
            "alerts": alerts[:5],  # Top 5 alerts
        }

    except Exception as e:
        logger.error(f"Error getting health summary: {e}")
        return {
            "total_niches": 0,
            "healthy_count": 0,
            "warning_count": 0,
            "overall_status": "error",
            "error": str(e),
        }
