"""
Multi-Output Prediction API - Endpoints for Multi-Objective Engagement Prediction

Provides endpoints for multi-output predictions with configurable weights:
- POST /multi-output/predict: Predict with custom weights
- POST /multi-output/predict/batch: Batch predictions
- GET /multi-output/status: Model status
- POST /multi-output/train: Train multi-output model

Author: BrandPulse AI
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.models.kpi_weights import UserKPIWeights
from app.schemas.kpi_weights import (
    EngagementWeights,
    MultiOutputPrediction,
    MultiOutputPredictionRequest,
    MultiOutputPredictionResponse,
)
from app.services.multi_output_predictor import (
    get_multi_output_predictor,
    MultiOutputEngagementPredictor,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Helper Functions
# =============================================================================

async def get_user_weights_for_business(
    db: AsyncSession,
    user_id: int,
    business_id: int,
) -> Optional[EngagementWeights]:
    """Get user's configured weights for a business."""
    query = select(UserKPIWeights).where(
        and_(
            UserKPIWeights.user_id == user_id,
            UserKPIWeights.business_id == business_id,
            UserKPIWeights.is_active == True,
        )
    )
    result = await db.execute(query)
    weights_model = result.scalar_one_or_none()

    if weights_model:
        return EngagementWeights(
            likes_weight=weights_model.likes_weight,
            comments_weight=weights_model.comments_weight,
            shares_weight=weights_model.shares_weight,
            saves_weight=weights_model.saves_weight,
            views_weight=weights_model.views_weight,
        )

    return None


def build_features_from_request(request: MultiOutputPredictionRequest) -> dict:
    """Extract features dictionary from prediction request."""
    features = {}

    # Caption features (will be processed by embedding extractor)
    if request.caption:
        features["caption"] = request.caption
        features["caption_length"] = len(request.caption)
        features["caption_words"] = len(request.caption.split())

    # Hashtags
    if request.hashtags:
        features["hashtag_count"] = len(request.hashtags)
        features["hashtag_density"] = len(request.hashtags) / max(len(request.caption or ""), 1)

    # Format
    features["is_reel"] = 1 if request.content_format == "reel" else 0
    features["is_carousel"] = 1 if request.content_format == "carousel" else 0
    features["is_static"] = 1 if request.content_format == "static" else 0

    # Video features
    if request.video_duration_seconds:
        features["video_duration"] = request.video_duration_seconds
        features["video_optimal_length"] = 1 if 15 <= request.video_duration_seconds <= 60 else 0

    if request.hook_energy is not None:
        features["hook_energy"] = request.hook_energy

    if request.retention_energy is not None:
        features["retention_energy"] = request.retention_energy

    if request.face_in_hook is not None:
        features["face_in_hook"] = 1 if request.face_in_hook else 0

    if request.tempo is not None:
        features["tempo"] = request.tempo

    # Timing
    if request.posted_at:
        features["hour_of_day"] = request.posted_at.hour
        features["day_of_week"] = request.posted_at.weekday()
        features["is_weekend"] = 1 if request.posted_at.weekday() >= 5 else 0
        features["is_prime_time"] = 1 if 12 <= request.posted_at.hour <= 21 else 0

    # Business type
    if request.business_type:
        features["business_type"] = request.business_type
        features[f"niche_{request.business_type}"] = 1

    return features


# =============================================================================
# Endpoints
# =============================================================================

@router.post(
    "/predict",
    response_model=MultiOutputPredictionResponse,
    summary="Multi-Output Prediction with Configurable Weights",
    description="""
    Predicts engagement across multiple metrics simultaneously:
    - likes, comments, shares, saves, views

    The prediction uses the user's configured weights (if any) or defaults.
    Custom weights can be overridden per-request.

    **RPI Calculation:**
    RPI = weighted_sum(log_predictions) / total_weight * scale_factor

    **SHAP Explanations:**
    Returns top contributing factors for EACH metric separately,
    allowing users to understand what drives saves vs comments.
    """
)
async def predict_multi_output(
    request: MultiOutputPredictionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> MultiOutputPredictionResponse:
    """Multi-output prediction with configurable weights."""
    # Get predictor
    niche = request.business_type or "general"
    predictor = get_multi_output_predictor(niche)

    if not predictor.is_trained:
        # Try general model as fallback
        predictor = get_multi_output_predictor("general")
        if not predictor.is_trained:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Multi-output model not trained. Please train using POST /multi-output/train"
            )

    # Determine weights to use
    weights_dict = None

    # 1. Check request for custom weights
    if request.custom_weights:
        weights_dict = {
            "likes_weight": request.custom_weights.likes_weight,
            "comments_weight": request.custom_weights.comments_weight,
            "shares_weight": request.custom_weights.shares_weight,
            "saves_weight": request.custom_weights.saves_weight,
            "views_weight": request.custom_weights.views_weight,
        }
        logger.info("Using custom weights from request")

    # 2. Check user's saved weights for business
    elif request.business_id:
        user_weights = await get_user_weights_for_business(
            db, current_user.id, request.business_id
        )
        if user_weights:
            weights_dict = {
                "likes_weight": user_weights.likes_weight,
                "comments_weight": user_weights.comments_weight,
                "shares_weight": user_weights.shares_weight,
                "saves_weight": user_weights.saves_weight,
                "views_weight": user_weights.views_weight,
            }
            logger.info(f"Using saved weights for business {request.business_id}")

    # 3. Use defaults (handled by predictor)

    # Build features
    features = build_features_from_request(request)

    # Build author baseline if provided
    author_baseline = None
    if any([
        request.author_avg_likes,
        request.author_avg_comments,
        request.author_avg_shares,
        request.author_avg_saves,
        request.author_avg_views,
    ]):
        author_baseline = {
            "avg_likes": request.author_avg_likes or 0,
            "avg_comments": request.author_avg_comments or 0,
            "avg_shares": request.author_avg_shares or 0,
            "avg_saves": request.author_avg_saves or 0,
            "avg_views": request.author_avg_views or 0,
        }

    try:
        # Predict
        result = predictor.predict(
            features=features,
            weights=weights_dict,
            author_baseline=author_baseline,
            compute_shap=True,
        )

        # Build response
        prediction = MultiOutputPrediction(
            log_likes=result.log_likes,
            log_comments=result.log_comments,
            log_shares=result.log_shares,
            log_saves=result.log_saves,
            log_views=result.log_views,
            predicted_likes=result.predicted_likes,
            predicted_comments=result.predicted_comments,
            predicted_shares=result.predicted_shares,
            predicted_saves=result.predicted_saves,
            predicted_views=result.predicted_views,
            weighted_rpi=result.weighted_rpi,
            weights_used=EngagementWeights(**result.weights_used),
            relative_to_baseline=result.relative_to_baseline,
        )

        # Build explanation
        explanation_parts = []

        if result.weighted_rpi >= 70:
            explanation_parts.append(f"RPI alto ({result.weighted_rpi:.0f}/100)")
        elif result.weighted_rpi >= 40:
            explanation_parts.append(f"RPI moderado ({result.weighted_rpi:.0f}/100)")
        else:
            explanation_parts.append(f"RPI bajo ({result.weighted_rpi:.0f}/100)")

        # Add metric highlights
        metrics_sorted = sorted([
            ("likes", result.predicted_likes),
            ("comments", result.predicted_comments),
            ("shares", result.predicted_shares),
            ("saves", result.predicted_saves),
            ("views", result.predicted_views),
        ], key=lambda x: x[1], reverse=True)

        top_metric = metrics_sorted[0]
        explanation_parts.append(f"Mayor predicción en {top_metric[0]} ({top_metric[1]:.0f})")

        if result.relative_to_baseline:
            if result.relative_to_baseline > 1.2:
                explanation_parts.append(f"Por encima del promedio del autor ({result.relative_to_baseline:.1f}x)")
            elif result.relative_to_baseline < 0.8:
                explanation_parts.append(f"Por debajo del promedio del autor ({result.relative_to_baseline:.1f}x)")

        explanation = ". ".join(explanation_parts) + "."

        return MultiOutputPredictionResponse(
            prediction=prediction,
            shap_likes=result.shap_likes,
            shap_comments=result.shap_comments,
            shap_shares=result.shap_shares,
            shap_saves=result.shap_saves,
            shap_views=result.shap_views,
            explanation=explanation,
            confidence_interval={},  # TODO: Add confidence intervals
            model_version=predictor.MODEL_VERSION,
            is_multi_output=True,
        )

    except Exception as e:
        logger.error(f"Multi-output prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post(
    "/predict/batch",
    response_model=List[MultiOutputPredictionResponse],
    summary="Batch Multi-Output Predictions",
    description="Predict engagement for multiple content pieces."
)
async def predict_batch(
    requests: List[MultiOutputPredictionRequest],
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> List[MultiOutputPredictionResponse]:
    """Batch predictions for multiple contents."""
    results = []

    for req in requests:
        try:
            result = await predict_multi_output(req, current_user, db)
            results.append(result)
        except HTTPException as e:
            # Continue with partial results
            logger.warning(f"Batch item failed: {e.detail}")

    return results


@router.get(
    "/status",
    summary="Multi-Output Model Status",
    description="Get status of multi-output models."
)
async def get_model_status(
    niche: str = Query(default="general", description="Business niche"),
):
    """Get multi-output model status."""
    predictor = get_multi_output_predictor(niche)
    return predictor.get_model_status()


@router.post(
    "/train",
    summary="Train Multi-Output Model",
    description="""
    Train or retrain the multi-output engagement model.

    Requires training data with engagement metrics:
    - likes, comments, shares, saves, views

    Training creates a MultiOutputRegressor with 5 XGBoost estimators.
    """
)
async def train_model(
    niche: str = Query(default="general", description="Business niche"),
    use_synthetic: bool = Query(default=True, description="Use synthetic data if no real data"),
    n_samples: int = Query(default=1000, ge=100, le=10000, description="Number of synthetic samples"),
    current_user: User = Depends(get_current_user),
):
    """Train multi-output model."""
    import numpy as np
    import pandas as pd

    predictor = get_multi_output_predictor(niche)

    # Generate synthetic training data
    if use_synthetic:
        logger.info(f"Generating {n_samples} synthetic samples for training")

        np.random.seed(42)

        # Generate features
        X = pd.DataFrame({
            "hook_energy": np.random.uniform(0.2, 0.9, n_samples),
            "retention_energy": np.random.uniform(0.3, 0.8, n_samples),
            "caption_length": np.random.randint(50, 500, n_samples),
            "hashtag_count": np.random.randint(3, 20, n_samples),
            "is_reel": np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            "hour_of_day": np.random.randint(6, 23, n_samples),
            "semantic_hook_score": np.random.uniform(0.3, 0.95, n_samples),
            "sentiment_compound": np.random.uniform(-0.5, 0.8, n_samples),
            "has_strong_cta": np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        })

        # Generate correlated engagement targets
        base_engagement = (
            X["hook_energy"] * 2 +
            X["semantic_hook_score"] * 1.5 +
            X["is_reel"] * 0.5 +
            np.random.normal(0, 0.3, n_samples)
        )

        y = pd.DataFrame({
            "log_likes": np.clip(base_engagement + np.random.normal(4, 0.5, n_samples), 0, 10),
            "log_comments": np.clip(base_engagement * 0.6 + np.random.normal(2, 0.4, n_samples), 0, 8),
            "log_shares": np.clip(base_engagement * 0.3 + np.random.normal(1, 0.3, n_samples), 0, 6),
            "log_saves": np.clip(base_engagement * 0.5 + np.random.normal(2.5, 0.4, n_samples), 0, 7),
            "log_views": np.clip(base_engagement * 1.2 + np.random.normal(6, 0.6, n_samples), 0, 12),
        })

        try:
            metrics = predictor.train(X, y, feature_columns=list(X.columns))

            return {
                "success": True,
                "message": f"Multi-output model trained for niche={niche}",
                "metrics": metrics.to_dict(),
            }

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Training failed: {str(e)}"
            )

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Real data training not implemented. Set use_synthetic=true."
        )


@router.post(
    "/compare-weights",
    summary="Compare Different Weight Configurations",
    description="""
    Compare how different weight configurations affect RPI for the same content.

    Useful for A/B testing weight configurations or understanding
    the impact of prioritizing different KPIs.
    """
)
async def compare_weights(
    request: MultiOutputPredictionRequest,
    weights_configs: List[EngagementWeights],
    current_user: User = Depends(get_current_user),
):
    """Compare predictions across different weight configurations."""
    niche = request.business_type or "general"
    predictor = get_multi_output_predictor(niche)

    if not predictor.is_trained:
        predictor = get_multi_output_predictor("general")
        if not predictor.is_trained:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not trained"
            )

    features = build_features_from_request(request)

    results = []
    for weights in weights_configs:
        weights_dict = {
            "likes_weight": weights.likes_weight,
            "comments_weight": weights.comments_weight,
            "shares_weight": weights.shares_weight,
            "saves_weight": weights.saves_weight,
            "views_weight": weights.views_weight,
        }

        result = predictor.predict(features, weights_dict, compute_shap=False)

        results.append({
            "weights": weights_dict,
            "weighted_rpi": result.weighted_rpi,
            "predicted_metrics": {
                "likes": result.predicted_likes,
                "comments": result.predicted_comments,
                "shares": result.predicted_shares,
                "saves": result.predicted_saves,
                "views": result.predicted_views,
            }
        })

    # Sort by RPI
    results.sort(key=lambda x: x["weighted_rpi"], reverse=True)

    return {
        "comparisons": results,
        "best_config": results[0] if results else None,
        "rpi_range": {
            "min": min(r["weighted_rpi"] for r in results) if results else 0,
            "max": max(r["weighted_rpi"] for r in results) if results else 0,
        }
    }
