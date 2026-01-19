"""
A/B Test API Endpoints
======================

Endpoints for logging and analyzing A/B test results.
Enables continuous model improvement through real performance feedback.

Endpoints:
- POST /api/v1/abtest/log-result - Log actual engagement for published content
- GET /api/v1/abtest/stats - Get statistics on predictions vs actuals
- GET /api/v1/abtest/high-priority - Get high-delta samples for retraining
- POST /api/v1/abtest/feedback - Submit engagement feedback with online learning
- GET /api/v1/abtest/{experiment_id}/recommend - Get Thompson Sampling recommendation

NEW: Online Learning Integration
================================
When actual engagement metrics are logged, the system automatically:
1. Extracts features from the original content
2. Performs incremental online learning update
3. Tracks improvement and signals when full retrain is needed
"""

import logging
import random
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.abtest import ABTestLog, PredictionLog, ABTestExperiment, ABTestVariant
from app.models.content import GeneratedContent
from app.models.business import Business
from app.models.user import User
from app.api.deps import get_current_user
from app.services.abtest_service import perform_thompson_sampling
from app.schemas.abtest import ABTestExperimentCreate, ABTestExperimentResponse, ABTestVariantResponse

# Online learning integration
try:
    from backend.ml.online_update import (
        online_update,
        online_update_single,
        get_online_predictor,
        OnlineUpdateResult,
        RIVER_AVAILABLE,
    )
    ONLINE_LEARNING_AVAILABLE = True
except ImportError:
    ONLINE_LEARNING_AVAILABLE = False
    RIVER_AVAILABLE = False

# Feature extraction for online learning
try:
    from app.services.ml_service import FeatureExtractor
    FEATURE_EXTRACTOR_AVAILABLE = True
except ImportError:
    FEATURE_EXTRACTOR_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Schemas
# =============================================================================

class LogResultRequest(BaseModel):
    """Request schema for logging A/B test results."""
    post_id: int = Field(..., description="ID of the GeneratedContent record")
    actual_likes: int = Field(0, ge=0, description="Actual likes count")
    actual_comments: int = Field(0, ge=0, description="Actual comments count")
    actual_saves: Optional[int] = Field(None, ge=0, description="Actual saves count")
    actual_shares: Optional[int] = Field(None, ge=0, description="Actual shares count")
    actual_views: Optional[int] = Field(None, ge=0, description="Actual views/reach count")
    actual_reach: Optional[int] = Field(None, ge=0, description="Actual reach")
    published_date: Optional[datetime] = Field(None, description="When the content was published")


class LogResultResponse(BaseModel):
    """Response schema for logged A/B test result."""
    id: int
    post_id: int
    predicted_rpi: float
    actual_engagement: float
    delta_percent: float
    is_high_priority: bool
    status: str
    message: str


class ABTestStatsResponse(BaseModel):
    """Response schema for A/B test statistics."""
    total_logs: int
    logs_with_results: int
    high_priority_count: int
    avg_delta_percent: float
    model_bias: str  # "underestimating", "overestimating", or "calibrated"
    rmse: float
    mae: float
    recent_accuracy: float  # % of predictions within 20% of actual


class HighPrioritySample(BaseModel):
    """Schema for high-priority training sample."""
    id: int
    post_id: int
    predicted_rpi: float
    actual_engagement: float
    delta_percent: float
    content_format: Optional[str]
    platform: Optional[str]
    published_date: Optional[datetime]


class RecommendationResponse(BaseModel):
    """Response schema for variant recommendation."""
    experiment_id: int
    recommended_variant: str
    exploration_factor: float  # Just for info
    variants_status: dict  # {variant_name: {alpha: int, beta: int}}


class OnlineFeedbackRequest(BaseModel):
    """Request schema for online learning feedback."""
    post_id: int = Field(..., description="ID of the GeneratedContent record")
    actual_likes: int = Field(0, ge=0)
    actual_comments: int = Field(0, ge=0)
    actual_saves: Optional[int] = Field(None, ge=0)
    actual_shares: Optional[int] = Field(None, ge=0)
    actual_views: Optional[int] = Field(None, ge=0)
    niche: Optional[str] = Field(None, description="Business niche override")


class OnlineFeedbackResponse(BaseModel):
    """Response schema for online learning feedback."""
    post_id: int
    online_learning_enabled: bool
    samples_processed: int
    total_samples: int
    current_mae: Optional[float]
    improvement_detected: bool
    trigger_full_retrain: bool
    update_time_ms: float
    message: str


# =============================================================================
# Router Initialization
# =============================================================================

router = APIRouter(prefix="/abtest", tags=["A/B Testing"])


# =============================================================================
# Helper Functions
# =============================================================================

async def update_bandit_params(
    db: AsyncSession,
    business_id: int,
    variant_name: str,
    is_success: bool,
    test_name: str = "Format Optimization",
    content_id: Optional[int] = None
):
    """
    Update Alpha/Beta parameters for the bandit.
    Handles both global format optimization and content-specific experiments.
    """
    # 1. Update Content-Specific Experiment (if content_id provided)
    if content_id:
        # Check if content is part of an experiment
        # Either it's the original content or a variation
        content_result = await db.execute(
            select(GeneratedContent).where(GeneratedContent.id == content_id)
        )
        content = content_result.scalar_one_or_none()

        if content:
            original_id = content.variation_of if content.variation_of else content.id
            variant_label = content.variation_label if content.variation_label else "A" # Default to A if not labeled

            # Find experiment linked to this content
            exp_result = await db.execute(
                select(ABTestExperiment)
                .options(selectinload(ABTestExperiment.variants))
                .where(ABTestExperiment.original_content_id == original_id)
                .where(ABTestExperiment.is_active == True)
            )
            specific_experiment = exp_result.scalar_one_or_none()

            if specific_experiment:
                target_variant = None
                for v in specific_experiment.variants:
                    if v.variant_name == variant_label:
                        target_variant = v
                        break

                if target_variant:
                    if is_success:
                        target_variant.alpha_param += 1
                    else:
                        target_variant.beta_param += 1
                    db.add(target_variant)
                    logger.info(f"Updated specific experiment {specific_experiment.id} variant {variant_label}")

    # 2. Update Global Format Optimization (Fallback/Parallel)
    # Find active global experiment
    result = await db.execute(
        select(ABTestExperiment)
        .options(selectinload(ABTestExperiment.variants))
        .where(ABTestExperiment.business_id == business_id)
        .where(ABTestExperiment.test_name == test_name)
        .where(ABTestExperiment.is_active == True)
    )
    experiment = result.scalar_one_or_none()

    if experiment:
        # Find matching variant
        target_variant = None
        for v in experiment.variants:
            # Basic mapping logic - could be more sophisticated
            if v.variant_name.lower() == variant_name.lower():
                target_variant = v
                break

        # If variant doesn't exist but experiment does, maybe we should create it?
        # For now, we only update if it exists.
        if target_variant:
            if is_success:
                target_variant.alpha_param += 1
            else:
                target_variant.beta_param += 1

            db.add(target_variant)
            # Experiment updated implicitly via session commit in caller
            logger.info(f"Updated bandit for {variant_name}: +{'Success' if is_success else 'Fail'}")


# =============================================================================
# Online Learning Helper
# =============================================================================

async def _perform_online_update(
    content: GeneratedContent,
    actual_metrics: dict,
    niche: str
) -> Optional[dict]:
    """
    Perform online learning update in background.

    Extracts features from content and targets from actual metrics,
    then calls the online_update function.

    Args:
        content: The original GeneratedContent record
        actual_metrics: Dict with likes, comments, shares, saves, views
        niche: Business niche for model selection

    Returns:
        OnlineUpdateResult dict or None if failed
    """
    if not ONLINE_LEARNING_AVAILABLE or not RIVER_AVAILABLE:
        logger.debug("Online learning not available")
        return None

    if not FEATURE_EXTRACTOR_AVAILABLE:
        logger.warning("FeatureExtractor not available for online learning")
        return None

    try:
        # Build content dict for feature extraction
        content_dict = {
            "caption": content.caption or "",
            "content_format": content.content_format or "reel",
            "business_type": niche,
            "hashtags": content.hashtags or [],
            "posted_at": content.created_at.isoformat() if content.created_at else None,
            # Add any additional fields from content
            "whisper_transcript": getattr(content, "whisper_transcript", "") or "",
            "easyocr_text": getattr(content, "easyocr_text", "") or "",
        }

        # Extract features
        features = FeatureExtractor.extract_features(content_dict)

        # Remove non-numeric features
        numeric_features = {
            k: float(v) if isinstance(v, (int, float, np.number)) else 0.0
            for k, v in features.items()
            if k != "business_type"
        }

        # Prepare targets (log-transformed)
        targets = {
            "log_likes": float(np.log1p(actual_metrics.get("likes", 0))),
            "log_comments": float(np.log1p(actual_metrics.get("comments", 0))),
            "log_shares": float(np.log1p(actual_metrics.get("shares", 0))),
            "log_saves": float(np.log1p(actual_metrics.get("saves", 0))),
            "log_views": float(np.log1p(actual_metrics.get("views", 0))),
        }

        # Perform online update
        result = online_update_single(
            niche=niche,
            features=numeric_features,
            targets=targets,
            save_model=True  # Persist after each update
        )

        logger.info(
            f"Online update completed: niche={niche}, "
            f"total_samples={result.total_samples}, "
            f"MAE={result.current_mae:.4f if result.current_mae else 'N/A'}"
        )

        return result.to_dict()

    except Exception as e:
        logger.error(f"Online learning update failed: {e}", exc_info=True)
        return None


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/create", response_model=ABTestExperimentResponse)
async def create_ab_test(
    experiment_data: ABTestExperimentCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create a new A/B test experiment.
    """
    # Verify business access (assuming business_id is inferred or passed.
    # For now, we'll use the user's business. In a real app, we might need business_id in request)
    # Get user's business
    result = await db.execute(
        select(Business).where(Business.user_id == current_user.id)
    )
    business = result.scalar_one_or_none()

    if not business:
        raise HTTPException(status_code=404, detail="Business not found for user")

    # Create Experiment
    experiment = ABTestExperiment(
        business_id=business.id,
        test_name=experiment_data.test_name,
        original_content_id=experiment_data.original_content_id,
        is_active=True
    )
    db.add(experiment)
    await db.flush() # Get ID

    # Create Variants
    for v_data in experiment_data.variants:
        variant = ABTestVariant(
            experiment_id=experiment.id,
            variant_name=v_data.variant_name,
            content_structure=v_data.content_structure,
            alpha_param=1,
            beta_param=1
        )
        db.add(variant)

    await db.commit()
    await db.refresh(experiment)
    # Eager load variants for response
    result = await db.execute(
        select(ABTestExperiment)
        .options(selectinload(ABTestExperiment.variants))
        .where(ABTestExperiment.id == experiment.id)
    )
    experiment = result.scalar_one()

    return experiment


@router.get("/{experiment_id}", response_model=ABTestExperimentResponse)
async def get_experiment(
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get details of a specific experiment.
    """
    result = await db.execute(
        select(ABTestExperiment)
        .options(selectinload(ABTestExperiment.variants))
        .where(ABTestExperiment.id == experiment_id)
    )
    experiment = result.scalar_one_or_none()

    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    return experiment


@router.post("/{experiment_id}/winner")
async def set_experiment_winner(
    experiment_id: int,
    variant_name: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Manually set a winner and stop the experiment.
    """
    result = await db.execute(
        select(ABTestExperiment)
        .where(ABTestExperiment.id == experiment_id)
    )
    experiment = result.scalar_one_or_none()

    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # Stop experiment
    experiment.is_active = False

    # We could log the manual winner somewhere, but for now just stopping is enough
    # as the stats (alpha/beta) tell the history.

    await db.commit()

    return {"status": "success", "message": f"Experiment stopped. Winner: {variant_name}"}


@router.get("/{experiment_id}/recommend", response_model=RecommendationResponse)
async def recommend_variant_endpoint(
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get a recommendation using Thompson Sampling.
    """
    result = await db.execute(
        select(ABTestExperiment)
        .options(selectinload(ABTestExperiment.variants))
        .where(ABTestExperiment.id == experiment_id)
    )
    experiment = result.scalar_one_or_none()

    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    recommended = perform_thompson_sampling(experiment)

    return RecommendationResponse(
        experiment_id=experiment.id,
        recommended_variant=recommended,
        exploration_factor=0.0,  # Thompson sampling handles this implicitly
        variants_status={
            v.variant_name: {"alpha": v.alpha_param, "beta": v.beta_param}
            for v in experiment.variants
        }
    )


@router.post("/log-result", response_model=LogResultResponse)
async def log_abtest_result(
    request: LogResultRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Log actual engagement results for a published content piece.
    """
    # Get the content piece
    result = await db.execute(
        select(GeneratedContent).where(GeneratedContent.id == request.post_id)
    )
    content = result.scalar_one_or_none()

    if not content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Content with ID {request.post_id} not found"
        )

    # Calculate actual engagement score (same formula as training)
    actual_engagement = (
        request.actual_likes +
        request.actual_comments * 3 +
        (request.actual_saves or 0) * 5 +
        (request.actual_shares or 0) * 4
    )

    # Normalize based on views if available
    if request.actual_views and request.actual_views > 0:
        # Engagement rate based normalization
        engagement_rate = actual_engagement / request.actual_views
        actual_engagement_normalized = min(100, engagement_rate * 1000)
    else:
        # Absolute scale normalization
        actual_engagement_normalized = min(100, (actual_engagement / 100) * 10)

    # Get predicted RPI from content
    predicted_rpi = content.engagement_score or 50.0

    # Calculate delta
    if predicted_rpi > 0:
        delta_percent = ((actual_engagement_normalized - predicted_rpi) / predicted_rpi) * 100
    else:
        delta_percent = 100.0 if actual_engagement_normalized > 0 else 0.0

    is_high_priority = abs(delta_percent) > 20.0

    # Create AB test log entry
    ab_log = ABTestLog(
        post_id=request.post_id,
        business_id=content.business_id,
        predicted_rpi=predicted_rpi,
        predicted_engagement_score=content.engagement_score,
        actual_engagement=actual_engagement_normalized,
        actual_likes=request.actual_likes,
        actual_comments=request.actual_comments,
        actual_saves=request.actual_saves,
        actual_shares=request.actual_shares,
        actual_views=request.actual_views,
        actual_reach=request.actual_reach,
        delta_percent=delta_percent,
        is_high_priority=is_high_priority,
        content_format=content.content_format,
        platform=content.platform,
        published_date=request.published_date or datetime.utcnow(),
        metrics_collected_at=datetime.utcnow(),
        status="collected"
    )

    db.add(ab_log)

    # Also update the content record with actual performance
    content.actual_performance_metrics = {
        "likes": request.actual_likes,
        "comments": request.actual_comments,
        "saves": request.actual_saves,
        "shares": request.actual_shares,
        "views": request.actual_views,
        "reach": request.actual_reach,
        "collected_at": datetime.utcnow().isoformat(),
    }
    content.actual_engagement_score = actual_engagement_normalized
    content.performance_delta_percent = delta_percent
    content.performance_collected_at = datetime.utcnow()

    # === UPDATE BANDIT PARAMS ===
    # Success threshold: Normalized engagement >= 50 (approx 5% engagement rate)
    is_success = actual_engagement_normalized >= 50.0
    await update_bandit_params(
        db,
        content.business_id,
        content.content_format,
        is_success,
        content_id=content.id
    )

    await db.commit()
    await db.refresh(ab_log)

    # Determine message based on delta
    if is_high_priority:
        if delta_percent > 0:
            message = f"High priority: Model underestimated by {delta_percent:.1f}%. Content performed better than predicted!"
        else:
            message = f"High priority: Model overestimated by {abs(delta_percent):.1f}%. Content underperformed predictions."
    else:
        message = f"Prediction within acceptable range (delta: {delta_percent:.1f}%)"

    return LogResultResponse(
        id=ab_log.id,
        post_id=ab_log.post_id,
        predicted_rpi=ab_log.predicted_rpi,
        actual_engagement=ab_log.actual_engagement,
        delta_percent=ab_log.delta_percent,
        is_high_priority=ab_log.is_high_priority,
        status=ab_log.status,
        message=message
    )


@router.get("/stats", response_model=ABTestStatsResponse)
async def get_abtest_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get statistics on A/B test results and model performance.

    Returns:
    - Total logs and logs with results
    - High priority sample count
    - Average delta and model bias direction
    - RMSE and MAE metrics
    - Recent prediction accuracy (% within 20% of actual)
    """
    # Get all logs with results
    result = await db.execute(
        select(ABTestLog).where(ABTestLog.actual_engagement.isnot(None))
    )
    logs = result.scalars().all()

    total_result = await db.execute(select(func.count(ABTestLog.id)))
    total_logs = total_result.scalar() or 0

    if not logs:
        return ABTestStatsResponse(
            total_logs=total_logs,
            logs_with_results=0,
            high_priority_count=0,
            avg_delta_percent=0.0,
            model_bias="unknown",
            rmse=0.0,
            mae=0.0,
            recent_accuracy=0.0
        )

    # Calculate metrics
    deltas = [log.delta_percent for log in logs if log.delta_percent is not None]
    predictions = [log.predicted_rpi for log in logs]
    actuals = [log.actual_engagement for log in logs]

    avg_delta = sum(deltas) / len(deltas) if deltas else 0
    high_priority_count = sum(1 for log in logs if log.is_high_priority)

    # Determine model bias
    if avg_delta > 10:
        model_bias = "underestimating"
    elif avg_delta < -10:
        model_bias = "overestimating"
    else:
        model_bias = "calibrated"

    # Calculate RMSE and MAE
    if predictions and actuals:
        import numpy as np
        predictions_arr = np.array(predictions)
        actuals_arr = np.array(actuals)
        rmse = float(np.sqrt(np.mean((predictions_arr - actuals_arr) ** 2)))
        mae = float(np.mean(np.abs(predictions_arr - actuals_arr)))
    else:
        rmse = 0.0
        mae = 0.0

    # Calculate accuracy (% within 20% of actual)
    accurate_predictions = sum(1 for d in deltas if abs(d) <= 20)
    recent_accuracy = (accurate_predictions / len(deltas) * 100) if deltas else 0

    return ABTestStatsResponse(
        total_logs=total_logs,
        logs_with_results=len(logs),
        high_priority_count=high_priority_count,
        avg_delta_percent=round(avg_delta, 2),
        model_bias=model_bias,
        rmse=round(rmse, 2),
        mae=round(mae, 2),
        recent_accuracy=round(recent_accuracy, 1)
    )


@router.get("/high-priority", response_model=List[HighPrioritySample])
async def get_high_priority_samples(
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get high-priority samples for model retraining.

    Returns samples where prediction delta exceeded 20%.
    These are the most valuable samples for improving model accuracy.
    """
    result = await db.execute(
        select(ABTestLog)
        .where(ABTestLog.is_high_priority == True)
        .where(ABTestLog.actual_engagement.isnot(None))
        .order_by(ABTestLog.metrics_collected_at.desc())
        .limit(limit)
    )
    logs = result.scalars().all()

    return [
        HighPrioritySample(
            id=log.id,
            post_id=log.post_id,
            predicted_rpi=log.predicted_rpi,
            actual_engagement=log.actual_engagement,
            delta_percent=log.delta_percent,
            content_format=log.content_format,
            platform=log.platform,
            published_date=log.published_date
        )
        for log in logs
    ]


@router.post("/feedback", response_model=OnlineFeedbackResponse)
async def submit_online_feedback(
    request: OnlineFeedbackRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Submit engagement feedback with online learning update.

    This endpoint:
    1. Logs the actual engagement metrics (like log-result)
    2. Performs incremental online learning update
    3. Returns update statistics and improvement signals

    The online model learns from each feedback sample in real-time,
    enabling the system to adapt to changing engagement patterns
    without waiting for full batch retraining.

    When improvement_detected is True, consider triggering a full
    XGBoost retrain to capture the improved patterns.
    """
    # Get the content piece
    result = await db.execute(
        select(GeneratedContent).where(GeneratedContent.id == request.post_id)
    )
    content = result.scalar_one_or_none()

    if not content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Content with ID {request.post_id} not found"
        )

    # Determine niche
    niche = request.niche or getattr(content, "business_type", "general") or "general"

    # Calculate actual engagement score
    actual_engagement = (
        request.actual_likes +
        request.actual_comments * 3 +
        (request.actual_saves or 0) * 5 +
        (request.actual_shares or 0) * 4
    )

    # Normalize
    if request.actual_views and request.actual_views > 0:
        engagement_rate = actual_engagement / request.actual_views
        actual_engagement_normalized = min(100, engagement_rate * 1000)
    else:
        actual_engagement_normalized = min(100, (actual_engagement / 100) * 10)

    # Prepare metrics dict
    actual_metrics = {
        "likes": request.actual_likes,
        "comments": request.actual_comments,
        "saves": request.actual_saves or 0,
        "shares": request.actual_shares or 0,
        "views": request.actual_views or 0,
    }

    # Perform online learning update
    online_result = None
    if ONLINE_LEARNING_AVAILABLE and RIVER_AVAILABLE:
        online_result = await _perform_online_update(content, actual_metrics, niche)

    # Also log to ABTestLog for consistency
    predicted_rpi = content.engagement_score or 50.0
    delta_percent = ((actual_engagement_normalized - predicted_rpi) / predicted_rpi) * 100 if predicted_rpi > 0 else 0.0
    is_high_priority = abs(delta_percent) > 20.0

    ab_log = ABTestLog(
        post_id=request.post_id,
        business_id=content.business_id,
        predicted_rpi=predicted_rpi,
        predicted_engagement_score=content.engagement_score,
        actual_engagement=actual_engagement_normalized,
        actual_likes=request.actual_likes,
        actual_comments=request.actual_comments,
        actual_saves=request.actual_saves,
        actual_shares=request.actual_shares,
        actual_views=request.actual_views,
        delta_percent=delta_percent,
        is_high_priority=is_high_priority,
        content_format=content.content_format,
        platform=content.platform,
        published_date=datetime.utcnow(),
        metrics_collected_at=datetime.utcnow(),
        status="collected_with_online_learning" if online_result else "collected"
    )

    db.add(ab_log)

    # === UPDATE BANDIT PARAMS ===
    # Success threshold: Normalized engagement >= 50
    is_success = actual_engagement_normalized >= 50.0
    await update_bandit_params(
        db,
        content.business_id,
        content.content_format,
        is_success,
        content_id=content.id
    )

    await db.commit()

    # Build response
    if online_result:
        return OnlineFeedbackResponse(
            post_id=request.post_id,
            online_learning_enabled=True,
            samples_processed=online_result.get("samples_processed", 1),
            total_samples=online_result.get("total_samples", 1),
            current_mae=online_result.get("current_mae"),
            improvement_detected=online_result.get("improvement_detected", False),
            trigger_full_retrain=online_result.get("trigger_full_retrain", False),
            update_time_ms=online_result.get("update_time_ms", 0.0),
            message=f"Online learning updated for niche '{niche}'. "
                    f"Total samples: {online_result.get('total_samples', 1)}"
        )
    else:
        return OnlineFeedbackResponse(
            post_id=request.post_id,
            online_learning_enabled=False,
            samples_processed=0,
            total_samples=0,
            current_mae=None,
            improvement_detected=False,
            trigger_full_retrain=False,
            update_time_ms=0.0,
            message="Feedback logged. Online learning not available."
        )


@router.get("/online-status")
async def get_online_model_status(
    niche: str = "general",
    current_user: User = Depends(get_current_user)
):
    """
    Get status of online learning model for a niche.

    Returns:
    - samples_seen: Total training samples processed
    - current_mae: Current Mean Absolute Error
    - best_mae: Best MAE achieved
    - is_initialized: Whether model has been initialized
    """
    if not ONLINE_LEARNING_AVAILABLE or not RIVER_AVAILABLE:
        return {
            "online_learning_available": False,
            "river_installed": RIVER_AVAILABLE,
            "message": "Online learning not available. Install river: pip install river"
        }

    try:
        predictor = get_online_predictor(niche)
        status = predictor.get_status()
        status["online_learning_available"] = True
        return status
    except Exception as e:
        return {
            "online_learning_available": True,
            "error": str(e),
            "niche": niche
        }
