"""
A/B Test API Endpoints
======================

Endpoints for logging and analyzing A/B test results.
Enables continuous model improvement through real performance feedback.

Endpoints:
- POST /api/v1/abtest/log-result - Log actual engagement for published content
- GET /api/v1/abtest/stats - Get statistics on predictions vs actuals
- GET /api/v1/abtest/high-priority - Get high-delta samples for retraining
"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.abtest import ABTestLog, PredictionLog
from app.models.content import GeneratedContent
from app.models.user import User
from app.api.auth import get_current_user

router = APIRouter(prefix="/abtest", tags=["A/B Testing"])


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


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/log-result", response_model=LogResultResponse)
async def log_abtest_result(
    request: LogResultRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Log actual engagement results for a published content piece.

    This endpoint records the real performance of content that was previously
    predicted by the ML model. The data is used for:
    - Model performance monitoring
    - Identifying high-delta samples for retraining
    - A/B test analysis

    The system automatically calculates:
    - actual_engagement: Weighted engagement score (likes + comments*3 + saves*5 + shares*4)
    - delta_percent: Difference from predicted RPI
    - is_high_priority: True if delta > 20% (valuable for retraining)
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
