"""
KPI Weights API - Endpoints for Multi-Objective Engagement Configuration

Endpoints for managing user-configurable engagement weights:
- GET /kpi/weights: Get weights for a business
- POST /kpi/weights: Create/update weights
- GET /kpi/templates: Get available weight templates
- POST /kpi/preview: Preview how weights affect RPI

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
    KPIWeightsCreate,
    KPIWeightsResponse,
    KPIWeightsPreview,
    KPITemplate,
    KPITemplatesResponse,
    KPI_TEMPLATES,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Helper Functions
# =============================================================================

async def get_user_weights(
    db: AsyncSession,
    user_id: int,
    business_id: int,
    niche: Optional[str] = None
) -> Optional[UserKPIWeights]:
    """Get active weights for a user+business+niche combination."""
    query = select(UserKPIWeights).where(
        and_(
            UserKPIWeights.user_id == user_id,
            UserKPIWeights.business_id == business_id,
            UserKPIWeights.is_active == True,
        )
    )
    if niche:
        query = query.where(UserKPIWeights.niche == niche)
    else:
        query = query.where(UserKPIWeights.niche.is_(None))

    result = await db.execute(query)
    return result.scalar_one_or_none()


def model_to_response(model: UserKPIWeights) -> KPIWeightsResponse:
    """Convert SQLAlchemy model to Pydantic response."""
    return KPIWeightsResponse(
        id=model.id,
        user_id=model.user_id,
        business_id=model.business_id,
        niche=model.niche,
        weights=EngagementWeights(
            likes_weight=model.likes_weight,
            comments_weight=model.comments_weight,
            shares_weight=model.shares_weight,
            saves_weight=model.saves_weight,
            views_weight=model.views_weight,
        ),
        template_name=model.template_name,
        description=model.description,
        is_active=model.is_active,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


# =============================================================================
# Endpoints
# =============================================================================

@router.get(
    "/weights",
    response_model=KPIWeightsResponse,
    summary="Get KPI Weights",
    description="""
    Retrieves the active engagement weights configuration for a business.

    If no custom weights are configured, returns default weights.

    **Parameters:**
    - `business_id`: The business to get weights for (required)
    - `niche`: Optional niche for niche-specific weights

    **Returns:**
    - Weights configuration with all metric weights
    - Template name if created from a template
    - Timestamps for tracking changes
    """
)
async def get_weights(
    business_id: int = Query(..., description="Business ID"),
    niche: Optional[str] = Query(None, description="Optional niche filter"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> KPIWeightsResponse:
    """Get active KPI weights for a business."""
    weights = await get_user_weights(db, current_user.id, business_id, niche)

    if weights is None:
        # Return default weights if none configured
        logger.info(f"No custom weights found for user={current_user.id}, business={business_id}. Using defaults.")
        from datetime import datetime
        return KPIWeightsResponse(
            id=0,
            user_id=current_user.id,
            business_id=business_id,
            niche=niche,
            weights=EngagementWeights(),  # Defaults
            template_name=None,
            description="Default weights (not customized)",
            is_active=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    return model_to_response(weights)


@router.get(
    "/weights/all",
    response_model=List[KPIWeightsResponse],
    summary="Get All KPI Weights for User",
    description="Retrieves all weight configurations for the current user across all businesses."
)
async def get_all_weights(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> List[KPIWeightsResponse]:
    """Get all KPI weight configurations for the user."""
    query = select(UserKPIWeights).where(
        and_(
            UserKPIWeights.user_id == current_user.id,
            UserKPIWeights.is_active == True,
        )
    )
    result = await db.execute(query)
    weights_list = result.scalars().all()

    return [model_to_response(w) for w in weights_list]


@router.post(
    "/weights",
    response_model=KPIWeightsResponse,
    summary="Create or Update KPI Weights",
    description="""
    Creates or updates engagement weights for a business.

    **Options:**
    1. Provide custom weights directly via `weights` object
    2. Use a template via `template_name` (overwrites custom weights)

    **Templates available:**
    - `brand_awareness`: Prioritizes likes and views
    - `leads_conversions`: Prioritizes saves and shares
    - `community`: Prioritizes comments
    - `viral`: Prioritizes shares
    - `balanced`: Equal emphasis on all metrics

    If weights already exist for this business+niche, they are updated.
    """
)
async def create_or_update_weights(
    request: KPIWeightsCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> KPIWeightsResponse:
    """Create or update KPI weights."""
    # Check if weights already exist
    existing = await get_user_weights(
        db, current_user.id, request.business_id, request.niche
    )

    if existing:
        # Update existing weights
        existing.likes_weight = request.weights.likes_weight
        existing.comments_weight = request.weights.comments_weight
        existing.shares_weight = request.weights.shares_weight
        existing.saves_weight = request.weights.saves_weight
        existing.views_weight = request.weights.views_weight
        existing.template_name = request.template_name
        existing.description = request.description

        await db.commit()
        await db.refresh(existing)
        logger.info(f"Updated KPI weights for user={current_user.id}, business={request.business_id}")
        return model_to_response(existing)

    # Create new weights
    new_weights = UserKPIWeights(
        user_id=current_user.id,
        business_id=request.business_id,
        niche=request.niche,
        likes_weight=request.weights.likes_weight,
        comments_weight=request.weights.comments_weight,
        shares_weight=request.weights.shares_weight,
        saves_weight=request.weights.saves_weight,
        views_weight=request.weights.views_weight,
        template_name=request.template_name,
        description=request.description,
        is_active=True,
    )

    db.add(new_weights)
    await db.commit()
    await db.refresh(new_weights)

    logger.info(f"Created KPI weights for user={current_user.id}, business={request.business_id}")
    return model_to_response(new_weights)


@router.delete(
    "/weights/{weights_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete KPI Weights",
    description="Deactivates (soft delete) a weight configuration."
)
async def delete_weights(
    weights_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Soft delete (deactivate) KPI weights."""
    query = select(UserKPIWeights).where(
        and_(
            UserKPIWeights.id == weights_id,
            UserKPIWeights.user_id == current_user.id,
        )
    )
    result = await db.execute(query)
    weights = result.scalar_one_or_none()

    if weights is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Weights with id {weights_id} not found"
        )

    weights.is_active = False
    await db.commit()
    logger.info(f"Deactivated KPI weights id={weights_id} for user={current_user.id}")


@router.get(
    "/templates",
    response_model=KPITemplatesResponse,
    summary="Get Available KPI Templates",
    description="""
    Returns all available predefined weight templates.

    Templates provide quick-start configurations for common use cases:
    - Brand Awareness: Best for new businesses or brand building
    - Leads/Conversions: Best for sales-focused accounts
    - Community: Best for local businesses building relationships
    - Viral: Best for content creators seeking reach
    - Balanced: Equal weight on all metrics
    """
)
async def get_templates() -> KPITemplatesResponse:
    """Get available KPI weight templates."""
    return KPITemplatesResponse(templates=KPI_TEMPLATES)


@router.post(
    "/preview",
    response_model=KPIWeightsPreview,
    summary="Preview Weight Impact",
    description="""
    Preview how custom weights would affect RPI calculation.

    Given a set of weights and sample metric predictions, calculates
    what the weighted RPI would be and compares to default weights.

    Useful for users to understand the impact before saving changes.
    """
)
async def preview_weights(
    weights: EngagementWeights,
    sample_likes: float = Query(default=100, description="Sample predicted likes"),
    sample_comments: float = Query(default=10, description="Sample predicted comments"),
    sample_shares: float = Query(default=5, description="Sample predicted shares"),
    sample_saves: float = Query(default=15, description="Sample predicted saves"),
    sample_views: float = Query(default=1000, description="Sample predicted views"),
) -> KPIWeightsPreview:
    """Preview how weights affect RPI calculation."""
    import numpy as np

    # Sample predictions (log scale for typical ML output)
    predictions = {
        "log_likes": np.log1p(sample_likes),
        "log_comments": np.log1p(sample_comments),
        "log_shares": np.log1p(sample_shares),
        "log_saves": np.log1p(sample_saves),
        "log_views": np.log1p(sample_views),
    }

    # Calculate weighted RPI with custom weights
    weights_vector = weights.to_vector()
    pred_vector = [
        predictions["log_likes"],
        predictions["log_comments"],
        predictions["log_shares"],
        predictions["log_saves"],
        predictions["log_views"],
    ]
    weighted_sum = np.dot(weights_vector, pred_vector)
    weighted_rpi = float(np.clip(weighted_sum / sum(weights_vector) * 20, 0, 100))

    # Calculate with default weights for comparison
    default_weights = EngagementWeights()
    default_vector = default_weights.to_vector()
    default_weighted_sum = np.dot(default_vector, pred_vector)
    default_rpi = float(np.clip(default_weighted_sum / sum(default_vector) * 20, 0, 100))

    # Build explanation
    normalized = weights.normalize()
    top_metric = max(
        [("likes", normalized.likes_weight),
         ("comments", normalized.comments_weight),
         ("shares", normalized.shares_weight),
         ("saves", normalized.saves_weight),
         ("views", normalized.views_weight)],
        key=lambda x: x[1]
    )

    explanation = (
        f"Con estos pesos, tu RPI prioriza {top_metric[0]} "
        f"({top_metric[1]*100:.0f}% del peso total). "
        f"RPI calculado: {weighted_rpi:.1f}/100."
    )

    return KPIWeightsPreview(
        weights=weights,
        sample_prediction=predictions,
        weighted_rpi=weighted_rpi,
        explanation=explanation,
        comparison_to_default={
            "default_rpi": default_rpi,
            "difference": weighted_rpi - default_rpi,
            "percent_change": ((weighted_rpi - default_rpi) / max(default_rpi, 1)) * 100,
        }
    )


@router.post(
    "/weights/from-template",
    response_model=KPIWeightsResponse,
    summary="Create Weights from Template",
    description="Creates new weights configuration using a predefined template."
)
async def create_from_template(
    business_id: int,
    template_name: str = Query(
        ...,
        description="Template name",
        pattern="^(brand_awareness|leads_conversions|community|viral|balanced)$"
    ),
    niche: Optional[str] = Query(None, description="Optional niche"),
    description: Optional[str] = Query(None, max_length=500, description="Custom description"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> KPIWeightsResponse:
    """Create weights from a template."""
    # Create request with template
    request = KPIWeightsCreate(
        business_id=business_id,
        niche=niche,
        template_name=template_name,
        description=description or f"Created from '{template_name}' template",
    )

    # Use the main create/update endpoint
    return await create_or_update_weights(request, current_user, db)
