"""
Embedding Configuration API Endpoints

Provides endpoints for configuring embedding precision levels per business.

Precision Levels:
- "low" (128 dims): Ultra fast, lower precision
- "medium" (256 dims): Balanced precision/speed
- "high" (384 dims): Full precision
- "max" (384 dims): Full raw embeddings - mejor matices creativos/locales (default)

Default is "max" - safe for typical SMB data volumes (100-2000 posts, train <1min, RAM <2GB)
"""
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.models.business import Business, EmbeddingPrecision
from app.api.deps import get_current_user
from app.models.user import User

router = APIRouter(prefix="/embeddings", tags=["Embedding Configuration"])


# =============================================================================
# Schema Models
# =============================================================================

class EmbeddingPrecisionUpdate(BaseModel):
    """Request to update embedding precision."""
    precision: EmbeddingPrecision = Field(
        default=EmbeddingPrecision.MAX,
        description="Embedding precision level"
    )


class EmbeddingPrecisionResponse(BaseModel):
    """Response with current embedding precision."""
    business_id: int
    precision: EmbeddingPrecision
    dimensions: int
    description: str
    performance_notes: str


class EmbeddingConfigInfo(BaseModel):
    """Information about available embedding configurations."""
    available_precisions: list[Dict[str, Any]]
    current_precision: Optional[EmbeddingPrecision] = None
    recommended: str = "max"


# =============================================================================
# Helper Functions
# =============================================================================

PRECISION_INFO = {
    "low": {
        "dimensions": 128,
        "label": "Baja",
        "description": "Ultra rapido, menor precision semantica",
        "performance": "Mas rapido, menor RAM. Para datasets muy grandes (>5000 posts).",
        "use_case": "Volumen alto, velocidad prioritaria"
    },
    "medium": {
        "dimensions": 256,
        "label": "Media",
        "description": "Balance entre precision y velocidad",
        "performance": "Buen balance. Adecuado para 2000-5000 posts.",
        "use_case": "Balance precision/velocidad"
    },
    "high": {
        "dimensions": 384,
        "label": "Alta",
        "description": "Precision completa, todas las dimensiones",
        "performance": "Todas las dimensiones. Training rapido con <2000 posts.",
        "use_case": "Precision maxima"
    },
    "max": {
        "dimensions": 384,
        "label": "Maxima (Recomendada)",
        "description": "Embeddings raw completos - captura mejor matices creativos y slang local (Almeria/andaluz)",
        "performance": "Seguro en PC normal: train <1min, RAM <2GB con 100-2000 posts tipicos de SMB.",
        "use_case": "SMB tipico, mejor precision semantica regional"
    }
}


def get_precision_response(business_id: int, precision: EmbeddingPrecision) -> EmbeddingPrecisionResponse:
    """Build response for a precision level."""
    info = PRECISION_INFO.get(precision.value, PRECISION_INFO["max"])
    return EmbeddingPrecisionResponse(
        business_id=business_id,
        precision=precision,
        dimensions=info["dimensions"],
        description=info["description"],
        performance_notes=info["performance"]
    )


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("/info", response_model=EmbeddingConfigInfo)
async def get_embedding_info():
    """
    Get information about available embedding precision levels.

    Returns detailed info about each precision option including:
    - Dimensions
    - Performance characteristics
    - Use cases
    - Recommended default
    """
    available = []
    for key, info in PRECISION_INFO.items():
        available.append({
            "value": key,
            "label": info["label"],
            "dimensions": info["dimensions"],
            "description": info["description"],
            "performance": info["performance"],
            "use_case": info["use_case"],
            "is_recommended": key == "max"
        })

    return EmbeddingConfigInfo(
        available_precisions=available,
        recommended="max"
    )


@router.get("/{business_id}", response_model=EmbeddingPrecisionResponse)
async def get_business_embedding_precision(
    business_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get current embedding precision configuration for a business.

    Returns:
    - Current precision level
    - Number of dimensions
    - Description and performance notes
    """
    # Fetch business
    result = await db.execute(
        select(Business).where(
            Business.id == business_id,
            Business.user_id == current_user.id
        )
    )
    business = result.scalar_one_or_none()

    if not business:
        raise HTTPException(status_code=404, detail="Negocio no encontrado")

    return get_precision_response(business_id, business.embedding_precision)


@router.put("/{business_id}", response_model=EmbeddingPrecisionResponse)
async def update_business_embedding_precision(
    business_id: int,
    update: EmbeddingPrecisionUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update embedding precision configuration for a business.

    Available precision levels:
    - **low** (128 dims): Ultra fast, lower semantic precision
    - **medium** (256 dims): Balanced precision and speed
    - **high** (384 dims): Full precision
    - **max** (384 dims): Full raw embeddings, best for regional nuances (default)

    Note: Changing precision requires model retraining for optimal results.
    """
    # Fetch business
    result = await db.execute(
        select(Business).where(
            Business.id == business_id,
            Business.user_id == current_user.id
        )
    )
    business = result.scalar_one_or_none()

    if not business:
        raise HTTPException(status_code=404, detail="Negocio no encontrado")

    # Update precision
    old_precision = business.embedding_precision
    business.embedding_precision = update.precision
    await db.commit()
    await db.refresh(business)

    response = get_precision_response(business_id, business.embedding_precision)

    # Add note about retraining if precision changed
    if old_precision != update.precision:
        response.performance_notes += " | Nota: Reentrenar modelo para mejores resultados."

    return response


@router.get("/{business_id}/benchmark", response_model=Dict[str, Any])
async def get_embedding_benchmark_estimate(
    business_id: int,
    n_samples: int = 500,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get estimated performance benchmarks for different precision levels.

    Provides estimates for:
    - Training time
    - RAM usage
    - Feature count

    Based on typical SMB data volumes (default 500 samples).
    """
    # Fetch business
    result = await db.execute(
        select(Business).where(
            Business.id == business_id,
            Business.user_id == current_user.id
        )
    )
    business = result.scalar_one_or_none()

    if not business:
        raise HTTPException(status_code=404, detail="Negocio no encontrado")

    # Estimate benchmarks for each precision
    benchmarks = {}
    base_heuristic_features = 50  # Approximate non-embedding features

    for precision, info in PRECISION_INFO.items():
        dims = info["dimensions"]
        # Caption + transcript + OCR embeddings
        total_embedding_dims = dims * 3
        total_features = base_heuristic_features + total_embedding_dims + 10  # +10 interaction features

        # Rough estimates based on typical XGBoost performance
        # These are conservative estimates for a normal desktop
        bytes_per_sample = total_features * 4  # float32
        total_bytes = n_samples * bytes_per_sample
        ram_mb = total_bytes / (1024 * 1024)

        # Training time estimates (very rough - depends heavily on hardware)
        # XGBoost is fast, these are conservative
        train_time_seconds = (n_samples * total_features) / 500000  # ~rough heuristic

        benchmarks[precision] = {
            "dimensions_per_modality": dims,
            "total_embedding_features": total_embedding_dims,
            "total_features": total_features,
            "estimated_ram_mb": round(ram_mb, 2),
            "estimated_train_seconds": round(max(train_time_seconds, 1), 1),
            "is_lightweight": ram_mb < 500,  # < 500MB is definitely lightweight
            "is_current": precision == business.embedding_precision.value
        }

    return {
        "business_id": business_id,
        "current_precision": business.embedding_precision.value,
        "n_samples": n_samples,
        "benchmarks": benchmarks,
        "recommendation": {
            "precision": "max",
            "reason": f"Con {n_samples} samples, 'max' es seguro y ofrece mejor precision semantica."
        }
    }
