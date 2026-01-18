"""
Embedding Configuration API Endpoints

Provides endpoints for configuring embedding precision levels per business.

Precision Levels (precision es REQUERIDO):
- "ultra_low" (64 dims): Ultra rapido para PC modesto/sobremesa Almeria
- "low" (128 dims): Recomendado sobremesa normal (NEW DEFAULT)
- "medium" (256 dims): Balance precision/velocidad
- "high" (384 dims): Full dims con TruncatedSVD
- "max" (full raw 384): Sin reduccion - mejor matices creativos/slang local

Default changed to "low" - seguro para sobremesa normal Almeria (100-2000 posts, train <1min, RAM <1GB)
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
        default=EmbeddingPrecision.LOW,  # Changed from MAX - safer for typical sobremesa
        description="Embedding precision level: ultra_low (64), low (128), medium (256), high (384), max (full raw)"
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
    recommended: str = "low"  # Changed from "max" - safer for typical sobremesa


# =============================================================================
# Helper Functions
# =============================================================================

PRECISION_INFO = {
    "ultra_low": {
        "dimensions": 64,
        "label": "Ultra Baja (64 dims - ultra rapido)",
        "description": "Ultra rapido para PC modesto/sobremesa Almeria",
        "performance": "Muy rapido, minimo RAM (~0.3GB). Ideal para PC antiguo o volumen muy alto.",
        "use_case": "PC modesto, datasets muy grandes (>10000 posts)"
    },
    "low": {
        "dimensions": 128,
        "label": "Baja (128 dims - Recomendada)",
        "description": "Recomendado para sobremesa normal Almeria - balance seguro",
        "performance": "Rapido, bajo RAM (~0.5GB). Train <30s. Ideal para SMB tipico.",
        "use_case": "Sobremesa normal, volumen tipico SMB (100-2000 posts)"
    },
    "medium": {
        "dimensions": 256,
        "label": "Media (256 dims)",
        "description": "Balance entre precision y velocidad",
        "performance": "Buen balance, RAM moderado (~1GB). Para 2000-5000 posts.",
        "use_case": "Balance precision/velocidad"
    },
    "high": {
        "dimensions": 384,
        "label": "Alta (384 dims)",
        "description": "Precision completa con TruncatedSVD (preserva varianza)",
        "performance": "Full dims con reduccion. Train <1min con <2000 posts.",
        "use_case": "Precision maxima con reduccion"
    },
    "max": {
        "dimensions": 384,
        "label": "Maxima (full raw)",
        "description": "Embeddings raw completos sin reduccion - mejor matices creativos y slang local (Almeria/andaluz)",
        "performance": "Sin reduccion, RAM ~2GB. Solo si tienes buen hardware.",
        "use_case": "Hardware potente, maxima precision semantica regional"
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
    - Recommended default (now "low" for typical sobremesa)
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
            "is_recommended": key == "low"  # Changed from "max" - safer for typical sobremesa
        })

    return EmbeddingConfigInfo(
        available_precisions=available,
        recommended="low"  # Changed from "max"
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
    - **ultra_low** (64 dims): Ultra rapido para PC modesto
    - **low** (128 dims): Recomendado sobremesa normal (NEW DEFAULT)
    - **medium** (256 dims): Balance precision/velocidad
    - **high** (384 dims): Full dims con TruncatedSVD
    - **max** (full raw 384): Sin reduccion - mejor matices creativos/slang

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
            "precision": "low",
            "reason": f"Con {n_samples} samples, 'low' (128 dims) es seguro para sobremesa normal Almeria y ofrece buen balance precision/velocidad."
        }
    }
