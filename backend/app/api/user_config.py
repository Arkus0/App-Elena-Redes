"""
User Config API - Unified Configuration Endpoints

Endpoints unificados para gestionar TODAS las configuraciones de usuario
que afectan los pipelines ML (ingest/train/inference).

Endpoints:
- GET  /api/v1/user/config             - Obtener config actual
- POST /api/v1/user/config             - Crear/actualizar config
- GET  /api/v1/user/config/info        - Obtener opciones disponibles
- GET  /api/v1/user/config/{config_id} - Obtener config por ID

SINCRONIZACION REAL-TIME:
Cuando el frontend guarda aqui, TODOS los pipelines usan la nueva config
automaticamente (ingest, train, inference, growth prediction).

Author: BrandPulse AI
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.core.database import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.schemas.user_config import (
    UserConfigCreate, UserConfigUpdate, UserConfigResponse,
    UserConfigInfoResponse, PipelineConfig
)
from app.services.user_config_service import user_config_service

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Helper to get current user (mock if auth not fully implemented)
# =============================================================================

async def get_current_user_id(
    user_id: Optional[int] = Query(None, description="User ID (for testing)")
) -> int:
    """
    Obtiene el user_id actual.

    En produccion, esto vendria del JWT token.
    Para desarrollo/testing, acepta query param.
    """
    if user_id is not None:
        return user_id
    # Default for testing
    return 1


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("/config/info", response_model=UserConfigInfoResponse)
async def get_config_info():
    """
    Obtiene opciones disponibles para configuracion.

    Retorna:
    - precision_options: Niveles de precision de embeddings
    - multimodal_options: Modos de procesamiento multimodal
    - kpi_templates: Plantillas predefinidas de KPI weights
    - default_config: Configuracion por defecto

    Usar para popular dropdowns y opciones en el frontend.
    """
    return user_config_service.get_config_info()


@router.get("/config", response_model=UserConfigResponse)
async def get_user_config(
    business_id: int = Query(..., description="Business ID"),
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Obtiene la configuracion actual del usuario.

    Si no existe, crea una con defaults seguros:
    - embedding_precision: "low" (128 dims)
    - multimodal_mode: "light" (~5s por video)
    - kpi_weights: balanced (shares > saves > views > comments > likes)

    Esta config es usada por TODOS los pipelines:
    - ingest.py: own_username para feedback, light_mode para procesamiento
    - ml_service.py: embedding_precision, kpi_weights para RPI
    - growth_prediction_engine.py: embeddings dims, kpi weights
    - train.py: embedding precision para features
    """
    config = await user_config_service.get_or_create_config(db, user_id, business_id)

    logger.info(
        f"Config retrieved: user={user_id}, business={business_id}, "
        f"precision={config.embedding_precision.value}, "
        f"multimodal={config.multimodal_mode.value}"
    )

    return UserConfigResponse.from_orm_with_dims(config)


@router.post("/config", response_model=UserConfigResponse)
async def save_user_config(
    config_data: UserConfigUpdate,
    business_id: int = Query(..., description="Business ID"),
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Guarda la configuracion del usuario.

    SINCRONIZACION REAL-TIME:
    Cuando guardas aqui, la nueva config afecta inmediatamente:
    - Ingestion: own_username detecta posts propios para feedback ML
    - Embeddings: precision cambia dimensiones de features
    - KPI Weights: RPI se calcula con tus pesos personalizados
    - Multimodal: light mode procesa videos en ~5s vs ~20s

    Campos actualizables:
    - discovery: hashtags, location_keywords, niche_keywords, follower filters
    - kpi_weights: likes, comments, shares, saves, views (0-20 cada uno)
    - embedding_precision: ultra_low/low/medium/high/max
    - multimodal_mode: light/full
    - light_mode_config: whisper_model, ocr_max_frames, etc.
    - own_instagram_username: para feedback loop (sin @)
    - own_tiktok_username: para feedback loop (sin @)
    """
    config = await user_config_service.update_config(
        db, user_id, business_id, config_data
    )

    logger.info(
        f"Config saved: user={user_id}, business={business_id}, "
        f"precision={config.embedding_precision.value} ({config.get_embedding_dims()} dims), "
        f"multimodal={config.multimodal_mode.value}, "
        f"own=@{config.own_instagram_username or 'N/A'}"
    )

    return UserConfigResponse.from_orm_with_dims(config)


@router.get("/config/pipeline", response_model=PipelineConfig)
async def get_pipeline_config(
    business_id: int = Query(..., description="Business ID"),
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Obtiene config en formato optimizado para pipelines ML.

    Este endpoint es para uso interno de los pipelines.
    Retorna formato simplificado con:
    - embedding_precision y dims
    - kpi_weights (raw y normalized)
    - multimodal config
    - own_username para feedback
    - helper methods (is_own_profile, get_weighted_rpi)
    """
    config = await user_config_service.get_pipeline_config(db, user_id, business_id)

    logger.info(
        f"Pipeline config loaded: user={user_id}, business={business_id}, "
        f"precision={config.embedding_precision} ({config.embedding_dims} dims), "
        f"multimodal={config.multimodal_mode}, "
        f"own=@{config.own_instagram_username or 'N/A'}"
    )

    return config


@router.post("/config/validate")
async def validate_config(
    config_data: UserConfigUpdate
):
    """
    Valida una configuracion sin guardarla.

    Util para preview en frontend antes de guardar.
    Retorna warnings si la config tiene valores extremos.
    """
    warnings = []
    info = []

    # Validate KPI weights
    if config_data.kpi_weights:
        w = config_data.kpi_weights
        total = w.likes + w.comments + w.shares + w.saves + w.views

        if total == 0:
            warnings.append("Todos los pesos KPI son 0 - RPI sera siempre 0")

        if w.shares > 15:
            info.append("Shares muy alto - enfoque extremo en viralidad")

        if w.comments > 10 and w.likes < 2:
            info.append("Comentarios muy alto vs likes - enfoque en engagement profundo")

    # Validate embedding precision
    if config_data.embedding_precision:
        p = config_data.embedding_precision.value
        if p == "max":
            info.append("Precision 'max' requiere mas RAM (~2GB) - solo para hardware potente")
        elif p == "ultra_low":
            info.append("Precision 'ultra_low' es muy rapida pero menos precisa")

    # Validate multimodal mode
    if config_data.multimodal_mode:
        if config_data.multimodal_mode.value == "full":
            info.append("Modo 'full' procesa ~20s por video - mas lento pero mas preciso")

    # Validate own username
    if config_data.own_instagram_username:
        username = config_data.own_instagram_username
        if "@" in username:
            info.append("El @ se elimina automaticamente del username")
        if len(username) < 3:
            warnings.append("Username muy corto - verifica que sea correcto")

    return {
        "valid": len(warnings) == 0,
        "warnings": warnings,
        "info": info,
        "config_preview": {
            "embedding_precision": config_data.embedding_precision.value if config_data.embedding_precision else None,
            "multimodal_mode": config_data.multimodal_mode.value if config_data.multimodal_mode else None,
            "own_instagram_username": config_data.own_instagram_username,
        }
    }


@router.delete("/config")
async def reset_config(
    business_id: int = Query(..., description="Business ID"),
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Resetea la configuracion a valores por defecto.

    Defaults seguros:
    - embedding_precision: "low" (128 dims)
    - multimodal_mode: "light"
    - kpi_weights: balanced (1, 2, 10, 5, 3)
    """
    from app.schemas.user_config import UserConfigCreate, EmbeddingPrecision, MultimodalMode

    # Create fresh default config
    default_data = UserConfigCreate(
        embedding_precision=EmbeddingPrecision.LOW,
        multimodal_mode=MultimodalMode.LIGHT,
    )

    config = await user_config_service.create_config(
        db, user_id, business_id, default_data
    )

    logger.info(f"Config reset to defaults: user={user_id}, business={business_id}")

    return {
        "success": True,
        "message": "Configuracion reseteada a valores por defecto",
        "config": UserConfigResponse.from_orm_with_dims(config)
    }
