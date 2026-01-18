"""
Light Mode Configuration API
=============================

API endpoints for configuring lightweight multimodal processing.

Endpoints:
- GET /api/light-mode/info - Get available light mode options
- GET /api/light-mode/{business_id} - Get current light mode config
- PUT /api/light-mode/{business_id} - Update light mode config
- GET /api/light-mode/{business_id}/benchmark - Get performance estimates
- POST /api/light-mode/benchmark - Run benchmark on sample media

Author: ML Engineering Team
"""

from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
import logging

from app.core.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()
settings = get_settings()


# =============================================================================
# Pydantic Models
# =============================================================================

class LightModeOption(BaseModel):
    """A light mode configuration option."""
    value: str
    label: str
    description: str
    whisper_model: str
    max_duration_seconds: float
    ocr_max_frames: int
    estimated_time_seconds: float
    estimated_ram_mb: float
    is_recommended: bool


class LightModeConfig(BaseModel):
    """Current light mode configuration for a business."""
    business_id: int
    light_mode_enabled: bool
    whisper_model: str
    max_duration_seconds: float
    ocr_max_frames: int
    use_thumbnail: bool
    cache_enabled: bool
    skip_non_video: bool
    description: str


class LightModeUpdate(BaseModel):
    """Request to update light mode configuration."""
    light_mode_enabled: Optional[bool] = None
    whisper_model: Optional[str] = Field(None, description="'tiny' or 'base'")
    max_duration_seconds: Optional[float] = Field(None, ge=1.0, le=30.0)
    ocr_max_frames: Optional[int] = Field(None, ge=1, le=30)
    use_thumbnail: Optional[bool] = None
    cache_enabled: Optional[bool] = None
    skip_non_video: Optional[bool] = None


class LightModeBenchmark(BaseModel):
    """Benchmark estimate for light mode processing."""
    mode: str
    estimated_time_seconds: float
    estimated_ram_mb: float
    whisper_model: str
    ocr_frames: int
    hook_duration: float
    is_current: bool
    description: str


class LightModeBenchmarkResponse(BaseModel):
    """Response with benchmark estimates."""
    business_id: int
    current_mode: str
    benchmarks: Dict[str, LightModeBenchmark]
    recommendation: Dict[str, str]
    time_comparison: Dict[str, Any]


class LightModeInfoResponse(BaseModel):
    """Response with available light mode options."""
    available_modes: List[LightModeOption]
    current_default: str
    description: str


class LightModeCacheStats(BaseModel):
    """Cache statistics for light mode processing."""
    total_entries: int
    cache_enabled: bool
    cache_file: str
    ttl_hours: float


# =============================================================================
# In-Memory Configuration Store (per business)
# =============================================================================
# In production, this would be stored in the database

_business_light_configs: Dict[int, Dict[str, Any]] = {}


def get_business_light_config(business_id: int) -> Dict[str, Any]:
    """Get light mode config for a business, with defaults from settings."""
    if business_id not in _business_light_configs:
        _business_light_configs[business_id] = {
            "light_mode_enabled": settings.LIGHT_MODE_ENABLED,
            "whisper_model": settings.LIGHT_WHISPER_MODEL,
            "max_duration_seconds": settings.LIGHT_WHISPER_MAX_DURATION,
            "ocr_max_frames": settings.LIGHT_OCR_MAX_FRAMES,
            "use_thumbnail": settings.LIGHT_OCR_USE_THUMBNAIL,
            "cache_enabled": settings.LIGHT_CACHE_ENABLED,
            "skip_non_video": settings.LIGHT_SKIP_NON_VIDEO,
        }
    return _business_light_configs[business_id]


def update_business_light_config(business_id: int, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update light mode config for a business."""
    config = get_business_light_config(business_id)
    for key, value in updates.items():
        if value is not None and key in config:
            config[key] = value
    _business_light_configs[business_id] = config
    return config


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("/info", response_model=LightModeInfoResponse)
async def get_light_mode_info():
    """
    Get available light mode configuration options.

    Returns available modes with their performance characteristics.
    """
    modes = [
        LightModeOption(
            value="ultra_light",
            label="Ultra Ligero",
            description="Whisper tiny, solo 3s audio, 3 frames OCR. Maximo ahorro.",
            whisper_model="tiny",
            max_duration_seconds=3.0,
            ocr_max_frames=3,
            estimated_time_seconds=3.0,
            estimated_ram_mb=150,
            is_recommended=False,
        ),
        LightModeOption(
            value="light",
            label="Ligero (Recomendado)",
            description="Whisper tiny, primeros 3s, 5 frames OCR. Balance optimo.",
            whisper_model="tiny",
            max_duration_seconds=3.0,
            ocr_max_frames=5,
            estimated_time_seconds=5.0,
            estimated_ram_mb=200,
            is_recommended=True,
        ),
        LightModeOption(
            value="balanced",
            label="Balanceado",
            description="Whisper base, primeros 5s, 10 frames OCR. Mejor precision.",
            whisper_model="base",
            max_duration_seconds=5.0,
            ocr_max_frames=10,
            estimated_time_seconds=10.0,
            estimated_ram_mb=350,
            is_recommended=False,
        ),
        LightModeOption(
            value="full",
            label="Completo",
            description="Whisper base, audio completo, todos los frames. Maxima precision.",
            whisper_model="base",
            max_duration_seconds=30.0,
            ocr_max_frames=30,
            estimated_time_seconds=20.0,
            estimated_ram_mb=500,
            is_recommended=False,
        ),
    ]

    return LightModeInfoResponse(
        available_modes=modes,
        current_default="light",
        description=(
            "Modo Ligero optimiza el procesamiento de Whisper/EasyOCR para "
            "reducir tiempo y recursos sin perder calidad significativa. "
            "Recomendado para sobremesa normal."
        ),
    )


@router.get("/{business_id}", response_model=LightModeConfig)
async def get_light_mode_config(business_id: int):
    """
    Get current light mode configuration for a business.

    Args:
        business_id: Business ID

    Returns:
        Current light mode configuration
    """
    config = get_business_light_config(business_id)

    # Determine description based on config
    if not config["light_mode_enabled"]:
        description = "Modo completo - procesamiento full (maximo 20s)"
    elif config["whisper_model"] == "tiny" and config["max_duration_seconds"] <= 3.0:
        description = "Modo ligero - procesamiento rapido (~5s)"
    else:
        description = "Modo balanceado - procesamiento moderado (~10s)"

    return LightModeConfig(
        business_id=business_id,
        light_mode_enabled=config["light_mode_enabled"],
        whisper_model=config["whisper_model"],
        max_duration_seconds=config["max_duration_seconds"],
        ocr_max_frames=config["ocr_max_frames"],
        use_thumbnail=config["use_thumbnail"],
        cache_enabled=config["cache_enabled"],
        skip_non_video=config["skip_non_video"],
        description=description,
    )


@router.put("/{business_id}", response_model=LightModeConfig)
async def update_light_mode_config(business_id: int, update: LightModeUpdate):
    """
    Update light mode configuration for a business.

    Args:
        business_id: Business ID
        update: Configuration updates

    Returns:
        Updated configuration
    """
    # Validate whisper model
    if update.whisper_model and update.whisper_model not in ("tiny", "base"):
        raise HTTPException(
            status_code=400,
            detail="whisper_model must be 'tiny' or 'base'"
        )

    updates = {k: v for k, v in update.dict().items() if v is not None}
    config = update_business_light_config(business_id, updates)

    logger.info(
        f"Light mode config updated for business {business_id}: "
        f"enabled={config['light_mode_enabled']}, model={config['whisper_model']}"
    )

    # Determine description
    if not config["light_mode_enabled"]:
        description = "Modo completo - procesamiento full (maximo 20s)"
    elif config["whisper_model"] == "tiny" and config["max_duration_seconds"] <= 3.0:
        description = "Modo ligero - procesamiento rapido (~5s)"
    else:
        description = "Modo balanceado - procesamiento moderado (~10s)"

    return LightModeConfig(
        business_id=business_id,
        light_mode_enabled=config["light_mode_enabled"],
        whisper_model=config["whisper_model"],
        max_duration_seconds=config["max_duration_seconds"],
        ocr_max_frames=config["ocr_max_frames"],
        use_thumbnail=config["use_thumbnail"],
        cache_enabled=config["cache_enabled"],
        skip_non_video=config["skip_non_video"],
        description=description,
    )


@router.get("/{business_id}/benchmark", response_model=LightModeBenchmarkResponse)
async def get_light_mode_benchmark(business_id: int):
    """
    Get performance benchmark estimates for different light modes.

    Compares processing time and RAM usage across modes.
    """
    config = get_business_light_config(business_id)

    # Determine current mode
    if not config["light_mode_enabled"]:
        current_mode = "full"
    elif config["whisper_model"] == "tiny" and config["max_duration_seconds"] <= 3.0:
        current_mode = "light"
    elif config["max_duration_seconds"] <= 5.0:
        current_mode = "balanced"
    else:
        current_mode = "full"

    benchmarks = {
        "ultra_light": LightModeBenchmark(
            mode="ultra_light",
            estimated_time_seconds=3.0,
            estimated_ram_mb=150,
            whisper_model="tiny",
            ocr_frames=3,
            hook_duration=3.0,
            is_current=(current_mode == "ultra_light"),
            description="Maximo ahorro de recursos",
        ),
        "light": LightModeBenchmark(
            mode="light",
            estimated_time_seconds=5.0,
            estimated_ram_mb=200,
            whisper_model="tiny",
            ocr_frames=5,
            hook_duration=3.0,
            is_current=(current_mode == "light"),
            description="Balance optimo (recomendado)",
        ),
        "balanced": LightModeBenchmark(
            mode="balanced",
            estimated_time_seconds=10.0,
            estimated_ram_mb=350,
            whisper_model="base",
            ocr_frames=10,
            hook_duration=5.0,
            is_current=(current_mode == "balanced"),
            description="Mejor precision, mas recursos",
        ),
        "full": LightModeBenchmark(
            mode="full",
            estimated_time_seconds=20.0,
            estimated_ram_mb=500,
            whisper_model="base",
            ocr_frames=30,
            hook_duration=30.0,
            is_current=(current_mode == "full"),
            description="Maxima precision, maximo tiempo",
        ),
    }

    return LightModeBenchmarkResponse(
        business_id=business_id,
        current_mode=current_mode,
        benchmarks={k: v.dict() for k, v in benchmarks.items()},
        recommendation={
            "mode": "light",
            "reason": (
                "Modo ligero recomendado para sobremesa normal. "
                "75% mas rapido con minima perdida de calidad en hook detection."
            ),
        },
        time_comparison={
            "light_vs_full_seconds_saved": 15.0,
            "light_vs_full_percent_saved": 75.0,
            "example": "Reel procesado: 5s (light) vs 20s (full)",
        },
    )


@router.get("/cache/stats", response_model=LightModeCacheStats)
async def get_cache_stats():
    """
    Get light mode media cache statistics.

    Returns cache size and configuration.
    """
    try:
        from ml.light_processors import get_light_processor

        processor = get_light_processor()
        stats = processor.get_cache_stats()

        return LightModeCacheStats(
            total_entries=stats.get("total_entries", 0),
            cache_enabled=settings.LIGHT_CACHE_ENABLED,
            cache_file=stats.get("cache_file", "N/A"),
            ttl_hours=settings.LIGHT_CACHE_TTL_HOURS,
        )
    except Exception as e:
        logger.warning(f"Could not get cache stats: {e}")
        return LightModeCacheStats(
            total_entries=0,
            cache_enabled=settings.LIGHT_CACHE_ENABLED,
            cache_file="N/A",
            ttl_hours=settings.LIGHT_CACHE_TTL_HOURS,
        )


@router.post("/cache/clear")
async def clear_cache():
    """
    Clear the light mode media cache.

    Use when you want to force reprocessing of all media.
    """
    try:
        from ml.light_processors import get_light_processor

        processor = get_light_processor()
        if processor._cache:
            processor._cache.clear()
            logger.info("Light mode cache cleared")
            return {"success": True, "message": "Cache cleared successfully"}
        return {"success": True, "message": "Cache not enabled"}
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quick-toggle/{business_id}")
async def quick_toggle_light_mode(
    business_id: int,
    enabled: bool = Query(..., description="Enable or disable light mode")
):
    """
    Quick toggle for light mode on/off.

    Convenient endpoint for dashboard toggle.
    """
    config = update_business_light_config(business_id, {"light_mode_enabled": enabled})

    logger.info(
        f"Light mode {'enabled' if enabled else 'disabled'} for business {business_id}"
    )

    return {
        "success": True,
        "business_id": business_id,
        "light_mode_enabled": enabled,
        "message": f"Modo ligero {'activado' if enabled else 'desactivado'}",
        "processing_estimate": "~5s por video" if enabled else "~20s por video",
    }
