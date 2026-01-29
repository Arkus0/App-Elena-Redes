"""
Trend Velocity API - Endpoints for Real-Time Trend Freshness Validation

Endpoints para verificar la frescura de audios y hashtags antes de usarlos:
- POST /check: Verifica velocidad de una tendencia individual
- POST /check/batch: Verifica múltiples tendencias
- POST /validate-recipe: Valida y filtra elementos de una receta viral
- GET /status/{status}: Descripción de un estado de tendencia
"""
import asyncio
import logging
from typing import List

from fastapi import APIRouter, HTTPException, status

from app.schemas.trend_velocity import (
    TrendCheckRequest,
    TrendVelocityResponse,
    VelocityMetricsSchema,
    VelocityPercentagesSchema,
    VelocityScoreSchema,
    EngagementStatsSchema,
    BatchTrendCheckRequest,
    BatchTrendCheckResponse,
    ValidateRecipeRequest,
    ValidateRecipeResponse,
    ValidatedElementSchema,
    FilteredElementSchema,
    ValidationSummarySchema,
    QuickCheckResponse,
    TrendStatusSummaryResponse,
    TrendStatusEnum,
    TrendTypeEnum,
)
from app.services.trend_velocity import (
    get_trend_velocity_checker,
    TrendVelocityChecker,
    TrendType,
    TrendStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_checker() -> TrendVelocityChecker:
    """Obtiene la instancia del checker de velocidad."""
    return get_trend_velocity_checker()


def _convert_trend_type(trend_type: TrendTypeEnum) -> TrendType:
    """Convierte enum de schema a enum de servicio."""
    mapping = {
        TrendTypeEnum.AUDIO: TrendType.AUDIO,
        TrendTypeEnum.HASHTAG: TrendType.HASHTAG,
        TrendTypeEnum.CHALLENGE: TrendType.CHALLENGE,
        TrendTypeEnum.EFFECT: TrendType.EFFECT,
    }
    return mapping[trend_type]


@router.post(
    "/check",
    response_model=TrendVelocityResponse,
    summary="Check Trend Velocity",
    description="""
    Verifica la velocidad/frescura de una tendencia específica (audio, hashtag, etc.).

    **Lógica de clasificación:**
    - TRENDING: >= 50% de muestras en últimas 48h → MUY FRESCO, usar
    - RISING: >= 30% de muestras en última semana → En crecimiento, usar
    - STABLE: Distribución normal → Estable, usar con precaución
    - STALE: >= 80% de muestras hace > 2 semanas → CADUCADO, NO USAR

    **Uso recomendado:**
    Llamar este endpoint ANTES de incluir un audio o hashtag en una receta.
    Si el resultado es STALE, buscar una alternativa más fresca.
    """
)
async def check_trend_velocity(request: TrendCheckRequest) -> TrendVelocityResponse:
    """Verifica la velocidad de una tendencia."""
    try:
        checker = _get_checker()

        result = await checker.check_trend_velocity(
            trend_type=_convert_trend_type(request.trend_type),
            trend_identifier=request.trend_identifier,
            platform=request.platform,
            sample_size=request.sample_size
        )

        return TrendVelocityResponse(
            trend_type=request.trend_type,
            trend_identifier=result.trend_identifier,
            status=TrendStatusEnum(result.status.value),
            is_fresh=result.is_fresh,
            is_stale=result.is_stale,
            metrics=VelocityMetricsSchema(
                samples_analyzed=result.samples_analyzed,
                samples_last_48h=result.samples_last_48h,
                samples_last_week=result.samples_last_week,
                samples_older_2weeks=result.samples_older_2weeks
            ),
            percentages=VelocityPercentagesSchema(
                pct_last_48h=round(result.pct_last_48h * 100, 1),
                pct_last_week=round(result.pct_last_week * 100, 1),
                pct_older_2weeks=round(result.pct_older_2weeks * 100, 1)
            ),
            velocity=VelocityScoreSchema(
                score=round(result.velocity_score, 3),
                acceleration=round(result.acceleration, 3)
            ),
            engagement=EngagementStatsSchema(
                avg_engagement=round(result.avg_engagement, 2),
                peak_date=result.peak_date.isoformat() if result.peak_date else None
            ),
            recommendation=result.recommendation,
            analyzed_at=result.analyzed_at
        )

    except Exception as e:
        logger.error(f"Trend velocity check error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check trend velocity: {str(e)}"
        )


@router.post(
    "/check/batch",
    response_model=BatchTrendCheckResponse,
    summary="Batch Check Multiple Trends",
    description="""
    Verifica múltiples tendencias en paralelo.

    **Optimizado para:**
    - Validar una lista completa de hashtags recomendados
    - Verificar varios audios antes de elegir uno
    - Filtrar tendencias caducadas de un conjunto

    **Resultado:**
    - fresh_trends: Ordenadas por velocity_score (más trending primero)
    - stale_trends: Tendencias a evitar
    """
)
async def check_trends_batch(request: BatchTrendCheckRequest) -> BatchTrendCheckResponse:
    """Verifica múltiples tendencias en paralelo."""
    try:
        checker = _get_checker()

        # Convertir requests a formato interno
        trends_to_check = [
            {
                "identifier": t.trend_identifier,
                "type": _convert_trend_type(t.trend_type),
                "platform": t.platform
            }
            for t in request.trends
        ]

        # Procesar en paralelo por tipo
        # Bolt Optimization: Use asyncio.gather for concurrent execution
        tasks = []
        for trend_req in request.trends:
            tasks.append(
                checker.check_trend_velocity(
                    trend_type=_convert_trend_type(trend_req.trend_type),
                    trend_identifier=trend_req.trend_identifier,
                    platform=trend_req.platform,
                    sample_size=trend_req.sample_size
                )
            )

        results = await asyncio.gather(*tasks)

        all_fresh = []
        all_stale = []

        for result, trend_req in zip(results, request.trends):
            response_item = TrendVelocityResponse(
                trend_type=trend_req.trend_type,
                trend_identifier=result.trend_identifier,
                status=TrendStatusEnum(result.status.value),
                is_fresh=result.is_fresh,
                is_stale=result.is_stale,
                metrics=VelocityMetricsSchema(
                    samples_analyzed=result.samples_analyzed,
                    samples_last_48h=result.samples_last_48h,
                    samples_last_week=result.samples_last_week,
                    samples_older_2weeks=result.samples_older_2weeks
                ),
                percentages=VelocityPercentagesSchema(
                    pct_last_48h=round(result.pct_last_48h * 100, 1),
                    pct_last_week=round(result.pct_last_week * 100, 1),
                    pct_older_2weeks=round(result.pct_older_2weeks * 100, 1)
                ),
                velocity=VelocityScoreSchema(
                    score=round(result.velocity_score, 3),
                    acceleration=round(result.acceleration, 3)
                ),
                engagement=EngagementStatsSchema(
                    avg_engagement=round(result.avg_engagement, 2),
                    peak_date=result.peak_date.isoformat() if result.peak_date else None
                ),
                recommendation=result.recommendation,
                analyzed_at=result.analyzed_at
            )

            if result.is_fresh:
                all_fresh.append(response_item)
            else:
                all_stale.append(response_item)

        # Ordenar frescos por velocity_score
        all_fresh.sort(key=lambda x: x.velocity.score, reverse=True)

        return BatchTrendCheckResponse(
            total_analyzed=len(request.trends),
            fresh_count=len(all_fresh),
            stale_count=len(all_stale),
            filtered_count=len(all_stale),
            fresh_trends=all_fresh,
            stale_trends=all_stale
        )

    except Exception as e:
        logger.error(f"Batch trend check error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to batch check trends: {str(e)}"
        )


@router.post(
    "/validate-recipe",
    response_model=ValidateRecipeResponse,
    summary="Validate Viral Recipe Elements",
    description="""
    Valida todos los elementos de una receta viral antes de devolverla al usuario.

    **CRÍTICO:** Este endpoint DEBE llamarse antes de entregar una receta.
    Filtra automáticamente audios y hashtags STALE (caducados) para evitar
    recomendar tendencias pasadas de moda.

    **Proceso:**
    1. Verifica cada audio en paralelo
    2. Verifica cada hashtag en paralelo
    3. Filtra elementos STALE
    4. Ordena elementos frescos por velocity_score

    **Resultado:**
    - validated_audios/hashtags: Elementos seguros para usar
    - filtered_audios/hashtags: Elementos eliminados con razón
    - has_stale_elements: Flag para alertar al usuario
    """
)
async def validate_recipe(request: ValidateRecipeRequest) -> ValidateRecipeResponse:
    """Valida y filtra elementos de una receta viral."""
    try:
        checker = _get_checker()

        result = await checker.validate_recipe_elements(
            audios=request.audios,
            hashtags=request.hashtags,
            platform=request.platform
        )

        # Convertir a schemas de response
        validated_audios = [
            ValidatedElementSchema(
                name=a["name"],
                status=a["status"],
                velocity_score=a.get("velocity_score")
            )
            for a in result["validated_audios"]
        ]

        filtered_audios = [
            FilteredElementSchema(
                name=a["name"],
                status=a["status"],
                reason=a["reason"]
            )
            for a in result["filtered_audios"]
        ]

        validated_hashtags = [
            ValidatedElementSchema(
                name=h["name"],
                status=h["status"],
                velocity_score=h.get("velocity_score")
            )
            for h in result["validated_hashtags"]
        ]

        filtered_hashtags = [
            FilteredElementSchema(
                name=h["name"],
                status=h["status"],
                reason=h["reason"]
            )
            for h in result["filtered_hashtags"]
        ]

        summary = ValidationSummarySchema(**result["summary"])

        return ValidateRecipeResponse(
            validated_audios=validated_audios,
            filtered_audios=filtered_audios,
            validated_hashtags=validated_hashtags,
            filtered_hashtags=filtered_hashtags,
            summary=summary,
            has_stale_elements=result["has_stale_elements"]
        )

    except Exception as e:
        logger.error(f"Recipe validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate recipe: {str(e)}"
        )


@router.get(
    "/quick/{trend_type}/{identifier}",
    response_model=QuickCheckResponse,
    summary="Quick Trend Check",
    description="Verificación rápida de una tendencia por URL path."
)
async def quick_check(
    trend_type: TrendTypeEnum,
    identifier: str,
    platform: str = "instagram"
) -> QuickCheckResponse:
    """Verificación rápida de tendencia."""
    try:
        checker = _get_checker()

        result = await checker.check_trend_velocity(
            trend_type=_convert_trend_type(trend_type),
            trend_identifier=identifier,
            platform=platform
        )

        return QuickCheckResponse(
            identifier=identifier,
            status=TrendStatusEnum(result.status.value),
            is_fresh=result.is_fresh,
            is_stale=result.is_stale,
            velocity_score=round(result.velocity_score, 3),
            recommendation=result.recommendation
        )

    except Exception as e:
        logger.error(f"Quick check error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quick check failed: {str(e)}"
        )


@router.get(
    "/status/{status_name}",
    response_model=TrendStatusSummaryResponse,
    summary="Get Status Description",
    description="Obtiene la descripción y umbrales de un estado de tendencia."
)
async def get_status_description(status_name: str) -> TrendStatusSummaryResponse:
    """Obtiene descripción de un estado de tendencia."""
    status_info = {
        "trending": {
            "status": TrendStatusEnum.TRENDING,
            "description": "Tendencia MUY FRESCA. El algoritmo está priorizando activamente este contenido. Úsala ahora para máximo alcance.",
            "threshold": ">= 50% de muestras en últimas 48 horas",
            "usable": True,
            "priority": 1
        },
        "rising": {
            "status": TrendStatusEnum.RISING,
            "description": "Tendencia en CRECIMIENTO. Buen momento para usarla. Todavía tiene momentum pero no está saturada.",
            "threshold": ">= 30% de muestras en última semana",
            "usable": True,
            "priority": 2
        },
        "stable": {
            "status": TrendStatusEnum.STABLE,
            "description": "Tendencia ESTABLE. Puede usarse pero no garantiza boost algorítmico. Considera combinar con elemento trending.",
            "threshold": "Distribución temporal normal",
            "usable": True,
            "priority": 3
        },
        "stale": {
            "status": TrendStatusEnum.STALE,
            "description": "TENDENCIA CADUCADA. NO USAR. El algoritmo ya no prioriza este contenido. Los nuevos posts no rendirán igual aunque tenga likes históricos.",
            "threshold": ">= 80% de muestras hace más de 2 semanas",
            "usable": False,
            "priority": 4
        },
        "unknown": {
            "status": TrendStatusEnum.UNKNOWN,
            "description": "No hay suficientes datos para evaluar. Procede con precaución o busca alternativa más establecida.",
            "threshold": "< 5 muestras encontradas",
            "usable": True,
            "priority": 5
        }
    }

    if status_name.lower() not in status_info:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid status. Valid values: {list(status_info.keys())}"
        )

    info = status_info[status_name.lower()]
    return TrendStatusSummaryResponse(**info)
