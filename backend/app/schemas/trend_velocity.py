"""
Trend Velocity Schemas - Pydantic models for TrendVelocity API

Define los modelos de request/response para los endpoints de verificación
de frescura de tendencias en tiempo real.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class TrendStatusEnum(str, Enum):
    """Estados de frescura de una tendencia."""
    TRENDING = "trending"      # >= 50% en últimas 48h
    RISING = "rising"          # >= 30% en última semana
    STABLE = "stable"          # Distribución normal
    STALE = "stale"            # >= 80% hace > 2 semanas
    UNKNOWN = "unknown"        # Datos insuficientes


class TrendTypeEnum(str, Enum):
    """Tipos de elementos de tendencia."""
    AUDIO = "audio"
    HASHTAG = "hashtag"
    CHALLENGE = "challenge"
    EFFECT = "effect"


class TrendCheckRequest(BaseModel):
    """Request para verificar la frescura de una tendencia."""
    trend_type: TrendTypeEnum = Field(
        ...,
        description="Tipo de tendencia a verificar"
    )
    trend_identifier: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Identificador de la tendencia (nombre del audio, hashtag, etc.)"
    )
    platform: str = Field(
        "instagram",
        description="Plataforma a consultar: instagram, tiktok"
    )
    sample_size: int = Field(
        50,
        ge=10,
        le=100,
        description="Número de muestras a recolectar para el análisis"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "trend_type": "hashtag",
                    "trend_identifier": "#smallbusiness",
                    "platform": "instagram",
                    "sample_size": 50
                },
                {
                    "trend_type": "audio",
                    "trend_identifier": "original sound - viral creator",
                    "platform": "tiktok",
                    "sample_size": 50
                }
            ]
        }
    }


class VelocityMetricsSchema(BaseModel):
    """Métricas de muestras analizadas."""
    samples_analyzed: int = Field(..., description="Total de muestras analizadas")
    samples_last_48h: int = Field(..., description="Muestras de las últimas 48 horas")
    samples_last_week: int = Field(..., description="Muestras de la última semana")
    samples_older_2weeks: int = Field(..., description="Muestras de hace más de 2 semanas")


class VelocityPercentagesSchema(BaseModel):
    """Porcentajes de distribución temporal."""
    pct_last_48h: float = Field(..., description="Porcentaje en últimas 48h (0-100)")
    pct_last_week: float = Field(..., description="Porcentaje en última semana (0-100)")
    pct_older_2weeks: float = Field(..., description="Porcentaje hace >2 semanas (0-100)")


class VelocityScoreSchema(BaseModel):
    """Scores de velocidad y aceleración."""
    score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Score de velocidad (-1 decayendo, +1 creciendo)"
    )
    acceleration: float = Field(
        ...,
        description="Aceleración (segunda derivada)"
    )


class EngagementStatsSchema(BaseModel):
    """Estadísticas de engagement."""
    avg_engagement: float = Field(..., description="Engagement promedio de las muestras")
    peak_date: Optional[str] = Field(None, description="Fecha pico de actividad (ISO)")


class TrendVelocityResponse(BaseModel):
    """Response completa del análisis de velocidad de una tendencia."""
    trend_type: TrendTypeEnum = Field(..., description="Tipo de tendencia")
    trend_identifier: str = Field(..., description="Identificador de la tendencia")
    status: TrendStatusEnum = Field(..., description="Estado de frescura")
    is_fresh: bool = Field(..., description="Si la tendencia es fresca (usable)")
    is_stale: bool = Field(..., description="Si la tendencia está caducada")

    metrics: VelocityMetricsSchema = Field(..., description="Métricas de muestras")
    percentages: VelocityPercentagesSchema = Field(..., description="Distribución temporal")
    velocity: VelocityScoreSchema = Field(..., description="Scores de velocidad")
    engagement: EngagementStatsSchema = Field(..., description="Stats de engagement")

    recommendation: str = Field(..., description="Recomendación para el usuario")
    analyzed_at: str = Field(..., description="Timestamp del análisis (ISO)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "trend_type": "hashtag",
                    "trend_identifier": "#smallbusiness",
                    "status": "trending",
                    "is_fresh": True,
                    "is_stale": False,
                    "metrics": {
                        "samples_analyzed": 50,
                        "samples_last_48h": 28,
                        "samples_last_week": 12,
                        "samples_older_2weeks": 10
                    },
                    "percentages": {
                        "pct_last_48h": 56.0,
                        "pct_last_week": 24.0,
                        "pct_older_2weeks": 20.0
                    },
                    "velocity": {
                        "score": 0.72,
                        "acceleration": 0.15
                    },
                    "engagement": {
                        "avg_engagement": 4523.5,
                        "peak_date": "2024-03-14"
                    },
                    "recommendation": "Tendencia MUY FRESCA. Úsala ahora...",
                    "analyzed_at": "2024-03-15T14:30:00"
                }
            ]
        }
    }


class BatchTrendCheckRequest(BaseModel):
    """Request para verificar múltiples tendencias."""
    trends: List[TrendCheckRequest] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Lista de tendencias a verificar (máx 20)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "trends": [
                        {"trend_type": "hashtag", "trend_identifier": "#smallbusiness", "platform": "instagram"},
                        {"trend_type": "audio", "trend_identifier": "trending sound", "platform": "tiktok"}
                    ]
                }
            ]
        }
    }


class BatchTrendCheckResponse(BaseModel):
    """Response del análisis de múltiples tendencias."""
    total_analyzed: int = Field(..., description="Total de tendencias analizadas")
    fresh_count: int = Field(..., description="Cantidad de tendencias frescas")
    stale_count: int = Field(..., description="Cantidad de tendencias caducadas")
    filtered_count: int = Field(..., description="Cantidad filtrada por ser STALE")

    fresh_trends: List[TrendVelocityResponse] = Field(
        default_factory=list,
        description="Tendencias frescas (ordenadas por velocity_score)"
    )
    stale_trends: List[TrendVelocityResponse] = Field(
        default_factory=list,
        description="Tendencias caducadas (NO USAR)"
    )


class ValidateRecipeRequest(BaseModel):
    """Request para validar elementos de una receta viral."""
    audios: List[str] = Field(
        default_factory=list,
        max_length=10,
        description="Lista de audios a validar"
    )
    hashtags: List[str] = Field(
        default_factory=list,
        max_length=30,
        description="Lista de hashtags a validar"
    )
    platform: str = Field(
        "instagram",
        description="Plataforma objetivo"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "audios": ["trending audio 2024", "viral sound effect"],
                    "hashtags": ["#smallbusiness", "#entrepreneur", "#motivation"],
                    "platform": "instagram"
                }
            ]
        }
    }


class ValidatedElementSchema(BaseModel):
    """Elemento validado (fresco)."""
    name: str = Field(..., description="Nombre del elemento")
    status: str = Field(..., description="Estado de frescura")
    velocity_score: Optional[float] = Field(None, description="Score de velocidad")


class FilteredElementSchema(BaseModel):
    """Elemento filtrado (caducado)."""
    name: str = Field(..., description="Nombre del elemento")
    status: str = Field(..., description="Estado (siempre 'stale')")
    reason: str = Field(..., description="Razón del filtrado")


class ValidationSummarySchema(BaseModel):
    """Resumen de la validación."""
    total_audios: int
    fresh_audios: int
    stale_audios: int
    total_hashtags: int
    fresh_hashtags: int
    stale_hashtags: int


class ValidateRecipeResponse(BaseModel):
    """Response de validación de receta viral."""
    validated_audios: List[ValidatedElementSchema] = Field(
        default_factory=list,
        description="Audios validados (frescos, ordenados por velocity)"
    )
    filtered_audios: List[FilteredElementSchema] = Field(
        default_factory=list,
        description="Audios filtrados (caducados)"
    )
    validated_hashtags: List[ValidatedElementSchema] = Field(
        default_factory=list,
        description="Hashtags validados (frescos)"
    )
    filtered_hashtags: List[FilteredElementSchema] = Field(
        default_factory=list,
        description="Hashtags filtrados (caducados)"
    )
    summary: ValidationSummarySchema = Field(
        ...,
        description="Resumen de la validación"
    )
    has_stale_elements: bool = Field(
        ...,
        description="Si había elementos caducados"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "validated_audios": [
                        {"name": "trending audio 2024", "status": "trending", "velocity_score": 0.85}
                    ],
                    "filtered_audios": [
                        {"name": "old viral sound", "status": "stale", "reason": "TENDENCIA CADUCADA..."}
                    ],
                    "validated_hashtags": [
                        {"name": "#smallbusiness", "status": "rising", "velocity_score": 0.45}
                    ],
                    "filtered_hashtags": [],
                    "summary": {
                        "total_audios": 2,
                        "fresh_audios": 1,
                        "stale_audios": 1,
                        "total_hashtags": 1,
                        "fresh_hashtags": 1,
                        "stale_hashtags": 0
                    },
                    "has_stale_elements": True
                }
            ]
        }
    }


class QuickCheckResponse(BaseModel):
    """Response de verificación rápida de tendencia."""
    identifier: str = Field(..., description="Identificador de la tendencia")
    status: TrendStatusEnum = Field(..., description="Estado de frescura")
    is_fresh: bool = Field(..., description="Si es usable")
    is_stale: bool = Field(..., description="Si está caducada")
    velocity_score: float = Field(..., description="Score de velocidad")
    recommendation: str = Field(..., description="Recomendación")


class TrendStatusSummaryResponse(BaseModel):
    """Response con descripción de un estado de tendencia."""
    status: TrendStatusEnum
    description: str
    threshold: str
    usable: bool
    priority: int
