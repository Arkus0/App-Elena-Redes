"""
Account Health Schemas - Pydantic models for AccountHealthScoring API

Define los modelos de request/response para los endpoints de salud de cuenta.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class AccountHealthStatusEnum(str, Enum):
    """Estados de salud de la cuenta."""
    HEALTHY = "healthy"
    LOW_AUTHORITY = "low_authority"
    POSSIBLE_SHADOWBAN = "possible_shadowban"
    INSUFFICIENT_DATA = "insufficient_data"


class AuthorityLevelEnum(str, Enum):
    """Niveles de autoridad basados en la relación views/seguidores."""
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    CRITICAL = "critical"


class PostMetricsSchema(BaseModel):
    """Métricas de un post individual para análisis de salud."""
    post_id: Optional[str] = Field(None, description="ID único del post")
    views_count: int = Field(..., ge=0, description="Número de views del post")
    likes_count: int = Field(0, ge=0, description="Número de likes")
    comments_count: int = Field(0, ge=0, description="Número de comentarios")
    shares_count: int = Field(0, ge=0, description="Número de shares")
    saves_count: int = Field(0, ge=0, description="Número de guardados")
    posted_at: Optional[str] = Field(None, description="Timestamp ISO de publicación")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "post_id": "post_123",
                    "views_count": 15000,
                    "likes_count": 1200,
                    "comments_count": 45,
                    "shares_count": 30,
                    "saves_count": 80,
                    "posted_at": "2024-03-15T14:30:00Z"
                }
            ]
        }
    }


class AccountHealthRequest(BaseModel):
    """Request para evaluar la salud de una cuenta."""
    recent_posts: List[PostMetricsSchema] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Lista de posts recientes (últimos 10-20 recomendados)"
    )
    follower_count: int = Field(
        ...,
        gt=0,
        description="Número total de seguidores de la cuenta"
    )
    posts_to_analyze: int = Field(
        10,
        ge=1,
        le=20,
        description="Número de posts a analizar (default: 10)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "recent_posts": [
                        {"views_count": 15000, "likes_count": 1200},
                        {"views_count": 12000, "likes_count": 950},
                        {"views_count": 18000, "likes_count": 1500},
                        {"views_count": 14000, "likes_count": 1100},
                        {"views_count": 16000, "likes_count": 1300}
                    ],
                    "follower_count": 50000,
                    "posts_to_analyze": 10
                }
            ]
        }
    }


class AccountHealthMetricsSchema(BaseModel):
    """Métricas calculadas del análisis de salud."""
    avg_views: float = Field(..., description="Promedio de views de los posts analizados")
    std_views: float = Field(..., description="Desviación estándar de views")
    follower_count: int = Field(..., description="Número de seguidores")
    views_to_followers_ratio: float = Field(
        ...,
        description="Porcentaje de seguidores que ven los posts (0-100)"
    )


class CalibrationInfoSchema(BaseModel):
    """Información sobre la calibración aplicada."""
    prediction_penalty_factor: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Factor de penalización (1.0 = sin penalización, 0.3 = Low Authority)"
    )
    is_penalized: bool = Field(..., description="Si se aplicó penalización")


class AnalysisInfoSchema(BaseModel):
    """Información sobre el análisis realizado."""
    posts_analyzed: int = Field(..., description="Número de posts analizados")
    min_posts_required: int = Field(..., description="Mínimo de posts requeridos")
    has_sufficient_data: bool = Field(..., description="Si hay suficientes datos")


class UserFeedbackSchema(BaseModel):
    """Mensajes de feedback para el usuario."""
    warning_message: Optional[str] = Field(
        None,
        description="Mensaje de advertencia si la cuenta tiene problemas"
    )
    recommendation: Optional[str] = Field(
        None,
        description="Recomendación para mejorar la salud de la cuenta"
    )


class AccountHealthResponse(BaseModel):
    """Response completa del análisis de salud de cuenta."""
    health_status: AccountHealthStatusEnum = Field(
        ...,
        description="Estado de salud de la cuenta"
    )
    authority_level: AuthorityLevelEnum = Field(
        ...,
        description="Nivel de autoridad de la cuenta"
    )
    metrics: AccountHealthMetricsSchema = Field(
        ...,
        description="Métricas calculadas del análisis"
    )
    calibration: CalibrationInfoSchema = Field(
        ...,
        description="Información de calibración para predicciones"
    )
    analysis: AnalysisInfoSchema = Field(
        ...,
        description="Información sobre el análisis realizado"
    )
    user_feedback: UserFeedbackSchema = Field(
        ...,
        description="Mensajes de feedback para el usuario"
    )
    analyzed_at: str = Field(..., description="Timestamp ISO del análisis")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "health_status": "low_authority",
                    "authority_level": "low",
                    "metrics": {
                        "avg_views": 3500.0,
                        "std_views": 1200.0,
                        "follower_count": 50000,
                        "views_to_followers_ratio": 7.0
                    },
                    "calibration": {
                        "prediction_penalty_factor": 0.3,
                        "is_penalized": True
                    },
                    "analysis": {
                        "posts_analyzed": 10,
                        "min_posts_required": 5,
                        "has_sufficient_data": True
                    },
                    "user_feedback": {
                        "warning_message": "Tu cuenta tiene baja tracción actualmente...",
                        "recommendation": "Recomendamos: 1) Publicar consistentemente..."
                    },
                    "analyzed_at": "2024-03-15T14:30:00"
                }
            ]
        }
    }


class CalibratedPredictionRequest(BaseModel):
    """Request para obtener predicción calibrada con salud de cuenta."""
    # Información de la cuenta
    recent_posts: List[PostMetricsSchema] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Posts recientes para evaluar salud"
    )
    follower_count: int = Field(
        ...,
        gt=0,
        description="Número de seguidores"
    )

    # Features del contenido a predecir (mismo formato que GrowthPredictionRequest)
    posted_at: Optional[str] = Field(None, description="Timestamp de publicación")
    post_type: str = Field("reel", description="Tipo de contenido")

    # Temporal Features (Hook Theory)
    hook_energy: float = Field(0.0, ge=0.0, le=1.0, description="Energía visual 0-3s")
    retention_energy: float = Field(0.0, ge=0.0, le=1.0, description="Energía visual 3s+")
    hook_cut_rate: float = Field(0.0, ge=0.0, description="Cortes/min en hook")
    retention_cut_rate: float = Field(0.0, ge=0.0, description="Cortes/min post-hook")
    face_in_hook: int = Field(0, ge=0, le=1, description="Cara en hook (0/1)")

    # Global Features
    tempo: float = Field(0.0, ge=0.0, le=300.0, description="BPM del audio")
    brightness_variance: float = Field(0.0, ge=0.0, le=1.0, description="Variación de brillo")

    # Semantic PCA
    sem_pca_1: float = Field(0.0)
    sem_pca_2: float = Field(0.0)
    sem_pca_3: float = Field(0.0)
    sem_pca_4: float = Field(0.0)
    sem_pca_5: float = Field(0.0)
    sem_pca_6: float = Field(0.0)
    sem_pca_7: float = Field(0.0)
    sem_pca_8: float = Field(0.0)
    sem_pca_9: float = Field(0.0)
    sem_pca_10: float = Field(0.0)


class OriginalPredictionSchema(BaseModel):
    """Predicción original sin calibrar."""
    rpi_score: float = Field(..., description="RPI score original (log-transformed)")
    rpi_raw: float = Field(..., description="RPI raw original (views predichas)")


class CalibratedPredictionSchema(BaseModel):
    """Predicción calibrada según salud de cuenta."""
    rpi_score: float = Field(..., description="RPI score calibrado")
    rpi_raw: float = Field(..., description="RPI raw calibrado (views ajustadas)")


class CalibrationAppliedSchema(BaseModel):
    """Información sobre la calibración aplicada."""
    penalty_factor: float = Field(..., description="Factor de penalización aplicado")
    was_penalized: bool = Field(..., description="Si se aplicó penalización")


class CalibratedPredictionResponse(BaseModel):
    """Response de predicción calibrada con información de salud."""
    original_prediction: OriginalPredictionSchema = Field(
        ...,
        description="Predicción original del modelo"
    )
    calibrated_prediction: CalibratedPredictionSchema = Field(
        ...,
        description="Predicción calibrada según salud de cuenta"
    )
    calibration_applied: CalibrationAppliedSchema = Field(
        ...,
        description="Información de la calibración aplicada"
    )
    account_health: AccountHealthResponse = Field(
        ...,
        description="Estado de salud de la cuenta"
    )
    combined_message: str = Field(
        ...,
        description="Mensaje combinado con predicción y advertencias"
    )

    # SHAP explanation (heredado de GrowthPredictionResponse)
    explanation: str = Field(..., description="Explicación SHAP del modelo")
    top_positive_factors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Factores positivos principales"
    )
    top_negative_factors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Factores negativos principales"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "original_prediction": {
                        "rpi_score": 1.5,
                        "rpi_raw": 3.48
                    },
                    "calibrated_prediction": {
                        "rpi_score": 0.3,
                        "rpi_raw": 1.04
                    },
                    "calibration_applied": {
                        "penalty_factor": 0.3,
                        "was_penalized": True
                    },
                    "account_health": {
                        "health_status": "low_authority",
                        "authority_level": "low",
                        "metrics": {
                            "avg_views": 3500.0,
                            "std_views": 1200.0,
                            "follower_count": 50000,
                            "views_to_followers_ratio": 7.0
                        },
                        "calibration": {
                            "prediction_penalty_factor": 0.3,
                            "is_penalized": True
                        },
                        "analysis": {
                            "posts_analyzed": 10,
                            "min_posts_required": 5,
                            "has_sufficient_data": True
                        },
                        "user_feedback": {
                            "warning_message": "Tu cuenta tiene baja tracción...",
                            "recommendation": "Recomendamos: 1) Publicar..."
                        },
                        "analyzed_at": "2024-03-15T14:30:00"
                    },
                    "combined_message": "**Predicción ajustada:** Este contenido...",
                    "explanation": "El score es alto porque...",
                    "top_positive_factors": [],
                    "top_negative_factors": []
                }
            ]
        }
    }


class AccountHealthSummaryResponse(BaseModel):
    """Response con resumen legible de salud de cuenta."""
    summary: str = Field(..., description="Resumen formateado de la salud de la cuenta")
    health_status: AccountHealthStatusEnum = Field(..., description="Estado de salud")
    authority_level: AuthorityLevelEnum = Field(..., description="Nivel de autoridad")
    penalty_factor: float = Field(..., description="Factor de penalización aplicado")
    needs_attention: bool = Field(
        ...,
        description="Si la cuenta necesita atención (Low Authority o Shadowban)"
    )
