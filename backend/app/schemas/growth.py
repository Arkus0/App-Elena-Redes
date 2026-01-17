"""
Growth Prediction Schemas - Pydantic models for GrowthPredictionEngine API

Define los modelos de request/response para los endpoints de predicción de crecimiento.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FeatureContributionSchema(BaseModel):
    """Contribución de una feature individual al score predicho."""
    feature: str = Field(..., description="Nombre de la feature")
    contribution: float = Field(..., description="Contribución SHAP al score")
    value: float = Field(..., description="Valor de la feature")
    direction: str = Field(..., description="Dirección del impacto: 'positive' o 'negative'")


class ConfidenceIntervalSchema(BaseModel):
    """Intervalo de confianza de la predicción."""
    lower: float = Field(..., description="Límite inferior del intervalo (95%)")
    upper: float = Field(..., description="Límite superior del intervalo (95%)")


class GrowthPredictionRequest(BaseModel):
    """
    Request para obtener predicción de crecimiento (RPI score).

    Implementa HOOK THEORY: El algoritmo TikTok/IG evalúa videos como
    secuencias temporales. Los primeros 3 segundos determinan el 90% del éxito.
    """
    # Metadata temporal
    posted_at: Optional[str] = Field(
        None,
        description="Timestamp ISO de publicación (opcional, usa hora actual si no se provee)"
    )
    post_type: str = Field(
        "reel",
        description="Tipo de contenido: reel, carousel, static, story, video"
    )

    # =========================================================================
    # TEMPORAL FEATURES (Hook Theory) - CRÍTICO para el algoritmo
    # =========================================================================
    hook_energy: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Energía visual en los segundos 0-3 (CRÍTICO - determina retención)"
    )
    retention_energy: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Energía visual del resto del video (segundos 3+)"
    )
    hook_cut_rate: float = Field(
        0.0, ge=0.0,
        description="Cortes por minuto en el hook (ponderados 10x por el algoritmo)"
    )
    retention_cut_rate: float = Field(
        0.0, ge=0.0,
        description="Cortes por minuto después del hook"
    )
    face_in_hook: int = Field(
        0, ge=0, le=1,
        description="¿Hay cara en los primeros 3 segundos? (0=No, 1=Sí)"
    )

    # =========================================================================
    # GLOBAL FEATURES
    # =========================================================================
    tempo: float = Field(
        0.0, ge=0.0, le=300.0,
        description="BPM del audio (beats per minute)"
    )
    bpm: Optional[float] = Field(
        None,
        description="Alias para tempo (BPM)"
    )
    brightness_variance: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Variación de brillo en el video (0-1)"
    )

    # Semantic features (del TextIntelligence - PCA components)
    sem_pca_1: float = Field(0.0, description="Componente semántico PCA 1")
    sem_pca_2: float = Field(0.0, description="Componente semántico PCA 2")
    sem_pca_3: float = Field(0.0, description="Componente semántico PCA 3")
    sem_pca_4: float = Field(0.0, description="Componente semántico PCA 4")
    sem_pca_5: float = Field(0.0, description="Componente semántico PCA 5")
    sem_pca_6: float = Field(0.0, description="Componente semántico PCA 6")
    sem_pca_7: float = Field(0.0, description="Componente semántico PCA 7")
    sem_pca_8: float = Field(0.0, description="Componente semántico PCA 8")
    sem_pca_9: float = Field(0.0, description="Componente semántico PCA 9")
    sem_pca_10: float = Field(0.0, description="Componente semántico PCA 10")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "posted_at": "2024-03-15T14:30:00Z",
                    "post_type": "reel",
                    "hook_energy": 0.85,
                    "retention_energy": 0.45,
                    "hook_cut_rate": 40.0,
                    "retention_cut_rate": 8.0,
                    "face_in_hook": 1,
                    "tempo": 128.0,
                    "brightness_variance": 0.45,
                    "sem_pca_1": 0.23,
                    "sem_pca_2": -0.15
                }
            ]
        }
    }


class GrowthPredictionResponse(BaseModel):
    """Response completa de predicción de crecimiento con explicabilidad SHAP."""
    # Predicción principal
    predicted_rpi_score: float = Field(
        ...,
        description="RPI score predicho (log-transformed)"
    )
    predicted_rpi_raw: float = Field(
        ...,
        description="RPI sin transformar (exp(score) - 1)"
    )

    # Intervalo de confianza
    confidence_interval: ConfidenceIntervalSchema = Field(
        ...,
        description="Intervalo de confianza del 95%"
    )

    # Explicabilidad SHAP
    top_positive_factors: List[FeatureContributionSchema] = Field(
        default_factory=list,
        description="Features con mayor contribución positiva"
    )
    top_negative_factors: List[FeatureContributionSchema] = Field(
        default_factory=list,
        description="Features con mayor contribución negativa"
    )

    # Explicación en lenguaje natural
    explanation: str = Field(
        ...,
        description="Explicación en lenguaje natural del resultado"
    )

    # Valores de features procesados
    feature_values: Dict[str, float] = Field(
        default_factory=dict,
        description="Valores de features utilizados en la predicción"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "predicted_rpi_score": 0.92,
                    "predicted_rpi_raw": 1.51,
                    "confidence_interval": {"lower": 0.72, "upper": 1.12},
                    "top_positive_factors": [
                        {"feature": "BPM (ritmo)", "contribution": 0.42, "value": 128.0, "direction": "positive"},
                        {"feature": "Energía visual", "contribution": 0.31, "value": 0.75, "direction": "positive"}
                    ],
                    "top_negative_factors": [
                        {"feature": "Hora de publicación", "contribution": -0.08, "value": 3.0, "direction": "negative"}
                    ],
                    "explanation": "El score es alto porque factores positivos: BPM (ritmo) (+0.42), Energía visual (+0.31)",
                    "feature_values": {
                        "visual_energy": 0.75,
                        "tempo": 128.0,
                        "hour": 14
                    }
                }
            ]
        }
    }


class BatchPredictionRequest(BaseModel):
    """Request para predicción en batch de múltiples contenidos."""
    items: List[GrowthPredictionRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Lista de contenidos a predecir (máximo 100)"
    )


class BatchPredictionResponse(BaseModel):
    """Response de predicción en batch."""
    predictions: List[GrowthPredictionResponse] = Field(
        ...,
        description="Lista de predicciones para cada contenido"
    )
    total_items: int = Field(..., description="Total de items procesados")


class TrainingDataItem(BaseModel):
    """Item individual de datos de entrenamiento."""
    # Metadata
    posted_at: Optional[str] = None
    post_type: str = "unknown"

    # Sensory features
    visual_energy: float = 0.0
    tempo: float = 0.0
    brightness_variance: float = 0.0
    cut_density: float = 0.0

    # Semantic PCA features
    sem_pca_1: float = 0.0
    sem_pca_2: float = 0.0
    sem_pca_3: float = 0.0
    sem_pca_4: float = 0.0
    sem_pca_5: float = 0.0
    sem_pca_6: float = 0.0
    sem_pca_7: float = 0.0
    sem_pca_8: float = 0.0
    sem_pca_9: float = 0.0
    sem_pca_10: float = 0.0

    # Target variable
    rpi_score: float = Field(..., description="RPI score (log-transformed) - variable objetivo")


class GrowthTrainingRequest(BaseModel):
    """Request para entrenar o re-entrenar el modelo."""
    use_synthetic_data: bool = Field(
        True,
        description="Si True, genera datos sintéticos para entrenamiento inicial"
    )
    synthetic_samples: int = Field(
        1000,
        ge=100,
        le=10000,
        description="Número de muestras sintéticas a generar (si use_synthetic_data=True)"
    )
    training_data: Optional[List[TrainingDataItem]] = Field(
        None,
        description="Datos de entrenamiento reales (opcional)"
    )
    save_model: bool = Field(
        True,
        description="Si True, guarda el modelo después de entrenar"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "use_synthetic_data": True,
                    "synthetic_samples": 1000,
                    "save_model": True
                }
            ]
        }
    }


class CVScoreSchema(BaseModel):
    """Scores de validación cruzada."""
    fold_scores: List[float] = Field(..., description="RMSE por fold")
    mean: float = Field(..., description="Media de RMSE")
    std: float = Field(..., description="Desviación estándar")


class FeatureImportanceSchema(BaseModel):
    """Importancia de features del modelo."""
    features: Dict[str, float] = Field(
        ...,
        description="Diccionario feature -> importancia"
    )


class GrowthTrainingResponse(BaseModel):
    """Response del entrenamiento del modelo."""
    success: bool = Field(..., description="Si el entrenamiento fue exitoso")
    message: str = Field(..., description="Mensaje de estado")

    # Métricas de rendimiento
    train_rmse: float = Field(..., description="RMSE en datos de entrenamiento")
    test_rmse: float = Field(..., description="RMSE en datos de test")
    train_mae: float = Field(..., description="MAE en datos de entrenamiento")
    test_mae: float = Field(..., description="MAE en datos de test")
    train_r2: float = Field(..., description="R² en datos de entrenamiento")
    test_r2: float = Field(..., description="R² en datos de test")

    # Validación cruzada
    cv_rmse_mean: float = Field(..., description="RMSE medio de validación cruzada")
    cv_rmse_std: float = Field(..., description="Desviación estándar de CV")
    cv_scores: List[float] = Field(..., description="RMSE por fold de CV")

    # Metadata
    training_samples: int = Field(..., description="Número de muestras de entrenamiento")
    feature_count: int = Field(..., description="Número de features")
    feature_importance: Dict[str, float] = Field(
        ...,
        description="Importancia de cada feature"
    )
    training_date: str = Field(..., description="Fecha de entrenamiento")
    model_version: str = Field(..., description="Versión del modelo")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": True,
                    "message": "Model trained successfully",
                    "train_rmse": 0.1523,
                    "test_rmse": 0.1845,
                    "train_mae": 0.1234,
                    "test_mae": 0.1456,
                    "train_r2": 0.85,
                    "test_r2": 0.78,
                    "cv_rmse_mean": 0.1789,
                    "cv_rmse_std": 0.0234,
                    "cv_scores": [0.17, 0.18, 0.19, 0.17, 0.18],
                    "training_samples": 1000,
                    "feature_count": 17,
                    "feature_importance": {
                        "tempo": 0.25,
                        "visual_energy": 0.18,
                        "hour": 0.12
                    },
                    "training_date": "2024-03-15T14:30:00",
                    "model_version": "1.0.0"
                }
            ]
        }
    }


class ModelConfigSchema(BaseModel):
    """Configuración del modelo."""
    n_estimators: int
    max_depth: int
    learning_rate: float
    cv_folds: int


class GrowthModelStatusResponse(BaseModel):
    """Response con el estado del modelo de predicción de crecimiento."""
    is_trained: bool = Field(..., description="Si el modelo está entrenado")
    model_version: str = Field(..., description="Versión del modelo")
    feature_count: int = Field(..., description="Número de features")
    features: List[str] = Field(..., description="Lista de features usadas")
    config: ModelConfigSchema = Field(..., description="Configuración del modelo")
    training_metrics: Optional[GrowthTrainingResponse] = Field(
        None,
        description="Métricas del último entrenamiento"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "is_trained": True,
                    "model_version": "1.0.0",
                    "feature_count": 17,
                    "features": ["hour", "day_of_week", "post_type_encoded", "visual_energy", "tempo"],
                    "config": {
                        "n_estimators": 200,
                        "max_depth": 6,
                        "learning_rate": 0.05,
                        "cv_folds": 5
                    }
                }
            ]
        }
    }


class FeatureImportanceResponse(BaseModel):
    """Response con la importancia de features."""
    importance: Dict[str, float] = Field(
        ...,
        description="Features ordenadas por importancia (mayor a menor)"
    )
    top_features: List[str] = Field(
        ...,
        description="Top 5 features más importantes"
    )
