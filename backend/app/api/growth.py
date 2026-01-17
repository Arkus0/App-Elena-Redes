"""
Growth Prediction API - Endpoints for RPI Score Prediction and Model Training

Endpoints para el motor de predicción de crecimiento (GrowthPredictionEngine):
- POST /predict: Predice RPI score con explicación SHAP
- POST /predict/batch: Predicción en batch
- POST /predict/calibrated: Predicción calibrada con salud de cuenta
- POST /account-health: Evalúa la salud de una cuenta
- POST /train: Entrena o re-entrena el modelo
- GET /status: Estado del modelo
- GET /features/importance: Importancia de features
"""
import logging
from typing import List

from fastapi import APIRouter, HTTPException, status

from app.schemas.growth import (
    GrowthPredictionRequest,
    GrowthPredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    GrowthTrainingRequest,
    GrowthTrainingResponse,
    GrowthModelStatusResponse,
    FeatureImportanceResponse,
    FeatureContributionSchema,
    ConfidenceIntervalSchema,
    ModelConfigSchema,
)
from app.schemas.account_health import (
    AccountHealthRequest,
    AccountHealthResponse,
    AccountHealthMetricsSchema,
    CalibrationInfoSchema,
    AnalysisInfoSchema,
    UserFeedbackSchema,
    CalibratedPredictionRequest,
    CalibratedPredictionResponse,
    OriginalPredictionSchema,
    CalibratedPredictionSchema,
    CalibrationAppliedSchema,
    AccountHealthSummaryResponse,
)
from app.services.growth_prediction_engine import (
    get_growth_prediction_engine,
    GrowthPredictionEngine,
)
from app.services.account_health_scoring import (
    get_account_health_scoring,
    AccountHealthScoring,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_engine() -> GrowthPredictionEngine:
    """Obtiene la instancia del motor de predicción."""
    return get_growth_prediction_engine()


@router.post(
    "/predict",
    response_model=GrowthPredictionResponse,
    summary="Predict RPI Score with SHAP Explanation",
    description="""
    Predice el RPI score (Relative Performance Index) para un contenido
    y proporciona explicación detallada usando SHAP values.

    **Features de entrada:**
    - Metadata: hora, día de la semana, tipo de post
    - Sensoriales: visual_energy, tempo (BPM), brightness_variance, cut_density
    - Semánticas: 10 componentes PCA del embedding semántico

    **Output:**
    - RPI score predicho (log-transformed y raw)
    - Contribución de cada feature (SHAP values)
    - Explicación en lenguaje natural

    **Ejemplo de explicación:**
    "El score es alto porque BPM (+0.4) y visual_energy (+0.3) son altos"
    """
)
async def predict_growth(request: GrowthPredictionRequest) -> GrowthPredictionResponse:
    """Predice RPI score con explicación SHAP completa."""
    engine = _get_engine()

    if not engine.is_trained:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not trained. Please train the model first using POST /growth/train"
        )

    try:
        # Construir features dict con Hook Theory features
        features = {
            "posted_at": request.posted_at,
            "post_type": request.post_type,

            # === TEMPORAL FEATURES (Hook Theory) ===
            "hook_energy": request.hook_energy,
            "retention_energy": request.retention_energy,
            "hook_cut_rate": request.hook_cut_rate,
            "retention_cut_rate": request.retention_cut_rate,
            "face_in_hook": request.face_in_hook,

            # === GLOBAL FEATURES ===
            "tempo": request.tempo or request.bpm or 0.0,
            "brightness_variance": request.brightness_variance,

            # === SEMANTIC PCA ===
            "sem_pca_1": request.sem_pca_1,
            "sem_pca_2": request.sem_pca_2,
            "sem_pca_3": request.sem_pca_3,
            "sem_pca_4": request.sem_pca_4,
            "sem_pca_5": request.sem_pca_5,
            "sem_pca_6": request.sem_pca_6,
            "sem_pca_7": request.sem_pca_7,
            "sem_pca_8": request.sem_pca_8,
            "sem_pca_9": request.sem_pca_9,
            "sem_pca_10": request.sem_pca_10,
        }

        # Obtener predicción con explicación
        result = engine.predict_with_explanation(features)

        # Convertir a response schema
        return GrowthPredictionResponse(
            predicted_rpi_score=result.predicted_rpi_score,
            predicted_rpi_raw=result.predicted_rpi_raw,
            confidence_interval=ConfidenceIntervalSchema(
                lower=result.confidence_interval[0],
                upper=result.confidence_interval[1]
            ),
            top_positive_factors=[
                FeatureContributionSchema(
                    feature=c.feature_name,
                    contribution=c.contribution,
                    value=c.feature_value,
                    direction=c.direction
                )
                for c in result.top_positive_contributions
            ],
            top_negative_factors=[
                FeatureContributionSchema(
                    feature=c.feature_name,
                    contribution=c.contribution,
                    value=c.feature_value,
                    direction=c.direction
                )
                for c in result.top_negative_contributions
            ],
            explanation=result.explanation_text,
            feature_values=result.feature_values
        )

    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch Prediction for Multiple Contents",
    description="Predice RPI scores para múltiples contenidos en una sola llamada."
)
async def predict_growth_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Predicción en batch para múltiples contenidos."""
    engine = _get_engine()

    if not engine.is_trained:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not trained. Please train the model first."
        )

    try:
        predictions = []

        for item in request.items:
            features = {
                "posted_at": item.posted_at,
                "post_type": item.post_type,

                # === TEMPORAL FEATURES (Hook Theory) ===
                "hook_energy": item.hook_energy,
                "retention_energy": item.retention_energy,
                "hook_cut_rate": item.hook_cut_rate,
                "retention_cut_rate": item.retention_cut_rate,
                "face_in_hook": item.face_in_hook,

                # === GLOBAL FEATURES ===
                "tempo": item.tempo or item.bpm or 0.0,
                "brightness_variance": item.brightness_variance,

                # === SEMANTIC PCA ===
                "sem_pca_1": item.sem_pca_1,
                "sem_pca_2": item.sem_pca_2,
                "sem_pca_3": item.sem_pca_3,
                "sem_pca_4": item.sem_pca_4,
                "sem_pca_5": item.sem_pca_5,
                "sem_pca_6": item.sem_pca_6,
                "sem_pca_7": item.sem_pca_7,
                "sem_pca_8": item.sem_pca_8,
                "sem_pca_9": item.sem_pca_9,
                "sem_pca_10": item.sem_pca_10,
            }

            result = engine.predict_with_explanation(features)

            predictions.append(GrowthPredictionResponse(
                predicted_rpi_score=result.predicted_rpi_score,
                predicted_rpi_raw=result.predicted_rpi_raw,
                confidence_interval=ConfidenceIntervalSchema(
                    lower=result.confidence_interval[0],
                    upper=result.confidence_interval[1]
                ),
                top_positive_factors=[
                    FeatureContributionSchema(
                        feature=c.feature_name,
                        contribution=c.contribution,
                        value=c.feature_value,
                        direction=c.direction
                    )
                    for c in result.top_positive_contributions
                ],
                top_negative_factors=[
                    FeatureContributionSchema(
                        feature=c.feature_name,
                        contribution=c.contribution,
                        value=c.feature_value,
                        direction=c.direction
                    )
                    for c in result.top_negative_contributions
                ],
                explanation=result.explanation_text,
                feature_values=result.feature_values
            ))

        return BatchPredictionResponse(
            predictions=predictions,
            total_items=len(predictions)
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.post(
    "/train",
    response_model=GrowthTrainingResponse,
    summary="Train or Retrain the Growth Prediction Model",
    description="""
    Entrena o re-entrena el modelo XGBoost para predicción de RPI score.

    **Proceso de entrenamiento:**
    1. Preparación de features (metadata, sensoriales, semánticas)
    2. Split train/test (80/20)
    3. Validación cruzada K-Fold (5 folds por defecto)
    4. Entrenamiento final del modelo XGBoost
    5. Cálculo de métricas y feature importance
    6. Serialización con joblib

    **Opciones:**
    - use_synthetic_data: Genera datos sintéticos para entrenamiento inicial
    - training_data: Datos reales de entrenamiento (opcional)
    """
)
async def train_model(request: GrowthTrainingRequest) -> GrowthTrainingResponse:
    """Entrena o re-entrena el modelo de predicción de crecimiento."""
    engine = _get_engine()

    try:
        # Preparar datos de entrenamiento
        if request.training_data:
            # Usar datos reales proporcionados
            training_data = [item.model_dump() for item in request.training_data]
            logger.info(f"Training with {len(training_data)} real samples")
        elif request.use_synthetic_data:
            # Generar datos sintéticos
            training_data = engine.generate_synthetic_training_data(
                n_samples=request.synthetic_samples
            )
            logger.info(f"Training with {len(training_data)} synthetic samples")
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either provide training_data or set use_synthetic_data=True"
            )

        # Entrenar modelo
        metrics = engine.train(
            training_data=training_data,
            save_model=request.save_model
        )

        return GrowthTrainingResponse(
            success=True,
            message=f"Model trained successfully with {metrics.training_samples} samples",
            train_rmse=metrics.train_rmse,
            test_rmse=metrics.test_rmse,
            train_mae=metrics.train_mae,
            test_mae=metrics.test_mae,
            train_r2=metrics.train_r2,
            test_r2=metrics.test_r2,
            cv_rmse_mean=metrics.cv_rmse_mean,
            cv_rmse_std=metrics.cv_rmse_std,
            cv_scores=metrics.cv_scores,
            training_samples=metrics.training_samples,
            feature_count=len(engine.feature_columns),
            feature_importance=metrics.feature_importance,
            training_date=metrics.training_date,
            model_version=metrics.model_version
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(e)}"
        )


@router.get(
    "/status",
    response_model=GrowthModelStatusResponse,
    summary="Get Model Status",
    description="Obtiene el estado actual del modelo de predicción de crecimiento."
)
async def get_model_status() -> GrowthModelStatusResponse:
    """Obtiene el estado del modelo."""
    engine = _get_engine()
    status_data = engine.get_model_status()

    training_metrics = None
    if status_data.get("training_metrics"):
        tm = status_data["training_metrics"]
        training_metrics = GrowthTrainingResponse(
            success=True,
            message="Model trained",
            train_rmse=tm.get("train_rmse", 0),
            test_rmse=tm.get("test_rmse", 0),
            train_mae=tm.get("train_mae", 0),
            test_mae=tm.get("test_mae", 0),
            train_r2=tm.get("train_r2", 0),
            test_r2=tm.get("test_r2", 0),
            cv_rmse_mean=tm.get("cv_rmse_mean", 0),
            cv_rmse_std=tm.get("cv_rmse_std", 0),
            cv_scores=tm.get("cv_scores", []),
            training_samples=tm.get("training_samples", 0),
            feature_count=tm.get("feature_count", 0) or len(engine.feature_columns),
            feature_importance=tm.get("feature_importance", {}),
            training_date=tm.get("training_date", ""),
            model_version=tm.get("model_version", "")
        )

    return GrowthModelStatusResponse(
        is_trained=status_data["is_trained"],
        model_version=status_data["model_version"],
        feature_count=status_data["feature_count"],
        features=status_data["features"],
        config=ModelConfigSchema(**status_data["config"]),
        training_metrics=training_metrics
    )


@router.get(
    "/features/importance",
    response_model=FeatureImportanceResponse,
    summary="Get Feature Importance",
    description="Obtiene la importancia de cada feature en el modelo entrenado."
)
async def get_feature_importance() -> FeatureImportanceResponse:
    """Obtiene la importancia de features del modelo."""
    engine = _get_engine()

    if not engine.is_trained:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not trained. Please train the model first."
        )

    importance = engine.get_feature_importance()

    if not importance:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve feature importance"
        )

    top_features = list(importance.keys())[:5]

    return FeatureImportanceResponse(
        importance=importance,
        top_features=top_features
    )


# =============================================================================
# ACCOUNT HEALTH ENDPOINTS
# =============================================================================

def _get_health_scoring() -> AccountHealthScoring:
    """Obtiene la instancia del motor de scoring de salud de cuenta."""
    return get_account_health_scoring()


@router.post(
    "/account-health",
    response_model=AccountHealthResponse,
    summary="Evaluate Account Health",
    description="""
    Evalúa la salud de una cuenta basándose en sus posts recientes.

    **Lógica de Autoridad:**
    - Analiza los últimos 10 posts de la cuenta del usuario
    - Calcula la media de views y la desviación estándar
    - Clasifica según ratio views/seguidores:
      - >= 10%: HEALTHY (sin penalización)
      - < 10%: LOW_AUTHORITY (factor 0.3x)
      - < 2%: POSSIBLE_SHADOWBAN (factor 0.1x)

    **Uso recomendado:**
    Llamar este endpoint ANTES de generar predicciones para calibrar
    expectativas del usuario sobre su alcance real.
    """
)
async def evaluate_account_health(request: AccountHealthRequest) -> AccountHealthResponse:
    """Evalúa la salud de una cuenta de usuario."""
    try:
        health_scoring = _get_health_scoring()

        # Convertir posts a formato esperado
        posts_data = [
            {
                "post_id": p.post_id,
                "views_count": p.views_count,
                "likes_count": p.likes_count,
                "comments_count": p.comments_count,
                "shares_count": p.shares_count,
                "saves_count": p.saves_count,
                "posted_at": p.posted_at,
            }
            for p in request.recent_posts
        ]

        # Evaluar salud
        result = health_scoring.evaluate_account_health(
            recent_posts=posts_data,
            follower_count=request.follower_count,
            posts_to_analyze=request.posts_to_analyze
        )

        # Convertir a response schema
        return AccountHealthResponse(
            health_status=result.health_status.value,
            authority_level=result.authority_level.value,
            metrics=AccountHealthMetricsSchema(
                avg_views=round(result.avg_views, 2),
                std_views=round(result.std_views, 2),
                follower_count=result.follower_count,
                views_to_followers_ratio=round(result.views_to_followers_ratio * 100, 2)
            ),
            calibration=CalibrationInfoSchema(
                prediction_penalty_factor=result.prediction_penalty_factor,
                is_penalized=result.prediction_penalty_factor < 1.0
            ),
            analysis=AnalysisInfoSchema(
                posts_analyzed=result.posts_analyzed,
                min_posts_required=result.min_posts_required,
                has_sufficient_data=result.posts_analyzed >= result.min_posts_required
            ),
            user_feedback=UserFeedbackSchema(
                warning_message=result.warning_message,
                recommendation=result.recommendation
            ),
            analyzed_at=result.analyzed_at
        )

    except Exception as e:
        logger.error(f"Account health evaluation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to evaluate account health: {str(e)}"
        )


@router.post(
    "/predict/calibrated",
    response_model=CalibratedPredictionResponse,
    summary="Calibrated Prediction with Account Health",
    description="""
    Predice RPI score CALIBRADO según la salud de la cuenta del usuario.

    **IMPORTANTE:** Este endpoint combina:
    1. Predicción XGBoost estándar con explicación SHAP
    2. Análisis de salud de cuenta (últimos 10 posts)
    3. Calibración automática de la predicción

    **Calibración aplicada:**
    - LOW_AUTHORITY: Factor 0.3x en views predichas
    - POSSIBLE_SHADOWBAN: Factor 0.1x + alerta crítica
    - HEALTHY: Sin modificación (factor 1.0x)

    **Mensaje al Usuario:**
    Si detectamos Low Authority, el response incluye una advertencia:
    "Tu cuenta tiene baja tracción actualmente. Este video está optimizado,
    pero necesitarás subir 5-10 así de constantes para reactivar el algoritmo."
    """
)
async def predict_calibrated(request: CalibratedPredictionRequest) -> CalibratedPredictionResponse:
    """Predicción calibrada con información de salud de cuenta."""
    engine = _get_engine()

    if not engine.is_trained:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not trained. Please train the model first using POST /growth/train"
        )

    try:
        # Preparar features del contenido
        features = {
            "posted_at": request.posted_at,
            "post_type": request.post_type,

            # Temporal Features (Hook Theory)
            "hook_energy": request.hook_energy,
            "retention_energy": request.retention_energy,
            "hook_cut_rate": request.hook_cut_rate,
            "retention_cut_rate": request.retention_cut_rate,
            "face_in_hook": request.face_in_hook,

            # Global Features
            "tempo": request.tempo,
            "brightness_variance": request.brightness_variance,

            # Semantic PCA
            "sem_pca_1": request.sem_pca_1,
            "sem_pca_2": request.sem_pca_2,
            "sem_pca_3": request.sem_pca_3,
            "sem_pca_4": request.sem_pca_4,
            "sem_pca_5": request.sem_pca_5,
            "sem_pca_6": request.sem_pca_6,
            "sem_pca_7": request.sem_pca_7,
            "sem_pca_8": request.sem_pca_8,
            "sem_pca_9": request.sem_pca_9,
            "sem_pca_10": request.sem_pca_10,
        }

        # Preparar posts para análisis de salud
        posts_data = [
            {
                "post_id": p.post_id,
                "views_count": p.views_count,
                "likes_count": p.likes_count,
                "comments_count": p.comments_count,
                "shares_count": p.shares_count,
                "saves_count": p.saves_count,
            }
            for p in request.recent_posts
        ]

        # Obtener predicción calibrada
        calibrated = engine.predict_with_account_health(
            features=features,
            recent_posts=posts_data,
            follower_count=request.follower_count
        )

        # Obtener también los factores SHAP de la predicción base
        base_result = engine.predict_with_explanation(features)

        # Construir response de salud de cuenta
        health = calibrated.account_health
        account_health_response = AccountHealthResponse(
            health_status=health.health_status.value,
            authority_level=health.authority_level.value,
            metrics=AccountHealthMetricsSchema(
                avg_views=round(health.avg_views, 2),
                std_views=round(health.std_views, 2),
                follower_count=health.follower_count,
                views_to_followers_ratio=round(health.views_to_followers_ratio * 100, 2)
            ),
            calibration=CalibrationInfoSchema(
                prediction_penalty_factor=health.prediction_penalty_factor,
                is_penalized=health.prediction_penalty_factor < 1.0
            ),
            analysis=AnalysisInfoSchema(
                posts_analyzed=health.posts_analyzed,
                min_posts_required=health.min_posts_required,
                has_sufficient_data=health.posts_analyzed >= health.min_posts_required
            ),
            user_feedback=UserFeedbackSchema(
                warning_message=health.warning_message,
                recommendation=health.recommendation
            ),
            analyzed_at=health.analyzed_at
        )

        return CalibratedPredictionResponse(
            original_prediction=OriginalPredictionSchema(
                rpi_score=round(calibrated.original_rpi_score, 4),
                rpi_raw=round(calibrated.original_rpi_raw, 4)
            ),
            calibrated_prediction=CalibratedPredictionSchema(
                rpi_score=round(calibrated.calibrated_rpi_score, 4),
                rpi_raw=round(calibrated.calibrated_rpi_raw, 4)
            ),
            calibration_applied=CalibrationAppliedSchema(
                penalty_factor=calibrated.penalty_factor,
                was_penalized=calibrated.penalty_factor < 1.0
            ),
            account_health=account_health_response,
            combined_message=calibrated.combined_message,
            explanation=base_result.explanation_text,
            top_positive_factors=[
                {
                    "feature": c.feature_name,
                    "contribution": c.contribution,
                    "value": c.feature_value,
                    "direction": c.direction
                }
                for c in base_result.top_positive_contributions
            ],
            top_negative_factors=[
                {
                    "feature": c.feature_name,
                    "contribution": c.contribution,
                    "value": c.feature_value,
                    "direction": c.direction
                }
                for c in base_result.top_negative_contributions
            ]
        )

    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Calibrated prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Calibrated prediction failed: {str(e)}"
        )


@router.get(
    "/account-health/summary/{health_status}",
    response_model=AccountHealthSummaryResponse,
    summary="Get Health Status Summary",
    description="Obtiene un resumen descriptivo de un estado de salud específico."
)
async def get_health_summary(health_status: str) -> AccountHealthSummaryResponse:
    """Obtiene descripción de un estado de salud."""
    from app.services.account_health_scoring import AccountHealthStatus, AuthorityLevel

    status_map = {
        "healthy": (AccountHealthStatus.HEALTHY, AuthorityLevel.NORMAL, 1.0),
        "low_authority": (AccountHealthStatus.LOW_AUTHORITY, AuthorityLevel.LOW, 0.3),
        "possible_shadowban": (AccountHealthStatus.POSSIBLE_SHADOWBAN, AuthorityLevel.CRITICAL, 0.1),
        "insufficient_data": (AccountHealthStatus.INSUFFICIENT_DATA, AuthorityLevel.NORMAL, 1.0),
    }

    if health_status.lower() not in status_map:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid health status. Valid values: {list(status_map.keys())}"
        )

    health_enum, authority, penalty = status_map[health_status.lower()]

    summaries = {
        "healthy": "La cuenta tiene buena tracción. Las predicciones reflejan el potencial real del contenido sin ajustes.",
        "low_authority": "La cuenta tiene baja tracción. Las predicciones se ajustan con factor 0.3x. Se recomienda publicar 5-10 videos consistentes para reactivar el algoritmo.",
        "possible_shadowban": "ALERTA: La cuenta muestra señales de posible shadowban. Views muy por debajo de lo esperado. Factor de ajuste 0.1x aplicado.",
        "insufficient_data": "No hay suficientes datos para evaluar la cuenta. Se necesitan al menos 5 posts recientes con métricas de views.",
    }

    return AccountHealthSummaryResponse(
        summary=summaries[health_status.lower()],
        health_status=health_enum.value,
        authority_level=authority.value,
        penalty_factor=penalty,
        needs_attention=health_status.lower() in ["low_authority", "possible_shadowban"]
    )
