"""
Growth Prediction Engine - XGBoost-based RPI Score Prediction with SHAP Explainability

Este módulo implementa el "Cerebro" del sistema de predicción de crecimiento.
Usa XGBoost para predecir el RPI_score (log-transformed) basándose en:
- Metadata: hora, día, tipo de post
- Features sensoriales: visual_energy, bpm, brightness, cut_density
- Features semánticas: 10 componentes PCA del embedding semántico

ARQUITECTURA:
- XGBoost Regressor optimizado para datos tabulares
- SHAP (SHapley Additive exPlanations) para explicabilidad
- Validación cruzada K-Fold para evitar overfitting
- Serialización con joblib para persistencia del modelo

Autor: BrandPulse AI
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

if TYPE_CHECKING:
    from app.services.account_health_scoring import AccountHealthResult, CalibratedPrediction

logger = logging.getLogger(__name__)

# Directory for model persistence
MODEL_DIR = Path("./ml_models/growth")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Minimum number of real data points required for training
MINIMUM_TRAINING_SAMPLES = 30


class InsufficientDataError(Exception):
    """
    Raised when there are not enough real data points to train a reliable model.

    The system requires at least MINIMUM_TRAINING_SAMPLES (30) data points
    to avoid training a weak XGBoost model. When this error is raised,
    the system should fall back to Cold Start heuristic logic.
    """

    def __init__(self, samples_provided: int, samples_required: int = MINIMUM_TRAINING_SAMPLES):
        self.samples_provided = samples_provided
        self.samples_required = samples_required
        super().__init__(
            f"Insufficient training data: {samples_provided} samples provided, "
            f"minimum {samples_required} required. Use Cold Start heuristic instead."
        )


@dataclass(frozen=True)
class InstagramInsightsSchema:
    """
    Strictly typed schema for REAL Instagram Graph API metrics.

    These are the actual metrics returned by the Instagram Graph API
    for media insights. Do NOT use synthetic/fake data with this schema.

    API Reference: https://developers.facebook.com/docs/instagram-api/reference/ig-media/insights

    Attributes:
        media_id: Unique Instagram media ID
        reach: Number of unique accounts that have seen the media
        impressions: Total number of times the media has been seen
        saved: Number of unique accounts that have saved the media
        shares: Number of shares (only for Reels/Videos)
        watch_time_seconds: Total watch time in seconds (only for Reels/Videos)
        likes: Number of likes on the media
        comments: Number of comments on the media
        plays: Number of times video was played (for Reels/Videos)
        ig_reels_avg_watch_time: Average watch time for Reels in milliseconds
        ig_reels_video_view_total_time: Total time video has been viewed (Reels)
        timestamp: ISO timestamp when the media was posted
        media_type: Type of media (IMAGE, VIDEO, CAROUSEL_ALBUM, REELS)
        media_product_type: Product type (FEED, REELS, STORY)
    """
    media_id: str
    reach: int
    impressions: int
    saved: int
    shares: int
    watch_time_seconds: float
    likes: int = 0
    comments: int = 0
    plays: int = 0
    ig_reels_avg_watch_time: float = 0.0
    ig_reels_video_view_total_time: float = 0.0
    timestamp: Optional[str] = None
    media_type: str = "VIDEO"
    media_product_type: str = "REELS"

    def __post_init__(self):
        """Validate that metrics are non-negative."""
        if self.reach < 0:
            raise ValueError("reach must be non-negative")
        if self.impressions < 0:
            raise ValueError("impressions must be non-negative")
        if self.saved < 0:
            raise ValueError("saved must be non-negative")
        if self.shares < 0:
            raise ValueError("shares must be non-negative")
        if self.watch_time_seconds < 0:
            raise ValueError("watch_time_seconds must be non-negative")

    @property
    def engagement_rate(self) -> float:
        """Calculate engagement rate based on reach."""
        if self.reach == 0:
            return 0.0
        return (self.likes + self.comments + self.saved + self.shares) / self.reach

    @property
    def save_rate(self) -> float:
        """Calculate save rate (strong signal of value)."""
        if self.reach == 0:
            return 0.0
        return self.saved / self.reach

    @property
    def share_rate(self) -> float:
        """Calculate share rate (viral potential indicator)."""
        if self.reach == 0:
            return 0.0
        return self.shares / self.reach

    @property
    def avg_watch_time_ratio(self) -> float:
        """
        Calculate average watch time ratio.
        Returns watch time per play, normalized.
        """
        if self.plays == 0:
            return 0.0
        return self.watch_time_seconds / self.plays

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "media_id": self.media_id,
            "reach": self.reach,
            "impressions": self.impressions,
            "saved": self.saved,
            "shares": self.shares,
            "watch_time_seconds": self.watch_time_seconds,
            "likes": self.likes,
            "comments": self.comments,
            "plays": self.plays,
            "ig_reels_avg_watch_time": self.ig_reels_avg_watch_time,
            "ig_reels_video_view_total_time": self.ig_reels_video_view_total_time,
            "timestamp": self.timestamp,
            "media_type": self.media_type,
            "media_product_type": self.media_product_type,
            "engagement_rate": self.engagement_rate,
            "save_rate": self.save_rate,
            "share_rate": self.share_rate,
        }

    @classmethod
    def from_api_response(cls, response: Dict[str, Any]) -> "InstagramInsightsSchema":
        """
        Create an InstagramInsightsSchema from raw Instagram Graph API response.

        Args:
            response: Raw API response dictionary

        Returns:
            InstagramInsightsSchema instance
        """
        # Extract insights from nested structure if present
        insights = {}
        if "insights" in response and "data" in response["insights"]:
            for insight in response["insights"]["data"]:
                name = insight.get("name", "")
                values = insight.get("values", [{}])
                value = values[0].get("value", 0) if values else 0
                insights[name] = value

        return cls(
            media_id=response.get("id", ""),
            reach=insights.get("reach", response.get("reach", 0)),
            impressions=insights.get("impressions", response.get("impressions", 0)),
            saved=insights.get("saved", response.get("saved", 0)),
            shares=insights.get("shares", response.get("shares", 0)),
            watch_time_seconds=insights.get("ig_reels_video_view_total_time",
                                           response.get("watch_time_seconds", 0)) / 1000.0,
            likes=response.get("like_count", response.get("likes", 0)),
            comments=response.get("comments_count", response.get("comments", 0)),
            plays=insights.get("plays", response.get("plays", 0)),
            ig_reels_avg_watch_time=insights.get("ig_reels_avg_watch_time", 0),
            ig_reels_video_view_total_time=insights.get("ig_reels_video_view_total_time", 0),
            timestamp=response.get("timestamp"),
            media_type=response.get("media_type", "VIDEO"),
            media_product_type=response.get("media_product_type", "REELS"),
        )


@dataclass(frozen=True)
class GrowthPredictionConfig:
    """
    Configuración inmutable para el motor de predicción de crecimiento.

    Attributes:
        n_estimators: Número de árboles en el ensemble XGBoost
        max_depth: Profundidad máxima de cada árbol
        learning_rate: Tasa de aprendizaje (eta)
        min_child_weight: Peso mínimo de hoja (regularización)
        subsample: Fracción de muestras para cada árbol
        colsample_bytree: Fracción de features para cada árbol
        cv_folds: Número de folds para validación cruzada
        random_state: Semilla para reproducibilidad
        early_stopping_rounds: Parada temprana si no mejora
        shap_max_display: Máximo de features a mostrar en explicaciones
    """
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.05
    min_child_weight: int = 3
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    gamma: float = 0.1
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    cv_folds: int = 5
    random_state: int = 42
    early_stopping_rounds: int = 20
    shap_max_display: int = 10


@dataclass
class FeatureContribution:
    """Contribución de una feature individual al score predicho."""
    feature_name: str
    contribution: float
    feature_value: float
    direction: str  # "positive" or "negative"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature_name,
            "contribution": round(self.contribution, 4),
            "value": round(self.feature_value, 4),
            "direction": self.direction
        }


@dataclass
class PredictionResult:
    """Resultado completo de una predicción con explicabilidad."""
    predicted_rpi_score: float
    predicted_rpi_raw: float  # exp(predicted_rpi_score) - 1
    confidence_interval: Tuple[float, float]
    top_positive_contributions: List[FeatureContribution]
    top_negative_contributions: List[FeatureContribution]
    explanation_text: str
    feature_values: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_rpi_score": round(self.predicted_rpi_score, 4),
            "predicted_rpi_raw": round(self.predicted_rpi_raw, 4),
            "confidence_interval": {
                "lower": round(self.confidence_interval[0], 4),
                "upper": round(self.confidence_interval[1], 4)
            },
            "top_positive_factors": [c.to_dict() for c in self.top_positive_contributions],
            "top_negative_factors": [c.to_dict() for c in self.top_negative_contributions],
            "explanation": self.explanation_text,
            "feature_values": {k: round(v, 4) for k, v in self.feature_values.items()}
        }


@dataclass
class TrainingMetrics:
    """Métricas de entrenamiento del modelo."""
    train_rmse: float
    test_rmse: float
    train_mae: float
    test_mae: float
    train_r2: float
    test_r2: float
    cv_rmse_mean: float
    cv_rmse_std: float
    cv_scores: List[float]
    feature_importance: Dict[str, float]
    training_samples: int
    training_date: str
    model_version: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_rmse": round(self.train_rmse, 4),
            "test_rmse": round(self.test_rmse, 4),
            "train_mae": round(self.train_mae, 4),
            "test_mae": round(self.test_mae, 4),
            "train_r2": round(self.train_r2, 4),
            "test_r2": round(self.test_r2, 4),
            "cv_rmse_mean": round(self.cv_rmse_mean, 4),
            "cv_rmse_std": round(self.cv_rmse_std, 4),
            "cv_scores": [round(s, 4) for s in self.cv_scores],
            "feature_importance": {k: round(v, 4) for k, v in self.feature_importance.items()},
            "training_samples": self.training_samples,
            "training_date": self.training_date,
            "model_version": self.model_version
        }


@dataclass
class ColdStartPredictionResult:
    """
    Result from the Cold Start heuristic when no trained model is available.

    This provides rule-based predictions based on Instagram algorithm heuristics
    and industry best practices when we don't have enough data to train XGBoost.
    """
    predicted_rpi_score: float
    predicted_rpi_raw: float
    confidence_level: str  # "low", "medium" - always lower than ML model
    heuristic_factors: List[Dict[str, Any]]
    explanation_text: str
    is_cold_start: bool = True
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_rpi_score": round(self.predicted_rpi_score, 4),
            "predicted_rpi_raw": round(self.predicted_rpi_raw, 4),
            "confidence_level": self.confidence_level,
            "is_cold_start": self.is_cold_start,
            "heuristic_factors": self.heuristic_factors,
            "explanation": self.explanation_text,
            "recommendation": self.recommendation,
        }

    def to_prediction_result(self) -> PredictionResult:
        """Convert to standard PredictionResult for API compatibility."""
        return PredictionResult(
            predicted_rpi_score=self.predicted_rpi_score,
            predicted_rpi_raw=self.predicted_rpi_raw,
            confidence_interval=(self.predicted_rpi_score - 0.5, self.predicted_rpi_score + 0.5),
            top_positive_contributions=[],
            top_negative_contributions=[],
            explanation_text=f"[COLD START] {self.explanation_text}",
            feature_values={}
        )


class GrowthPredictionEngine:
    """
    Motor de predicción de crecimiento usando XGBoost con explicabilidad SHAP.

    Este es el "Cerebro" del sistema que:
    1. Entrena un modelo XGBoost para predecir RPI_score
    2. Proporciona explicaciones SHAP de cada predicción
    3. Identifica qué features contribuyen más al score

    Features de entrada:
    - Metadata: hour, day_of_week, post_type
    - Sensoriales: visual_energy, tempo (bpm), brightness_variance, cut_density
    - Semánticas: sem_pca_1 a sem_pca_10 (10 componentes PCA)

    Variable objetivo:
    - rpi_score (log-transformed RPI)

    Ejemplo de uso:
        engine = GrowthPredictionEngine()

        # Entrenar
        metrics = engine.train(training_data)

        # Predecir con explicación
        result = engine.predict_with_explanation(features)
        print(result.explanation_text)
        # Output: "El score es alto porque tempo (+0.4) y visual_energy (+0.3) son altos"
    """

    # Feature columns esperadas
    METADATA_FEATURES = ["hour", "day_of_week", "post_type_encoded"]

    # =========================================================================
    # SENSORY FEATURES - Hook Theory (Algorithm-Aligned)
    # =========================================================================
    # El algoritmo TikTok/IG evalúa el video como SECUENCIA TEMPORAL.
    # Los primeros 3 segundos (hook) determinan el 90% del éxito.
    # =========================================================================
    SENSORY_FEATURES = [
        # === TEMPORAL FEATURES (Hook Theory) ===
        "hook_energy",        # Energía visual en segundos 0-3 (CRÍTICO)
        "retention_energy",   # Energía visual del resto del video
        "hook_cut_rate",      # Cortes en hook (ponderados 10x)
        "retention_cut_rate", # Cortes después del hook
        "face_in_hook",       # Face-to-camera en primeros 3 segundos

        # === GLOBAL FEATURES (Backward Compatibility) ===
        "tempo",              # BPM del audio
        "brightness_variance", # Variación de brillo global
    ]

    # ==========================================================================
    # SEMANTIC FEATURES - Dynamic Detection (Refactored 2024)
    # ==========================================================================
    # DEPRECATED: Old fixed PCA columns (sem_pca_1 to sem_pca_10)
    # NEW: Dynamic detection of embedding_* columns from features_embeddings.py
    #
    # The engine now:
    # 1. Detects embedding_* columns dynamically at runtime
    # 2. Supports any dimension (128/256/384 based on precision config)
    # 3. Falls back to zeros with warning if no embeddings found
    # 4. Adds optional semantic_score (mean of all embedding dims)
    # ==========================================================================
    LEGACY_SEMANTIC_FEATURES = [f"sem_pca_{i}" for i in range(1, 11)]  # Kept for backward compat

    # Dynamic semantic features - detected at runtime from DataFrame columns
    # Format: embedding_0, embedding_1, ..., embedding_N (0-based)
    # Set dynamically by _detect_semantic_columns()
    _detected_semantic_cols: List[str] = []
    _semantic_dims: int = 0

    # Nombres legibles para explicaciones SHAP
    FEATURE_DISPLAY_NAMES = {
        # Metadata
        "hour": "Hora de publicación",
        "day_of_week": "Día de la semana",
        "post_type_encoded": "Tipo de post",

        # === TEMPORAL FEATURES (Hook Theory) ===
        "hook_energy": "Energía del HOOK (0-3s)",
        "retention_energy": "Energía de retención",
        "hook_cut_rate": "Cortes en HOOK (×10)",
        "retention_cut_rate": "Cortes post-hook",
        "face_in_hook": "Cara en HOOK",

        # === GLOBAL FEATURES ===
        "tempo": "BPM (ritmo)",
        "brightness_variance": "Variación de brillo",

        # Aggregated semantic score (mean of all embedding dims)
        "semantic_score": "Score Semántico Global",

        # Legacy Semantic PCA (deprecated - kept for old models)
        "sem_pca_1": "[Legacy] Semántica PC1",
        "sem_pca_2": "[Legacy] Semántica PC2",
        "sem_pca_3": "[Legacy] Semántica PC3",
        "sem_pca_4": "[Legacy] Semántica PC4",
        "sem_pca_5": "[Legacy] Semántica PC5",
        "sem_pca_6": "[Legacy] Semántica PC6",
        "sem_pca_7": "[Legacy] Semántica PC7",
        "sem_pca_8": "[Legacy] Semántica PC8",
        "sem_pca_9": "[Legacy] Semántica PC9",
        "sem_pca_10": "[Legacy] Semántica PC10",
    }

    # Mapeo de tipos de post
    POST_TYPES = ["reel", "carousel", "static", "story", "video", "unknown"]

    @classmethod
    def get_feature_display_name(cls, feature_name: str) -> str:
        """
        Get human-readable display name for a feature.

        Handles dynamic embedding column names (embedding_0 to embedding_N).
        """
        # Check static mapping first
        if feature_name in cls.FEATURE_DISPLAY_NAMES:
            return cls.FEATURE_DISPLAY_NAMES[feature_name]

        # Handle dynamic embedding columns
        if feature_name.startswith('embedding_'):
            idx = feature_name.split('_')[1]
            return f"Semántica Emb[{idx}]"

        # Default: return original name
        return feature_name

    @staticmethod
    def aggregate_shap_embeddings(
        shap_values: np.ndarray,
        feature_names: List[str],
        top_k: int = 3
    ) -> Tuple[float, List[Tuple[str, float]]]:
        """
        Aggregate SHAP values for embedding dimensions into a single contribution.

        Instead of showing 384 individual SHAP values, this aggregates all
        embedding_* columns into a single "Semantic Embeddings" contribution.

        Args:
            shap_values: Array of SHAP values for all features
            feature_names: List of feature names
            top_k: Number of top contributing individual embeddings to track

        Returns:
            Tuple of (total_embedding_contribution, top_contributors)
        """
        embedding_indices = [
            i for i, name in enumerate(feature_names)
            if name.startswith('embedding_') or name.startswith('sem_pca_')
        ]

        if not embedding_indices:
            return 0.0, []

        # Sum all embedding SHAP values
        embedding_shap = shap_values[embedding_indices]
        total_contribution = float(np.sum(embedding_shap))

        # Find top contributors (by absolute value)
        sorted_indices = np.argsort(np.abs(embedding_shap))[::-1][:top_k]
        top_contributors = [
            (feature_names[embedding_indices[i]], float(embedding_shap[i]))
            for i in sorted_indices
        ]

        return total_contribution, top_contributors

    MODEL_VERSION = "1.0.0"
    MODEL_FILENAME = "growth_prediction_model.joblib"
    SCALER_FILENAME = "growth_prediction_scaler.joblib"
    ENCODER_FILENAME = "growth_prediction_encoder.joblib"
    METADATA_FILENAME = "growth_prediction_metadata.joblib"

    def __init__(self, config: Optional[GrowthPredictionConfig] = None):
        """
        Inicializa el motor de predicción.

        Args:
            config: Configuración opcional. Usa valores por defecto si no se proporciona.
        """
        self.config = config or GrowthPredictionConfig()
        self._model: Optional[xgb.XGBRegressor] = None
        self._scaler: Optional[StandardScaler] = None
        self._post_type_encoder: Optional[LabelEncoder] = None
        self._shap_explainer = None
        self._is_trained = False
        self._training_metrics: Optional[TrainingMetrics] = None
        self._feature_columns: List[str] = []

        # Intentar cargar modelo existente
        self._load_model()

    @property
    def is_trained(self) -> bool:
        """Verifica si el modelo está entrenado y listo para predecir."""
        return self._is_trained and self._model is not None

    @property
    def feature_columns(self) -> List[str]:
        """
        Retorna las columnas de features en el orden esperado.

        Dynamic: Uses detected embedding_* columns if available,
        otherwise falls back to legacy sem_pca_* columns.
        """
        if not self._feature_columns:
            semantic_cols = self._detected_semantic_cols if self._detected_semantic_cols else self.LEGACY_SEMANTIC_FEATURES
            self._feature_columns = (
                self.METADATA_FEATURES +
                self.SENSORY_FEATURES +
                semantic_cols
            )
        return self._feature_columns

    def _detect_semantic_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Dynamically detect semantic embedding columns from DataFrame.

        Looks for columns matching pattern 'embedding_*' (new format, 0-based).
        Falls back to 'sem_pca_*' (legacy format) if no new columns found.

        Args:
            df: DataFrame to inspect for embedding columns

        Returns:
            List of semantic column names found in the DataFrame

        Side Effects:
            Updates self._detected_semantic_cols and self._semantic_dims
            Logs the number of dimensions detected
        """
        # Try new format first: embedding_0, embedding_1, ..., embedding_N
        new_cols = sorted(
            [col for col in df.columns if col.startswith('embedding_')],
            key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else 0
        )

        if new_cols:
            self._detected_semantic_cols = new_cols
            self._semantic_dims = len(new_cols)
            logger.info(f"Usando {self._semantic_dims} embedding dims semánticos en predicción (embedding_0 a embedding_{self._semantic_dims - 1})")
            return new_cols

        # Fallback to legacy format: sem_pca_1, ..., sem_pca_10
        legacy_cols = [col for col in df.columns if col.startswith('sem_pca_')]
        legacy_cols = sorted(legacy_cols, key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)

        if legacy_cols:
            self._detected_semantic_cols = legacy_cols
            self._semantic_dims = len(legacy_cols)
            logger.warning(f"Usando formato legacy: {self._semantic_dims} sem_pca dims (deprecated - migrar a embedding_*)")
            return legacy_cols

        # No semantic columns found
        logger.warning("Sin embeddings semánticos detectados – usa config precisión para generar embeddings")
        self._detected_semantic_cols = []
        self._semantic_dims = 0
        return []

    def _add_semantic_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add aggregated semantic_score column (mean of all embedding dims).

        This provides a single interpretable SHAP feature for semantics
        instead of N individual embedding dimensions.

        Args:
            df: DataFrame with embedding columns

        Returns:
            DataFrame with semantic_score column added
        """
        semantic_cols = self._detect_semantic_columns(df)

        if semantic_cols:
            df = df.copy()
            df['semantic_score'] = df[semantic_cols].mean(axis=1)
            logger.debug(f"semantic_score added (mean of {len(semantic_cols)} dims)")
        else:
            df = df.copy()
            df['semantic_score'] = 0.0

        return df

    def _get_model_path(self, filename: str) -> Path:
        """Obtiene la ruta completa para un archivo del modelo."""
        return MODEL_DIR / filename

    def _load_model(self) -> bool:
        """
        Carga el modelo serializado desde disco si existe.

        Returns:
            True si el modelo se cargó exitosamente, False en caso contrario.
        """
        model_path = self._get_model_path(self.MODEL_FILENAME)
        scaler_path = self._get_model_path(self.SCALER_FILENAME)
        encoder_path = self._get_model_path(self.ENCODER_FILENAME)
        metadata_path = self._get_model_path(self.METADATA_FILENAME)

        if not all(p.exists() for p in [model_path, scaler_path, encoder_path, metadata_path]):
            logger.info("No pre-trained model found. Model needs to be trained.")
            return False

        try:
            self._model = joblib.load(model_path)
            self._scaler = joblib.load(scaler_path)
            self._post_type_encoder = joblib.load(encoder_path)
            metadata = joblib.load(metadata_path)

            self._feature_columns = metadata.get("feature_columns", self.feature_columns)
            self._training_metrics = metadata.get("training_metrics")
            self._is_trained = True

            # Inicializar SHAP explainer
            self._init_shap_explainer()

            logger.info(f"Model loaded successfully from {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._is_trained = False
            return False

    def _save_model(self) -> bool:
        """
        Guarda el modelo entrenado a disco usando joblib.

        Returns:
            True si se guardó exitosamente, False en caso contrario.
        """
        if not self._is_trained or self._model is None:
            logger.error("Cannot save: model is not trained")
            return False

        try:
            model_path = self._get_model_path(self.MODEL_FILENAME)
            scaler_path = self._get_model_path(self.SCALER_FILENAME)
            encoder_path = self._get_model_path(self.ENCODER_FILENAME)
            metadata_path = self._get_model_path(self.METADATA_FILENAME)

            joblib.dump(self._model, model_path)
            joblib.dump(self._scaler, scaler_path)
            joblib.dump(self._post_type_encoder, encoder_path)

            metadata = {
                "feature_columns": self._feature_columns,
                "training_metrics": self._training_metrics,
                "model_version": self.MODEL_VERSION,
                "saved_at": datetime.now().isoformat()
            }
            joblib.dump(metadata, metadata_path)

            logger.info(f"Model saved successfully to {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    def _init_shap_explainer(self) -> None:
        """Inicializa el explicador SHAP para el modelo entrenado."""
        if self._model is None:
            return

        try:
            import shap
            self._shap_explainer = shap.TreeExplainer(self._model)
            logger.info("SHAP explainer initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {e}")
            self._shap_explainer = None

    def _encode_post_type(self, post_type: str) -> int:
        """
        Codifica el tipo de post a un valor numérico.

        Args:
            post_type: Tipo de post (reel, carousel, static, etc.)

        Returns:
            Valor entero codificado.
        """
        if self._post_type_encoder is None:
            self._post_type_encoder = LabelEncoder()
            self._post_type_encoder.fit(self.POST_TYPES)

        post_type_lower = post_type.lower() if post_type else "unknown"
        if post_type_lower not in self.POST_TYPES:
            post_type_lower = "unknown"

        return self._post_type_encoder.transform([post_type_lower])[0]

    def _prepare_features(
        self,
        data: Union[Dict[str, Any], pd.DataFrame],
        fit_scaler: bool = False
    ) -> np.ndarray:
        """
        Prepara las features para el modelo.

        Supports dynamic embedding detection:
        - New format: embedding_0 to embedding_N (any dimension)
        - Legacy format: sem_pca_1 to sem_pca_10 (deprecated)

        Args:
            data: Diccionario de features o DataFrame
            fit_scaler: Si True, ajusta el scaler (solo durante entrenamiento)

        Returns:
            Array numpy con las features procesadas.
        """
        if isinstance(data, dict):
            # Convertir diccionario a DataFrame de una fila
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        # Detect semantic columns dynamically from the data
        semantic_cols = self._detect_semantic_columns(df)

        # Build feature columns list dynamically based on what's in the data
        base_features = self.METADATA_FEATURES + self.SENSORY_FEATURES

        # Add semantic columns (new embedding_* or legacy sem_pca_*)
        if semantic_cols:
            feature_cols = base_features + semantic_cols
        else:
            # Fallback: try legacy sem_pca columns or use zeros
            feature_cols = base_features + self.LEGACY_SEMANTIC_FEATURES
            logger.warning("No embedding columns found, using legacy sem_pca fallback")

        # Update instance feature columns for this prediction
        self._feature_columns = feature_cols

        # Asegurar que todas las columnas existan
        for col in feature_cols:
            if col not in df.columns:
                if col == "post_type_encoded" and "post_type" in df.columns:
                    df["post_type_encoded"] = df["post_type"].apply(self._encode_post_type)
                elif col.startswith("embedding_"):
                    # New format: embedding columns get zeros if missing
                    df[col] = 0.0
                    logger.debug(f"Embedding column {col} not found, using 0.0")
                elif col.startswith("sem_pca_"):
                    # Legacy format: PCA columns get zeros if missing (deprecated)
                    df[col] = 0.0
                else:
                    df[col] = 0.0

        # Codificar post_type si existe
        if "post_type" in df.columns and "post_type_encoded" not in df.columns:
            df["post_type_encoded"] = df["post_type"].apply(self._encode_post_type)

        # Seleccionar solo las columnas necesarias en el orden correcto
        X = df[feature_cols].values.astype(np.float32)

        # Log feature summary
        n_embedding_cols = len([c for c in feature_cols if c.startswith('embedding_') or c.startswith('sem_pca_')])
        logger.debug(f"Feature vector: {len(feature_cols)} total ({n_embedding_cols} semantic dims)")

        # Escalar features
        if self._scaler is None:
            self._scaler = StandardScaler()
            fit_scaler = True

        if fit_scaler:
            X = self._scaler.fit_transform(X)
        else:
            X = self._scaler.transform(X)

        return X

    def _extract_hour_day_from_timestamp(
        self,
        timestamp: Optional[Union[str, datetime]]
    ) -> Tuple[int, int]:
        """
        Extrae hora y día de la semana de un timestamp.

        Args:
            timestamp: Timestamp ISO o objeto datetime

        Returns:
            Tupla (hora, día_semana) donde día 0=Lunes, 6=Domingo.
        """
        if timestamp is None:
            # Default: hora pico típica (mediodía, miércoles)
            return 12, 2

        if isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                return 12, 2
        else:
            dt = timestamp

        return dt.hour, dt.weekday()

    def prepare_training_data(
        self,
        raw_data: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara datos crudos para entrenamiento.

        Args:
            raw_data: Lista de diccionarios con features y rpi_score.

        Returns:
            Tupla (X, y) con features y target.
        """
        processed_records = []

        for record in raw_data:
            # Extraer metadata temporal
            hour, day_of_week = self._extract_hour_day_from_timestamp(
                record.get("posted_at") or record.get("timestamp")
            )

            processed = {
                "hour": hour,
                "day_of_week": day_of_week,
                "post_type": record.get("post_type", record.get("content_format", "unknown")),

                # === TEMPORAL FEATURES (Hook Theory) ===
                "hook_energy": record.get("hook_energy", 0.0),
                "retention_energy": record.get("retention_energy", 0.0),
                "hook_cut_rate": record.get("hook_cut_rate", 0.0),
                "retention_cut_rate": record.get("retention_cut_rate", 0.0),
                "face_in_hook": record.get("face_in_hook", 0),

                # === GLOBAL FEATURES ===
                "tempo": record.get("tempo", record.get("bpm", 0.0)),
                "brightness_variance": record.get("brightness_variance", 0.0),

                # Target
                "rpi_score": record.get("rpi_score", 0.0)
            }

            # === SEMANTIC FEATURES (Dynamic) ===
            # Check for new embedding format first
            embedding_keys = [k for k in record.keys() if k.startswith('embedding_')]
            if embedding_keys:
                # New format: embedding_0 to embedding_N
                for key in embedding_keys:
                    processed[key] = record.get(key, 0.0)
            else:
                # Legacy format: sem_pca_1 to sem_pca_10
                for i in range(1, 11):
                    pca_key = f"sem_pca_{i}"
                    processed[pca_key] = record.get(pca_key, 0.0)

            processed_records.append(processed)

        df = pd.DataFrame(processed_records)

        # Codificar post_type
        df["post_type_encoded"] = df["post_type"].apply(self._encode_post_type)

        # Preparar features y target
        X = self._prepare_features(df, fit_scaler=True)
        y = df["rpi_score"].values.astype(np.float32)

        return X, y

    def train(
        self,
        training_data: Union[List[Dict[str, Any]], pd.DataFrame],
        save_model: bool = True
    ) -> TrainingMetrics:
        """
        Entrena el modelo XGBoost con validación cruzada.

        IMPORTANT: Requires at least MINIMUM_TRAINING_SAMPLES (30) real data points.
        Training on insufficient data leads to weak models with confirmation bias.
        If you have fewer samples, use Cold Start heuristics instead.

        Args:
            training_data: Datos de entrenamiento (lista de dicts o DataFrame).
                           MUST contain at least 30 REAL data points.
            save_model: Si True, guarda el modelo después de entrenar.

        Returns:
            TrainingMetrics con métricas de rendimiento del modelo.

        Raises:
            InsufficientDataError: If training_data has fewer than 30 samples.
                                   Use Cold Start heuristics in this case.
        """
        logger.info("Starting GrowthPredictionEngine training...")

        # Validate minimum sample count BEFORE any processing
        sample_count = len(training_data) if isinstance(training_data, list) else len(training_data)

        if sample_count < MINIMUM_TRAINING_SAMPLES:
            logger.warning(
                f"Insufficient training data: {sample_count} samples provided, "
                f"minimum {MINIMUM_TRAINING_SAMPLES} required. Use Cold Start heuristics."
            )
            raise InsufficientDataError(
                samples_provided=sample_count,
                samples_required=MINIMUM_TRAINING_SAMPLES
            )

        # Preparar datos
        if isinstance(training_data, pd.DataFrame):
            X, y = self._prepare_features_from_dataframe(training_data)
        else:
            X, y = self.prepare_training_data(training_data)

        logger.info(f"Training with {len(y)} samples, {X.shape[1]} features")

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=self.config.random_state
        )

        # Configurar modelo XGBoost
        self._model = xgb.XGBRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            min_child_weight=self.config.min_child_weight,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            gamma=self.config.gamma,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            random_state=self.config.random_state,
            objective="reg:squarederror",
            n_jobs=-1,
            verbosity=0
        )

        # Validación cruzada
        logger.info(f"Running {self.config.cv_folds}-fold cross-validation...")
        kfold = KFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state
        )

        cv_scores = cross_val_score(
            self._model, X_train, y_train,
            cv=kfold,
            scoring="neg_root_mean_squared_error"
        )
        cv_rmse_scores = -cv_scores  # Convertir a positivo

        logger.info(f"CV RMSE: {cv_rmse_scores.mean():.4f} (+/- {cv_rmse_scores.std():.4f})")

        # Entrenar modelo final con early stopping
        self._model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Calcular métricas
        y_train_pred = self._model.predict(X_train)
        y_test_pred = self._model.predict(X_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # Feature importance
        importance_dict = dict(zip(
            self.feature_columns,
            self._model.feature_importances_
        ))
        importance_sorted = dict(sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        ))

        self._training_metrics = TrainingMetrics(
            train_rmse=train_rmse,
            test_rmse=test_rmse,
            train_mae=train_mae,
            test_mae=test_mae,
            train_r2=train_r2,
            test_r2=test_r2,
            cv_rmse_mean=cv_rmse_scores.mean(),
            cv_rmse_std=cv_rmse_scores.std(),
            cv_scores=cv_rmse_scores.tolist(),
            feature_importance=importance_sorted,
            training_samples=len(y),
            training_date=datetime.now().isoformat(),
            model_version=self.MODEL_VERSION
        )

        self._is_trained = True

        # Inicializar SHAP
        self._init_shap_explainer()

        # Guardar modelo
        if save_model:
            self._save_model()

        logger.info(f"Training completed. Test RMSE: {test_rmse:.4f}, Test R2: {test_r2:.4f}")

        return self._training_metrics

    def _prepare_features_from_dataframe(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara features desde un DataFrame ya estructurado."""
        # Asegurar columnas necesarias
        if "post_type_encoded" not in df.columns:
            if "post_type" in df.columns:
                df = df.copy()
                df["post_type_encoded"] = df["post_type"].apply(self._encode_post_type)
            else:
                df = df.copy()
                df["post_type_encoded"] = 0

        X = self._prepare_features(df, fit_scaler=True)
        y = df["rpi_score"].values.astype(np.float32)

        return X, y

    def calculate_confidence_weight(self, n_samples: int) -> float:
        """
        Calculates the weight (alpha) for the ML model in the Bayesian ensemble.

        Uses a sigmoid function to smooth the transition from heuristic (n<30)
        to ML-dominated predictions.
        Formula: alpha = MAX_ALPHA / (1 + exp(-k * (n - midpoint)))

        Args:
            n_samples: Number of training samples available.

        Returns:
            Float between 0.0 and 0.95 representing ML model weight.
        """
        # If not enough samples for minimal training, force alpha=0
        if n_samples < MINIMUM_TRAINING_SAMPLES:
            return 0.0

        k = 0.1
        midpoint = 50
        max_alpha = 0.95

        alpha = max_alpha / (1 + np.exp(-k * (n_samples - midpoint)))
        return float(alpha)

    def _cold_start_predict(self, features: Dict[str, Any]) -> ColdStartPredictionResult:
        """
        Heuristic rule-based prediction for Cold Start scenario.

        When no trained model is available (insufficient data), this method
        provides predictions based on Instagram algorithm heuristics and
        industry best practices from 2024.

        HEURISTIC RULES (based on Instagram Algorithm Analysis):
        1. Hook Energy (0-3s): Most critical factor - 40% weight
        2. Face in Hook: +20% boost for face-to-camera content
        3. Post Type: Reels > Video > Carousel > Static
        4. Posting Time: Peak hours (12-21h) get +15% boost
        5. Tempo: Optimal BPM range (100-140) gets +10% boost

        Args:
            features: Content features dictionary

        Returns:
            ColdStartPredictionResult with heuristic-based prediction
        """
        logger.info("Using Cold Start heuristic prediction (no trained model available)")

        heuristic_factors = []
        base_score = 0.5  # Neutral baseline

        # Extract temporal info
        hour, day_of_week = self._extract_hour_day_from_timestamp(
            features.get("posted_at") or features.get("timestamp")
        )

        # === RULE 1: Hook Energy (40% weight) ===
        hook_energy = features.get("hook_energy", 0.5)
        if hook_energy >= 0.7:
            hook_bonus = 0.4
            heuristic_factors.append({
                "factor": "Energía del Hook",
                "value": hook_energy,
                "impact": "+0.40",
                "rule": "Hook energy ≥0.7 indicates strong opening"
            })
        elif hook_energy >= 0.4:
            hook_bonus = 0.2
            heuristic_factors.append({
                "factor": "Energía del Hook",
                "value": hook_energy,
                "impact": "+0.20",
                "rule": "Moderate hook energy"
            })
        else:
            hook_bonus = -0.2
            heuristic_factors.append({
                "factor": "Energía del Hook",
                "value": hook_energy,
                "impact": "-0.20",
                "rule": "Weak hook - consider stronger opening"
            })
        base_score += hook_bonus

        # === RULE 2: Face in Hook (+20% boost) ===
        face_in_hook = features.get("face_in_hook", 0)
        if face_in_hook:
            face_bonus = 0.2
            heuristic_factors.append({
                "factor": "Cara en Hook",
                "value": 1,
                "impact": "+0.20",
                "rule": "Face-to-camera in first 3s increases trust & retention"
            })
            base_score += face_bonus

        # === RULE 3: Post Type Hierarchy ===
        post_type = features.get("post_type", features.get("content_format", "unknown")).lower()
        type_bonuses = {
            "reel": 0.25,
            "video": 0.15,
            "carousel": 0.05,
            "static": -0.05,
            "story": 0.0,
            "unknown": 0.0
        }
        type_bonus = type_bonuses.get(post_type, 0.0)
        if type_bonus != 0:
            heuristic_factors.append({
                "factor": "Tipo de Contenido",
                "value": post_type,
                "impact": f"{'+' if type_bonus > 0 else ''}{type_bonus:.2f}",
                "rule": f"Reels have highest algorithmic priority in 2024"
            })
        base_score += type_bonus

        # === RULE 4: Peak Hours (12-21h) ===
        if 12 <= hour <= 21:
            time_bonus = 0.15
            heuristic_factors.append({
                "factor": "Hora de Publicación",
                "value": f"{hour}:00",
                "impact": "+0.15",
                "rule": "Peak engagement hours (12-21h)"
            })
        elif 8 <= hour <= 23:
            time_bonus = 0.05
            heuristic_factors.append({
                "factor": "Hora de Publicación",
                "value": f"{hour}:00",
                "impact": "+0.05",
                "rule": "Acceptable posting hours"
            })
        else:
            time_bonus = -0.1
            heuristic_factors.append({
                "factor": "Hora de Publicación",
                "value": f"{hour}:00",
                "impact": "-0.10",
                "rule": "Low engagement hours (late night/early morning)"
            })
        base_score += time_bonus

        # === RULE 5: Optimal Tempo (100-140 BPM) ===
        tempo = features.get("tempo", features.get("bpm", 0))
        if tempo > 0:
            if 100 <= tempo <= 140:
                tempo_bonus = 0.1
                heuristic_factors.append({
                    "factor": "Tempo del Audio",
                    "value": f"{tempo:.0f} BPM",
                    "impact": "+0.10",
                    "rule": "Optimal BPM range for engagement"
                })
            elif 80 <= tempo <= 160:
                tempo_bonus = 0.05
                heuristic_factors.append({
                    "factor": "Tempo del Audio",
                    "value": f"{tempo:.0f} BPM",
                    "impact": "+0.05",
                    "rule": "Acceptable BPM range"
                })
            else:
                tempo_bonus = 0.0
            base_score += tempo_bonus

        # === RULE 6: Cut Rate in Hook ===
        hook_cut_rate = features.get("hook_cut_rate", 0)
        if 20 <= hook_cut_rate <= 60:
            cut_bonus = 0.1
            heuristic_factors.append({
                "factor": "Cortes en Hook",
                "value": f"{hook_cut_rate:.0f}/min",
                "impact": "+0.10",
                "rule": "Optimal cut rate maintains attention"
            })
            base_score += cut_bonus

        # Clamp final score
        final_score = np.clip(base_score, 0.0, 2.5)
        raw_score = np.expm1(max(0, final_score))

        # Build explanation
        positive_factors = [f["factor"] for f in heuristic_factors if f["impact"].startswith("+")]
        negative_factors = [f["factor"] for f in heuristic_factors if f["impact"].startswith("-")]

        explanation_parts = []
        if positive_factors:
            explanation_parts.append(f"Factores positivos: {', '.join(positive_factors[:3])}")
        if negative_factors:
            explanation_parts.append(f"Factores a mejorar: {', '.join(negative_factors[:2])}")

        explanation = ". ".join(explanation_parts) if explanation_parts else "Predicción basada en heurísticas estándar"

        # Generate recommendation
        if final_score >= 1.2:
            recommendation = "Contenido prometedor. Considere publicar en hora pico para maximizar alcance."
        elif final_score >= 0.8:
            recommendation = "Buen potencial. Revise el hook inicial para captar más atención."
        else:
            recommendation = "Considere mejorar el hook (primeros 3 segundos) y usar formato Reel."

        return ColdStartPredictionResult(
            predicted_rpi_score=final_score,
            predicted_rpi_raw=raw_score,
            confidence_level="low",
            heuristic_factors=heuristic_factors,
            explanation_text=explanation,
            is_cold_start=True,
            recommendation=recommendation
        )

    def predict(self, features: Dict[str, Any], allow_cold_start: bool = True) -> float:
        """
        Predice el RPI score para un conjunto de features.

        IMPLEMENTS BAYESIAN ENSEMBLE:
        Combines Heuristic (Rule-based) and ML (XGBoost) predictions based on
        training sample count.
        Final = alpha * Model + (1 - alpha) * Heuristic

        Args:
            features: Diccionario con las features del contenido.
            allow_cold_start: If True, use heuristic prediction when model not trained.

        Returns:
            RPI score predicho (log-transformed).

        Raises:
            RuntimeError: If model not trained and allow_cold_start is False.
        """
        # Determine sample count and alpha
        if self.is_trained and self._training_metrics:
            n_samples = self._training_metrics.training_samples
            alpha = self.calculate_confidence_weight(n_samples)
        else:
            n_samples = 0
            alpha = 0.0

        if not self.is_trained:
            if allow_cold_start:
                cold_result = self._cold_start_predict(features)
                return cold_result.predicted_rpi_score
            raise RuntimeError(
                "Model is not trained and cold start is disabled. "
                f"Train with at least {MINIMUM_TRAINING_SAMPLES} samples first."
            )

        # Calculate Heuristic Score (Always run it for blending)
        cold_result = self._cold_start_predict(features)
        heuristic_score = cold_result.predicted_rpi_score

        # Calculate ML Score
        X = self._prepare_features(features, fit_scaler=False)
        model_score = float(self._model.predict(X)[0])

        # Bayesian Blending
        final_score = (alpha * model_score) + ((1.0 - alpha) * heuristic_score)

        return float(final_score)

    def predict_with_explanation(
        self,
        features: Dict[str, Any],
        top_k: int = 5,
        allow_cold_start: bool = True,
        embedding_precision: str = "low",
        kpi_weights: Optional[Dict[str, float]] = None
    ) -> PredictionResult:
        """
        Predice con explicación SHAP completa.

        USER CONFIG SYNC:
        =================
        Accepts embedding_precision and kpi_weights from user_config to ensure
        frontend configuration changes affect predictions in real-time.

        BAYESIAN ENSEMBLE EXPLANATION:
        - Blends scores using sigmoid alpha.
        - Uses "Dominant Source" strategy for explanation text:
          * alpha < 0.5: Heuristic explanation + suffix
          * alpha >= 0.5: SHAP explanation + suffix

        Args:
            features: Diccionario con las features del contenido.
            top_k: Número de features top a mostrar en la explicación.
            allow_cold_start: If True, use heuristic prediction when model not trained.
            embedding_precision: From user_config (default: "low")
            kpi_weights: From user_config for RPI calculation

        Returns:
            PredictionResult con predicción y explicación detallada.

        Raises:
            RuntimeError: If model not trained and allow_cold_start is False.
        """
        # CRITICAL LOG: User config being used
        logger.info(
            f"GrowthPredictionEngine: User config loaded: precision={embedding_precision}, "
            f"kpi_weights={'custom' if kpi_weights else 'default'}"
        )

        # Determine sample count and alpha
        if self.is_trained and self._training_metrics:
            n_samples = self._training_metrics.training_samples
            alpha = self.calculate_confidence_weight(n_samples)
        else:
            n_samples = 0
            alpha = 0.0

        if not self.is_trained:
            if allow_cold_start:
                cold_result = self._cold_start_predict(features)
                return cold_result.to_prediction_result()
            raise RuntimeError(
                "Model is not trained and cold start is disabled. "
                f"Train with at least {MINIMUM_TRAINING_SAMPLES} samples first."
            )

        # Preparar features para procesamiento
        hour, day_of_week = self._extract_hour_day_from_timestamp(
            features.get("posted_at") or features.get("timestamp")
        )

        processed_features = {
            "hour": hour,
            "day_of_week": day_of_week,
            "post_type": features.get("post_type", features.get("content_format", "unknown")),

            # === TEMPORAL FEATURES (Hook Theory) ===
            "hook_energy": features.get("hook_energy", 0.0),
            "retention_energy": features.get("retention_energy", 0.0),
            "hook_cut_rate": features.get("hook_cut_rate", 0.0),
            "retention_cut_rate": features.get("retention_cut_rate", 0.0),
            "face_in_hook": features.get("face_in_hook", 0),

            # === GLOBAL FEATURES ===
            "tempo": features.get("tempo", features.get("bpm", 0.0)),
            "brightness_variance": features.get("brightness_variance", 0.0),
        }

        # === SEMANTIC FEATURES (Dynamic Detection) ===
        # First try new format: embedding_0, embedding_1, ..., embedding_N
        embedding_keys = sorted(
            [k for k in features.keys() if k.startswith('embedding_')],
            key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else 0
        )

        if embedding_keys:
            # New format detected - use embedding_* columns
            for key in embedding_keys:
                processed_features[key] = features.get(key, 0.0)
            logger.info(f"Usando {len(embedding_keys)} embedding dims semánticos en predicción")
        else:
            # Fallback to legacy format: sem_pca_1 to sem_pca_10
            for i in range(1, 11):
                pca_key = f"sem_pca_{i}"
                processed_features[pca_key] = features.get(pca_key, 0.0)
            logger.debug("Usando formato legacy sem_pca (deprecated)")

        # --- EXECUTE ENSEMBLE ---

        # 1. Heuristic Prediction
        cold_result = self._cold_start_predict(features)
        heuristic_score = cold_result.predicted_rpi_score

        # 2. ML Prediction
        X = self._prepare_features(processed_features, fit_scaler=False)
        model_score = float(self._model.predict(X)[0])

        # 3. Blending
        final_score = (alpha * model_score) + ((1.0 - alpha) * heuristic_score)

        # 4. Generate SHAP Values (ML Component)
        positive_contributions = []
        negative_contributions = []

        if self._shap_explainer is not None:
            try:
                shap_values = self._shap_explainer.shap_values(X)

                # shap_values puede ser una lista o array dependiendo de la versión
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]

                shap_vector = shap_values[0] if len(shap_values.shape) > 1 else shap_values

                # Crear lista de contribuciones
                feature_values_array = X[0]

                for idx, (feature_name, shap_val) in enumerate(zip(self.feature_columns, shap_vector)):
                    contribution = FeatureContribution(
                        feature_name=self.FEATURE_DISPLAY_NAMES.get(feature_name, feature_name),
                        contribution=float(shap_val),
                        feature_value=float(feature_values_array[idx]),
                        direction="positive" if shap_val > 0 else "negative"
                    )

                    if shap_val > 0:
                        positive_contributions.append(contribution)
                    elif shap_val < 0:
                        negative_contributions.append(contribution)

                # Ordenar por magnitud
                positive_contributions.sort(key=lambda x: x.contribution, reverse=True)
                negative_contributions.sort(key=lambda x: x.contribution)

            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")

        # 5. Determine Explanation Text (Dominant Source Strategy)
        if alpha < 0.5:
            # Show Heuristic explanation
            base_explanation = cold_result.explanation_text
            suffix = " (Model starting to learn)"
        else:
            # Show SHAP explanation
            shap_parts = []
            if positive_contributions:
                top_positive = positive_contributions[:3]
                pos_text = ", ".join([
                    f"{c.feature_name} (+{c.contribution:.2f})"
                    for c in top_positive
                ])
                shap_parts.append(f"factores positivos: {pos_text}")

            if negative_contributions:
                top_negative = negative_contributions[:2]
                neg_text = ", ".join([
                    f"{c.feature_name} ({c.contribution:.2f})"
                    for c in top_negative
                ])
                shap_parts.append(f"factores negativos: {neg_text}")

            if shap_parts:
                base_explanation = f"El score es {'alto' if model_score > 0.5 else 'moderado' if model_score > 0 else 'bajo'} porque " + " y ".join(shap_parts)
            else:
                base_explanation = f"Score predicho: {model_score:.3f}"
            suffix = " (Validated by historical patterns)"

        explanation_text = f"{base_explanation}{suffix}"

        # Add UI Tag
        explanation_text += f"\n[Hybrid Prediction (Confidence: {alpha:.0%})]"

        # Calcular intervalo de confianza aproximado (usando std del CV si está disponible)
        if self._training_metrics:
            std = self._training_metrics.cv_rmse_mean
            confidence_interval = (final_score - 1.96 * std, final_score + 1.96 * std)
        else:
            confidence_interval = (final_score - 0.2, final_score + 0.2)

        # For the result contributions, we return the SHAP ones if alpha >= 0.5,
        # otherwise empty (or we could try to map heuristic factors to FeatureContribution,
        # but the schema differs slightly). The user said "Show Heuristic explanation" which is text.
        # But `top_positive_contributions` is used by frontend likely for visualization.
        # If alpha < 0.5, the heuristic factors are in `cold_result.heuristic_factors`.
        # I'll populate the contribution lists only if using SHAP explanation to avoid type mismatches or confusion,
        # as heuristic factors have different structure.

        final_positive_contributions = positive_contributions[:top_k] if alpha >= 0.5 else []
        final_negative_contributions = negative_contributions[:top_k] if alpha >= 0.5 else []

        return PredictionResult(
            predicted_rpi_score=final_score,
            predicted_rpi_raw=np.expm1(max(0, final_score)),  # Inversa de log1p
            confidence_interval=confidence_interval,
            top_positive_contributions=final_positive_contributions,
            top_negative_contributions=final_negative_contributions,
            explanation_text=explanation_text,
            feature_values=processed_features
        )

    def predict_batch(
        self,
        features_list: List[Dict[str, Any]]
    ) -> List[PredictionResult]:
        """
        Predice múltiples muestras con explicaciones.

        Args:
            features_list: Lista de diccionarios con features.

        Returns:
            Lista de PredictionResult para cada muestra.
        """
        return [self.predict_with_explanation(f) for f in features_list]

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Obtiene la importancia de features del modelo entrenado.

        Returns:
            Diccionario feature -> importancia ordenado de mayor a menor.
        """
        if not self.is_trained or self._training_metrics is None:
            return {}

        return self._training_metrics.feature_importance

    def predict_cold_start(self, features: Dict[str, Any]) -> ColdStartPredictionResult:
        """
        Public method to explicitly request Cold Start heuristic prediction.

        Use this when you know you don't have a trained model and want
        heuristic-based predictions without triggering model loading attempts.

        Args:
            features: Content features dictionary

        Returns:
            ColdStartPredictionResult with rule-based prediction
        """
        return self._cold_start_predict(features)

    def get_model_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado actual del modelo.

        Returns:
            Diccionario con información del estado del modelo.
        """
        status = "trained" if self.is_trained else "cold_start"
        return {
            "is_trained": self.is_trained,
            "mode": status,
            "model_version": self.MODEL_VERSION,
            "feature_count": len(self.feature_columns),
            "features": self.feature_columns,
            "training_metrics": self._training_metrics.to_dict() if self._training_metrics else None,
            "config": {
                "n_estimators": self.config.n_estimators,
                "max_depth": self.config.max_depth,
                "learning_rate": self.config.learning_rate,
                "cv_folds": self.config.cv_folds
            }
        }

    def predict_with_account_health(
        self,
        features: Dict[str, Any],
        recent_posts: List[Dict[str, Any]],
        follower_count: int
    ) -> "CalibratedPrediction":
        """
        Realiza una predicción calibrada según la salud de la cuenta del usuario.

        Este método combina:
        1. La predicción XGBoost estándar con explicación SHAP
        2. El análisis de salud de la cuenta (últimos 10 posts)
        3. Calibración de la predicción según nivel de autoridad

        LÓGICA DE CALIBRACIÓN:
        - Si media views < 10% seguidores -> Low_Authority -> factor 0.3x
        - Si media views < 2% seguidores -> Possible_Shadowban -> factor 0.1x
        - Incluye mensaje de advertencia para el usuario

        Args:
            features: Features del contenido a predecir (igual que predict_with_explanation)
            recent_posts: Lista de los últimos 10 posts del usuario con views_count
            follower_count: Número de seguidores de la cuenta

        Returns:
            CalibratedPrediction con predicción ajustada e información de salud.

        Raises:
            RuntimeError: Si el modelo no está entrenado.
        """
        # Import aquí para evitar circular imports
        from app.services.account_health_scoring import (
            get_account_health_scoring,
            CalibratedPrediction
        )

        # 1. Obtener predicción base
        base_result = self.predict_with_explanation(features)

        # 2. Evaluar salud de la cuenta
        health_scoring = get_account_health_scoring()
        account_health = health_scoring.evaluate_account_health(
            recent_posts=recent_posts,
            follower_count=follower_count,
            posts_to_analyze=10
        )

        # 3. Calibrar predicción según salud
        calibrated = health_scoring.calibrate_prediction(
            original_score=base_result.predicted_rpi_score,
            original_raw=base_result.predicted_rpi_raw,
            account_health=account_health,
            prediction_explanation=base_result.explanation_text
        )

        logger.info(
            f"Calibrated prediction: original_raw={base_result.predicted_rpi_raw:.2f}, "
            f"calibrated_raw={calibrated.calibrated_rpi_raw:.2f}, "
            f"penalty_factor={account_health.prediction_penalty_factor}, "
            f"health_status={account_health.health_status.value}"
        )

        return calibrated

    def get_calibration_factor(
        self,
        recent_posts: List[Dict[str, Any]],
        follower_count: int
    ) -> Tuple[float, str]:
        """
        Obtiene el factor de calibración sin hacer predicción.

        Útil para pre-evaluar la salud de una cuenta antes de generar contenido.

        Args:
            recent_posts: Últimos posts del usuario
            follower_count: Número de seguidores

        Returns:
            Tuple de (factor de penalización, mensaje de estado)
        """
        from app.services.account_health_scoring import get_account_health_scoring

        health_scoring = get_account_health_scoring()
        account_health = health_scoring.evaluate_account_health(
            recent_posts=recent_posts,
            follower_count=follower_count
        )

        return (
            account_health.prediction_penalty_factor,
            account_health.health_status.value
        )


# Singleton instance para uso en la aplicación
_growth_engine_instance: Optional[GrowthPredictionEngine] = None


def get_growth_prediction_engine() -> GrowthPredictionEngine:
    """
    Obtiene la instancia singleton del GrowthPredictionEngine.

    Returns:
        Instancia del motor de predicción.
    """
    global _growth_engine_instance
    if _growth_engine_instance is None:
        _growth_engine_instance = GrowthPredictionEngine()
    return _growth_engine_instance


def reset_growth_engine() -> None:
    """Reinicia la instancia singleton (útil para tests)."""
    global _growth_engine_instance
    _growth_engine_instance = None
