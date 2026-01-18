"""
Multi-Output Engagement Predictor - XGBoost with MultiOutputRegressor

Predicts multiple engagement metrics simultaneously:
- log_likes, log_comments, log_shares, log_saves, log_views

Uses sklearn.multioutput.MultiOutputRegressor wrapping XGBRegressor
for independent prediction of each target with shared features.

Key Features:
- Multi-target prediction in single inference pass
- Configurable weights for RPI calculation
- Per-metric SHAP explanations
- Backward compatible with single-output (falls back to weighted sum)

Author: BrandPulse AI
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Directory for model persistence
MODEL_DIR = Path("./ml_models/multi_output")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Target columns (in order)
TARGET_COLUMNS = ["log_likes", "log_comments", "log_shares", "log_saves", "log_views"]
TARGET_NAMES = ["likes", "comments", "shares", "saves", "views"]

# Default weights (matching schema defaults)
DEFAULT_WEIGHTS = {
    "likes_weight": 1.0,
    "comments_weight": 2.0,
    "shares_weight": 10.0,
    "saves_weight": 5.0,
    "views_weight": 3.0,
}

# Minimum samples for multi-output training
MINIMUM_SAMPLES_MULTI_OUTPUT = 50


@dataclass
class MultiOutputConfig:
    """Configuration for multi-output predictor."""
    n_estimators: int = 150
    max_depth: int = 6
    learning_rate: float = 0.08
    min_child_weight: int = 3
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    gamma: float = 0.1
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class MultiOutputPredictionResult:
    """Result of multi-output prediction."""
    # Log-scale predictions
    log_likes: float
    log_comments: float
    log_shares: float
    log_saves: float
    log_views: float

    # Raw predictions (exp-transformed)
    predicted_likes: float
    predicted_comments: float
    predicted_shares: float
    predicted_saves: float
    predicted_views: float

    # Weighted RPI
    weighted_rpi: float
    weights_used: Dict[str, float]

    # Relative to baseline (optional)
    relative_to_baseline: Optional[float] = None

    # Per-metric SHAP explanations (top 5 factors each)
    shap_likes: Optional[Dict[str, float]] = None
    shap_comments: Optional[Dict[str, float]] = None
    shap_shares: Optional[Dict[str, float]] = None
    shap_saves: Optional[Dict[str, float]] = None
    shap_views: Optional[Dict[str, float]] = None

    # Confidence intervals
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "log_likes": round(self.log_likes, 4),
            "log_comments": round(self.log_comments, 4),
            "log_shares": round(self.log_shares, 4),
            "log_saves": round(self.log_saves, 4),
            "log_views": round(self.log_views, 4),
            "predicted_likes": round(self.predicted_likes, 2),
            "predicted_comments": round(self.predicted_comments, 2),
            "predicted_shares": round(self.predicted_shares, 2),
            "predicted_saves": round(self.predicted_saves, 2),
            "predicted_views": round(self.predicted_views, 2),
            "weighted_rpi": round(self.weighted_rpi, 2),
            "weights_used": self.weights_used,
            "relative_to_baseline": self.relative_to_baseline,
            "shap_likes": self.shap_likes,
            "shap_comments": self.shap_comments,
            "shap_shares": self.shap_shares,
            "shap_saves": self.shap_saves,
            "shap_views": self.shap_views,
        }


@dataclass
class MultiOutputTrainingMetrics:
    """Training metrics for multi-output model."""
    # Per-metric metrics
    likes_rmse: float
    likes_r2: float
    comments_rmse: float
    comments_r2: float
    shares_rmse: float
    shares_r2: float
    saves_rmse: float
    saves_r2: float
    views_rmse: float
    views_r2: float

    # Combined metrics
    combined_rmse: float
    combined_r2: float

    # Training info
    training_samples: int
    feature_count: int
    training_date: str
    model_version: str
    is_multi_output: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "likes": {"rmse": self.likes_rmse, "r2": self.likes_r2},
            "comments": {"rmse": self.comments_rmse, "r2": self.comments_r2},
            "shares": {"rmse": self.shares_rmse, "r2": self.shares_r2},
            "saves": {"rmse": self.saves_rmse, "r2": self.saves_r2},
            "views": {"rmse": self.views_rmse, "r2": self.views_r2},
            "combined": {"rmse": self.combined_rmse, "r2": self.combined_r2},
            "training_samples": self.training_samples,
            "feature_count": self.feature_count,
            "training_date": self.training_date,
            "model_version": self.model_version,
            "is_multi_output": self.is_multi_output,
        }


class MultiOutputEngagementPredictor:
    """
    Multi-output engagement predictor using MultiOutputRegressor(XGBRegressor).

    Predicts [log_likes, log_comments, log_shares, log_saves, log_views]
    simultaneously from shared features.

    Usage:
        predictor = MultiOutputEngagementPredictor()

        # Train
        metrics = predictor.train(X, y_multi)

        # Predict with custom weights
        weights = {"likes_weight": 1.0, "comments_weight": 2.0, ...}
        result = predictor.predict(features, weights)
        print(f"Weighted RPI: {result.weighted_rpi}")
    """

    MODEL_VERSION = "multi-output-v1.0"
    MODEL_FILENAME = "multi_output_engagement.joblib"
    SCALER_FILENAME = "multi_output_scaler.joblib"
    METADATA_FILENAME = "multi_output_metadata.joblib"

    def __init__(self, config: Optional[MultiOutputConfig] = None, niche: str = "general"):
        """
        Initialize multi-output predictor.

        Args:
            config: Model configuration
            niche: Business niche for niche-specific models
        """
        self.config = config or MultiOutputConfig()
        self.niche = niche
        self._model: Optional[MultiOutputRegressor] = None
        self._scaler: Optional[StandardScaler] = None
        self._shap_explainers: Dict[str, Any] = {}
        self._is_trained = False
        self._training_metrics: Optional[MultiOutputTrainingMetrics] = None
        self._feature_columns: List[str] = []

        # Try to load existing model
        self._load_model()

    @property
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self._is_trained and self._model is not None

    def _get_model_path(self, filename: str) -> Path:
        """Get path for model file."""
        niche_dir = MODEL_DIR / self.niche
        niche_dir.mkdir(parents=True, exist_ok=True)
        return niche_dir / filename

    def _load_model(self) -> bool:
        """Load existing model from disk."""
        model_path = self._get_model_path(self.MODEL_FILENAME)
        scaler_path = self._get_model_path(self.SCALER_FILENAME)
        metadata_path = self._get_model_path(self.METADATA_FILENAME)

        if not all(p.exists() for p in [model_path, scaler_path, metadata_path]):
            logger.info(f"No multi-output model found for niche={self.niche}")
            return False

        try:
            self._model = joblib.load(model_path)
            self._scaler = joblib.load(scaler_path)
            metadata = joblib.load(metadata_path)

            self._feature_columns = metadata.get("feature_columns", [])
            self._training_metrics = metadata.get("training_metrics")
            self._is_trained = True

            # Initialize SHAP explainers
            self._init_shap_explainers()

            logger.info(f"Multi-output model loaded for niche={self.niche}")
            return True

        except Exception as e:
            logger.error(f"Error loading multi-output model: {e}")
            return False

    def _save_model(self) -> bool:
        """Save model to disk."""
        if not self._is_trained:
            return False

        try:
            model_path = self._get_model_path(self.MODEL_FILENAME)
            scaler_path = self._get_model_path(self.SCALER_FILENAME)
            metadata_path = self._get_model_path(self.METADATA_FILENAME)

            joblib.dump(self._model, model_path)
            joblib.dump(self._scaler, scaler_path)

            metadata = {
                "feature_columns": self._feature_columns,
                "training_metrics": self._training_metrics,
                "model_version": self.MODEL_VERSION,
                "niche": self.niche,
                "saved_at": datetime.now().isoformat(),
            }
            joblib.dump(metadata, metadata_path)

            logger.info(f"Multi-output model saved for niche={self.niche}")
            return True

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    def _init_shap_explainers(self) -> None:
        """Initialize SHAP explainers for each target."""
        if self._model is None:
            return

        try:
            import shap

            # MultiOutputRegressor stores individual estimators in estimators_
            for i, target_name in enumerate(TARGET_NAMES):
                estimator = self._model.estimators_[i]
                self._shap_explainers[target_name] = shap.TreeExplainer(estimator)

            logger.info("SHAP explainers initialized for all targets")

        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainers: {e}")

    def _create_base_estimator(self) -> xgb.XGBRegressor:
        """Create base XGBoost estimator."""
        return xgb.XGBRegressor(
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
            n_jobs=self.config.n_jobs,
            objective="reg:squarederror",
            verbosity=0,
        )

    def prepare_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare multi-output targets from raw engagement metrics.

        Applies log1p transformation to handle skewed distributions.

        Args:
            df: DataFrame with likes, comments, shares, saves, views columns

        Returns:
            DataFrame with log-transformed targets
        """
        targets = pd.DataFrame()

        # Required columns
        metrics = ["likes", "comments", "shares", "saves", "views"]

        for metric in metrics:
            col = metric
            if col not in df.columns:
                # Try alternative column names
                alternatives = [f"{metric}_count", f"num_{metric}", metric.upper()]
                for alt in alternatives:
                    if alt in df.columns:
                        col = alt
                        break

            if col in df.columns:
                # Log transform: log(x + 1) to handle zeros
                targets[f"log_{metric}"] = np.log1p(df[col].fillna(0).clip(lower=0))
            else:
                logger.warning(f"Missing target column: {metric}. Using zeros.")
                targets[f"log_{metric}"] = 0.0

        return targets

    def train(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        save_model: bool = True,
    ) -> MultiOutputTrainingMetrics:
        """
        Train multi-output model.

        Args:
            X: Features DataFrame
            y: Targets DataFrame with columns [log_likes, log_comments, log_shares, log_saves, log_views]
            feature_columns: Optional list of feature column names
            save_model: Whether to save after training

        Returns:
            Training metrics for all targets
        """
        logger.info(f"Training multi-output model for niche={self.niche}")

        # Validate data
        if len(X) < MINIMUM_SAMPLES_MULTI_OUTPUT:
            raise ValueError(
                f"Insufficient data: {len(X)} samples. "
                f"Multi-output requires at least {MINIMUM_SAMPLES_MULTI_OUTPUT}."
            )

        # Ensure targets have correct columns
        for col in TARGET_COLUMNS:
            if col not in y.columns:
                raise ValueError(f"Missing target column: {col}")

        # Store feature columns
        self._feature_columns = feature_columns or list(X.columns)

        # Prepare data
        X_arr = X[self._feature_columns].values.astype(np.float32)
        y_arr = y[TARGET_COLUMNS].values.astype(np.float32)

        # Replace NaN/inf
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
        y_arr = np.nan_to_num(y_arr, nan=0.0, posinf=10.0, neginf=0.0)

        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_arr)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_arr,
            test_size=0.2,
            random_state=self.config.random_state
        )

        logger.info(f"Training with {len(X_train)} samples, {X_arr.shape[1]} features, 5 targets")

        # Create multi-output model
        base_estimator = self._create_base_estimator()
        self._model = MultiOutputRegressor(base_estimator, n_jobs=1)

        # Train
        self._model.fit(X_train, y_train)

        # Evaluate
        y_pred = self._model.predict(X_test)

        # Calculate per-target metrics
        metrics_dict = {}
        rmse_values = []
        r2_values = []

        for i, target in enumerate(TARGET_NAMES):
            rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
            r2 = r2_score(y_test[:, i], y_pred[:, i])
            metrics_dict[f"{target}_rmse"] = rmse
            metrics_dict[f"{target}_r2"] = r2
            rmse_values.append(rmse)
            r2_values.append(r2)

            logger.info(f"  {target}: RMSE={rmse:.4f}, R2={r2:.4f}")

        # Combined metrics (average)
        combined_rmse = np.mean(rmse_values)
        combined_r2 = np.mean(r2_values)

        logger.info(f"Combined: RMSE={combined_rmse:.4f}, R2={combined_r2:.4f}")

        self._training_metrics = MultiOutputTrainingMetrics(
            likes_rmse=metrics_dict["likes_rmse"],
            likes_r2=metrics_dict["likes_r2"],
            comments_rmse=metrics_dict["comments_rmse"],
            comments_r2=metrics_dict["comments_r2"],
            shares_rmse=metrics_dict["shares_rmse"],
            shares_r2=metrics_dict["shares_r2"],
            saves_rmse=metrics_dict["saves_rmse"],
            saves_r2=metrics_dict["saves_r2"],
            views_rmse=metrics_dict["views_rmse"],
            views_r2=metrics_dict["views_r2"],
            combined_rmse=combined_rmse,
            combined_r2=combined_r2,
            training_samples=len(X),
            feature_count=X_arr.shape[1],
            training_date=datetime.now().isoformat(),
            model_version=self.MODEL_VERSION,
        )

        self._is_trained = True

        # Initialize SHAP
        self._init_shap_explainers()

        # Save
        if save_model:
            self._save_model()

        return self._training_metrics

    def predict(
        self,
        features: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
        author_baseline: Optional[Dict[str, float]] = None,
        compute_shap: bool = True,
    ) -> MultiOutputPredictionResult:
        """
        Predict engagement with configurable weights.

        Args:
            features: Feature dictionary
            weights: Custom weights (defaults to DEFAULT_WEIGHTS if None)
            author_baseline: Optional author average metrics for relative scoring
            compute_shap: Whether to compute SHAP explanations

        Returns:
            MultiOutputPredictionResult with all metrics and weighted RPI
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call train() first.")

        # Use default weights if not provided
        weights = weights or DEFAULT_WEIGHTS.copy()

        # Prepare features
        X = self._prepare_features(features)

        # Predict
        y_pred = self._model.predict(X)[0]

        # Extract predictions
        log_likes = float(y_pred[0])
        log_comments = float(y_pred[1])
        log_shares = float(y_pred[2])
        log_saves = float(y_pred[3])
        log_views = float(y_pred[4])

        # Convert to raw values (exp transform)
        pred_likes = float(np.expm1(max(0, log_likes)))
        pred_comments = float(np.expm1(max(0, log_comments)))
        pred_shares = float(np.expm1(max(0, log_shares)))
        pred_saves = float(np.expm1(max(0, log_saves)))
        pred_views = float(np.expm1(max(0, log_views)))

        # Calculate weighted RPI
        weighted_rpi = self._calculate_weighted_rpi(
            log_likes, log_comments, log_shares, log_saves, log_views,
            weights
        )

        # Calculate relative to baseline if provided
        relative = None
        if author_baseline:
            relative = self._calculate_relative_score(
                pred_likes, pred_comments, pred_shares, pred_saves, pred_views,
                author_baseline
            )

        # SHAP explanations
        shap_explanations = {}
        if compute_shap and self._shap_explainers:
            shap_explanations = self._compute_shap_explanations(X)

        return MultiOutputPredictionResult(
            log_likes=log_likes,
            log_comments=log_comments,
            log_shares=log_shares,
            log_saves=log_saves,
            log_views=log_views,
            predicted_likes=pred_likes,
            predicted_comments=pred_comments,
            predicted_shares=pred_shares,
            predicted_saves=pred_saves,
            predicted_views=pred_views,
            weighted_rpi=weighted_rpi,
            weights_used=weights,
            relative_to_baseline=relative,
            shap_likes=shap_explanations.get("likes"),
            shap_comments=shap_explanations.get("comments"),
            shap_shares=shap_explanations.get("shares"),
            shap_saves=shap_explanations.get("saves"),
            shap_views=shap_explanations.get("views"),
        )

    def _prepare_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare features for prediction."""
        # Create DataFrame with single row
        df = pd.DataFrame([features])

        # Ensure all expected columns exist
        for col in self._feature_columns:
            if col not in df.columns:
                df[col] = 0.0

        # Select and order columns
        X = df[self._feature_columns].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale
        X_scaled = self._scaler.transform(X)

        return X_scaled

    def _calculate_weighted_rpi(
        self,
        log_likes: float,
        log_comments: float,
        log_shares: float,
        log_saves: float,
        log_views: float,
        weights: Dict[str, float],
    ) -> float:
        """
        Calculate weighted RPI from log predictions.

        RPI = weighted_sum / total_weight * normalization_factor

        Normalized to 0-100 scale.
        """
        pred_vector = np.array([log_likes, log_comments, log_shares, log_saves, log_views])
        weight_vector = np.array([
            weights.get("likes_weight", 1.0),
            weights.get("comments_weight", 2.0),
            weights.get("shares_weight", 10.0),
            weights.get("saves_weight", 5.0),
            weights.get("views_weight", 3.0),
        ])

        # Weighted sum
        weighted_sum = np.dot(pred_vector, weight_vector)
        total_weight = np.sum(weight_vector)

        # Normalize: typical log values range 0-10, we want 0-100
        # Scale factor of 5 maps avg log value of ~4 to ~20 base, then weights scale up
        rpi = (weighted_sum / total_weight) * 15.0

        # Clip to 0-100
        return float(np.clip(rpi, 0, 100))

    def _calculate_relative_score(
        self,
        pred_likes: float,
        pred_comments: float,
        pred_shares: float,
        pred_saves: float,
        pred_views: float,
        baseline: Dict[str, float],
    ) -> float:
        """
        Calculate performance relative to author baseline.

        Returns ratio where 1.0 = average, >1.0 = above average.
        """
        comparisons = []

        if baseline.get("avg_likes", 0) > 0:
            comparisons.append(pred_likes / baseline["avg_likes"])
        if baseline.get("avg_comments", 0) > 0:
            comparisons.append(pred_comments / baseline["avg_comments"])
        if baseline.get("avg_shares", 0) > 0:
            comparisons.append(pred_shares / baseline["avg_shares"])
        if baseline.get("avg_saves", 0) > 0:
            comparisons.append(pred_saves / baseline["avg_saves"])
        if baseline.get("avg_views", 0) > 0:
            comparisons.append(pred_views / baseline["avg_views"])

        if not comparisons:
            return None

        return float(np.mean(comparisons))

    def _compute_shap_explanations(
        self,
        X: np.ndarray,
        top_k: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """Compute SHAP explanations for each target."""
        explanations = {}

        for target_name, explainer in self._shap_explainers.items():
            try:
                shap_values = explainer.shap_values(X)
                shap_vector = shap_values[0] if len(shap_values.shape) > 1 else shap_values

                # Create feature -> contribution mapping
                contributions = {}
                for i, (feature, value) in enumerate(zip(self._feature_columns, shap_vector)):
                    contributions[feature] = float(value)

                # Sort by absolute value and take top k
                sorted_contrib = sorted(
                    contributions.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:top_k]

                explanations[target_name] = dict(sorted_contrib)

            except Exception as e:
                logger.warning(f"SHAP computation failed for {target_name}: {e}")
                explanations[target_name] = {}

        return explanations

    def predict_batch(
        self,
        features_list: List[Dict[str, Any]],
        weights: Optional[Dict[str, float]] = None,
    ) -> List[MultiOutputPredictionResult]:
        """Predict for multiple samples."""
        return [
            self.predict(features, weights, compute_shap=False)
            for features in features_list
        ]

    def get_model_status(self) -> Dict[str, Any]:
        """Get model status info."""
        return {
            "is_trained": self.is_trained,
            "niche": self.niche,
            "model_version": self.MODEL_VERSION,
            "feature_count": len(self._feature_columns),
            "targets": TARGET_NAMES,
            "is_multi_output": True,
            "training_metrics": self._training_metrics.to_dict() if self._training_metrics else None,
        }


# =============================================================================
# Singleton Management
# =============================================================================

_predictor_instances: Dict[str, MultiOutputEngagementPredictor] = {}


def get_multi_output_predictor(niche: str = "general") -> MultiOutputEngagementPredictor:
    """
    Get or create multi-output predictor for a niche.

    Args:
        niche: Business niche

    Returns:
        MultiOutputEngagementPredictor instance
    """
    global _predictor_instances

    if niche not in _predictor_instances:
        _predictor_instances[niche] = MultiOutputEngagementPredictor(niche=niche)

    return _predictor_instances[niche]


def reset_predictor(niche: str = None) -> None:
    """Reset predictor instance(s)."""
    global _predictor_instances

    if niche:
        _predictor_instances.pop(niche, None)
    else:
        _predictor_instances.clear()
