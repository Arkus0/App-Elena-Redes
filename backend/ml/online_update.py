"""
Online/Incremental Learning Module for BrandPulse AI
=====================================================

Implements real-time model updates using River for live learning from
A/B test feedback without full batch retraining.

Architecture:
- Uses River's AdaptiveRandomForestRegressor as lightweight proxy tree model
- Per-niche online models stored in /models/online_{niche}.pkl
- Supports multi-output prediction (likes, comments, shares, saves, views)
- Automatic performance tracking with MAE/R2 evaluation
- Triggers full XGBoost retrain when significant improvement detected

Key Features:
- Incremental updates: learns from each feedback sample in real-time
- Multi-output support: separate River model per target metric
- Drift detection: flags when online model outperforms batch model
- Cold start friendly: works with as few as 1 sample
- Memory efficient: ~10KB per niche model

Usage:
    from backend.ml.online_update import online_update, get_online_predictor

    # Update with new feedback
    result = online_update(
        niche="inmobiliaria",
        new_features=feature_df,
        new_targets=target_array
    )

    # Get online predictor for inference fallback
    predictor = get_online_predictor("inmobiliaria")
    prediction = predictor.predict(features)

Author: BrandPulse AI
"""

import logging
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Model storage directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Import evaluation module for periodic evaluation
try:
    from backend.ml.evaluate_model import (
        should_evaluate_online,
        get_baseline_mae,
        compute_drift_score,
        save_evaluation_results,
        EvaluationResult,
        evaluate_global_metrics,
        generate_insights,
        ONLINE_EVAL_INTERVAL,
    )
    EVALUATION_MODULE_AVAILABLE = True
except ImportError:
    EVALUATION_MODULE_AVAILABLE = False
    ONLINE_EVAL_INTERVAL = 50

# Online model settings
EVALUATION_INTERVAL = 15  # Evaluate every N samples (internal River metrics)
IMPROVEMENT_THRESHOLD = 0.05  # 5% improvement triggers full retrain signal
MAX_SAMPLES_HISTORY = 1000  # Rolling window for metrics tracking
MIN_SAMPLES_FOR_EVALUATION = 10  # Minimum samples before evaluation

# Target metrics (multi-output)
TARGET_NAMES = ["log_likes", "log_comments", "log_shares", "log_saves", "log_views"]

# River availability flag
try:
    from river import ensemble, tree, metrics, preprocessing, compose, forest
    RIVER_AVAILABLE = True
    logger.info("River online learning library loaded successfully")
except ImportError:
    RIVER_AVAILABLE = False
    logger.warning(
        "River not installed. Online learning disabled. "
        "Install with: pip install river"
    )


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class OnlineUpdateResult:
    """Result of an online update operation."""
    niche: str
    samples_processed: int
    total_samples: int
    current_mae: Optional[float] = None
    current_r2: Optional[float] = None
    previous_mae: Optional[float] = None
    previous_r2: Optional[float] = None
    improvement_detected: bool = False
    improvement_percent: float = 0.0
    trigger_full_retrain: bool = False
    update_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "niche": self.niche,
            "samples_processed": self.samples_processed,
            "total_samples": self.total_samples,
            "current_mae": round(self.current_mae, 4) if self.current_mae else None,
            "current_r2": round(self.current_r2, 4) if self.current_r2 else None,
            "previous_mae": round(self.previous_mae, 4) if self.previous_mae else None,
            "previous_r2": round(self.previous_r2, 4) if self.previous_r2 else None,
            "improvement_detected": self.improvement_detected,
            "improvement_percent": round(self.improvement_percent, 2),
            "trigger_full_retrain": self.trigger_full_retrain,
            "update_time_ms": round(self.update_time_ms, 2),
            "timestamp": self.timestamp,
        }


@dataclass
class OnlineModelMetrics:
    """Tracking metrics for online model performance."""
    samples_seen: int = 0
    mae_history: List[float] = field(default_factory=list)
    r2_history: List[float] = field(default_factory=list)
    last_evaluation_sample: int = 0
    last_mae: Optional[float] = None
    last_r2: Optional[float] = None
    best_mae: Optional[float] = None
    best_r2: Optional[float] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def update(self, mae: float, r2: float):
        """Update metrics with new evaluation."""
        self.last_mae = mae
        self.last_r2 = r2
        self.mae_history.append(mae)
        self.r2_history.append(r2)
        self.updated_at = datetime.utcnow().isoformat()

        # Track best
        if self.best_mae is None or mae < self.best_mae:
            self.best_mae = mae
        if self.best_r2 is None or r2 > self.best_r2:
            self.best_r2 = r2

        # Limit history size
        if len(self.mae_history) > MAX_SAMPLES_HISTORY:
            self.mae_history = self.mae_history[-MAX_SAMPLES_HISTORY:]
            self.r2_history = self.r2_history[-MAX_SAMPLES_HISTORY:]


# =============================================================================
# Online Model Wrapper
# =============================================================================

class OnlineEngagementPredictor:
    """
    Online/incremental engagement predictor using River.

    Uses AdaptiveRandomForestRegressor for each target metric,
    providing real-time learning from feedback data.

    Features:
    - Multi-output: separate model per metric (likes, comments, etc.)
    - Adaptive: automatically adjusts to concept drift
    - Lightweight: minimal memory footprint
    - Persistent: saves/loads from disk per niche
    """

    MODEL_VERSION = "river-online-v1.0"

    def __init__(self, niche: str = "general"):
        """
        Initialize online predictor for a specific niche.

        Args:
            niche: Business niche identifier
        """
        self.niche = niche
        self._models: Dict[str, Any] = {}  # target -> River model
        self._scalers: Dict[str, Any] = {}  # target -> River scaler
        self._metrics: OnlineModelMetrics = OnlineModelMetrics()
        self._feature_names: List[str] = []
        self._is_initialized = False

        # River metrics for tracking
        self._river_metrics: Dict[str, Any] = {}

        # Try to load existing model
        self._load_model()

    def _get_model_path(self) -> Path:
        """Get path for online model file."""
        return MODELS_DIR / f"online_{self.niche}.pkl"

    def _create_model(self) -> Any:
        """
        Create a new River online model.

        Uses AdaptiveRandomForestRegressor which:
        - Handles concept drift automatically
        - Works well with few samples
        - Is lightweight (~10KB per model)
        """
        if not RIVER_AVAILABLE:
            return None

        # Create adaptive random forest regressor
        # This is a lightweight tree-based model suitable for online learning
        model = compose.Pipeline(
            ("scale", preprocessing.StandardScaler()),
            ("model", forest.ARFRegressor(
                n_models=10,  # Ensemble of 10 trees
                max_depth=6,  # Similar to XGBoost depth
                lambda_value=6,  # Poisson lambda for bootstrap
                grace_period=50,  # Samples before first split
                delta=0.01,  # Renamed from split_confidence
                seed=42
            ))
        )

        return model

    def _initialize_models(self, feature_names: List[str]):
        """Initialize models for all targets."""
        if not RIVER_AVAILABLE:
            logger.warning("River not available, cannot initialize online models")
            return

        self._feature_names = feature_names

        for target in TARGET_NAMES:
            self._models[target] = self._create_model()
            self._river_metrics[target] = {
                "mae": metrics.MAE(),
                "r2": metrics.R2()
            }

        self._is_initialized = True
        logger.info(f"Initialized online models for niche={self.niche}, targets={TARGET_NAMES}")

    def _load_model(self) -> bool:
        """Load existing online model from disk."""
        model_path = self._get_model_path()

        if not model_path.exists():
            logger.info(f"No online model found for niche={self.niche}")
            return False

        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)

            self._models = data.get("models", {})
            self._metrics = data.get("metrics", OnlineModelMetrics())
            self._feature_names = data.get("feature_names", [])
            self._river_metrics = data.get("river_metrics", {})
            self._is_initialized = bool(self._models)

            logger.info(
                f"Loaded online model for niche={self.niche}, "
                f"samples_seen={self._metrics.samples_seen}"
            )
            return True

        except Exception as e:
            logger.error(f"Error loading online model: {e}")
            return False

    def _save_model(self) -> bool:
        """Save online model to disk."""
        model_path = self._get_model_path()

        try:
            data = {
                "models": self._models,
                "metrics": self._metrics,
                "feature_names": self._feature_names,
                "river_metrics": self._river_metrics,
                "version": self.MODEL_VERSION,
                "niche": self.niche,
                "saved_at": datetime.utcnow().isoformat(),
            }

            with open(model_path, 'wb') as f:
                pickle.dump(data, f)

            logger.debug(f"Saved online model for niche={self.niche}")
            return True

        except Exception as e:
            logger.error(f"Error saving online model: {e}")
            return False

    def partial_fit(
        self,
        features: Dict[str, float],
        targets: Dict[str, float]
    ) -> None:
        """
        Perform incremental update with a single sample.

        This is the core online learning method - learns from one
        observation at a time without storing historical data.

        Args:
            features: Feature dict with embedding and heuristic features
            targets: Target dict with log_likes, log_comments, etc.
        """
        if not RIVER_AVAILABLE:
            return

        # Initialize if needed
        if not self._is_initialized:
            self._initialize_models(list(features.keys()))

        # Update each target model
        for target_name in TARGET_NAMES:
            if target_name not in targets:
                continue

            model = self._models.get(target_name)
            if model is None:
                continue

            y = targets[target_name]

            # Update metrics first (for evaluation)
            if target_name in self._river_metrics:
                try:
                    y_pred = model.predict_one(features)
                    if y_pred is not None:
                        self._river_metrics[target_name]["mae"].update(y, y_pred)
                        self._river_metrics[target_name]["r2"].update(y, y_pred)
                except:
                    pass  # Model not ready for prediction yet

            # Learn from this sample
            model.learn_one(features, y)

        self._metrics.samples_seen += 1
        self._metrics.updated_at = datetime.utcnow().isoformat()

    def partial_fit_batch(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame
    ) -> int:
        """
        Perform incremental update with multiple samples.

        Processes samples one at a time (true online learning).

        Args:
            features_df: DataFrame with feature columns
            targets_df: DataFrame with target columns

        Returns:
            Number of samples processed
        """
        if not RIVER_AVAILABLE:
            return 0

        processed = 0

        for idx in range(len(features_df)):
            features = features_df.iloc[idx].to_dict()
            targets = targets_df.iloc[idx].to_dict()

            self.partial_fit(features, targets)
            processed += 1

        return processed

    def predict_one(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Predict engagement metrics for a single sample.

        Args:
            features: Feature dictionary

        Returns:
            Dict with predicted log values for each target
        """
        if not RIVER_AVAILABLE or not self._is_initialized:
            return {t: 0.0 for t in TARGET_NAMES}

        predictions = {}

        for target_name in TARGET_NAMES:
            model = self._models.get(target_name)
            if model is None:
                predictions[target_name] = 0.0
                continue

            try:
                pred = model.predict_one(features)
                predictions[target_name] = float(pred) if pred is not None else 0.0
            except Exception as e:
                logger.debug(f"Prediction failed for {target_name}: {e}")
                predictions[target_name] = 0.0

        return predictions

    def predict_batch(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict engagement metrics for multiple samples.

        Args:
            features_df: DataFrame with feature columns

        Returns:
            DataFrame with predicted values
        """
        predictions = []

        for idx in range(len(features_df)):
            features = features_df.iloc[idx].to_dict()
            pred = self.predict_one(features)
            predictions.append(pred)

        return pd.DataFrame(predictions)

    def get_current_metrics(self) -> Dict[str, float]:
        """Get current performance metrics from River."""
        if not RIVER_AVAILABLE:
            return {"mae": None, "r2": None}

        mae_values = []
        r2_values = []

        for target_name in TARGET_NAMES:
            if target_name in self._river_metrics:
                try:
                    mae = self._river_metrics[target_name]["mae"].get()
                    r2 = self._river_metrics[target_name]["r2"].get()
                    if mae is not None:
                        mae_values.append(mae)
                    if r2 is not None:
                        r2_values.append(r2)
                except:
                    pass

        return {
            "mae": np.mean(mae_values) if mae_values else None,
            "r2": np.mean(r2_values) if r2_values else None,
        }

    def evaluate_improvement(self) -> Tuple[bool, float]:
        """
        Evaluate if model has improved significantly.

        Returns:
            Tuple of (improvement_detected, improvement_percent)
        """
        current = self.get_current_metrics()

        if current["mae"] is None:
            return False, 0.0

        # Compare with previous evaluation
        if self._metrics.last_mae is None:
            # First evaluation
            self._metrics.update(current["mae"], current["r2"] or 0.0)
            return False, 0.0

        # Calculate improvement (lower MAE = better)
        mae_improvement = (self._metrics.last_mae - current["mae"]) / self._metrics.last_mae

        # Update metrics
        self._metrics.update(current["mae"], current["r2"] or 0.0)

        improvement_detected = mae_improvement > IMPROVEMENT_THRESHOLD

        return improvement_detected, mae_improvement * 100

    def get_status(self) -> Dict[str, Any]:
        """Get status of the online model."""
        current = self.get_current_metrics()

        return {
            "niche": self.niche,
            "is_initialized": self._is_initialized,
            "samples_seen": self._metrics.samples_seen,
            "feature_count": len(self._feature_names),
            "targets": TARGET_NAMES,
            "current_mae": current["mae"],
            "current_r2": current["r2"],
            "best_mae": self._metrics.best_mae,
            "best_r2": self._metrics.best_r2,
            "created_at": self._metrics.created_at,
            "updated_at": self._metrics.updated_at,
            "model_version": self.MODEL_VERSION,
        }


# =============================================================================
# Global Predictor Registry
# =============================================================================

_online_predictors: Dict[str, OnlineEngagementPredictor] = {}


def get_online_predictor(niche: str = "general") -> OnlineEngagementPredictor:
    """
    Get or create online predictor for a niche.

    Args:
        niche: Business niche identifier

    Returns:
        OnlineEngagementPredictor instance
    """
    global _online_predictors

    if niche not in _online_predictors:
        _online_predictors[niche] = OnlineEngagementPredictor(niche=niche)

    return _online_predictors[niche]


def reset_online_predictor(niche: Optional[str] = None) -> None:
    """
    Reset online predictor(s).

    Args:
        niche: Specific niche to reset, or None to reset all
    """
    global _online_predictors

    if niche:
        _online_predictors.pop(niche, None)
    else:
        _online_predictors.clear()


# =============================================================================
# Main Online Update Function
# =============================================================================

def online_update(
    niche: str,
    new_features: pd.DataFrame,
    new_targets: Union[np.ndarray, pd.DataFrame],
    save_model: bool = True
) -> OnlineUpdateResult:
    """
    Perform incremental online update with new feedback data.

    This is the main entry point for online learning. Call this whenever
    new actual engagement data is available from A/B testing.

    Flow:
    1. Load or initialize online model for niche
    2. Perform partial_fit with new samples
    3. Every EVALUATION_INTERVAL samples, evaluate MAE/R2
    4. If improvement > IMPROVEMENT_THRESHOLD, signal full retrain
    5. Save model and return results

    Args:
        niche: Business niche identifier (e.g., "inmobiliaria", "floristeria")
        new_features: DataFrame with feature columns (embeddings + heuristics)
                     Must include columns matching the feature extractor output
        new_targets: Multi-output targets as array of shape (n, 5) or DataFrame
                    with columns [log_likes, log_comments, log_shares, log_saves, log_views]
        save_model: Whether to persist model after update (default True)

    Returns:
        OnlineUpdateResult with update statistics and improvement signals

    Example:
        # After receiving A/B test results
        from backend.ml.online_update import online_update

        # Prepare features (from FeatureExtractor)
        features_df = pd.DataFrame([feature_dict])

        # Prepare targets (log-transformed actual metrics)
        targets = np.array([[np.log1p(likes), np.log1p(comments), ...]])

        # Update
        result = online_update("inmobiliaria", features_df, targets)

        if result.trigger_full_retrain:
            # Online model improved significantly, trigger batch retrain
            trigger_xgboost_retrain(niche)
    """
    start_time = time.time()

    # Validate River availability
    if not RIVER_AVAILABLE:
        logger.warning("River not available, online update skipped")
        return OnlineUpdateResult(
            niche=niche,
            samples_processed=0,
            total_samples=0,
            update_time_ms=0.0
        )

    # Validate inputs
    if new_features is None or len(new_features) == 0:
        logger.warning("No features provided for online update")
        return OnlineUpdateResult(
            niche=niche,
            samples_processed=0,
            total_samples=0,
            update_time_ms=0.0
        )

    # Convert targets to DataFrame if needed
    if isinstance(new_targets, np.ndarray):
        if new_targets.ndim == 1:
            new_targets = new_targets.reshape(1, -1)
        targets_df = pd.DataFrame(new_targets, columns=TARGET_NAMES[:new_targets.shape[1]])
    else:
        targets_df = new_targets

    # Ensure target columns exist
    for col in TARGET_NAMES:
        if col not in targets_df.columns:
            targets_df[col] = 0.0

    # Get predictor
    predictor = get_online_predictor(niche)

    # Store previous metrics for comparison
    previous_mae = predictor._metrics.last_mae
    previous_r2 = predictor._metrics.last_r2

    # Perform incremental update
    samples_processed = predictor.partial_fit_batch(new_features, targets_df)

    # Check if evaluation is due
    samples_since_eval = predictor._metrics.samples_seen - predictor._metrics.last_evaluation_sample
    should_evaluate = (
        samples_since_eval >= EVALUATION_INTERVAL and
        predictor._metrics.samples_seen >= MIN_SAMPLES_FOR_EVALUATION
    )

    improvement_detected = False
    improvement_percent = 0.0
    trigger_retrain = False

    if should_evaluate:
        improvement_detected, improvement_percent = predictor.evaluate_improvement()
        predictor._metrics.last_evaluation_sample = predictor._metrics.samples_seen

        if improvement_detected:
            logger.info(
                f"Online model improvement detected for niche={niche}: "
                f"{improvement_percent:.1f}% MAE improvement"
            )
            trigger_retrain = True

    # Save model
    if save_model:
        predictor._save_model()

    # Calculate update time
    update_time_ms = (time.time() - start_time) * 1000

    # Get current metrics
    current = predictor.get_current_metrics()

    # Granular evaluation every ONLINE_EVAL_INTERVAL (50) samples
    granular_eval = None
    if EVALUATION_MODULE_AVAILABLE and should_evaluate_online(niche, samples_processed):
        try:
            logger.info(f"Running granular evaluation for niche={niche} (every {ONLINE_EVAL_INTERVAL} samples)")

            # Get baseline MAE for drift comparison
            baseline_mae = get_baseline_mae(niche, lookback=10)

            # Compute predictions for evaluation
            predictions_df = predictor.predict_batch(new_features)

            # Compute drift score
            drift_result = compute_drift_score(
                y_true=targets_df.values,
                y_pred=predictions_df.values,
                baseline_mae=baseline_mae
            )

            # Compute global metrics
            global_metrics = evaluate_global_metrics(
                targets_df.values,
                predictions_df.values,
                TARGET_NAMES[:min(targets_df.shape[1], predictions_df.shape[1])]
            )

            # Generate insights
            insights = generate_insights(
                global_metrics=global_metrics,
                format_metrics={},
                time_metrics={},
                drift=drift_result,
                calibration=None
            )

            # Build evaluation result
            granular_eval = EvaluationResult(
                niche=niche,
                timestamp=datetime.utcnow().isoformat(),
                global_metrics=global_metrics,
                format_metrics={},
                time_metrics={},
                rpi_metrics={},
                drift=drift_result,
                insights=insights,
                n_samples=samples_processed,
                evaluation_type="online"
            )

            # Save evaluation results
            save_evaluation_results(granular_eval)

            # Log insights
            for insight in insights:
                if insight.startswith("ALERTA"):
                    logger.warning(f"[ONLINE EVAL] {insight}")
                else:
                    logger.info(f"[ONLINE EVAL] {insight}")

            # Update trigger_retrain if drift detected
            if drift_result.drift_detected:
                trigger_retrain = True
                logger.warning(
                    f"DRIFT DETECTED in online evaluation for niche={niche}: "
                    f"score={drift_result.drift_score:.3f}"
                )

        except Exception as e:
            logger.warning(f"Granular online evaluation failed: {e}")

    # Build result
    result = OnlineUpdateResult(
        niche=niche,
        samples_processed=samples_processed,
        total_samples=predictor._metrics.samples_seen,
        current_mae=current["mae"],
        current_r2=current["r2"],
        previous_mae=previous_mae,
        previous_r2=previous_r2,
        improvement_detected=improvement_detected,
        improvement_percent=improvement_percent,
        trigger_full_retrain=trigger_retrain,
        update_time_ms=update_time_ms
    )

    # Log result
    mae_str = f"{current['mae']:.4f}" if current['mae'] is not None else "N/A"
    logger.info(
        f"Online update complete: niche={niche}, samples={samples_processed}, "
        f"total={predictor._metrics.samples_seen}, MAE={mae_str}, "
        f"time={update_time_ms:.1f}ms"
    )

    return result


def online_update_single(
    niche: str,
    features: Dict[str, float],
    targets: Dict[str, float],
    save_model: bool = False
) -> OnlineUpdateResult:
    """
    Convenience function for single-sample online update.

    Useful for real-time updates when each feedback sample arrives individually.

    Args:
        niche: Business niche
        features: Single feature dictionary
        targets: Single target dictionary with log_likes, log_comments, etc.
        save_model: Whether to save after this single update (default False for efficiency)

    Returns:
        OnlineUpdateResult
    """
    features_df = pd.DataFrame([features])
    targets_df = pd.DataFrame([targets])

    return online_update(niche, features_df, targets_df, save_model=save_model)


# =============================================================================
# Drift Detection
# =============================================================================

def detect_drift(
    niche: str,
    recent_predictions: List[float],
    recent_actuals: List[float],
    threshold: float = 0.3
) -> Tuple[bool, float]:
    """
    Detect if there's significant drift between predictions and actuals.

    This helps decide when to switch from batch XGBoost to online model.

    Args:
        niche: Business niche
        recent_predictions: Recent batch model predictions
        recent_actuals: Corresponding actual values
        threshold: MAE threshold for drift detection (default 0.3 = 30% error)

    Returns:
        Tuple of (drift_detected, drift_magnitude)
    """
    if len(recent_predictions) < 5:
        return False, 0.0

    predictions = np.array(recent_predictions)
    actuals = np.array(recent_actuals)

    # Calculate recent MAE
    mae = np.mean(np.abs(predictions - actuals))

    # Normalize by actual values range
    actual_range = np.ptp(actuals) if np.ptp(actuals) > 0 else 1.0
    normalized_mae = mae / actual_range

    drift_detected = normalized_mae > threshold

    if drift_detected:
        logger.warning(
            f"Drift detected for niche={niche}: normalized_MAE={normalized_mae:.3f} > {threshold}"
        )

    return drift_detected, normalized_mae


# =============================================================================
# Model Comparison
# =============================================================================

def compare_online_vs_batch(
    niche: str,
    test_features: pd.DataFrame,
    test_targets: pd.DataFrame,
    batch_model: Any = None
) -> Dict[str, Any]:
    """
    Compare online model performance vs batch XGBoost model.

    Useful for deciding which model to use for inference.

    Args:
        niche: Business niche
        test_features: Test feature DataFrame
        test_targets: Test target DataFrame
        batch_model: Optional batch XGBoost model for comparison

    Returns:
        Dict with comparison metrics
    """
    online_predictor = get_online_predictor(niche)

    # Online predictions
    online_preds = online_predictor.predict_batch(test_features)

    # Calculate online metrics
    online_mae_values = []
    for target in TARGET_NAMES:
        if target in test_targets.columns and target in online_preds.columns:
            mae = np.mean(np.abs(test_targets[target] - online_preds[target]))
            online_mae_values.append(mae)

    online_mae = np.mean(online_mae_values) if online_mae_values else None

    # Batch predictions (if model provided)
    batch_mae = None
    if batch_model is not None:
        try:
            batch_preds = batch_model.predict(test_features.values)
            batch_mae = np.mean(np.abs(test_targets.values - batch_preds))
        except Exception as e:
            logger.debug(f"Batch model prediction failed: {e}")

    result = {
        "niche": niche,
        "test_samples": len(test_features),
        "online_mae": online_mae,
        "batch_mae": batch_mae,
        "online_better": online_mae < batch_mae if (online_mae and batch_mae) else None,
        "online_samples_seen": online_predictor._metrics.samples_seen,
    }

    return result


# =============================================================================
# CLI for Testing
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("\n" + "=" * 70)
    print("Online Learning Module Test - BrandPulse AI")
    print("=" * 70 + "\n")

    if not RIVER_AVAILABLE:
        print("ERROR: River library not installed!")
        print("Install with: pip install river")
        sys.exit(1)

    # Test with synthetic data
    print("[1] Creating synthetic test data...")

    np.random.seed(42)
    n_samples = 50
    n_features = 20

    # Simulate features (would come from FeatureExtractor in production)
    feature_names = [f"feature_{i}" for i in range(n_features)]
    features_df = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=feature_names
    )

    # Simulate targets (log-transformed engagement metrics)
    targets_df = pd.DataFrame({
        "log_likes": np.random.uniform(2, 8, n_samples),
        "log_comments": np.random.uniform(0, 4, n_samples),
        "log_shares": np.random.uniform(0, 3, n_samples),
        "log_saves": np.random.uniform(0, 3, n_samples),
        "log_views": np.random.uniform(4, 10, n_samples),
    })

    print(f"   Features shape: {features_df.shape}")
    print(f"   Targets shape: {targets_df.shape}")

    # Test incremental updates
    print("\n[2] Testing incremental updates...")

    niche = "test_niche"
    reset_online_predictor(niche)  # Start fresh

    batch_size = 10
    for i in range(0, n_samples, batch_size):
        batch_features = features_df.iloc[i:i+batch_size]
        batch_targets = targets_df.iloc[i:i+batch_size]

        result = online_update(niche, batch_features, batch_targets, save_model=False)

        print(f"   Batch {i//batch_size + 1}: processed={result.samples_processed}, "
              f"total={result.total_samples}, MAE={result.current_mae:.4f if result.current_mae else 'N/A'}, "
              f"time={result.update_time_ms:.1f}ms")

    # Test prediction
    print("\n[3] Testing predictions...")

    predictor = get_online_predictor(niche)
    test_sample = features_df.iloc[0].to_dict()
    prediction = predictor.predict_one(test_sample)

    print(f"   Prediction for sample 0:")
    for target, value in prediction.items():
        print(f"      {target}: {value:.4f}")

    # Get status
    print("\n[4] Model status:")
    status = predictor.get_status()
    for key, value in status.items():
        print(f"   {key}: {value}")

    # Cleanup
    reset_online_predictor(niche)

    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70 + "\n")
