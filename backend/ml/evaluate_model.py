#!/usr/bin/env python3
"""
ML Model Evaluation with Granular Metrics and Drift Detection
=============================================================

Provides comprehensive model evaluation including:
- Global metrics (MAE, R2, RMSE) per target output
- Granular metrics by content format, hour/day, weighted RPI
- Calibration analysis: expected vs observed RPI percentile
- Drift detection using KS test and MAE vs historical baseline
- Automated logging with actionable insights

Author: BrandPulse AI
"""

import json
import logging
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

PROJECT_ROOT = Path(__file__).parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Target columns for multi-output models
TARGET_COLUMNS = ["log_likes", "log_comments", "log_shares", "log_saves", "log_views"]

# Default RPI weights (Relative Performance Index)
DEFAULT_RPI_WEIGHTS = {
    "log_likes": 1.0,
    "log_comments": 2.0,    # Comments = 2x more valuable
    "log_shares": 10.0,     # Shares = 10x more valuable
    "log_saves": 5.0,       # Saves = 5x more valuable
    "log_views": 0.5,       # Views = 0.5x (high volume, lower signal)
}

# Drift detection thresholds
DRIFT_ALERT_THRESHOLD = 0.3  # Score > 0.3 triggers alert
KS_CRITICAL_PVALUE = 0.05    # p-value below this indicates distribution shift

# Evaluation intervals
ONLINE_EVAL_INTERVAL = 50    # Evaluate every N online samples

# Segment definitions
FORMAT_SEGMENTS = ["reel", "carousel", "static_image", "story", "tiktok_video"]
HOUR_SEGMENTS = {
    "morning": (6, 12),
    "afternoon": (12, 18),
    "evening": (18, 22),
    "night": (22, 6),
}
DAY_SEGMENTS = {
    "weekday": [0, 1, 2, 3, 4],  # Mon-Fri
    "weekend": [5, 6],           # Sat-Sun
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MetricResult:
    """Single metric evaluation result."""
    name: str
    value: float
    target: Optional[str] = None
    segment: Optional[str] = None


@dataclass
class DriftResult:
    """Drift detection result."""
    drift_detected: bool
    drift_score: float  # 0-1, higher = more drift
    ks_statistic: Optional[float] = None
    ks_pvalue: Optional[float] = None
    mae_vs_baseline: Optional[float] = None
    baseline_mae: Optional[float] = None
    current_mae: Optional[float] = None
    alert_message: Optional[str] = None


@dataclass
class CalibrationResult:
    """Calibration analysis result."""
    expected_percentiles: List[float]
    observed_percentiles: List[float]
    calibration_error: float  # Mean absolute calibration error
    well_calibrated: bool


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    niche: str
    timestamp: str

    # Global metrics per target
    global_metrics: Dict[str, Dict[str, float]]

    # Granular metrics
    format_metrics: Dict[str, Dict[str, float]]
    time_metrics: Dict[str, Dict[str, float]]
    rpi_metrics: Dict[str, float]

    # Drift detection
    drift: DriftResult

    # Calibration
    calibration: Optional[CalibrationResult] = None

    # Insights
    insights: List[str] = field(default_factory=list)

    # Metadata
    n_samples: int = 0
    evaluation_type: str = "full"  # "full", "online", "retrain"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "niche": self.niche,
            "timestamp": self.timestamp,
            "evaluation_type": self.evaluation_type,
            "n_samples": self.n_samples,
            "global_metrics": self.global_metrics,
            "format_metrics": self.format_metrics,
            "time_metrics": self.time_metrics,
            "rpi_metrics": self.rpi_metrics,
            "drift": {
                "detected": self.drift.drift_detected,
                "score": round(self.drift.drift_score, 4),
                "ks_statistic": round(self.drift.ks_statistic, 4) if self.drift.ks_statistic else None,
                "ks_pvalue": round(self.drift.ks_pvalue, 4) if self.drift.ks_pvalue else None,
                "mae_vs_baseline": round(self.drift.mae_vs_baseline, 4) if self.drift.mae_vs_baseline else None,
                "alert": self.drift.alert_message,
            },
            "calibration": {
                "error": round(self.calibration.calibration_error, 4),
                "well_calibrated": self.calibration.well_calibrated,
            } if self.calibration else None,
            "insights": self.insights,
        }


# =============================================================================
# Metric Calculation Functions
# =============================================================================

def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - (ss_res / ss_tot))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Percentage Error."""
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_weighted_rpi(
    predictions: Dict[str, np.ndarray],
    weights: Dict[str, float] = None
) -> np.ndarray:
    """
    Compute weighted Relative Performance Index (RPI).

    RPI = weighted sum of predictions, normalized to 0-100 scale.
    """
    weights = weights or DEFAULT_RPI_WEIGHTS

    rpi = np.zeros(len(next(iter(predictions.values()))))
    total_weight = 0.0

    for target, pred in predictions.items():
        if target in weights:
            rpi += pred * weights[target]
            total_weight += weights[target]

    if total_weight > 0:
        rpi /= total_weight

    # Normalize to 0-100 using percentile scaling
    rpi_min, rpi_max = np.percentile(rpi, [5, 95])
    if rpi_max > rpi_min:
        rpi = (rpi - rpi_min) / (rpi_max - rpi_min) * 100
        rpi = np.clip(rpi, 0, 100)

    return rpi


# =============================================================================
# Granular Evaluation Functions
# =============================================================================

def evaluate_global_metrics(
    y_true: Union[np.ndarray, pd.DataFrame],
    y_pred: Union[np.ndarray, pd.DataFrame],
    target_columns: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute global metrics for each target output.

    Args:
        y_true: Ground truth values (n_samples, n_targets)
        y_pred: Predicted values (n_samples, n_targets)
        target_columns: Names of target columns

    Returns:
        Dict mapping target name to metrics dict
    """
    target_columns = target_columns or TARGET_COLUMNS

    # Convert to numpy if DataFrame
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values

    # Handle 1D arrays
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    metrics = {}
    n_targets = min(y_true.shape[1], y_pred.shape[1], len(target_columns))

    for i in range(n_targets):
        target = target_columns[i]
        true_col = y_true[:, i]
        pred_col = y_pred[:, i]

        metrics[target] = {
            "mae": round(compute_mae(true_col, pred_col), 4),
            "rmse": round(compute_rmse(true_col, pred_col), 4),
            "r2": round(compute_r2(true_col, pred_col), 4),
            "mape": round(compute_mape(true_col, pred_col), 2),
        }

    # Aggregate metrics
    all_maes = [m["mae"] for m in metrics.values()]
    all_rmses = [m["rmse"] for m in metrics.values()]
    all_r2s = [m["r2"] for m in metrics.values()]

    metrics["_aggregate"] = {
        "mae": round(np.mean(all_maes), 4),
        "rmse": round(np.mean(all_rmses), 4),
        "r2": round(np.mean(all_r2s), 4),
    }

    return metrics


def evaluate_by_format(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metadata: pd.DataFrame,
    format_column: str = "content_format"
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics segmented by content format (Reel, Image, etc.).

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        metadata: DataFrame with format column
        format_column: Name of format column

    Returns:
        Dict mapping format to metrics dict
    """
    metrics = {}

    if format_column not in metadata.columns:
        # Try alternative column names
        for alt in ["is_reel", "is_carousel", "is_static"]:
            if alt in metadata.columns:
                # Create format from binary flags
                formats = []
                for _, row in metadata.iterrows():
                    if row.get("is_reel", 0) == 1:
                        formats.append("reel")
                    elif row.get("is_carousel", 0) == 1:
                        formats.append("carousel")
                    else:
                        formats.append("static_image")
                metadata = metadata.copy()
                metadata[format_column] = formats
                break
        else:
            logger.warning(f"Format column '{format_column}' not found")
            return {"_all": {"mae": compute_mae(y_true.flatten(), y_pred.flatten())}}

    for fmt in metadata[format_column].unique():
        mask = metadata[format_column] == fmt
        if mask.sum() < 5:  # Skip small segments
            continue

        y_true_seg = y_true[mask].flatten()
        y_pred_seg = y_pred[mask].flatten()

        metrics[str(fmt)] = {
            "mae": round(compute_mae(y_true_seg, y_pred_seg), 4),
            "rmse": round(compute_rmse(y_true_seg, y_pred_seg), 4),
            "r2": round(compute_r2(y_true_seg, y_pred_seg), 4),
            "n_samples": int(mask.sum()),
        }

    return metrics


def evaluate_by_time(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metadata: pd.DataFrame,
    hour_column: str = "hour_of_day",
    day_column: str = "day_of_week"
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics segmented by time (hour segments, weekday/weekend).

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        metadata: DataFrame with time columns
        hour_column: Name of hour column (0-23)
        day_column: Name of day column (0-6, Monday=0)

    Returns:
        Dict mapping time segment to metrics dict
    """
    metrics = {}

    # Evaluate by hour segment
    if hour_column in metadata.columns:
        hours = metadata[hour_column].values
        for seg_name, (start, end) in HOUR_SEGMENTS.items():
            if start < end:
                mask = (hours >= start) & (hours < end)
            else:  # Night wraps around
                mask = (hours >= start) | (hours < end)

            if mask.sum() < 5:
                continue

            y_true_seg = y_true[mask].flatten()
            y_pred_seg = y_pred[mask].flatten()

            metrics[f"hour_{seg_name}"] = {
                "mae": round(compute_mae(y_true_seg, y_pred_seg), 4),
                "n_samples": int(mask.sum()),
            }

    # Evaluate by day segment
    if day_column in metadata.columns:
        days = metadata[day_column].values
        for seg_name, day_list in DAY_SEGMENTS.items():
            mask = np.isin(days, day_list)

            if mask.sum() < 5:
                continue

            y_true_seg = y_true[mask].flatten()
            y_pred_seg = y_pred[mask].flatten()

            metrics[f"day_{seg_name}"] = {
                "mae": round(compute_mae(y_true_seg, y_pred_seg), 4),
                "n_samples": int(mask.sum()),
            }

    # Prime time analysis (evening on weekdays)
    if hour_column in metadata.columns and day_column in metadata.columns:
        hours = metadata[hour_column].values
        days = metadata[day_column].values
        prime_mask = (hours >= 18) & (hours <= 21) & np.isin(days, [0, 1, 2, 3, 4])

        if prime_mask.sum() >= 5:
            y_true_seg = y_true[prime_mask].flatten()
            y_pred_seg = y_pred[prime_mask].flatten()

            metrics["prime_time"] = {
                "mae": round(compute_mae(y_true_seg, y_pred_seg), 4),
                "n_samples": int(prime_mask.sum()),
            }

    return metrics


def evaluate_rpi(
    y_true: Union[np.ndarray, pd.DataFrame],
    y_pred: Union[np.ndarray, pd.DataFrame],
    target_columns: List[str] = None,
    weights: Dict[str, float] = None
) -> Dict[str, float]:
    """
    Compute weighted RPI metrics.

    Args:
        y_true: Ground truth values (multi-output)
        y_pred: Predicted values (multi-output)
        target_columns: Names of target columns
        weights: RPI weights per target

    Returns:
        Dict with RPI metrics
    """
    target_columns = target_columns or TARGET_COLUMNS
    weights = weights or DEFAULT_RPI_WEIGHTS

    # Convert to DataFrame for easy handling
    if isinstance(y_true, np.ndarray):
        n_cols = min(y_true.shape[1] if y_true.ndim > 1 else 1, len(target_columns))
        y_true = pd.DataFrame(y_true[:, :n_cols] if y_true.ndim > 1 else y_true,
                             columns=target_columns[:n_cols])
    if isinstance(y_pred, np.ndarray):
        n_cols = min(y_pred.shape[1] if y_pred.ndim > 1 else 1, len(target_columns))
        y_pred = pd.DataFrame(y_pred[:, :n_cols] if y_pred.ndim > 1 else y_pred,
                             columns=target_columns[:n_cols])

    # Compute RPI for true and predicted
    true_preds = {col: y_true[col].values for col in y_true.columns if col in weights}
    pred_preds = {col: y_pred[col].values for col in y_pred.columns if col in weights}

    if not true_preds or not pred_preds:
        return {"rpi_mae": 0.0, "rpi_correlation": 0.0}

    rpi_true = compute_weighted_rpi(true_preds, weights)
    rpi_pred = compute_weighted_rpi(pred_preds, weights)

    # Compute RPI-specific metrics
    rpi_mae = compute_mae(rpi_true, rpi_pred)
    rpi_corr = np.corrcoef(rpi_true, rpi_pred)[0, 1] if len(rpi_true) > 1 else 0.0

    return {
        "rpi_mae": round(rpi_mae, 4),
        "rpi_correlation": round(float(rpi_corr), 4) if not np.isnan(rpi_corr) else 0.0,
        "rpi_rmse": round(compute_rmse(rpi_true, rpi_pred), 4),
    }


# =============================================================================
# Drift Detection
# =============================================================================

def detect_drift_ks_test(
    current_predictions: np.ndarray,
    baseline_predictions: np.ndarray
) -> Tuple[float, float]:
    """
    Detect distribution drift using Kolmogorov-Smirnov test.

    Args:
        current_predictions: Recent predictions
        baseline_predictions: Historical baseline predictions

    Returns:
        Tuple of (ks_statistic, p_value)
    """
    try:
        from scipy import stats
        ks_stat, p_value = stats.ks_2samp(current_predictions.flatten(),
                                          baseline_predictions.flatten())
        return float(ks_stat), float(p_value)
    except ImportError:
        logger.warning("scipy not available for KS test")
        return None, None


def detect_drift_mae(
    current_mae: float,
    baseline_mae: float,
    tolerance: float = 0.2
) -> Tuple[bool, float]:
    """
    Detect drift based on MAE degradation vs historical baseline.

    Args:
        current_mae: Current MAE
        baseline_mae: Historical baseline MAE
        tolerance: Relative increase tolerance (0.2 = 20% degradation)

    Returns:
        Tuple of (drift_detected, relative_increase)
    """
    if baseline_mae == 0:
        return False, 0.0

    relative_increase = (current_mae - baseline_mae) / baseline_mae
    drift_detected = relative_increase > tolerance

    return drift_detected, float(relative_increase)


def compute_drift_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_predictions: Optional[np.ndarray] = None,
    baseline_mae: Optional[float] = None
) -> DriftResult:
    """
    Compute comprehensive drift score (0-1).

    Combines:
    - KS test for distribution shift
    - MAE vs historical baseline

    Args:
        y_true: Current ground truth
        y_pred: Current predictions
        baseline_predictions: Historical predictions for KS test
        baseline_mae: Historical MAE baseline

    Returns:
        DriftResult with drift score and details
    """
    current_mae = compute_mae(y_true.flatten(), y_pred.flatten())

    # Initialize components
    ks_score = 0.0
    mae_score = 0.0
    ks_stat, ks_pvalue = None, None
    mae_vs_baseline = None

    # KS test component
    if baseline_predictions is not None and len(baseline_predictions) >= 10:
        ks_stat, ks_pvalue = detect_drift_ks_test(y_pred, baseline_predictions)
        if ks_stat is not None:
            # Convert KS statistic to 0-1 score (higher = more drift)
            ks_score = min(ks_stat * 2, 1.0)  # Scale: KS > 0.5 = max drift

    # MAE baseline component
    if baseline_mae is not None and baseline_mae > 0:
        drift_detected, relative_increase = detect_drift_mae(current_mae, baseline_mae)
        mae_vs_baseline = relative_increase
        # Convert relative increase to 0-1 score
        mae_score = min(max(relative_increase, 0), 1.0)

    # Combined drift score (weighted average)
    if ks_stat is not None and baseline_mae is not None:
        drift_score = 0.4 * ks_score + 0.6 * mae_score
    elif baseline_mae is not None:
        drift_score = mae_score
    elif ks_stat is not None:
        drift_score = ks_score
    else:
        drift_score = 0.0

    # Determine alert
    drift_detected = drift_score > DRIFT_ALERT_THRESHOLD
    alert_message = None

    if drift_detected:
        alert_parts = []
        if ks_pvalue is not None and ks_pvalue < KS_CRITICAL_PVALUE:
            alert_parts.append(f"Distribution shift detected (KS p={ks_pvalue:.4f})")
        if mae_vs_baseline is not None and mae_vs_baseline > 0.2:
            alert_parts.append(f"MAE degraded {mae_vs_baseline*100:.1f}% vs baseline")
        alert_message = "; ".join(alert_parts) if alert_parts else "Drift detected"

    return DriftResult(
        drift_detected=drift_detected,
        drift_score=round(drift_score, 4),
        ks_statistic=ks_stat,
        ks_pvalue=ks_pvalue,
        mae_vs_baseline=mae_vs_baseline,
        baseline_mae=baseline_mae,
        current_mae=current_mae,
        alert_message=alert_message,
    )


# =============================================================================
# Calibration Analysis
# =============================================================================

def analyze_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10
) -> CalibrationResult:
    """
    Analyze calibration: expected vs observed RPI percentiles.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        n_bins: Number of percentile bins

    Returns:
        CalibrationResult with percentile comparison
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Create percentile bins based on predictions
    percentiles = np.linspace(0, 100, n_bins + 1)
    pred_bins = np.percentile(y_pred, percentiles)

    expected_percentiles = []
    observed_percentiles = []

    for i in range(n_bins):
        # Get samples in this predicted percentile bin
        mask = (y_pred >= pred_bins[i]) & (y_pred < pred_bins[i + 1])
        if i == n_bins - 1:  # Include upper bound in last bin
            mask = (y_pred >= pred_bins[i]) & (y_pred <= pred_bins[i + 1])

        if mask.sum() == 0:
            continue

        # Expected: middle of bin
        expected = (percentiles[i] + percentiles[i + 1]) / 2

        # Observed: actual percentile of true values in this bin
        true_in_bin = y_true[mask]
        # What percentile of all true values are these?
        observed = np.mean([
            (y_true <= v).sum() / len(y_true) * 100 for v in true_in_bin
        ])

        expected_percentiles.append(expected)
        observed_percentiles.append(observed)

    # Compute calibration error (mean absolute difference)
    if expected_percentiles:
        calibration_error = np.mean(np.abs(
            np.array(expected_percentiles) - np.array(observed_percentiles)
        ))
    else:
        calibration_error = 0.0

    # Well-calibrated if error < 10 percentile points
    well_calibrated = calibration_error < 10

    return CalibrationResult(
        expected_percentiles=expected_percentiles,
        observed_percentiles=observed_percentiles,
        calibration_error=calibration_error,
        well_calibrated=well_calibrated,
    )


# =============================================================================
# Insight Generation
# =============================================================================

def generate_insights(
    global_metrics: Dict[str, Dict[str, float]],
    format_metrics: Dict[str, Dict[str, float]],
    time_metrics: Dict[str, Dict[str, float]],
    drift: DriftResult,
    calibration: Optional[CalibrationResult] = None
) -> List[str]:
    """
    Generate actionable insights from evaluation results.

    Args:
        global_metrics: Global metrics per target
        format_metrics: Metrics by content format
        time_metrics: Metrics by time segment
        drift: Drift detection result
        calibration: Calibration result

    Returns:
        List of insight strings
    """
    insights = []

    # Drift insights
    if drift.drift_detected:
        insights.append(f"ALERTA: Drift detectado (score={drift.drift_score:.2f}). "
                       f"Considerar reentrenamiento.")
        if drift.alert_message:
            insights.append(f"Detalle: {drift.alert_message}")

    # Per-target insights
    worst_target = None
    worst_mae = 0
    for target, metrics in global_metrics.items():
        if target == "_aggregate":
            continue
        if metrics.get("mae", 0) > worst_mae:
            worst_mae = metrics["mae"]
            worst_target = target

    if worst_target and worst_mae > 0.5:
        target_name = worst_target.replace("log_", "").capitalize()
        insights.append(f"{target_name}: MAE alto ({worst_mae:.3f}). "
                       f"Revisar features o datos de {target_name.lower()}.")

    # Format-specific insights
    for fmt, metrics in format_metrics.items():
        if fmt.startswith("_"):
            continue
        mae = metrics.get("mae", 0)
        n_samples = metrics.get("n_samples", 0)

        if mae > 0.6 and n_samples >= 10:
            insights.append(f"{fmt.capitalize()}: MAE elevado ({mae:.3f}, n={n_samples}). "
                          f"Posible subrepresentacion o patron diferente.")

    # Time-specific insights
    prime_metrics = time_metrics.get("prime_time", {})
    if prime_metrics.get("mae", 0) > 0.5:
        insights.append(f"Prime time (18-21h): MAE alto ({prime_metrics['mae']:.3f}). "
                       f"Mayor variabilidad en horas pico.")

    # Weekend vs weekday comparison
    weekday_mae = time_metrics.get("day_weekday", {}).get("mae", 0)
    weekend_mae = time_metrics.get("day_weekend", {}).get("mae", 0)
    if weekday_mae and weekend_mae:
        if weekend_mae > weekday_mae * 1.3:
            insights.append(f"Weekend: MAE {(weekend_mae/weekday_mae-1)*100:.0f}% mayor que weekday. "
                          f"Comportamiento diferente en fines de semana.")

    # Calibration insights
    if calibration and not calibration.well_calibrated:
        insights.append(f"Calibracion suboptima (error={calibration.calibration_error:.1f}%). "
                       f"Predicciones pueden ser sistematicamente optimistas/pesimistas.")

    # No issues found
    if not insights:
        insights.append("Modelo estable, sin drift significativo detectado.")

    return insights


# =============================================================================
# Main Evaluation Function
# =============================================================================

def evaluate(
    niche: str,
    model: Any,
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: Union[np.ndarray, pd.DataFrame],
    metadata: Optional[pd.DataFrame] = None,
    baseline_predictions: Optional[np.ndarray] = None,
    baseline_mae: Optional[float] = None,
    target_columns: List[str] = None,
    rpi_weights: Dict[str, float] = None,
    evaluation_type: str = "full",
    save_results: bool = True
) -> EvaluationResult:
    """
    Comprehensive model evaluation with granular metrics and drift detection.

    Args:
        niche: Business niche identifier
        model: Trained model with predict() method
        X_test: Test features
        y_test: Test targets (multi-output)
        metadata: Optional DataFrame with format/time columns for granular analysis
        baseline_predictions: Historical predictions for drift KS test
        baseline_mae: Historical MAE for drift comparison
        target_columns: Names of target columns
        rpi_weights: Weights for RPI calculation
        evaluation_type: "full", "online", or "retrain"
        save_results: Whether to save results to logs

    Returns:
        EvaluationResult with all metrics and insights
    """
    target_columns = target_columns or TARGET_COLUMNS
    timestamp = datetime.utcnow().isoformat()

    logger.info(f"Starting evaluation for niche={niche}, type={evaluation_type}")

    # Convert inputs to numpy
    if isinstance(X_test, pd.DataFrame):
        X_test_np = X_test.values
    else:
        X_test_np = X_test

    if isinstance(y_test, pd.DataFrame):
        y_test_np = y_test.values
        # Use metadata from y_test if not provided
        if metadata is None and any(col in y_test.columns for col in ["content_format", "is_reel", "hour_of_day"]):
            metadata = y_test
    else:
        y_test_np = y_test

    # Generate predictions
    try:
        y_pred = model.predict(X_test_np)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        # Return empty result
        return EvaluationResult(
            niche=niche,
            timestamp=timestamp,
            global_metrics={},
            format_metrics={},
            time_metrics={},
            rpi_metrics={},
            drift=DriftResult(drift_detected=False, drift_score=0.0),
            insights=[f"Evaluation failed: {str(e)}"],
            n_samples=len(X_test_np),
            evaluation_type=evaluation_type,
        )

    # Ensure 2D arrays
    if y_test_np.ndim == 1:
        y_test_np = y_test_np.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    n_samples = len(y_test_np)

    # 1. Global metrics per target
    global_metrics = evaluate_global_metrics(y_test_np, y_pred, target_columns)
    logger.info(f"Global metrics computed: aggregate MAE={global_metrics['_aggregate']['mae']:.4f}")

    # 2. Granular: by format
    if metadata is not None:
        format_metrics = evaluate_by_format(y_test_np, y_pred, metadata)
    else:
        format_metrics = {"_all": {"mae": global_metrics["_aggregate"]["mae"]}}

    # 3. Granular: by time
    if metadata is not None:
        time_metrics = evaluate_by_time(y_test_np, y_pred, metadata)
    else:
        time_metrics = {}

    # 4. RPI metrics
    rpi_metrics = evaluate_rpi(y_test_np, y_pred, target_columns, rpi_weights)

    # 5. Drift detection
    current_mae = global_metrics["_aggregate"]["mae"]
    drift = compute_drift_score(
        y_test_np, y_pred,
        baseline_predictions=baseline_predictions,
        baseline_mae=baseline_mae
    )

    # 6. Calibration analysis
    calibration = None
    if n_samples >= 50:
        calibration = analyze_calibration(y_test_np, y_pred)

    # 7. Generate insights
    insights = generate_insights(
        global_metrics, format_metrics, time_metrics, drift, calibration
    )

    # Log insights
    for insight in insights:
        if insight.startswith("ALERTA"):
            logger.warning(insight)
        else:
            logger.info(f"Insight: {insight}")

    # Build result
    result = EvaluationResult(
        niche=niche,
        timestamp=timestamp,
        global_metrics=global_metrics,
        format_metrics=format_metrics,
        time_metrics=time_metrics,
        rpi_metrics=rpi_metrics,
        drift=drift,
        calibration=calibration,
        insights=insights,
        n_samples=n_samples,
        evaluation_type=evaluation_type,
    )

    # Save results
    if save_results:
        save_evaluation_results(result)

    return result


# =============================================================================
# Result Storage
# =============================================================================

def save_evaluation_results(result: EvaluationResult) -> Path:
    """
    Save evaluation results to logs directory.

    Args:
        result: EvaluationResult to save

    Returns:
        Path to saved file
    """
    filename = f"eval_{result.niche}.json"
    filepath = LOGS_DIR / filename

    # Load existing results if any
    history = []
    if filepath.exists():
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                history = data.get("history", [])
        except Exception:
            pass

    # Add new result
    history.append(result.to_dict())

    # Keep last 100 evaluations
    history = history[-100:]

    # Save with history
    output = {
        "niche": result.niche,
        "last_updated": result.timestamp,
        "latest": result.to_dict(),
        "history": history,
    }

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"Saved evaluation results to: {filepath}")

    return filepath


def load_evaluation_history(niche: str) -> List[Dict[str, Any]]:
    """
    Load evaluation history for a niche.

    Args:
        niche: Business niche identifier

    Returns:
        List of historical evaluation results
    """
    filepath = LOGS_DIR / f"eval_{niche}.json"

    if not filepath.exists():
        return []

    try:
        with open(filepath, "r") as f:
            data = json.load(f)
            return data.get("history", [])
    except Exception as e:
        logger.error(f"Error loading evaluation history: {e}")
        return []


def get_baseline_mae(niche: str, lookback: int = 10) -> Optional[float]:
    """
    Get baseline MAE from recent evaluation history.

    Args:
        niche: Business niche identifier
        lookback: Number of recent evaluations to average

    Returns:
        Baseline MAE or None if no history
    """
    history = load_evaluation_history(niche)

    if not history:
        return None

    maes = []
    for eval_result in history[-lookback:]:
        global_metrics = eval_result.get("global_metrics", {})
        aggregate = global_metrics.get("_aggregate", {})
        if "mae" in aggregate:
            maes.append(aggregate["mae"])

    return np.mean(maes) if maes else None


# =============================================================================
# Online Evaluation Hook
# =============================================================================

# Counter for online samples
_online_sample_counter: Dict[str, int] = {}


def should_evaluate_online(niche: str, samples_added: int = 1) -> bool:
    """
    Check if online evaluation should be triggered.

    Triggers every ONLINE_EVAL_INTERVAL samples.

    Args:
        niche: Business niche identifier
        samples_added: Number of new samples added

    Returns:
        True if evaluation should run
    """
    global _online_sample_counter

    if niche not in _online_sample_counter:
        _online_sample_counter[niche] = 0

    _online_sample_counter[niche] += samples_added

    if _online_sample_counter[niche] >= ONLINE_EVAL_INTERVAL:
        _online_sample_counter[niche] = 0
        return True

    return False


def reset_online_counter(niche: str):
    """Reset online sample counter for a niche."""
    global _online_sample_counter
    _online_sample_counter[niche] = 0


# =============================================================================
# Calibration Plot (Text-Based)
# =============================================================================

def print_calibration_plot(calibration: CalibrationResult, niche: str):
    """
    Print text-based calibration plot to logs.

    Args:
        calibration: CalibrationResult to visualize
        niche: Niche name for logging
    """
    if not calibration.expected_percentiles:
        return

    logger.info(f"\n{'='*50}")
    logger.info(f"CALIBRATION PLOT - {niche.upper()}")
    logger.info(f"{'='*50}")
    logger.info(f"{'Expected %':<12} {'Observed %':<12} {'Delta':<10} {'Visual'}")
    logger.info(f"{'-'*50}")

    for exp, obs in zip(calibration.expected_percentiles, calibration.observed_percentiles):
        delta = obs - exp
        # Create visual bar
        bar_len = int(abs(delta) / 5)  # 5% = 1 char
        if delta >= 0:
            bar = "+" * min(bar_len, 10)
        else:
            bar = "-" * min(bar_len, 10)

        logger.info(f"{exp:<12.1f} {obs:<12.1f} {delta:+8.1f}   [{bar:<10}]")

    logger.info(f"{'-'*50}")
    logger.info(f"Calibration Error: {calibration.calibration_error:.2f}%")
    logger.info(f"Well Calibrated: {'Yes' if calibration.well_calibrated else 'No'}")
    logger.info(f"{'='*50}\n")


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
    print("ML Evaluation Module Test - BrandPulse AI")
    print("=" * 70 + "\n")

    # Test with synthetic data
    print("[1] Creating synthetic test data...")

    np.random.seed(42)
    n_samples = 200
    n_features = 50
    n_targets = 5

    # Synthetic features
    X_test = np.random.randn(n_samples, n_features)

    # Synthetic multi-output targets
    y_test = np.random.uniform(2, 8, (n_samples, n_targets))

    # Synthetic predictions (with some noise)
    y_pred = y_test + np.random.randn(n_samples, n_targets) * 0.3

    # Create metadata with format and time info
    metadata = pd.DataFrame({
        "content_format": np.random.choice(["reel", "carousel", "static_image"], n_samples),
        "hour_of_day": np.random.randint(0, 24, n_samples),
        "day_of_week": np.random.randint(0, 7, n_samples),
        "is_reel": [1 if f == "reel" else 0 for f in np.random.choice(["reel", "other"], n_samples)],
    })

    print(f"   X_test shape: {X_test.shape}")
    print(f"   y_test shape: {y_test.shape}")
    print(f"   Metadata: {list(metadata.columns)}")

    # Create a simple mock model
    class MockModel:
        def predict(self, X):
            # Return predictions with some noise
            np.random.seed(42)
            return y_pred

    model = MockModel()

    print("\n[2] Running evaluation...")

    result = evaluate(
        niche="test_niche",
        model=model,
        X_test=X_test,
        y_test=y_test,
        metadata=metadata,
        baseline_mae=0.25,  # Simulate baseline
        evaluation_type="full",
        save_results=False
    )

    print("\n[3] Results:")
    print(f"   Niche: {result.niche}")
    print(f"   Samples: {result.n_samples}")
    print(f"   Aggregate MAE: {result.global_metrics['_aggregate']['mae']:.4f}")
    print(f"   Aggregate R2: {result.global_metrics['_aggregate']['r2']:.4f}")
    print(f"   Drift Score: {result.drift.drift_score:.4f}")
    print(f"   Drift Detected: {result.drift.drift_detected}")
    print(f"   RPI MAE: {result.rpi_metrics['rpi_mae']:.4f}")
    print(f"   RPI Correlation: {result.rpi_metrics['rpi_correlation']:.4f}")

    print("\n[4] Per-target metrics:")
    for target, metrics in result.global_metrics.items():
        if target != "_aggregate":
            print(f"   {target}: MAE={metrics['mae']:.4f}, R2={metrics['r2']:.4f}")

    print("\n[5] Format metrics:")
    for fmt, metrics in result.format_metrics.items():
        if not fmt.startswith("_"):
            print(f"   {fmt}: MAE={metrics['mae']:.4f}, n={metrics.get('n_samples', '?')}")

    print("\n[6] Time metrics:")
    for seg, metrics in result.time_metrics.items():
        print(f"   {seg}: MAE={metrics['mae']:.4f}, n={metrics.get('n_samples', '?')}")

    if result.calibration:
        print("\n[7] Calibration:")
        print(f"   Error: {result.calibration.calibration_error:.2f}%")
        print(f"   Well Calibrated: {result.calibration.well_calibrated}")
        print_calibration_plot(result.calibration, "test_niche")

    print("\n[8] Insights:")
    for insight in result.insights:
        print(f"   - {insight}")

    print("\n" + "=" * 70)
    print("Evaluation Module Test Complete!")
    print("=" * 70 + "\n")
