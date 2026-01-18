#!/usr/bin/env python3
"""
BrandPulse AI - Model Retraining Script
========================================

This script retrains the ML models using collected performance data with:
- Time-based train/test split (prevents data leakage)
- Rigorous evaluation metrics: R², RMSE, MAE for regression; ROC-AUC for classification
- Baseline comparison (historical mean)
- SHAP feature importance analysis

Usage:
    python scripts/retrain_models.py [--min-samples 50] [--test-days 7]

Dependencies:
    - Collected performance feedback from ab_test_logs table
    - At least 50 samples recommended for meaningful retraining
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    roc_auc_score,
    accuracy_score,
    classification_report
)
import xgboost as xgb
import shap
import joblib

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ml_service import FeatureExtractor, MODEL_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelRetrainer:
    """
    Retrains ML models with time-based splitting and rigorous evaluation.

    EVALUATION METRICS:
    - Regression (RPI Score): R², RMSE, MAE
    - Classification (high_engagement): ROC-AUC, Accuracy

    BASELINE COMPARISON:
    - Compares model performance against historical mean baseline
    - Reports improvement percentage
    """

    def __init__(self, min_samples: int = 50, test_days: int = 7):
        self.min_samples = min_samples
        self.test_days = test_days
        self.feature_extractor = FeatureExtractor

    def load_training_data(self, data_source: str = "database") -> pd.DataFrame:
        """
        Load training data from database or file.

        Returns DataFrame with columns:
        - features (extracted from content)
        - engagement_score (actual performance)
        - created_at (for time-based splitting)
        """
        if data_source == "database":
            return self._load_from_database()
        else:
            return self._load_from_file(data_source)

    def _load_from_database(self) -> pd.DataFrame:
        """Load data from ab_test_logs and prediction_logs tables."""
        try:
            # This would connect to the actual database
            # For now, we'll simulate with sample data structure
            logger.info("Loading training data from database...")

            # In production, this would be:
            # from app.core.database import get_db
            # async with get_db() as db:
            #     results = await db.execute(select(ABTestLog).order_by(ABTestLog.published_date))
            #     logs = results.scalars().all()

            # Return empty DataFrame if no data (will be populated by actual DB)
            return pd.DataFrame(columns=[
                'content_id', 'caption', 'content_format', 'business_type',
                'predicted_rpi', 'actual_engagement', 'likes', 'comments',
                'saves', 'shares', 'views', 'published_date'
            ])

        except Exception as e:
            logger.error(f"Error loading from database: {e}")
            return pd.DataFrame()

    def _load_from_file(self, filepath: str) -> pd.DataFrame:
        """Load training data from CSV/JSON file."""
        path = Path(filepath)
        if path.suffix == '.csv':
            return pd.read_csv(filepath, parse_dates=['published_date'])
        elif path.suffix == '.json':
            return pd.read_json(filepath)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Extract features and prepare targets.

        Returns:
        - X: Feature DataFrame
        - y_regression: Engagement score (continuous)
        - y_classification: High engagement binary flag
        """
        logger.info(f"Preparing features from {len(df)} samples...")

        # Extract features for each content item
        features_list = []
        for _, row in df.iterrows():
            content = {
                'caption': row.get('caption', ''),
                'content_format': row.get('content_format', 'reel'),
                'hashtags': row.get('hashtags', []),
                'business_type': row.get('business_type', 'otros'),
                'posted_at': row.get('published_date'),
            }
            features = self.feature_extractor.extract_features(content)
            features['published_date'] = row.get('published_date')
            features_list.append(features)

        X = pd.DataFrame(features_list)

        # Calculate engagement score (target for regression)
        df['engagement_score'] = (
            df.get('likes', 0).fillna(0) +
            df.get('comments', 0).fillna(0) * 3 +
            df.get('saves', 0).fillna(0) * 5 +
            df.get('shares', 0).fillna(0) * 4
        )

        # Normalize to 0-100 scale
        max_engagement = df['engagement_score'].quantile(0.95)
        if max_engagement > 0:
            y_regression = (df['engagement_score'] / max_engagement * 100).clip(0, 100)
        else:
            y_regression = df['engagement_score']

        # Binary classification target (high engagement = above median)
        median_engagement = y_regression.median()
        y_classification = (y_regression > median_engagement).astype(int)

        return X, y_regression, y_classification

    def time_based_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_days: int = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Time-based train/test split to prevent data leakage.

        Uses the most recent `test_days` worth of data as test set.
        """
        test_days = test_days or self.test_days

        if 'published_date' not in X.columns:
            # Fallback to random split if no date column
            logger.warning("No date column found, using random 80/20 split")
            split_idx = int(len(X) * 0.8)
            return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]

        # Sort by date
        X = X.sort_values('published_date')
        y = y.loc[X.index]

        # Find split point
        max_date = X['published_date'].max()
        split_date = max_date - timedelta(days=test_days)

        train_mask = X['published_date'] <= split_date
        test_mask = X['published_date'] > split_date

        X_train = X[train_mask].drop(columns=['published_date'], errors='ignore')
        X_test = X[test_mask].drop(columns=['published_date'], errors='ignore')
        y_train = y[train_mask]
        y_test = y[test_mask]

        logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        logger.info(f"Train period: <= {split_date.date()}, Test period: > {split_date.date()}")

        return X_train, X_test, y_train, y_test

    def calculate_baseline(self, y_train: pd.Series, y_test: pd.Series) -> Dict[str, float]:
        """
        Calculate baseline metrics using historical mean.

        This serves as a comparison point to measure model improvement.
        """
        baseline_pred = np.full(len(y_test), y_train.mean())

        return {
            'baseline_rmse': np.sqrt(mean_squared_error(y_test, baseline_pred)),
            'baseline_mae': mean_absolute_error(y_test, baseline_pred),
            'baseline_mean': float(y_train.mean()),
        }

    def train_regression_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Tuple[xgb.XGBRegressor, Dict[str, float]]:
        """
        Train XGBoost regression model for RPI score prediction.

        Evaluation metrics: R², RMSE, MAE
        """
        logger.info("Training regression model (RPI Score)...")

        # Handle non-numeric columns
        X_train_numeric = X_train.select_dtypes(include=[np.number]).fillna(0)
        X_test_numeric = X_test.select_dtypes(include=[np.number]).fillna(0)

        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train_numeric, y_train)

        # Predictions
        y_pred = model.predict(X_test_numeric)

        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        # Calculate baseline for comparison
        baseline = self.calculate_baseline(y_train, y_test)

        metrics = {
            'r2_score': float(r2),
            'rmse': float(rmse),
            'mae': float(mae),
            'baseline_rmse': baseline['baseline_rmse'],
            'baseline_mae': baseline['baseline_mae'],
            'rmse_improvement': float((baseline['baseline_rmse'] - rmse) / baseline['baseline_rmse'] * 100),
            'mae_improvement': float((baseline['baseline_mae'] - mae) / baseline['baseline_mae'] * 100),
        }

        logger.info(f"Regression Metrics:")
        logger.info(f"  R² Score: {r2:.4f}")
        logger.info(f"  RMSE: {rmse:.4f} (baseline: {baseline['baseline_rmse']:.4f}, improvement: {metrics['rmse_improvement']:.1f}%)")
        logger.info(f"  MAE: {mae:.4f} (baseline: {baseline['baseline_mae']:.4f}, improvement: {metrics['mae_improvement']:.1f}%)")

        return model, metrics

    def train_classification_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Tuple[xgb.XGBClassifier, Dict[str, float]]:
        """
        Train XGBoost classifier for high_engagement prediction.

        Evaluation metrics: ROC-AUC, Accuracy
        """
        logger.info("Training classification model (high_engagement)...")

        # Handle non-numeric columns
        X_train_numeric = X_train.select_dtypes(include=[np.number]).fillna(0)
        X_test_numeric = X_test.select_dtypes(include=[np.number]).fillna(0)

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='binary:logistic',
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train_numeric, y_train)

        # Predictions
        y_pred = model.predict(X_test_numeric)
        y_pred_proba = model.predict_proba(X_test_numeric)[:, 1]

        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)

        # Baseline: predict majority class
        baseline_accuracy = max(y_test.mean(), 1 - y_test.mean())

        metrics = {
            'roc_auc': float(roc_auc),
            'accuracy': float(accuracy),
            'baseline_accuracy': float(baseline_accuracy),
            'accuracy_improvement': float((accuracy - baseline_accuracy) / baseline_accuracy * 100),
        }

        logger.info(f"Classification Metrics:")
        logger.info(f"  ROC-AUC: {roc_auc:.4f}")
        logger.info(f"  Accuracy: {accuracy:.4f} (baseline: {baseline_accuracy:.4f}, improvement: {metrics['accuracy_improvement']:.1f}%)")

        return model, metrics

    def compute_shap_importance(
        self,
        model: xgb.XGBRegressor,
        X_sample: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compute SHAP feature importance for model interpretability.

        Returns top 10 most important features with their mean |SHAP| values.
        """
        logger.info("Computing SHAP feature importance...")

        X_numeric = X_sample.select_dtypes(include=[np.number]).fillna(0)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_numeric)

        # Calculate mean absolute SHAP value for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'feature': X_numeric.columns,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)

        top_features = feature_importance.head(10).to_dict('records')

        logger.info("Top 10 Feature Importance (SHAP):")
        for i, f in enumerate(top_features, 1):
            logger.info(f"  {i}. {f['feature']}: {f['importance']:.4f}")

        return {
            'top_features': top_features,
            'all_features': feature_importance.to_dict('records'),
        }

    def save_models(
        self,
        regression_model: xgb.XGBRegressor,
        classification_model: xgb.XGBClassifier,
        metrics: Dict[str, Any],
        shap_importance: Dict[str, Any]
    ) -> Dict[str, str]:
        """Save trained models and metadata to disk."""
        logger.info("Saving models...")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save models
        regression_path = MODEL_DIR / f"engagement_model_{timestamp}.joblib"
        classification_path = MODEL_DIR / f"high_engagement_classifier_{timestamp}.joblib"

        joblib.dump(regression_model, regression_path)
        joblib.dump(classification_model, classification_path)

        # Save as "latest" for production use
        joblib.dump(regression_model, MODEL_DIR / "engagement_model.joblib")
        joblib.dump(classification_model, MODEL_DIR / "high_engagement_classifier.joblib")

        # Save training metadata
        metadata = {
            'timestamp': timestamp,
            'regression_metrics': metrics.get('regression', {}),
            'classification_metrics': metrics.get('classification', {}),
            'shap_importance': shap_importance,
        }
        joblib.dump(metadata, MODEL_DIR / f"training_metadata_{timestamp}.joblib")
        joblib.dump(metadata, MODEL_DIR / "training_metadata_latest.joblib")

        logger.info(f"Models saved to {MODEL_DIR}")

        return {
            'regression_model': str(regression_path),
            'classification_model': str(classification_path),
            'metadata': str(MODEL_DIR / f"training_metadata_{timestamp}.joblib"),
        }

    def retrain(self, data_source: str = "database") -> Dict[str, Any]:
        """
        Full retraining pipeline.

        Steps:
        1. Load training data
        2. Extract features
        3. Time-based train/test split
        4. Train regression model (RPI Score)
        5. Train classification model (high_engagement)
        6. Compute SHAP importance
        7. Save models and metrics

        Returns training report with all metrics.
        """
        logger.info("=" * 60)
        logger.info("Starting model retraining...")
        logger.info("=" * 60)

        # 1. Load data
        df = self.load_training_data(data_source)

        if len(df) < self.min_samples:
            logger.warning(f"Insufficient data: {len(df)} samples (minimum: {self.min_samples})")
            return {
                'success': False,
                'error': f'Insufficient data: {len(df)} samples (minimum: {self.min_samples})',
                'samples_available': len(df),
            }

        # 2. Prepare features
        X, y_regression, y_classification = self.prepare_features(df)

        # 3. Time-based split
        X_train, X_test, y_reg_train, y_reg_test = self.time_based_split(X, y_regression)
        _, _, y_cls_train, y_cls_test = self.time_based_split(X, y_classification)

        if len(X_test) < 10:
            logger.warning("Not enough test samples for reliable evaluation")

        # 4. Train regression model
        regression_model, regression_metrics = self.train_regression_model(
            X_train, y_reg_train, X_test, y_reg_test
        )

        # 5. Train classification model
        classification_model, classification_metrics = self.train_classification_model(
            X_train, y_cls_train, X_test, y_cls_test
        )

        # 6. Compute SHAP importance
        shap_importance = self.compute_shap_importance(regression_model, X_train.head(100))

        # 7. Save models
        saved_paths = self.save_models(
            regression_model,
            classification_model,
            {'regression': regression_metrics, 'classification': classification_metrics},
            shap_importance
        )

        # Generate training report
        report = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'samples_used': {
                'total': len(df),
                'train': len(X_train),
                'test': len(X_test),
            },
            'regression_metrics': regression_metrics,
            'classification_metrics': classification_metrics,
            'shap_top_features': shap_importance['top_features'][:5],
            'saved_paths': saved_paths,
            'baseline_comparison': {
                'rmse_vs_baseline': f"{regression_metrics['rmse_improvement']:.1f}% improvement",
                'accuracy_vs_baseline': f"{classification_metrics['accuracy_improvement']:.1f}% improvement",
            }
        }

        logger.info("=" * 60)
        logger.info("Retraining complete!")
        logger.info(f"  R² Score: {regression_metrics['r2_score']:.4f}")
        logger.info(f"  RMSE improvement vs baseline: {regression_metrics['rmse_improvement']:.1f}%")
        logger.info(f"  ROC-AUC: {classification_metrics['roc_auc']:.4f}")
        logger.info("=" * 60)

        return report


def main():
    parser = argparse.ArgumentParser(description='Retrain BrandPulse AI ML models')
    parser.add_argument('--min-samples', type=int, default=50,
                        help='Minimum samples required for retraining')
    parser.add_argument('--test-days', type=int, default=7,
                        help='Days of data to use for test set')
    parser.add_argument('--data-source', type=str, default='database',
                        help='Data source: "database" or path to CSV/JSON file')

    args = parser.parse_args()

    retrainer = ModelRetrainer(
        min_samples=args.min_samples,
        test_days=args.test_days
    )

    report = retrainer.retrain(data_source=args.data_source)

    if report['success']:
        print("\n✅ Retraining successful!")
        print(f"   R² Score: {report['regression_metrics']['r2_score']:.4f}")
        print(f"   ROC-AUC: {report['classification_metrics']['roc_auc']:.4f}")
    else:
        print(f"\n❌ Retraining failed: {report.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == '__main__':
    main()
