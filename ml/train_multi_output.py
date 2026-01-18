#!/usr/bin/env python3
"""
Multi-Output Model Training Script
===================================

Trains multi-output engagement prediction models that predict:
- log_likes, log_comments, log_shares, log_saves, log_views

This extends the standard train.py with multi-target support.

Usage:
    # Train multi-output model for a niche
    python ml/train_multi_output.py --niche restaurante --data-file data/posts.csv

    # Train with all engagement metrics
    python ml/train_multi_output.py --niche cafeteria --from-db

    # Train base multi-output model
    python ml/train_multi_output.py --base-model --data-file data/all_posts.csv

Author: BrandPulse AI
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

# Import from standard train.py
try:
    from ml.train import (
        load_data_from_csv,
        add_embedding_features,
        add_multimodal_features,
        add_semantic_hook_features,
        FEATURE_COLUMNS,
        BUSINESS_TYPES,
    )
except ImportError:
    # Fallback if running from different directory
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ml.train import (
        load_data_from_csv,
        add_embedding_features,
        add_multimodal_features,
        add_semantic_hook_features,
        FEATURE_COLUMNS,
        BUSINESS_TYPES,
    )

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MULTI_OUTPUT_DIR = PROJECT_ROOT / "backend" / "ml_models" / "multi_output"
MULTI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Target columns
TARGET_COLUMNS = ["log_likes", "log_comments", "log_shares", "log_saves", "log_views"]
RAW_METRIC_COLUMNS = ["likes", "comments", "shares", "saves", "views"]

# Training thresholds
MIN_SAMPLES_MULTI_OUTPUT = 50


def prepare_multi_output_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare multi-output targets from engagement metrics.

    Applies log1p transformation to normalize skewed distributions.

    Args:
        df: DataFrame with engagement columns

    Returns:
        DataFrame with log-transformed targets
    """
    targets = pd.DataFrame(index=df.index)

    # Map column names (handle different naming conventions)
    column_mappings = {
        "likes": ["likes", "likes_count", "num_likes", "like_count"],
        "comments": ["comments", "comments_count", "num_comments", "comment_count"],
        "shares": ["shares", "shares_count", "num_shares", "share_count"],
        "saves": ["saves", "saves_count", "num_saves", "save_count", "saved"],
        "views": ["views", "views_count", "num_views", "view_count", "plays", "video_views"],
    }

    for metric, alternatives in column_mappings.items():
        found_col = None
        for col in alternatives:
            if col in df.columns:
                found_col = col
                break

        if found_col:
            # Log transform: log(x + 1) to handle zeros
            values = df[found_col].fillna(0).clip(lower=0)
            targets[f"log_{metric}"] = np.log1p(values)
            logger.info(f"  {metric}: found in column '{found_col}', mean={values.mean():.1f}")
        else:
            logger.warning(f"  {metric}: NOT FOUND, using zeros")
            targets[f"log_{metric}"] = 0.0

    return targets


def prepare_features_multi_output(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare features and multi-output targets.

    Args:
        df: Raw DataFrame with content and engagement data

    Returns:
        Tuple of (features_df, targets_df)
    """
    logger.info("Preparing features and multi-output targets...")

    # Find caption column
    caption_col = None
    for col in ["caption", "text", "content", "post_caption", "description"]:
        if col in df.columns:
            caption_col = col
            break

    # Add embedding features
    if caption_col and not any(col.startswith("embedding_") for col in df.columns):
        logger.info("Adding embedding features...")
        df = add_embedding_features(df, caption_column=caption_col)

    # Add semantic hook features
    if caption_col and "semantic_hook_score" not in df.columns:
        logger.info("Adding semantic hook features...")
        df = add_semantic_hook_features(df, caption_column=caption_col)

    # Add multimodal features
    if not any(col.startswith("transcript_emb_") for col in df.columns):
        logger.info("Adding multimodal features...")
        df = add_multimodal_features(df)

    # Select feature columns that exist
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    logger.info(f"Using {len(feature_cols)} feature columns")

    X = df[feature_cols].fillna(0)

    # Prepare targets
    logger.info("Preparing targets:")
    y = prepare_multi_output_targets(df)

    return X, y


def train_multi_output_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    config: Optional[Dict] = None,
) -> Tuple[MultiOutputRegressor, Dict]:
    """
    Train multi-output XGBoost model.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        config: Optional hyperparameter config

    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    config = config or {}

    logger.info("=" * 60)
    logger.info("TRAINING MULTI-OUTPUT MODEL")
    logger.info("=" * 60)
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    logger.info(f"Features: {X_train.shape[1]}")
    logger.info(f"Targets: {y_train.shape[1]} ({', '.join(TARGET_COLUMNS)})")

    # Base XGBoost estimator
    base_estimator = xgb.XGBRegressor(
        n_estimators=config.get("n_estimators", 150),
        max_depth=config.get("max_depth", 6),
        learning_rate=config.get("learning_rate", 0.08),
        min_child_weight=config.get("min_child_weight", 3),
        subsample=config.get("subsample", 0.8),
        colsample_bytree=config.get("colsample_bytree", 0.8),
        gamma=config.get("gamma", 0.1),
        reg_alpha=config.get("reg_alpha", 0.1),
        reg_lambda=config.get("reg_lambda", 1.0),
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
        verbosity=0,
    )

    # Wrap in MultiOutputRegressor
    model = MultiOutputRegressor(base_estimator, n_jobs=1)

    # Train
    logger.info("Training...")
    model.fit(X_train.values, y_train.values)

    # Evaluate
    logger.info("Evaluating...")
    y_pred = model.predict(X_test.values)

    metrics = {}
    target_names = ["likes", "comments", "shares", "saves", "views"]

    print("\n" + "-" * 50)
    print("PER-TARGET METRICS")
    print("-" * 50)

    rmse_values = []
    r2_values = []

    for i, target in enumerate(target_names):
        rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
        r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])

        metrics[f"{target}_rmse"] = rmse
        metrics[f"{target}_r2"] = r2
        metrics[f"{target}_mae"] = mae

        rmse_values.append(rmse)
        r2_values.append(r2)

        print(f"{target:12s}: RMSE={rmse:.4f}, R2={r2:.4f}, MAE={mae:.4f}")

    # Combined metrics
    metrics["combined_rmse"] = np.mean(rmse_values)
    metrics["combined_r2"] = np.mean(r2_values)

    print("-" * 50)
    print(f"{'COMBINED':12s}: RMSE={metrics['combined_rmse']:.4f}, R2={metrics['combined_r2']:.4f}")
    print("-" * 50)

    return model, metrics


def save_multi_output_model(
    model: MultiOutputRegressor,
    metrics: Dict,
    niche: str,
    feature_columns: List[str],
) -> Path:
    """
    Save multi-output model and metadata.

    Args:
        model: Trained model
        metrics: Training metrics
        niche: Business niche
        feature_columns: List of feature column names

    Returns:
        Path to saved model
    """
    # Create niche directory
    niche_dir = MULTI_OUTPUT_DIR / niche
    niche_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = niche_dir / "multi_output_engagement.joblib"
    joblib.dump(model, model_path)

    # Save metadata
    metadata = {
        "niche": niche,
        "model_version": "multi-output-v1.0",
        "feature_columns": feature_columns,
        "target_columns": TARGET_COLUMNS,
        "metrics": metrics,
        "trained_at": datetime.now().isoformat(),
        "is_multi_output": True,
    }
    metadata_path = niche_dir / "multi_output_metadata.joblib"
    joblib.dump(metadata, metadata_path)

    # Save metrics as JSON for easy viewing
    import json
    metrics_path = niche_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "niche": niche,
            "metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
            "trained_at": metadata["trained_at"],
        }, f, indent=2)

    logger.info(f"Model saved to: {model_path}")
    return model_path


def main():
    parser = argparse.ArgumentParser(
        description="Train multi-output engagement prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train for a specific niche
    python ml/train_multi_output.py --niche restaurante --data-file data/posts.csv

    # Train base model with all data
    python ml/train_multi_output.py --base-model --data-file data/all_posts.csv
        """
    )

    parser.add_argument(
        "--niche",
        type=str,
        default="general",
        help="Business niche (default: general)"
    )

    parser.add_argument(
        "--data-file",
        type=str,
        required=True,
        help="Path to CSV with training data"
    )

    parser.add_argument(
        "--base-model",
        action="store_true",
        help="Train as base model (no niche filtering)"
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size (default: 0.2)"
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save the trained model"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("\n" + "=" * 70)
    print("MULTI-OUTPUT ENGAGEMENT MODEL TRAINING")
    print("=" * 70)

    try:
        # Load data
        niche = "general" if args.base_model else args.niche
        df = load_data_from_csv(args.data_file, niche=None if args.base_model else niche)

        if len(df) < MIN_SAMPLES_MULTI_OUTPUT:
            logger.error(
                f"Insufficient data: {len(df)} samples. "
                f"Multi-output training requires at least {MIN_SAMPLES_MULTI_OUTPUT}."
            )
            return 1

        logger.info(f"Loaded {len(df)} samples")

        # Check for engagement columns
        required_metrics = ["likes", "comments"]  # At minimum
        available = [m for m in RAW_METRIC_COLUMNS if any(c in df.columns for c in [m, f"{m}_count"])]

        if len(available) < 2:
            logger.error(
                f"Insufficient engagement columns. Found: {available}. "
                f"Need at least likes and comments."
            )
            return 1

        logger.info(f"Available engagement metrics: {available}")

        # Prepare data
        X, y = prepare_features_multi_output(df)

        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Targets shape: {y.shape}")

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=args.test_size,
            random_state=42
        )

        # Train
        model, metrics = train_multi_output_model(
            X_train, y_train,
            X_test, y_test
        )

        # Add training info to metrics
        metrics["niche"] = niche
        metrics["n_samples"] = len(df)
        metrics["n_features"] = X.shape[1]
        metrics["trained_at"] = datetime.now().isoformat()

        # Save
        if not args.no_save:
            model_path = save_multi_output_model(
                model=model,
                metrics=metrics,
                niche=niche,
                feature_columns=list(X.columns),
            )

        # Summary
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Niche:              {niche}")
        print(f"Samples:            {len(df)}")
        print(f"Features:           {X.shape[1]}")
        print(f"Targets:            {len(TARGET_COLUMNS)}")
        print(f"Combined RMSE:      {metrics['combined_rmse']:.4f}")
        print(f"Combined R2:        {metrics['combined_r2']:.4f}")
        if not args.no_save:
            print(f"Model saved to:     {model_path}")
        print("=" * 70)

        # Print per-target summary
        print("\nPer-target R2 scores:")
        for target in ["likes", "comments", "shares", "saves", "views"]:
            r2 = metrics.get(f"{target}_r2", 0)
            bar = "=" * int(r2 * 20) if r2 > 0 else ""
            print(f"  {target:12s}: {r2:6.3f} [{bar:<20s}]")

        return 0

    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
