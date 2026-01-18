#!/usr/bin/env python3
"""
Base Model Pretraining Script for Cold Start Mitigation
=========================================================

This script trains a robust base XGBoost model using a large synthetic dataset
that simulates realistic Instagram engagement patterns across multiple niches.

The base model serves as a foundation for fine-tuning on niche-specific data,
solving the cold start problem when a new niche has < 300 samples.

APPROACH:
=========
1. Generate 10,000 synthetic samples with realistic feature distributions
   - Features: emojis_count, hashtags, caption_length, hour_of_day, etc.
   - Target: engagement_rate following log-normal distribution (realistic for social media)

2. Train XGBoostRegressor with same hyperparameters as production model

3. Save as /models/base_xgboost.pkl for use in fine-tuning

Usage:
    python ml/pretrain_base_model.py
    python ml/pretrain_base_model.py --samples 20000 --output models/base_xgboost.pkl

Author: BrandPulse AI
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
BACKEND_MODELS_DIR = PROJECT_ROOT / "backend" / "ml_models"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
BACKEND_MODELS_DIR.mkdir(exist_ok=True, parents=True)


# =============================================================================
# FEATURE DEFINITIONS (matching ml_service.py FeatureExtractor)
# =============================================================================

FEATURE_COLUMNS = [
    "caption_length", "caption_words", "caption_lines", "avg_word_length",
    "emoji_count", "emoji_density", "hashtag_count", "hashtag_density",
    "mention_count", "lexical_richness", "has_question", "has_strong_cta",
    "sentiment_compound", "sentiment_positive", "sentiment_negative", "sentiment_neutral",
    "trigger_question", "trigger_urgency", "trigger_social_proof",
    "trigger_value", "trigger_curiosity", "trigger_action",
    "trigger_emotion", "trigger_transformation",
    "hook_pov", "hook_question", "hook_number", "hook_bold_claim",
    "hook_story", "hook_how_to", "hook_reveal",
    "cta_comment", "cta_save", "cta_share", "cta_follow", "cta_dm", "cta_link",
    "cta_count",
    "is_reel", "is_carousel", "is_static",
    "video_duration", "video_optimal_length",
    "has_audio", "is_trending_audio",
    "hour_of_day", "day_of_week", "is_weekend", "is_prime_time",
    "niche_inmobiliaria", "niche_floristeria", "niche_cafeteria",
    "niche_peluqueria", "niche_restaurante", "niche_gimnasio", "niche_clinica",
    "business_type_encoded",
]

BUSINESS_TYPES = [
    "inmobiliaria", "floristeria", "cafeteria", "peluqueria",
    "restaurante", "gimnasio", "clinica", "otros"
]

CONTENT_FORMATS = ["reel", "carousel", "static"]


# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================

def generate_synthetic_engagement_data(n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic Instagram engagement data.

    The engagement_rate follows a log-normal distribution which is typical
    for social media metrics (many low-engagement posts, few viral ones).

    Feature distributions are based on empirical observations from
    successful local business Instagram accounts.

    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with features and engagement_rate target
    """
    logger.info(f"Generating {n_samples} synthetic samples...")
    np.random.seed(seed)

    data = {}

    # === Caption Features ===
    # Caption length: typically 100-500 chars, some shorter/longer
    data["caption_length"] = np.clip(
        np.random.lognormal(mean=5.5, sigma=0.5, size=n_samples),
        20, 2200
    ).astype(int)

    # Words: roughly length / 6
    data["caption_words"] = (data["caption_length"] / 6 + np.random.normal(0, 5, n_samples)).clip(3, 400).astype(int)

    # Lines: 1-10 typically
    data["caption_lines"] = np.random.choice(range(1, 12), size=n_samples, p=[
        0.05, 0.15, 0.20, 0.20, 0.15, 0.10, 0.05, 0.04, 0.03, 0.02, 0.01
    ])

    # Avg word length: 4-8 chars for Spanish
    data["avg_word_length"] = np.clip(np.random.normal(5.5, 1.0, n_samples), 3, 10)

    # === Emoji Features ===
    # Most posts have 0-10 emojis
    data["emoji_count"] = np.random.choice(range(0, 15), size=n_samples, p=[
        0.10, 0.12, 0.15, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04, 0.03,
        0.02, 0.01, 0.01, 0.005, 0.005
    ])
    data["emoji_density"] = data["emoji_count"] / np.maximum(data["caption_length"], 1) * 100

    # === Hashtag Features ===
    # Typically 3-15 hashtags
    data["hashtag_count"] = np.random.choice(range(0, 20), size=n_samples, p=[
        0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.12, 0.10, 0.08, 0.07,
        0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01, 0.005, 0.003, 0.002
    ])
    data["hashtag_density"] = data["hashtag_count"] / np.maximum(data["caption_words"], 1)

    # === Mention Features ===
    data["mention_count"] = np.random.choice(range(0, 6), size=n_samples, p=[0.50, 0.25, 0.12, 0.08, 0.03, 0.02])

    # === Lexical Richness ===
    # Type-token ratio: 0.3-0.9 typically
    data["lexical_richness"] = np.clip(np.random.beta(5, 3, n_samples) * 0.8 + 0.2, 0.2, 0.95)

    # === Question & CTA Detection ===
    data["has_question"] = np.random.choice([0, 1], size=n_samples, p=[0.60, 0.40])
    data["has_strong_cta"] = np.random.choice([0, 1], size=n_samples, p=[0.55, 0.45])

    # === Sentiment Features (VADER-like distribution) ===
    # Compound: mostly positive for business content
    data["sentiment_compound"] = np.clip(np.random.beta(7, 3, n_samples) * 2 - 0.5, -1, 1)
    data["sentiment_positive"] = np.clip(np.random.beta(5, 2, n_samples), 0, 1)
    data["sentiment_negative"] = np.clip(np.random.beta(1, 10, n_samples), 0, 0.3)
    data["sentiment_neutral"] = 1 - data["sentiment_positive"] - data["sentiment_negative"]
    data["sentiment_neutral"] = np.clip(data["sentiment_neutral"], 0, 1)

    # === Trigger Word Features ===
    trigger_types = ["question", "urgency", "social_proof", "value", "curiosity", "action", "emotion", "transformation"]
    for trigger in trigger_types:
        # Most posts have 0-3 of each trigger type
        probs = [0.50, 0.25, 0.15, 0.07, 0.03]
        data[f"trigger_{trigger}"] = np.random.choice(range(5), size=n_samples, p=probs)

    # === Hook Features ===
    hook_types = ["pov", "question", "number", "bold_claim", "story", "how_to", "reveal"]
    for hook in hook_types:
        # Binary: does the first line have this hook type?
        prob_hook = {
            "pov": 0.08, "question": 0.25, "number": 0.15, "bold_claim": 0.10,
            "story": 0.12, "how_to": 0.15, "reveal": 0.10
        }
        data[f"hook_{hook}"] = np.random.choice([0, 1], size=n_samples, p=[1 - prob_hook[hook], prob_hook[hook]])

    # === CTA Features ===
    cta_types = ["comment", "save", "share", "follow", "dm", "link"]
    for cta in cta_types:
        prob_cta = {
            "comment": 0.30, "save": 0.25, "share": 0.15, "follow": 0.20, "dm": 0.10, "link": 0.15
        }
        data[f"cta_{cta}"] = np.random.choice([0, 1], size=n_samples, p=[1 - prob_cta[cta], prob_cta[cta]])

    # Total CTA count
    data["cta_count"] = sum(data[f"cta_{cta}"] for cta in cta_types)

    # === Format Features ===
    # Distribution: Reels dominate in 2024-2025
    format_probs = [0.55, 0.25, 0.20]  # reel, carousel, static
    formats = np.random.choice(CONTENT_FORMATS, size=n_samples, p=format_probs)
    data["is_reel"] = (formats == "reel").astype(int)
    data["is_carousel"] = (formats == "carousel").astype(int)
    data["is_static"] = (formats == "static").astype(int)

    # === Video Features ===
    # Duration: 0 for non-video, 15-90s for reels
    data["video_duration"] = np.where(
        data["is_reel"] == 1,
        np.clip(np.random.lognormal(3.5, 0.4, n_samples), 5, 180),
        0
    )
    # Optimal length: 15-60 seconds
    data["video_optimal_length"] = ((data["video_duration"] >= 15) & (data["video_duration"] <= 60)).astype(int)

    # === Audio Features ===
    data["has_audio"] = np.where(data["is_reel"] == 1, 1, np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]))
    data["is_trending_audio"] = np.where(
        data["has_audio"] == 1,
        np.random.choice([0, 1], size=n_samples, p=[0.60, 0.40]),
        0
    )

    # === Timing Features ===
    # Hour: peaks at 11-13 (lunch) and 19-21 (evening)
    hour_probs = np.array([
        0.01, 0.005, 0.005, 0.005, 0.005, 0.01,  # 0-5
        0.02, 0.03, 0.05, 0.07, 0.08, 0.10,       # 6-11
        0.09, 0.07, 0.05, 0.04, 0.05, 0.06,       # 12-17
        0.08, 0.10, 0.09, 0.06, 0.03, 0.02        # 18-23
    ])
    hour_probs = hour_probs / hour_probs.sum()
    data["hour_of_day"] = np.random.choice(range(24), size=n_samples, p=hour_probs)

    # Day of week: slightly higher activity on weekdays
    data["day_of_week"] = np.random.choice(range(7), size=n_samples)
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    data["is_prime_time"] = np.isin(data["hour_of_day"], [11, 12, 13, 19, 20, 21]).astype(int)

    # === Niche Features ===
    niche_distribution = {
        "inmobiliaria": 0.15, "floristeria": 0.10, "cafeteria": 0.12,
        "peluqueria": 0.12, "restaurante": 0.18, "gimnasio": 0.13,
        "clinica": 0.10, "otros": 0.10
    }
    niches = np.random.choice(
        list(niche_distribution.keys()),
        size=n_samples,
        p=list(niche_distribution.values())
    )

    for niche in BUSINESS_TYPES[:-1]:  # Exclude 'otros'
        data[f"niche_{niche}"] = (niches == niche).astype(int)

    # Business type encoded
    encoder = LabelEncoder()
    encoder.fit(BUSINESS_TYPES)
    data["business_type_encoded"] = encoder.transform(niches)
    data["business_type"] = niches  # Keep for reference

    # ==========================================================================
    # ENGAGEMENT RATE - Log-Normal Distribution (Realistic for Social Media)
    # ==========================================================================
    #
    # Social media engagement follows power law / log-normal:
    # - Most posts get low engagement (2-5%)
    # - Some get medium engagement (5-15%)
    # - Few go "viral" (15%+)
    #
    # We model this as: engagement_rate ~ LogNormal(mu, sigma) * feature_modifiers

    # Base log-normal engagement (log scale, will be exp'd later)
    base_log_engagement = np.random.normal(loc=1.5, scale=0.5, size=n_samples)

    # Feature multipliers (additive in log space)
    feature_boost = np.zeros(n_samples)

    # Format impact
    feature_boost += data["is_reel"] * 0.3          # Reels get ~35% more reach
    feature_boost += data["is_carousel"] * 0.15     # Carousels get ~16% boost

    # Hook impact
    feature_boost += data["hook_question"] * 0.2    # Questions boost engagement
    feature_boost += data["hook_pov"] * 0.15
    feature_boost += data["hook_number"] * 0.12
    feature_boost += data["hook_reveal"] * 0.10

    # CTA impact
    feature_boost += np.minimum(data["cta_count"], 3) * 0.08  # CTAs help, diminishing returns
    feature_boost += data["has_strong_cta"] * 0.12

    # Emoji & hashtag impact (inverted U-shape for hashtags)
    feature_boost += np.minimum(data["emoji_count"], 5) * 0.03
    hashtag_boost = -0.005 * (data["hashtag_count"] - 7) ** 2 + 0.1  # Optimal at ~7
    feature_boost += np.clip(hashtag_boost, -0.1, 0.15)

    # Timing impact
    feature_boost += data["is_prime_time"] * 0.15
    feature_boost -= data["is_weekend"] * 0.05  # Weekend typically lower for B2B

    # Sentiment impact
    feature_boost += data["sentiment_compound"] * 0.1

    # Video optimization
    feature_boost += data["video_optimal_length"] * 0.12
    feature_boost += data["is_trending_audio"] * 0.15

    # Trigger words (action and question most impactful)
    feature_boost += np.minimum(data["trigger_action"], 2) * 0.10
    feature_boost += np.minimum(data["trigger_question"], 2) * 0.08
    feature_boost += np.minimum(data["trigger_curiosity"], 2) * 0.06

    # Lexical richness (moderate is best)
    richness_boost = -2 * (data["lexical_richness"] - 0.6) ** 2 + 0.1
    feature_boost += np.clip(richness_boost, -0.05, 0.1)

    # Add noise
    noise = np.random.normal(0, 0.2, n_samples)

    # Calculate final engagement rate (log-normal)
    log_engagement = base_log_engagement + feature_boost + noise
    engagement_rate = np.exp(log_engagement)

    # Normalize to 0-100 scale (like engagement score in ml_service.py)
    engagement_rate_normalized = np.clip(engagement_rate * 10, 0, 100)

    data["engagement_rate"] = engagement_rate_normalized

    # Create DataFrame
    df = pd.DataFrame(data)

    # Remove helper columns not in feature set
    if "business_type" in df.columns:
        df = df.drop(columns=["business_type"])

    logger.info(f"Generated {len(df)} samples")
    logger.info(f"Engagement rate stats: mean={df['engagement_rate'].mean():.2f}, "
                f"std={df['engagement_rate'].std():.2f}, "
                f"min={df['engagement_rate'].min():.2f}, "
                f"max={df['engagement_rate'].max():.2f}")

    return df


# =============================================================================
# BASE MODEL TRAINING
# =============================================================================

def train_base_model(
    df: pd.DataFrame,
    feature_cols: List[str] = None,
    target_col: str = "engagement_rate"
) -> Tuple[xgb.XGBRegressor, Dict[str, Any]]:
    """
    Train the base XGBoost model with robust hyperparameters.

    Uses the same hyperparameters as the production MLPredictor for consistency.

    Args:
        df: Training DataFrame with features and target
        feature_cols: List of feature column names (default: FEATURE_COLUMNS)
        target_col: Name of target column

    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    logger.info("Training base XGBoost model...")

    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    X = df[feature_cols].fillna(0)
    y = df[target_col]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    # XGBoost with same hyperparameters as MLPredictor
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective="reg:squarederror",
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        "train_r2": float(r2_score(y_train, y_pred_train)),
        "test_r2": float(r2_score(y_test, y_pred_test)),
        "test_mae": float(mean_absolute_error(y_test, y_pred_test)),
        "n_samples": len(df),
        "n_features": len(feature_cols),
        "trained_at": datetime.now().isoformat(),
        "model_type": "base",
        "version": "1.0.0",
    }

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="neg_root_mean_squared_error")
    metrics["cv_rmse_mean"] = float(-cv_scores.mean())
    metrics["cv_rmse_std"] = float(cv_scores.std())

    # Feature importance
    feature_importance = dict(zip(feature_cols, model.feature_importances_))
    sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    metrics["feature_importance"] = {k: float(v) for k, v in list(sorted_importance.items())[:20]}

    logger.info(f"Training complete!")
    logger.info(f"  Train RMSE: {metrics['train_rmse']:.4f}")
    logger.info(f"  Test RMSE: {metrics['test_rmse']:.4f}")
    logger.info(f"  Test R2: {metrics['test_r2']:.4f}")
    logger.info(f"  CV RMSE: {metrics['cv_rmse_mean']:.4f} (+/- {metrics['cv_rmse_std']:.4f})")

    return model, metrics


def save_base_model(
    model: xgb.XGBRegressor,
    metrics: Dict[str, Any],
    output_path: Path = None
) -> Path:
    """
    Save the base model and metadata.

    Args:
        model: Trained XGBoost model
        metrics: Training metrics dictionary
        output_path: Output file path (default: models/base_xgboost.pkl)

    Returns:
        Path to saved model
    """
    if output_path is None:
        output_path = MODELS_DIR / "base_xgboost.pkl"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save model bundle
    model_bundle = {
        "model": model,
        "metrics": metrics,
        "feature_columns": FEATURE_COLUMNS,
        "version": "1.0.0",
        "created_at": datetime.now().isoformat(),
    }

    joblib.dump(model_bundle, output_path)
    logger.info(f"Base model saved to: {output_path}")

    # Also save to backend ml_models for easy access
    backend_path = BACKEND_MODELS_DIR / "base_xgboost.pkl"
    joblib.dump(model_bundle, backend_path)
    logger.info(f"Base model also saved to: {backend_path}")

    # Save metadata separately as JSON for easy inspection
    metadata_path = output_path.with_suffix(".json")
    import json
    with open(metadata_path, "w") as f:
        # Remove non-serializable items
        json_metrics = {k: v for k, v in metrics.items() if k != "feature_importance"}
        json_metrics["top_features"] = list(metrics.get("feature_importance", {}).keys())[:10]
        json.dump(json_metrics, f, indent=2)
    logger.info(f"Metadata saved to: {metadata_path}")

    return output_path


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pretrain base XGBoost model for cold start mitigation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with default 10k samples
    python ml/pretrain_base_model.py

    # Train with more samples
    python ml/pretrain_base_model.py --samples 20000

    # Custom output path
    python ml/pretrain_base_model.py --output models/my_base_model.pkl
        """
    )

    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=10000,
        help="Number of synthetic samples to generate (default: 10000)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for the model (default: models/base_xgboost.pkl)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("\n" + "=" * 70)
    print("BASE MODEL PRETRAINING - Cold Start Mitigation")
    print("=" * 70 + "\n")

    try:
        # Generate synthetic data
        df = generate_synthetic_engagement_data(n_samples=args.samples, seed=args.seed)

        # Train base model
        model, metrics = train_base_model(df)

        # Save model
        output_path = Path(args.output) if args.output else None
        saved_path = save_base_model(model, metrics, output_path)

        # Summary
        print("\n" + "=" * 70)
        print("PRETRAINING COMPLETE")
        print("=" * 70)
        print(f"Samples generated:  {args.samples}")
        print(f"Test RMSE:          {metrics['test_rmse']:.4f}")
        print(f"Test R2:            {metrics['test_r2']:.4f}")
        print(f"CV RMSE:            {metrics['cv_rmse_mean']:.4f} (+/- {metrics['cv_rmse_std']:.4f})")
        print(f"Model saved to:     {saved_path}")
        print("=" * 70)

        print("\nTop 10 Feature Importance:")
        for i, (feature, importance) in enumerate(list(metrics["feature_importance"].items())[:10], 1):
            print(f"  {i:2d}. {feature:30s} {importance:.4f}")

        print("\nBase model ready for fine-tuning on niche-specific data!")
        print("Use: python ml/train.py --niche <niche_name> --data-file <data.csv>")

        return 0

    except Exception as e:
        logger.exception(f"Pretraining failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
