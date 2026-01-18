#!/usr/bin/env python3
"""
Niche-Specific Model Training with Cold Start Handling
=======================================================

This script trains engagement prediction models for specific business niches,
with intelligent handling of the cold start problem.

COLD START STRATEGY:
====================
- If niche has >= 300 samples: Train from scratch (full training)
- If niche has < 300 samples: Load base model + fine-tune with low learning rate

Fine-tuning uses:
- 10-20 boosting rounds (not full 100)
- Low learning rate (0.01 instead of 0.1)
- Early stopping to prevent overfitting

Usage:
    # Train model for a specific niche
    python ml/train.py --niche restaurante --data-file data/restaurante_posts.csv

    # Train with auto-detection of fine-tune vs full training
    python ml/train.py --niche cafeteria --from-db

    # Force fine-tuning even with enough data (for testing)
    python ml/train.py --niche gimnasio --data-file data.csv --force-finetune

    # Train all niches from database
    python ml/train.py --all-niches --from-db

Author: BrandPulse AI
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

# Import embedding feature extractor
try:
    from ml.features_embeddings import (
        EmbeddingExtractor,
        get_embedding_extractor,
        EMBEDDING_FEATURE_COLUMNS,
        PCA_COMPONENTS
    )
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    EMBEDDING_FEATURE_COLUMNS = [f"embedding_{i+1}" for i in range(30)]
    PCA_COMPONENTS = 30

# Import multimodal fusion module
try:
    from backend.ml.multimodal_fusion import (
        fuse_multimodal_features,
        add_multimodal_features_conditional,
        TRANSCRIPT_FEATURE_COLUMNS,
        OCR_FEATURE_COLUMNS,
        INTERACTION_FEATURE_COLUMNS,
        MULTIMODAL_FEATURE_COLUMNS,
    )
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False
    TRANSCRIPT_FEATURE_COLUMNS = [f"transcript_emb_{i+1}" for i in range(20)]
    OCR_FEATURE_COLUMNS = [f"ocr_emb_{i+1}" for i in range(20)]
    INTERACTION_FEATURE_COLUMNS = [
        "interaction_hook_x_sentiment",
        "interaction_hook_x_is_reel",
        "interaction_hook_x_cta_count",
        "interaction_sentiment_x_cta_count",
        "interaction_is_reel_x_video_optimal",
        "interaction_transcript_richness",
        "interaction_ocr_richness",
        "interaction_multimodal_text_density",
    ]
    MULTIMODAL_FEATURE_COLUMNS = TRANSCRIPT_FEATURE_COLUMNS + OCR_FEATURE_COLUMNS + INTERACTION_FEATURE_COLUMNS

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
BASE_MODEL_PATH = MODELS_DIR / "base_xgboost.pkl"
BACKEND_BASE_MODEL_PATH = BACKEND_MODELS_DIR / "base_xgboost.pkl"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
BACKEND_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Training thresholds
COLD_START_THRESHOLD = 300  # Samples below this trigger fine-tuning
MIN_SAMPLES_FOR_TRAINING = 30  # Absolute minimum for any training

# Fine-tuning hyperparameters
FINETUNE_N_ESTIMATORS = 20  # Few additional rounds
FINETUNE_LEARNING_RATE = 0.01  # Low learning rate for gentle updates
FINETUNE_EARLY_STOPPING_ROUNDS = 5  # Stop if no improvement

# Full training hyperparameters (same as base model)
FULL_TRAIN_N_ESTIMATORS = 100
FULL_TRAIN_LEARNING_RATE = 0.1

# Manual heuristic features (kept as backup)
MANUAL_FEATURE_COLUMNS = [
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

# Combined feature columns: embeddings + multimodal + manual heuristics
# Total: 30 (caption) + 20 (transcript) + 20 (ocr) + 8 (interactions) + ~58 (manual) ≈ 136 features
FEATURE_COLUMNS = EMBEDDING_FEATURE_COLUMNS + MULTIMODAL_FEATURE_COLUMNS + MANUAL_FEATURE_COLUMNS

BUSINESS_TYPES = [
    "inmobiliaria", "floristeria", "cafeteria", "peluqueria",
    "restaurante", "gimnasio", "clinica", "otros"
]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data_from_csv(file_path: str, niche: str = None) -> pd.DataFrame:
    """
    Load training data from CSV file.

    Args:
        file_path: Path to CSV file
        niche: Optional niche to filter by

    Returns:
        DataFrame with training data
    """
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)

    # Filter by niche if specified
    if niche and "business_type" in df.columns:
        df = df[df["business_type"] == niche]
        logger.info(f"Filtered to {len(df)} samples for niche: {niche}")
    elif niche and f"niche_{niche}" in df.columns:
        df = df[df[f"niche_{niche}"] == 1]
        logger.info(f"Filtered to {len(df)} samples for niche: {niche}")

    return df


def load_data_from_database(niche: str = None) -> pd.DataFrame:
    """
    Load training data from database.

    Args:
        niche: Optional niche to filter by

    Returns:
        DataFrame with training data
    """
    logger.info(f"Loading data from database for niche: {niche or 'all'}")

    # TODO: Implement actual database loading
    # For now, return empty DataFrame as placeholder
    logger.warning("Database connection not configured. Please provide CSV data.")
    return pd.DataFrame()


def add_embedding_features(df: pd.DataFrame, caption_column: str = "caption") -> pd.DataFrame:
    """
    Add semantic embedding features to DataFrame.

    Generates embeddings for each caption and adds embedding_1 to embedding_30 columns.
    Also fits PCA on the corpus if not already fitted.

    Args:
        df: DataFrame with caption column
        caption_column: Name of column containing text

    Returns:
        DataFrame with embedding features added
    """
    if not EMBEDDINGS_AVAILABLE:
        logger.warning("Embeddings not available. Adding zero columns.")
        for col in EMBEDDING_FEATURE_COLUMNS:
            df[col] = 0.0
        return df

    logger.info(f"Generating semantic embeddings for {len(df)} samples...")

    try:
        extractor = get_embedding_extractor()

        # Get captions
        captions = df[caption_column].fillna("").astype(str).tolist()

        # Fit PCA on this corpus if not already fitted
        if not extractor.is_pca_fitted:
            logger.info("Fitting PCA on training corpus...")
            extractor.fit_pca_from_texts(captions, save=True)

        # Generate embedding features for all texts
        features_list = extractor.get_embedding_features_batch(captions)

        # Add to DataFrame
        for i, col in enumerate(EMBEDDING_FEATURE_COLUMNS):
            df[col] = [f.get(col, 0.0) for f in features_list]

        logger.info(f"Added {len(EMBEDDING_FEATURE_COLUMNS)} embedding features")

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}. Using zeros.")
        for col in EMBEDDING_FEATURE_COLUMNS:
            df[col] = 0.0

    return df


def add_multimodal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add multimodal fusion features for video/reel content.

    Applies late fusion combining:
    - Transcript embeddings (20 dims from Whisper transcription)
    - OCR embeddings (20 dims from EasyOCR text detection)
    - Cross-modal interaction features (8 features)

    Args:
        df: DataFrame with optional columns:
            - whisper_transcript: Audio transcription text
            - easyocr_text: Visual text overlay
            - media_type or is_reel: To detect video content

    Returns:
        DataFrame with multimodal features added
    """
    if not MULTIMODAL_AVAILABLE:
        logger.warning("Multimodal fusion not available. Adding zero columns.")
        for col in MULTIMODAL_FEATURE_COLUMNS:
            df[col] = 0.0
        return df

    # Check if we have video content
    has_video = False
    if 'media_type' in df.columns:
        has_video = df['media_type'].str.lower().isin(['reel', 'video', 'tiktok']).any()
    elif 'is_reel' in df.columns:
        has_video = df['is_reel'].sum() > 0

    # Check if we have multimodal text columns
    has_transcript = 'whisper_transcript' in df.columns
    has_ocr = 'easyocr_text' in df.columns

    if has_video and (has_transcript or has_ocr):
        logger.info("Applying multimodal late fusion...")
        logger.info(f"  - Transcript column: {has_transcript}")
        logger.info(f"  - OCR column: {has_ocr}")

        try:
            df = fuse_multimodal_features(df, fit_pca_if_needed=True, save_pca=True)
            logger.info(f"Multimodal features added: {len(MULTIMODAL_FEATURE_COLUMNS)} columns")
        except Exception as e:
            logger.error(f"Multimodal fusion failed: {e}. Using zeros.")
            for col in MULTIMODAL_FEATURE_COLUMNS:
                df[col] = 0.0
    else:
        # No video content or no multimodal columns - add zeros for consistency
        logger.info("No multimodal data detected. Adding zero multimodal features.")
        for col in MULTIMODAL_FEATURE_COLUMNS:
            df[col] = 0.0

    return df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for training.

    Args:
        df: Raw DataFrame

    Returns:
        Tuple of (features_df, target_series)
    """
    # Ensure we have the target column
    if "engagement_rate" not in df.columns:
        if "engagement_score" in df.columns:
            df["engagement_rate"] = df["engagement_score"]
        elif all(col in df.columns for col in ["likes", "comments", "saves", "shares"]):
            # Calculate engagement score
            df["engagement_rate"] = (
                df["likes"] +
                df["comments"] * 3 +
                df["saves"] * 5 +
                df["shares"] * 4
            )
            # Normalize to 0-100
            max_eng = df["engagement_rate"].quantile(0.95)
            df["engagement_rate"] = (df["engagement_rate"] / max(max_eng, 1) * 100).clip(0, 100)
        else:
            raise ValueError("Cannot find or calculate engagement_rate target")

    # Encode business type if present
    if "business_type" in df.columns:
        encoder = LabelEncoder()
        encoder.fit(BUSINESS_TYPES)
        known_types = set(encoder.classes_)
        df["business_type_encoded"] = df["business_type"].apply(
            lambda x: encoder.transform([x])[0] if x in known_types else len(BUSINESS_TYPES) - 1
        )

    # Add embedding features if not present
    if not any(col in df.columns for col in EMBEDDING_FEATURE_COLUMNS):
        # Find caption column
        caption_col = None
        for col in ["caption", "text", "content", "post_caption", "description"]:
            if col in df.columns:
                caption_col = col
                break

        if caption_col:
            df = add_embedding_features(df, caption_column=caption_col)
        else:
            logger.warning("No caption column found. Adding zero embeddings.")
            for col in EMBEDDING_FEATURE_COLUMNS:
                df[col] = 0.0

    # Add multimodal features for video content
    if not any(col in df.columns for col in MULTIMODAL_FEATURE_COLUMNS):
        df = add_multimodal_features(df)

    # Select feature columns that exist
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[feature_cols].fillna(0)
    y = df["engagement_rate"]

    logger.info(f"Prepared features: {X.shape[1]} columns, {X.shape[0]} samples")

    return X, y


# =============================================================================
# BASE MODEL LOADING
# =============================================================================

def load_base_model() -> Optional[Dict[str, Any]]:
    """
    Load the pretrained base model.

    Returns:
        Model bundle dict or None if not found
    """
    # Try primary location
    if BASE_MODEL_PATH.exists():
        logger.info(f"Loading base model from: {BASE_MODEL_PATH}")
        return joblib.load(BASE_MODEL_PATH)

    # Try backend location
    if BACKEND_BASE_MODEL_PATH.exists():
        logger.info(f"Loading base model from: {BACKEND_BASE_MODEL_PATH}")
        return joblib.load(BACKEND_BASE_MODEL_PATH)

    logger.warning("Base model not found. Run pretrain_base_model.py first.")
    return None


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_from_scratch(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[xgb.XGBRegressor, Dict[str, Any]]:
    """
    Train a model from scratch with full hyperparameters.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data

    Returns:
        Tuple of (model, metrics)
    """
    logger.info("Training model from scratch (full training)...")
    logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Hyperparameters adjusted for multimodal features (~136 total features)
    # max_depth increased from 6 to 7 to capture cross-modal interactions
    model = xgb.XGBRegressor(
        n_estimators=FULL_TRAIN_N_ESTIMATORS,
        max_depth=7,  # Increased for multimodal feature interactions
        learning_rate=FULL_TRAIN_LEARNING_RATE,
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

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Evaluate
    y_pred = model.predict(X_test)
    metrics = {
        "training_type": "from_scratch",
        "n_estimators": FULL_TRAIN_N_ESTIMATORS,
        "learning_rate": FULL_TRAIN_LEARNING_RATE,
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "test_r2": float(r2_score(y_test, y_pred)),
        "test_mae": float(mean_absolute_error(y_test, y_pred)),
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
    }

    logger.info(f"From-scratch training complete - RMSE: {metrics['test_rmse']:.4f}, R2: {metrics['test_r2']:.4f}")

    return model, metrics


def finetune_base_model(
    base_model: xgb.XGBRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_additional_rounds: int = FINETUNE_N_ESTIMATORS
) -> Tuple[xgb.XGBRegressor, Dict[str, Any]]:
    """
    Fine-tune the base model with niche-specific data.

    Uses low learning rate and few additional boosting rounds
    to gently adapt the base model without overfitting.

    Args:
        base_model: Pretrained base XGBoost model
        X_train, y_train: Niche training data
        X_test, y_test: Niche test data
        n_additional_rounds: Number of additional boosting rounds

    Returns:
        Tuple of (finetuned_model, metrics)
    """
    logger.info("=" * 60)
    logger.info("COLD START: Usando modelo base + fine-tuning")
    logger.info("=" * 60)
    logger.info(f"Fine-tuning with {len(X_train)} samples (low learning rate: {FINETUNE_LEARNING_RATE})")

    # Ensure feature alignment with base model
    base_features = base_model.get_booster().feature_names
    if base_features:
        # Align features - add missing ones as zeros
        for feat in base_features:
            if feat not in X_train.columns:
                X_train[feat] = 0
                X_test[feat] = 0
        X_train = X_train[base_features]
        X_test = X_test[base_features]

    # XGBoost fine-tuning: continue training from existing model
    # We create a new model with same params but add trees to the existing ones
    finetuned_model = xgb.XGBRegressor(
        n_estimators=base_model.n_estimators + n_additional_rounds,
        max_depth=base_model.max_depth,
        learning_rate=FINETUNE_LEARNING_RATE,  # Low LR for fine-tuning
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

    # Continue training from base model
    finetuned_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        xgb_model=base_model.get_booster(),  # Start from base model
        verbose=False
    )

    # Evaluate
    y_pred_base = base_model.predict(X_test)
    y_pred_finetuned = finetuned_model.predict(X_test)

    base_rmse = np.sqrt(mean_squared_error(y_test, y_pred_base))
    finetuned_rmse = np.sqrt(mean_squared_error(y_test, y_pred_finetuned))
    improvement = (base_rmse - finetuned_rmse) / base_rmse * 100

    metrics = {
        "training_type": "fine_tuned",
        "base_model_rmse": float(base_rmse),
        "finetuned_rmse": float(finetuned_rmse),
        "improvement_percent": float(improvement),
        "n_additional_rounds": n_additional_rounds,
        "learning_rate": FINETUNE_LEARNING_RATE,
        "test_rmse": float(finetuned_rmse),
        "test_r2": float(r2_score(y_test, y_pred_finetuned)),
        "test_mae": float(mean_absolute_error(y_test, y_pred_finetuned)),
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
    }

    logger.info(f"Fine-tuning complete!")
    logger.info(f"  Base model RMSE:      {base_rmse:.4f}")
    logger.info(f"  Fine-tuned RMSE:      {finetuned_rmse:.4f}")
    logger.info(f"  Improvement:          {improvement:+.2f}%")

    return finetuned_model, metrics


def train_niche_model(
    niche: str,
    df: pd.DataFrame,
    force_finetune: bool = False,
    force_scratch: bool = False,
    cold_start_threshold: int = COLD_START_THRESHOLD
) -> Tuple[xgb.XGBRegressor, Dict[str, Any]]:
    """
    Train a model for a specific niche with automatic cold start handling.

    Decision logic:
    - If < MIN_SAMPLES_FOR_TRAINING (30): Fail, use base model directly
    - If < cold_start_threshold (default 300): Fine-tune base model
    - If >= cold_start_threshold: Train from scratch

    Args:
        niche: Business niche name
        df: Training DataFrame for this niche
        force_finetune: Force fine-tuning even with enough data
        force_scratch: Force from-scratch training even with little data
        cold_start_threshold: Threshold below which to use fine-tuning

    Returns:
        Tuple of (model, metrics)
    """
    n_samples = len(df)
    logger.info(f"\nTraining model for niche: {niche}")
    logger.info(f"Available samples: {n_samples}")

    # Validate minimum samples
    if n_samples < MIN_SAMPLES_FOR_TRAINING:
        raise ValueError(
            f"Insufficient data for niche '{niche}': {n_samples} samples "
            f"(minimum: {MIN_SAMPLES_FOR_TRAINING}). Use base model directly for inference."
        )

    # Prepare features
    X, y = prepare_features(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Decide training strategy
    use_finetune = (n_samples < cold_start_threshold or force_finetune) and not force_scratch

    if use_finetune:
        # Cold start: fine-tune base model
        base_bundle = load_base_model()
        if base_bundle is None:
            logger.warning("Base model not found. Falling back to from-scratch training.")
            use_finetune = False
        else:
            base_model = base_bundle["model"]
            model, metrics = finetune_base_model(
                base_model, X_train, y_train, X_test, y_test
            )

    if not use_finetune:
        # Full training from scratch
        model, metrics = train_from_scratch(X_train, y_train, X_test, y_test)

    # Add niche info to metrics
    metrics["niche"] = niche
    metrics["n_samples"] = n_samples
    metrics["cold_start"] = use_finetune
    metrics["trained_at"] = datetime.now().isoformat()

    return model, metrics


# =============================================================================
# MODEL SAVING
# =============================================================================

def save_niche_model(
    model: xgb.XGBRegressor,
    metrics: Dict[str, Any],
    niche: str
) -> Tuple[Path, Path]:
    """
    Save niche-specific model to both models/ and backend/ml_models/.

    Args:
        model: Trained model
        metrics: Training metrics
        niche: Niche name

    Returns:
        Tuple of (primary_path, backend_path)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Model bundle
    bundle = {
        "model": model,
        "metrics": metrics,
        "niche": niche,
        "feature_columns": FEATURE_COLUMNS,
        "created_at": datetime.now().isoformat(),
        "version": "1.0.0",
    }

    # Save to primary location with versioning
    primary_path = MODELS_DIR / f"niche_{niche}_{timestamp}.pkl"
    joblib.dump(bundle, primary_path)
    logger.info(f"Saved versioned model to: {primary_path}")

    # Save to backend as "current" model for this niche
    backend_path = BACKEND_MODELS_DIR / f"niche_{niche}.pkl"
    joblib.dump(bundle, backend_path)
    logger.info(f"Saved current model to: {backend_path}")

    # Save metadata
    import json
    metadata = {k: v for k, v in metrics.items() if not isinstance(v, (np.ndarray, pd.DataFrame))}
    metadata_path = backend_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return primary_path, backend_path


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train niche-specific engagement models with cold start handling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train for a specific niche from CSV
    python ml/train.py --niche restaurante --data-file data/posts.csv

    # Force fine-tuning for testing
    python ml/train.py --niche cafeteria --data-file data.csv --force-finetune

    # Train from database
    python ml/train.py --niche gimnasio --from-db
        """
    )

    # Required arguments
    parser.add_argument(
        "--niche",
        type=str,
        required=True,
        choices=BUSINESS_TYPES,
        help="Business niche to train model for"
    )

    # Data source
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--data-file",
        type=str,
        help="Path to CSV file with training data"
    )
    data_group.add_argument(
        "--from-db",
        action="store_true",
        help="Load training data from database"
    )

    # Training options
    parser.add_argument(
        "--force-finetune",
        action="store_true",
        help="Force fine-tuning even with sufficient data"
    )
    parser.add_argument(
        "--force-scratch",
        action="store_true",
        help="Force from-scratch training even with little data"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=COLD_START_THRESHOLD,
        help=f"Cold start threshold (default: {COLD_START_THRESHOLD})"
    )

    # Output options
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save the trained model"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Use threshold from args (allows override via command line)
    cold_start_threshold = args.threshold

    print("\n" + "=" * 70)
    print(f"NICHE MODEL TRAINING - {args.niche.upper()}")
    print("=" * 70 + "\n")

    try:
        # Load data
        if args.data_file:
            df = load_data_from_csv(args.data_file, args.niche)
        else:
            df = load_data_from_database(args.niche)

        if len(df) == 0:
            logger.error("No data loaded. Please provide valid training data.")
            return 1

        # Train model
        model, metrics = train_niche_model(
            niche=args.niche,
            df=df,
            force_finetune=args.force_finetune,
            force_scratch=args.force_scratch,
            cold_start_threshold=cold_start_threshold
        )

        # Save model
        if not args.no_save:
            primary_path, backend_path = save_niche_model(model, metrics, args.niche)

        # Summary
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Niche:              {args.niche}")
        print(f"Training type:      {metrics['training_type']}")
        print(f"Samples used:       {metrics['n_samples']}")
        print(f"Cold start mode:    {'Yes (fine-tuned)' if metrics.get('cold_start') else 'No (from scratch)'}")
        print(f"Test RMSE:          {metrics['test_rmse']:.4f}")
        print(f"Test R2:            {metrics['test_r2']:.4f}")
        if metrics.get('improvement_percent'):
            print(f"Improvement vs base: {metrics['improvement_percent']:+.2f}%")
        if not args.no_save:
            print(f"Model saved to:     {backend_path}")
        print("=" * 70)

        if metrics.get("cold_start"):
            print("\nNOTE: Model was fine-tuned from base model due to cold start.")
            print(f"      When you have >= {cold_start_threshold} samples, retrain for better accuracy.")

        return 0

    except ValueError as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
