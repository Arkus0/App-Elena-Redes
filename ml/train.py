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

# Import embedding feature extractor (now with configurable precision)
try:
    from ml.features_embeddings import (
        EmbeddingExtractor,
        EmbeddingPrecision,
        get_embedding_extractor,
        get_embedding_feature_names,
        EMBEDDING_DIM,
        PRECISION_TO_DIMS,
        DEFAULT_PRECISION,
    )
    EMBEDDINGS_AVAILABLE = True
    # Default: full 384 dims (safe for SMB volumes 100-2000 posts)
    EMBEDDING_FEATURE_COLUMNS = get_embedding_feature_names(EMBEDDING_DIM)
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    EMBEDDING_DIM = 384  # Full dims by default
    EMBEDDING_FEATURE_COLUMNS = [f"embedding_{i+1}" for i in range(EMBEDDING_DIM)]
    PRECISION_TO_DIMS = {"low": 128, "medium": 256, "high": 384, "max": 384}
    DEFAULT_PRECISION = "max"

# Import multimodal fusion module (now with configurable precision)
try:
    from backend.ml.multimodal_fusion import (
        fuse_multimodal_features,
        add_multimodal_features_conditional,
        get_transcript_feature_names,
        get_ocr_feature_names,
        get_interaction_feature_names,
        get_multimodal_feature_names,
        TRANSCRIPT_DEFAULT_DIM,
        OCR_DEFAULT_DIM,
    )
    MULTIMODAL_AVAILABLE = True
    # Default: full dims for all modalities
    TRANSCRIPT_FEATURE_COLUMNS = get_transcript_feature_names(TRANSCRIPT_DEFAULT_DIM)
    OCR_FEATURE_COLUMNS = get_ocr_feature_names(OCR_DEFAULT_DIM)
    INTERACTION_FEATURE_COLUMNS = get_interaction_feature_names()
    MULTIMODAL_FEATURE_COLUMNS = get_multimodal_feature_names()
except ImportError:
    MULTIMODAL_AVAILABLE = False
    TRANSCRIPT_DEFAULT_DIM = 384  # Full dims by default
    OCR_DEFAULT_DIM = 384
    TRANSCRIPT_FEATURE_COLUMNS = [f"transcript_emb_{i+1}" for i in range(384)]
    OCR_FEATURE_COLUMNS = [f"ocr_emb_{i+1}" for i in range(384)]
    INTERACTION_FEATURE_COLUMNS = [
        "interaction_hook_x_sentiment",
        "interaction_hook_x_is_reel",
        "interaction_hook_x_cta_count",
        "interaction_sentiment_x_cta_count",
        "interaction_is_reel_x_video_optimal",
        "interaction_transcript_richness",
        "interaction_ocr_richness",
        "interaction_multimodal_text_density",
        "interaction_semantic_hook_x_vader",
        "interaction_semantic_hook_x_cta_strong",
    ]
    MULTIMODAL_FEATURE_COLUMNS = TRANSCRIPT_FEATURE_COLUMNS + OCR_FEATURE_COLUMNS + INTERACTION_FEATURE_COLUMNS

# Import semantic hook detection module
try:
    from backend.ml.semantic_hooks import (
        compute_hook_score,
        get_semantic_hook_scorer,
        VIRAL_HOOKS_DATABASE,
    )
    SEMANTIC_HOOKS_AVAILABLE = True
except ImportError:
    SEMANTIC_HOOKS_AVAILABLE = False

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
# SEMANTIC HOOK FEATURES: semantic_hook_score is now the primary hook detection method
MANUAL_FEATURE_COLUMNS = [
    "caption_length", "caption_words", "caption_lines", "avg_word_length",
    "emoji_count", "emoji_density", "hashtag_count", "hashtag_density",
    "mention_count", "lexical_richness", "has_question", "has_strong_cta",
    "sentiment_compound", "sentiment_positive", "sentiment_negative", "sentiment_neutral",
    "trigger_question", "trigger_urgency", "trigger_social_proof",
    "trigger_value", "trigger_curiosity", "trigger_action",
    "trigger_emotion", "trigger_transformation",
    # RegEx hook features (auxiliary/backup)
    "hook_pov", "hook_question", "hook_number", "hook_bold_claim",
    "hook_story", "hook_how_to", "hook_reveal",
    "hook_regex_score",  # Normalized RegEx hook score (backup)
    # SEMANTIC HOOK FEATURES (primary - embedding-based detection)
    "semantic_hook_score",     # Primary hook score (0-1 continuous, SHAP-friendly)
    "semantic_hook_max_sim",   # Max cosine similarity to any base hook
    "semantic_hook_top3_avg",  # Average similarity to top 3 hooks
    # SEMANTIC HOOK INTERACTIONS
    "interaction_semantic_hook_x_vader",      # Hook * |sentiment| synergy
    "interaction_semantic_hook_x_cta_strong", # Hook * strong CTA synergy
    # CTA features
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
# With full dims (default "max" precision):
#   Total: 384 (caption) + 384 (transcript) + 384 (ocr) + 10 (interactions) + ~58 (manual) ≈ 1220 features
# XGBoost handles this efficiently with early stopping and max_depth auto-adjustment
# Safe for SMB volumes (100-2000 posts, train <1min, RAM <2GB on normal desktop)
FEATURE_COLUMNS = EMBEDDING_FEATURE_COLUMNS + MULTIMODAL_FEATURE_COLUMNS + MANUAL_FEATURE_COLUMNS

# Global precision setting (can be overridden via CLI --precision)
CURRENT_PRECISION = DEFAULT_PRECISION if EMBEDDINGS_AVAILABLE else "max"

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


def add_embedding_features(
    df: pd.DataFrame,
    caption_column: str = "caption",
    precision: str = None
) -> pd.DataFrame:
    """
    Add semantic embedding features to DataFrame.

    Generates embeddings for each caption with CONFIGURABLE precision:
    - "max" (default): Full 384 dims - mejor matices creativos/locales
    - "high": Full 384 dims
    - "medium": 256 dims via TruncatedSVD
    - "low": 128 dims via TruncatedSVD (ultra fast)

    Default is "max" (full 384 dims) - safe for SMB volumes (100-2000 posts).

    Args:
        df: DataFrame with caption column
        caption_column: Name of column containing text
        precision: Embedding precision ("low", "medium", "high", "max")
                  Default: CURRENT_PRECISION (typically "max")

    Returns:
        DataFrame with embedding features added
    """
    global CURRENT_PRECISION

    # Use specified precision or global default
    if precision is None:
        precision = CURRENT_PRECISION
    else:
        CURRENT_PRECISION = precision  # Update global for consistency

    # Get dimensions for this precision
    dims = PRECISION_TO_DIMS.get(precision.lower(), EMBEDDING_DIM)
    feature_cols = [f"embedding_{i+1}" for i in range(dims)]

    if not EMBEDDINGS_AVAILABLE:
        logger.warning("Embeddings not available. Adding zero columns.")
        for col in feature_cols:
            df[col] = 0.0
        return df

    logger.info("=" * 60)
    logger.info(f"EMBEDDING FEATURES (precision={precision}, dims={dims})")
    logger.info("=" * 60)
    logger.info(f"Generating semantic embeddings for {len(df)} samples...")

    try:
        # Get extractor with specified precision
        extractor = get_embedding_extractor(precision=precision)

        # Get captions
        captions = df[caption_column].fillna("").astype(str).tolist()

        # Fit reducer if needed (only for reduced precision)
        if extractor.uses_reduction and not extractor.is_pca_fitted:
            logger.info(f"Fitting TruncatedSVD reducer on training corpus ({dims} dims)...")
            extractor.fit_reducer_from_texts(captions, save=True)

        # Generate embedding features for all texts
        features_list = extractor.get_embedding_features_batch(captions)

        # Add to DataFrame
        for col in feature_cols:
            df[col] = [f.get(col, 0.0) for f in features_list]

        logger.info(f"Added {len(feature_cols)} embedding features (precision={precision})")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}. Using zeros.")
        for col in feature_cols:
            df[col] = 0.0

    return df


def add_semantic_hook_features(df: pd.DataFrame, caption_column: str = "caption") -> pd.DataFrame:
    """
    Add semantic hook detection features to DataFrame.

    Uses SentenceTransformer embeddings to compute cosine similarity
    against a corpus of 100+ known viral hooks.

    Args:
        df: DataFrame with caption column
        caption_column: Name of column containing text

    Returns:
        DataFrame with semantic hook features added:
        - semantic_hook_score: Primary hook score (0-1)
        - semantic_hook_max_sim: Max similarity to any base hook
        - semantic_hook_top3_avg: Average of top 3 similarities
        - hook_regex_score: Fallback RegEx-based score
    """
    if not SEMANTIC_HOOKS_AVAILABLE:
        logger.warning("Semantic hooks not available. Adding zero columns.")
        df["semantic_hook_score"] = 0.0
        df["semantic_hook_max_sim"] = 0.0
        df["semantic_hook_top3_avg"] = 0.0
        df["hook_regex_score"] = 0.0
        return df

    logger.info("=" * 60)
    logger.info("SEMANTIC HOOK DETECTION")
    logger.info("=" * 60)
    logger.info("Hook detection ahora semántica con embeddings")
    logger.info(f"Base de hooks virales: {sum(len(v) for v in VIRAL_HOOKS_DATABASE.values())} hooks en {len(VIRAL_HOOKS_DATABASE)} categorías")
    logger.info(f"Procesando {len(df)} samples...")

    try:
        scorer = get_semantic_hook_scorer()

        # Get captions
        captions = df[caption_column].fillna("").astype(str).tolist()

        # Get transcripts if available
        transcripts = None
        if "whisper_transcript" in df.columns:
            transcripts = df["whisper_transcript"].fillna("").astype(str).tolist()

        # Compute semantic hook scores for all samples
        semantic_scores = []
        max_sims = []
        top3_avgs = []

        for i, caption in enumerate(captions):
            transcript = transcripts[i] if transcripts else None
            result = compute_hook_score(
                caption=caption,
                transcript=transcript,
                return_details=True
            )
            semantic_scores.append(result.get("score", 0.0))
            max_sims.append(result.get("max_similarity", 0.0))
            top3_avgs.append(result.get("top3_avg", 0.0))

        df["semantic_hook_score"] = semantic_scores
        df["semantic_hook_max_sim"] = max_sims
        df["semantic_hook_top3_avg"] = top3_avgs

        # Calculate RegEx-based hook score as backup
        regex_hook_cols = ["hook_pov", "hook_question", "hook_number", "hook_bold_claim",
                          "hook_story", "hook_how_to", "hook_reveal"]
        existing_regex_cols = [c for c in regex_hook_cols if c in df.columns]
        if existing_regex_cols:
            df["hook_regex_score"] = df[existing_regex_cols].sum(axis=1) / 3.0
            df["hook_regex_score"] = df["hook_regex_score"].clip(0, 1)
        else:
            df["hook_regex_score"] = 0.0

        # Calculate semantic hook interactions
        sentiment = df.get("sentiment_compound", pd.Series([0.0] * len(df))).fillna(0)
        has_strong_cta = df.get("has_strong_cta", pd.Series([0] * len(df))).fillna(0)

        df["interaction_semantic_hook_x_vader"] = df["semantic_hook_score"] * abs(sentiment)
        df["interaction_semantic_hook_x_cta_strong"] = df["semantic_hook_score"] * has_strong_cta

        # Log statistics
        high_hooks = (df["semantic_hook_score"] >= 0.7).sum()
        medium_hooks = ((df["semantic_hook_score"] >= 0.5) & (df["semantic_hook_score"] < 0.7)).sum()
        low_hooks = (df["semantic_hook_score"] < 0.5).sum()

        logger.info(f"Semantic hook features added:")
        logger.info(f"  - High hooks (>=0.7): {high_hooks} ({high_hooks/len(df)*100:.1f}%)")
        logger.info(f"  - Medium hooks (0.5-0.7): {medium_hooks} ({medium_hooks/len(df)*100:.1f}%)")
        logger.info(f"  - Low hooks (<0.5): {low_hooks} ({low_hooks/len(df)*100:.1f}%)")
        logger.info(f"  - Mean semantic_hook_score: {df['semantic_hook_score'].mean():.3f}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Semantic hook detection failed: {e}. Using zeros.")
        df["semantic_hook_score"] = 0.0
        df["semantic_hook_max_sim"] = 0.0
        df["semantic_hook_top3_avg"] = 0.0
        df["hook_regex_score"] = 0.0
        df["interaction_semantic_hook_x_vader"] = 0.0
        df["interaction_semantic_hook_x_cta_strong"] = 0.0

    return df


def add_multimodal_features(df: pd.DataFrame, precision: str = None) -> pd.DataFrame:
    """
    Add multimodal fusion features for video/reel content.

    Applies late fusion combining with CONFIGURABLE precision:
    - Transcript embeddings (384 dims full, or reduced via TruncatedSVD)
    - OCR embeddings (384 dims full, or reduced via TruncatedSVD)
    - Cross-modal interaction features (10 features)

    Default is "max" (full dims) - safe for SMB volumes.

    Args:
        df: DataFrame with optional columns:
            - whisper_transcript: Audio transcription text
            - easyocr_text: Visual text overlay
            - media_type or is_reel: To detect video content
        precision: Embedding precision ("low", "medium", "high", "max")

    Returns:
        DataFrame with multimodal features added
    """
    # Use specified precision or global default
    if precision is None:
        precision = CURRENT_PRECISION

    # Get dimensions for this precision
    dims = PRECISION_TO_DIMS.get(precision.lower(), 384)
    transcript_cols = [f"transcript_emb_{i+1}" for i in range(dims)]
    ocr_cols = [f"ocr_emb_{i+1}" for i in range(dims)]
    interaction_cols = INTERACTION_FEATURE_COLUMNS
    all_multimodal_cols = transcript_cols + ocr_cols + interaction_cols

    if not MULTIMODAL_AVAILABLE:
        logger.warning("Multimodal fusion not available. Adding zero columns.")
        for col in all_multimodal_cols:
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
        logger.info(f"Applying multimodal late fusion (precision={precision}, dims={dims})...")
        logger.info(f"  - Transcript column: {has_transcript}")
        logger.info(f"  - OCR column: {has_ocr}")

        try:
            df = fuse_multimodal_features(
                df,
                fit_reducers_if_needed=True,
                save_reducers=True,
                precision=precision
            )
            logger.info(f"Multimodal features added: {len(all_multimodal_cols)} columns")
        except Exception as e:
            logger.error(f"Multimodal fusion failed: {e}. Using zeros.")
            for col in all_multimodal_cols:
                df[col] = 0.0
    else:
        # No video content or no multimodal columns - add zeros for consistency
        logger.info("No multimodal data detected. Adding zero multimodal features.")
        for col in all_multimodal_cols:
            df[col] = 0.0

    return df


def prepare_features(df: pd.DataFrame, precision: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for training.

    Args:
        df: Raw DataFrame
        precision: Embedding precision ("low", "medium", "high", "max")
                  Default: CURRENT_PRECISION (typically "max")

    Returns:
        Tuple of (features_df, target_series)
    """
    # Use specified precision or global default
    if precision is None:
        precision = CURRENT_PRECISION

    # Get dimensions for dynamic feature column generation
    dims = PRECISION_TO_DIMS.get(precision.lower(), 384)
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

    # Find caption column first (needed for embeddings and hooks)
    caption_col = None
    for col in ["caption", "text", "content", "post_caption", "description"]:
        if col in df.columns:
            caption_col = col
            break

    # Generate dynamic feature columns for this precision
    emb_cols = [f"embedding_{i+1}" for i in range(dims)]
    transcript_cols = [f"transcript_emb_{i+1}" for i in range(dims)]
    ocr_cols = [f"ocr_emb_{i+1}" for i in range(dims)]
    all_embedding_cols = emb_cols + transcript_cols + ocr_cols

    # Add embedding features if not present
    if not any(col in df.columns for col in emb_cols):
        if caption_col:
            df = add_embedding_features(df, caption_column=caption_col, precision=precision)
        else:
            logger.warning("No caption column found. Adding zero embeddings.")
            for col in emb_cols:
                df[col] = 0.0

    # Add semantic hook features (primary hook detection method)
    # This replaces naive RegEx with embedding-based similarity scoring
    if "semantic_hook_score" not in df.columns:
        if caption_col:
            df = add_semantic_hook_features(df, caption_column=caption_col)
        else:
            logger.warning("No caption column found. Adding zero semantic hook features.")
            df["semantic_hook_score"] = 0.0
            df["semantic_hook_max_sim"] = 0.0
            df["semantic_hook_top3_avg"] = 0.0
            df["hook_regex_score"] = 0.0
            df["interaction_semantic_hook_x_vader"] = 0.0
            df["interaction_semantic_hook_x_cta_strong"] = 0.0

    # Add multimodal features for video content
    if not any(col in df.columns for col in transcript_cols + ocr_cols):
        df = add_multimodal_features(df, precision=precision)

    # Build dynamic feature columns for this precision level
    dynamic_feature_cols = emb_cols + transcript_cols + ocr_cols + INTERACTION_FEATURE_COLUMNS + MANUAL_FEATURE_COLUMNS

    # Select feature columns that exist
    feature_cols = [c for c in dynamic_feature_cols if c in df.columns]
    X = df[feature_cols].fillna(0)
    y = df["engagement_rate"]

    logger.info(f"Prepared features: {X.shape[1]} columns, {X.shape[0]} samples (precision={precision})")

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
    cold_start_threshold: int = COLD_START_THRESHOLD,
    precision: str = None
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
        precision: Embedding precision ("low", "medium", "high", "max")
                  Default: "max" (full 384 dims - safe for SMB volumes)

    Returns:
        Tuple of (model, metrics)
    """
    global CURRENT_PRECISION

    # Use specified precision or global default
    if precision is None:
        precision = CURRENT_PRECISION
    else:
        CURRENT_PRECISION = precision

    # Get dims for logging
    dims = PRECISION_TO_DIMS.get(precision.lower(), 384)

    n_samples = len(df)
    logger.info(f"\nTraining model for niche: {niche}")
    logger.info(f"Available samples: {n_samples}")
    logger.info(f"Embedding precision: {precision} ({dims} dims)")

    # Validate minimum samples
    if n_samples < MIN_SAMPLES_FOR_TRAINING:
        raise ValueError(
            f"Insufficient data for niche '{niche}': {n_samples} samples "
            f"(minimum: {MIN_SAMPLES_FOR_TRAINING}). Use base model directly for inference."
        )

    # Prepare features with specified precision
    X, y = prepare_features(df, precision=precision)

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
    metrics["embedding_precision"] = precision
    metrics["embedding_dims"] = dims
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
    parser.add_argument(
        "--precision",
        type=str,
        default="max",
        choices=["low", "medium", "high", "max"],
        help="Embedding precision: low (128), medium (256), high (384), max (384 raw). "
             "Default: max - full dims, safe for SMB volumes (100-2000 posts, train <1min, RAM <2GB)"
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
    precision = args.precision

    # Get dims for display
    precision_dims = PRECISION_TO_DIMS.get(precision.lower(), 384)

    print("\n" + "=" * 70)
    print(f"NICHE MODEL TRAINING - {args.niche.upper()}")
    print(f"Embedding precision: {precision} ({precision_dims} dims)")
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
            cold_start_threshold=cold_start_threshold,
            precision=precision
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
        print(f"Embedding precision:{metrics.get('embedding_precision', 'max')} ({metrics.get('embedding_dims', 384)} dims)")
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
