#!/usr/bin/env python3
"""
Base Model Pretraining Script for Cold Start Mitigation
=========================================================

This script trains a robust base XGBoost model using either:
1. A real dataset from Kaggle (recommended if available)
2. Synthetic data simulating realistic Instagram engagement patterns

The base model serves as a foundation for fine-tuning on niche-specific data,
solving the cold start problem when a new niche has < 300 samples.

DATA SOURCES:
=============
1. Kaggle Datasets (recommended):
   - Instagram Reach: kaggle.com/datasets/rxsraghavagrawal/instagram-reach
   - Instagram Analytics: kaggle.com/datasets/kundanbedmutha/instagram-analytics-dataset
   - Social Media Engagement: kaggle.com/datasets/purnisharma/social-media-engagement-metrics

2. Synthetic Data (fallback):
   - 10,000 samples with realistic feature distributions
   - Target: engagement_rate following log-normal distribution

APPROACH:
=========
1. Load data from CSV (Kaggle) or generate synthetic samples
2. Apply feature engineering to extract missing features from captions
3. Train XGBoostRegressor with same hyperparameters as production model
4. Save as /models/base_xgboost.pkl for use in fine-tuning

Usage:
    # With Kaggle data (recommended)
    python ml/pretrain_base_model.py --data-file data/instagram_reach.csv

    # With synthetic data (fallback)
    python ml/pretrain_base_model.py --synthetic --samples 10000

    # Mix: Kaggle + synthetic augmentation
    python ml/pretrain_base_model.py --data-file data/kaggle.csv --augment 5000

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

# Combined feature columns: semantic embeddings (prioritized) + manual heuristics (backup)
FEATURE_COLUMNS = EMBEDDING_FEATURE_COLUMNS + MANUAL_FEATURE_COLUMNS

BUSINESS_TYPES = [
    "inmobiliaria", "floristeria", "cafeteria", "peluqueria",
    "restaurante", "gimnasio", "clinica", "otros"
]

CONTENT_FORMATS = ["reel", "carousel", "static"]

# Regex patterns for feature extraction from captions
import re

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE
)

QUESTION_PATTERN = re.compile(r'\?|¿|cuál|qué|cómo|por qué|quién|dónde|cuándo|what|how|why|who|where|when', re.IGNORECASE)

STRONG_CTA_KEYWORDS = ["comenta", "guarda", "dm", "visita", "taggea", "etiqueta", "escríbeme",
                       "comment", "save", "tag", "follow", "share", "click", "link"]

TRIGGER_PATTERNS = {
    "question": r"\?|¿|cuál|qué|cómo|what|how|why",
    "urgency": r"ahora|hoy|último|limitado|exclusivo|now|today|last|limited",
    "social_proof": r"clientes|testimonios|opiniones|reviews|people|customers",
    "value": r"gratis|regalo|descuento|oferta|free|gift|discount|offer",
    "curiosity": r"secreto|descubre|sorpresa|increíble|secret|discover|amazing",
    "action": r"comenta|guarda|comparte|sígueme|comment|save|share|follow",
    "emotion": r"amor|feliz|alegría|pasión|love|happy|joy|passion",
    "transformation": r"antes|después|transformación|cambio|before|after|transform",
}

HOOK_PATTERNS = {
    "pov": r"pov[:\s]|punto de vista|point of view",
    "question": r"^[¿?]|^\w+\s*\?",
    "number": r"^\d+\s+\w+|top\s*\d+|\d+\s*(things|tips|ways|cosas|tips|formas)",
    "bold_claim": r"nunca|siempre|todo|nadie|el mejor|never|always|best|worst",
    "story": r"historia|storytime|cuando|un día|story|once",
    "how_to": r"cómo\s+\w+|aprende\s+a|tutorial|how\s+to|learn",
    "reveal": r"secreto|te cuento|descubre|reveal|secret|discover",
}


# =============================================================================
# KAGGLE DATA LOADING AND FEATURE ENGINEERING
# =============================================================================

def load_kaggle_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a Kaggle CSV file and apply feature engineering.

    Supports common Kaggle Instagram datasets:
    - Instagram Reach (rxsraghavagrawal)
    - Instagram Analytics (kundanbedmutha)
    - Social Media Engagement (purnisharma)

    Args:
        file_path: Path to CSV file

    Returns:
        DataFrame with engineered features
    """
    logger.info(f"Loading Kaggle data from: {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")

    # Normalize column names
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')

    # Calculate engagement rate if not present
    df = _calculate_engagement_rate(df)

    # Apply feature engineering to extract missing features
    df = _engineer_features_from_caption(df)

    # Add missing features with defaults/random values
    df = _add_missing_features(df)

    logger.info(f"After feature engineering: {len(df)} samples with {len(df.columns)} features")
    return df


def _calculate_engagement_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate engagement_rate from available metrics."""
    if 'engagement_rate' in df.columns:
        return df

    # Try different column name variations
    likes = df.get('likes', df.get('like_count', df.get('likescount', pd.Series([0] * len(df)))))
    comments = df.get('comments', df.get('comment_count', df.get('commentscount', pd.Series([0] * len(df)))))
    saves = df.get('saves', df.get('save_count', pd.Series([0] * len(df))))
    shares = df.get('shares', df.get('share_count', pd.Series([0] * len(df))))

    # Weighted engagement score
    raw_engagement = likes + (comments * 3) + (saves * 5) + (shares * 4)

    # Normalize to 0-100 scale
    max_eng = raw_engagement.quantile(0.95)
    df['engagement_rate'] = (raw_engagement / max(max_eng, 1) * 100).clip(0, 100)

    logger.info(f"Calculated engagement_rate: mean={df['engagement_rate'].mean():.2f}")
    return df


def _engineer_features_from_caption(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features from caption text using regex patterns."""

    # Find caption column
    caption_col = None
    for col in ['caption', 'text', 'content', 'post_caption', 'description']:
        if col in df.columns:
            caption_col = col
            break

    if caption_col is None:
        logger.warning("No caption column found. Using empty strings.")
        df['caption'] = ''
        caption_col = 'caption'

    captions = df[caption_col].fillna('').astype(str)

    # Basic text features
    df['caption_length'] = captions.str.len()
    df['caption_words'] = captions.str.split().str.len().fillna(0)
    df['caption_lines'] = captions.str.count('\n') + 1

    words_list = captions.str.split()
    df['avg_word_length'] = words_list.apply(
        lambda x: np.mean([len(w) for w in x]) if x and len(x) > 0 else 0
    )

    # Lexical richness
    df['lexical_richness'] = words_list.apply(
        lambda x: len(set(x)) / len(x) if x and len(x) > 0 else 0
    )

    # Emoji features
    df['emoji_count'] = captions.apply(lambda x: len(EMOJI_PATTERN.findall(x)))
    df['emoji_density'] = df['emoji_count'] / df['caption_length'].clip(lower=1) * 100

    # Hashtag features (from caption or separate column)
    if 'hashtags' in df.columns:
        hashtag_col = df['hashtags'].fillna('').astype(str)
        df['hashtag_count'] = hashtag_col.str.count('#') + hashtag_col.str.count(',') + 1
        df.loc[hashtag_col == '', 'hashtag_count'] = 0
    else:
        df['hashtag_count'] = captions.str.count('#')

    df['hashtag_density'] = df['hashtag_count'] / df['caption_words'].clip(lower=1)

    # Mention features
    df['mention_count'] = captions.str.count('@')

    # Question detection
    df['has_question'] = captions.apply(lambda x: 1 if QUESTION_PATTERN.search(x) else 0)

    # Strong CTA detection
    df['has_strong_cta'] = captions.apply(
        lambda x: 1 if any(kw in x.lower() for kw in STRONG_CTA_KEYWORDS) else 0
    )

    # Trigger word features
    for trigger_type, pattern in TRIGGER_PATTERNS.items():
        df[f'trigger_{trigger_type}'] = captions.apply(
            lambda x: len(re.findall(pattern, x.lower(), re.IGNORECASE))
        ).clip(upper=5)

    # Hook detection (from first line)
    first_lines = captions.str.split('\n').str[0].fillna('')
    for hook_type, pattern in HOOK_PATTERNS.items():
        df[f'hook_{hook_type}'] = first_lines.apply(
            lambda x: 1 if re.search(pattern, x.lower(), re.IGNORECASE) else 0
        )

    # CTA detection
    cta_patterns = {
        "comment": r"comenta|cuéntame|opina|comment|tell",
        "save": r"guarda|guardar|save",
        "share": r"comparte|compartir|share|tag",
        "follow": r"sígueme|sigue|follow",
        "dm": r"dm|mensaje|escríbeme|message",
        "link": r"link|enlace|bio|click",
    }
    for cta_type, pattern in cta_patterns.items():
        df[f'cta_{cta_type}'] = captions.apply(
            lambda x: 1 if re.search(pattern, x.lower(), re.IGNORECASE) else 0
        )
    df['cta_count'] = sum(df[f'cta_{t}'] for t in cta_patterns.keys())

    logger.info("Feature engineering from captions complete")
    return df


def _add_embedding_features_from_captions(df: pd.DataFrame, caption_col: str = "caption") -> pd.DataFrame:
    """
    Generate semantic embedding features from caption text.

    If embeddings are available, generates real embeddings.
    Otherwise, generates synthetic random embeddings.

    Args:
        df: DataFrame with caption column
        caption_col: Name of caption column

    Returns:
        DataFrame with embedding_1 to embedding_30 columns added
    """
    n = len(df)

    if EMBEDDINGS_AVAILABLE and caption_col in df.columns:
        logger.info(f"Generating semantic embeddings for {n} samples...")
        try:
            extractor = get_embedding_extractor()

            # Get captions
            captions = df[caption_col].fillna("").astype(str).tolist()

            # Fit PCA if not fitted
            if not extractor.is_pca_fitted:
                logger.info("Fitting PCA on corpus...")
                extractor.fit_pca_from_texts(captions, save=True)

            # Generate features
            features_list = extractor.get_embedding_features_batch(captions)

            # Add to DataFrame
            for col in EMBEDDING_FEATURE_COLUMNS:
                df[col] = [f.get(col, 0.0) for f in features_list]

            logger.info(f"Added {len(EMBEDDING_FEATURE_COLUMNS)} real embedding features")
            return df

        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}. Using synthetic embeddings.")

    # Fallback: Generate synthetic embedding-like features
    # These simulate the statistical properties of real embeddings
    logger.info("Generating synthetic embedding features...")

    np.random.seed(42)

    # Synthetic embeddings: random normal values with gradual variance decay
    # (higher components explain less variance in real PCA)
    for i, col in enumerate(EMBEDDING_FEATURE_COLUMNS):
        # Variance decreases for higher components (like real PCA)
        variance = 1.0 / (1 + i * 0.1)
        df[col] = np.random.normal(0, np.sqrt(variance), n)

    logger.info(f"Added {len(EMBEDDING_FEATURE_COLUMNS)} synthetic embedding features")
    return df


def _add_missing_features(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Add missing features with realistic random values."""
    np.random.seed(seed)
    n = len(df)

    # Sentiment features (random since we can't compute without NLTK)
    if 'sentiment_compound' not in df.columns:
        df['sentiment_compound'] = np.clip(np.random.beta(7, 3, n) * 2 - 0.5, -1, 1)
        df['sentiment_positive'] = np.clip(np.random.beta(5, 2, n), 0, 1)
        df['sentiment_negative'] = np.clip(np.random.beta(1, 10, n), 0, 0.3)
        df['sentiment_neutral'] = np.clip(1 - df['sentiment_positive'] - df['sentiment_negative'], 0, 1)

    # Format features (assume all are static if not specified)
    if 'is_reel' not in df.columns:
        # Random format distribution: 55% reel, 25% carousel, 20% static
        formats = np.random.choice(['reel', 'carousel', 'static'], size=n, p=[0.55, 0.25, 0.20])
        df['is_reel'] = (formats == 'reel').astype(int)
        df['is_carousel'] = (formats == 'carousel').astype(int)
        df['is_static'] = (formats == 'static').astype(int)

    # Video features
    if 'video_duration' not in df.columns:
        df['video_duration'] = np.where(
            df['is_reel'] == 1,
            np.clip(np.random.lognormal(3.5, 0.4, n), 5, 180),
            0
        )
    df['video_optimal_length'] = ((df['video_duration'] >= 15) & (df['video_duration'] <= 60)).astype(int)

    # Audio features
    if 'has_audio' not in df.columns:
        df['has_audio'] = np.where(df['is_reel'] == 1, 1, np.random.choice([0, 1], n, p=[0.7, 0.3]))
    if 'is_trending_audio' not in df.columns:
        df['is_trending_audio'] = np.where(df['has_audio'] == 1, np.random.choice([0, 1], n, p=[0.6, 0.4]), 0)

    # Timing features (random if not present)
    if 'hour_of_day' not in df.columns:
        hour_probs = np.array([0.01, 0.005, 0.005, 0.005, 0.005, 0.01,
                               0.02, 0.03, 0.05, 0.07, 0.08, 0.10,
                               0.09, 0.07, 0.05, 0.04, 0.05, 0.06,
                               0.08, 0.10, 0.09, 0.06, 0.03, 0.02])
        hour_probs = hour_probs / hour_probs.sum()
        df['hour_of_day'] = np.random.choice(range(24), n, p=hour_probs)

    if 'day_of_week' not in df.columns:
        df['day_of_week'] = np.random.choice(range(7), n)

    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_prime_time'] = np.isin(df['hour_of_day'], [11, 12, 13, 19, 20, 21]).astype(int)

    # Niche features (random distribution)
    if 'niche_inmobiliaria' not in df.columns:
        niches = np.random.choice(
            BUSINESS_TYPES[:-1],  # Exclude 'otros'
            size=n,
            p=[0.15, 0.10, 0.12, 0.12, 0.18, 0.13, 0.10, 0.10]
        )
        for niche in BUSINESS_TYPES[:-1]:
            df[f'niche_{niche}'] = (niches == niche).astype(int)

    # Business type encoded
    if 'business_type_encoded' not in df.columns:
        encoder = LabelEncoder()
        encoder.fit(BUSINESS_TYPES)
        # Infer from niche flags
        niche_flags = [f'niche_{n}' for n in BUSINESS_TYPES[:-1]]
        if all(col in df.columns for col in niche_flags):
            df['business_type'] = df[niche_flags].idxmax(axis=1).str.replace('niche_', '')
        else:
            df['business_type'] = np.random.choice(BUSINESS_TYPES, n)
        df['business_type_encoded'] = df['business_type'].apply(
            lambda x: encoder.transform([x])[0] if x in encoder.classes_ else len(BUSINESS_TYPES) - 1
        )

    # Add embedding features if not present
    if not any(col in df.columns for col in EMBEDDING_FEATURE_COLUMNS):
        # Find caption column
        caption_col = None
        for col in ['caption', 'text', 'content', 'post_caption', 'description']:
            if col in df.columns:
                caption_col = col
                break

        df = _add_embedding_features_from_captions(df, caption_col or "caption")

    return df


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

    # ==========================================================================
    # SEMANTIC EMBEDDING FEATURES (Synthetic)
    # ==========================================================================
    # Generate synthetic embedding-like features
    # These simulate PCA-reduced sentence embeddings
    # Higher components have lower variance (like real PCA)

    for i in range(PCA_COMPONENTS):
        col_name = f"embedding_{i+1}"
        # Variance decreases for higher components
        variance = 1.0 / (1 + i * 0.1)
        data[col_name] = np.random.normal(0, np.sqrt(variance), n_samples)

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
    logger.info(f"Embedding features: {PCA_COMPONENTS} synthetic PCA components")

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
    # With Kaggle data (recommended)
    python ml/pretrain_base_model.py --data-file data/instagram_reach.csv

    # With synthetic data only
    python ml/pretrain_base_model.py --synthetic --samples 10000

    # Mix: Kaggle + synthetic augmentation
    python ml/pretrain_base_model.py --data-file data/kaggle.csv --augment 5000

    # Custom output path
    python ml/pretrain_base_model.py --synthetic --output models/my_base_model.pkl

Kaggle Datasets (download and use with --data-file):
    - Instagram Reach: kaggle.com/datasets/rxsraghavagrawal/instagram-reach
    - Instagram Analytics: kaggle.com/datasets/kundanbedmutha/instagram-analytics-dataset
    - Social Media Engagement: kaggle.com/datasets/purnisharma/social-media-engagement-metrics
        """
    )

    # Data source (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--data-file", "-f",
        type=str,
        help="Path to Kaggle CSV file with engagement data"
    )
    data_group.add_argument(
        "--synthetic", "-s",
        action="store_true",
        help="Use synthetic data only (default if no --data-file)"
    )

    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=10000,
        help="Number of synthetic samples to generate (default: 10000)"
    )
    parser.add_argument(
        "--augment", "-a",
        type=int,
        default=0,
        help="Add N synthetic samples to augment Kaggle data (default: 0)"
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
        # Determine data source
        if args.data_file:
            # Load Kaggle data
            print(f"Loading data from: {args.data_file}")
            df = load_kaggle_data(args.data_file)
            data_source = f"Kaggle ({args.data_file})"

            # Optionally augment with synthetic data
            if args.augment > 0:
                print(f"Augmenting with {args.augment} synthetic samples...")
                synthetic_df = generate_synthetic_engagement_data(n_samples=args.augment, seed=args.seed)
                df = pd.concat([df, synthetic_df], ignore_index=True)
                data_source += f" + {args.augment} synthetic"

        else:
            # Generate synthetic data
            print(f"Generating {args.samples} synthetic samples...")
            df = generate_synthetic_engagement_data(n_samples=args.samples, seed=args.seed)
            data_source = f"Synthetic ({args.samples} samples)"

        print(f"Total samples: {len(df)}")

        # Train base model
        model, metrics = train_base_model(df)
        metrics["data_source"] = data_source

        # Save model
        output_path = Path(args.output) if args.output else None
        saved_path = save_base_model(model, metrics, output_path)

        # Summary
        print("\n" + "=" * 70)
        print("PRETRAINING COMPLETE")
        print("=" * 70)
        print(f"Data source:        {data_source}")
        print(f"Total samples:      {len(df)}")
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

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print("\nTo download Kaggle data:")
        print("  kaggle datasets download -d rxsraghavagrawal/instagram-reach")
        print("  unzip instagram-reach.zip")
        return 1
    except Exception as e:
        logger.exception(f"Pretraining failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
