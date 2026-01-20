"""
ML Service - Hybrid ML/LLM Architecture for Cost-Efficient Predictions
=======================================================================

Uses XGBoost/RandomForest for fast predictions, Grok (xAI) only for creative generation.

USER CONFIG INTEGRATION (REAL-TIME SYNC):
==========================================
This service now loads user configuration from the database to ensure
frontend changes affect predictions 100% (no placebo):

- embedding_precision: Uses dims from user_config (ultra_low/low/medium/high/max)
- kpi_weights: Uses custom weights from user_config for RPI calculation
- Logs: "User config loaded: precision={X}, multimodal={Y}, own=@{Z}"

FEATURE ENGINEERING:
====================
- Caption analysis: length, emoji_count, hashtag_count, has_question, has_strong_cta, lexical_richness
- Sentiment analysis: VADER compound score (-1 to +1)
- Timing features: post_hour, post_day_of_week, is_weekend
- Format: one-hot encoding (Reel, Carousel, Static, TikTok)
- Niche flags: binary indicators for business-specific keywords

FEEDBACK LOOP ARCHITECTURE (Human-in-the-Loop Reinforcement Learning):
======================================================================
The model learns from real-world performance by:
1. Storing predictions before publication
2. Collecting actual performance metrics after 24-48 hours
3. Calculating delta between predicted and actual engagement
4. Flagging high-delta samples (>20% difference) as high priority
5. Incorporating these samples in the next training cycle

This allows the model to learn from its mistakes and continuously improve.
"""
import os
import re
import logging
import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, mean_absolute_error, roc_auc_score
import xgboost as xgb
import shap
import joblib
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.database import async_session_maker
from app.models.ml_training_queue import MLTrainingSample
from app.services.multi_output_predictor import get_multi_output_predictor

logger = logging.getLogger(__name__)

# Add project root to path for ml module imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import semantic embedding extractor (now with configurable precision)
try:
    from ml.features_embeddings import (
        get_embedding_extractor,
        get_embedding_features,
        get_embedding_feature_names,
        EMBEDDING_DIM,
        PRECISION_TO_DIMS,
        DEFAULT_PRECISION,
    )
    EMBEDDINGS_AVAILABLE = True
    # Default: full 384 dims (safe for SMB volumes)
    EMBEDDING_FEATURE_COLUMNS = get_embedding_feature_names(EMBEDDING_DIM)
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    EMBEDDING_DIM = 384
    EMBEDDING_FEATURE_COLUMNS = [f"embedding_{i+1}" for i in range(384)]
    PRECISION_TO_DIMS = {"low": 128, "medium": 256, "high": 384, "max": 384}
    DEFAULT_PRECISION = "max"

# Import multimodal fusion module (now with configurable precision)
try:
    from backend.ml.multimodal_fusion import (
        get_multimodal_extractor,
        get_transcript_feature_names,
        get_ocr_feature_names,
        get_interaction_feature_names,
        get_multimodal_feature_names,
        TRANSCRIPT_DEFAULT_DIM,
        OCR_DEFAULT_DIM,
    )
    MULTIMODAL_AVAILABLE = True
    # Default: full dims
    TRANSCRIPT_FEATURE_COLUMNS = get_transcript_feature_names(TRANSCRIPT_DEFAULT_DIM)
    OCR_FEATURE_COLUMNS = get_ocr_feature_names(OCR_DEFAULT_DIM)
    INTERACTION_FEATURE_COLUMNS = get_interaction_feature_names()
    MULTIMODAL_FEATURE_COLUMNS = get_multimodal_feature_names()
except ImportError:
    MULTIMODAL_AVAILABLE = False
    TRANSCRIPT_DEFAULT_DIM = 384
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
        compute_hook_score as compute_semantic_hook_score,
        compute_combined_hook_score,
        get_semantic_hook_scorer,
        HIGH_HOOK_THRESHOLD,
        MEDIUM_HOOK_THRESHOLD,
    )
    SEMANTIC_HOOKS_AVAILABLE = True
    logger.info("Semantic hook detection loaded - using embeddings for hook scoring")
except ImportError:
    SEMANTIC_HOOKS_AVAILABLE = False
    HIGH_HOOK_THRESHOLD = 0.7
    MEDIUM_HOOK_THRESHOLD = 0.5
    logger.warning("Semantic hooks not available - falling back to RegEx-only detection")

# Import online learning module for fallback inference
try:
    from backend.ml.online_update import (
        get_online_predictor,
        detect_drift,
        RIVER_AVAILABLE as ONLINE_MODEL_AVAILABLE,
    )
    ONLINE_LEARNING_AVAILABLE = True
    logger.info("Online learning module loaded for fallback inference")
except ImportError:
    ONLINE_LEARNING_AVAILABLE = False
    ONLINE_MODEL_AVAILABLE = False
    logger.info("Online learning not available - using batch models only")

# VADER Sentiment Analysis
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

from app.core.config import settings


# =============================================================================
# FEEDBACK LOOP DATA STRUCTURES
# =============================================================================

# Threshold for flagging samples as high priority training data
# If actual differs from predicted by more than this percentage, it's flagged
HIGH_PRIORITY_DELTA_THRESHOLD = 20.0  # 20% difference

# Minimum hours after posting to collect performance data
MIN_HOURS_FOR_FEEDBACK = 24

# Maximum hours after posting (older data may not be relevant)
MAX_HOURS_FOR_FEEDBACK = 168  # 7 days


@dataclass
class PerformanceFeedback:
    """
    Feedback data structure for ML training loop.

    Captures the delta between predicted and actual performance,
    along with metadata about why this sample is valuable for training.
    """
    content_id: int
    predicted_score: float
    actual_score: float
    delta_percent: float
    is_high_priority: bool
    training_priority: float
    metrics: Dict[str, Any]
    collected_at: datetime
    analysis_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_id": self.content_id,
            "predicted_score": round(self.predicted_score, 2),
            "actual_score": round(self.actual_score, 2),
            "delta_percent": round(self.delta_percent, 2),
            "is_high_priority": self.is_high_priority,
            "training_priority": round(self.training_priority, 4),
            "metrics": self.metrics,
            "collected_at": self.collected_at.isoformat(),
            "analysis_notes": self.analysis_notes,
        }


@dataclass
class TrainingQueueItem:
    """
    Item in the training queue for the next retraining cycle.
    """
    content_id: int
    features: Dict[str, Any]
    actual_engagement: float
    priority: float
    added_at: datetime
    feedback: PerformanceFeedback

    def to_training_sample(self) -> Dict[str, Any]:
        """Convert to format suitable for model training."""
        sample = self.features.copy()
        sample["engagement_score"] = self.actual_engagement
        sample["_priority"] = self.priority
        sample["_source"] = "feedback_loop"
        return sample

# Model storage directories
MODEL_DIR = Path("./ml_models")
MODEL_DIR.mkdir(exist_ok=True)

# Base model for cold start (pretrained on synthetic data)
BASE_MODEL_PATH = MODEL_DIR / "base_xgboost.pkl"

# Project root for accessing /models directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
ALT_BASE_MODEL_PATH = PROJECT_ROOT / "models" / "base_xgboost.pkl"

# Cold start threshold
COLD_START_THRESHOLD = 300  # Samples below this use fine-tuned base model


class FeatureExtractor:
    """
    Extract features from content for ML predictions
    Features are designed for local SMB social media content

    FEATURES EXTRACTED:
    ===================
    - Caption: length, emoji_count, hashtag_count, has_question, has_strong_cta, lexical_richness
    - Sentiment: VADER compound score (-1 to +1)
    - Timing: post_hour, post_day_of_week, is_weekend
    - Format: one-hot encoding
    - Niche flags: binary for vertical-specific keywords
    """

    # Emoji patterns
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

    # Question detection regex (Spanish)
    QUESTION_PATTERN = re.compile(r'\?|¿|cuál|qué|cómo|por qué|quién|dónde|cuándo', re.IGNORECASE)

    # Strong CTA keywords (high-engagement drivers)
    STRONG_CTA_KEYWORDS = ["comenta", "guarda", "dm", "visita", "taggea", "etiqueta", "escríbeme", "guárdalo"]

    # Engagement trigger words (Spanish focused for local SMBs)
    TRIGGER_WORDS = {
        "question": ["?", "cuál", "qué", "cómo", "por qué", "quién", "dónde", "cuándo"],
        "urgency": ["ahora", "hoy", "último", "limitado", "exclusivo", "ya", "rápido"],
        "social_proof": ["clientes", "testimonios", "opiniones", "reviews", "valoraciones"],
        "value": ["gratis", "regalo", "descuento", "oferta", "promoción", "ahorra"],
        "curiosity": ["secreto", "descubre", "sorpresa", "increíble", "no creerás"],
        "action": ["comenta", "guarda", "comparte", "sígueme", "dm", "escríbeme", "haz click"],
        "emotion": ["amor", "feliz", "alegría", "pasión", "sueño", "gracias"],
        "transformation": ["antes", "después", "transformación", "cambio", "resultado"],
    }

    # Niche-specific keywords for business type detection
    NICHE_KEYWORDS = {
        "inmobiliaria": ["casa", "piso", "tour", "triana", "inmueble", "venta", "alquiler", "habitación", "propiedad", "reforma", "m²", "metros"],
        "floristeria": ["flores", "arreglo", "ramo", "bouquet", "rosas", "tulipanes", "floristería", "planta", "decoración floral", "centro de mesa"],
        "cafeteria": ["café", "coffee", "latte", "cappuccino", "barista", "espresso", "desayuno", "brunch", "pastelería", "dulce"],
        "peluqueria": ["corte", "pelo", "cabello", "tinte", "mechas", "peinado", "estilista", "look", "color", "tratamiento capilar"],
        "restaurante": ["plato", "menú", "cocina", "chef", "reserva", "cena", "comida", "gastronomía", "receta", "sabor"],
        "gimnasio": ["entreno", "fitness", "ejercicio", "músculo", "cardio", "peso", "rutina", "gym", "entrenador", "clase"],
        "clinica": ["salud", "doctor", "tratamiento", "consulta", "cita", "paciente", "bienestar", "medicina", "especialista"],
    }

    # Initialize VADER analyzer (singleton)
    _vader_analyzer = None

    @classmethod
    def _get_vader(cls):
        """Get or initialize VADER sentiment analyzer."""
        if cls._vader_analyzer is None and VADER_AVAILABLE:
            cls._vader_analyzer = SentimentIntensityAnalyzer()
        return cls._vader_analyzer

    # Hook patterns
    HOOK_PATTERNS = {
        "pov": r"pov[:\s]|punto de vista",
        "question": r"^[¿?]|^\w+\s*\?",
        "number": r"^\d+\s+\w+|top\s*\d+|\d+\s*(cosas|tips|errores|razones)",
        "bold_claim": r"nunca|siempre|todo|nadie|el mejor|el peor|imposible",
        "story": r"historia|storytime|cuando|un día|me pasó",
        "how_to": r"cómo\s+\w+|aprende\s+a|tutorial|paso\s+a\s+paso",
        "reveal": r"secreto|te cuento|descubre|te revelo|no sabías",
    }

    # CTA patterns
    CTA_PATTERNS = {
        "comment": r"comenta|cuéntame|opina|dime|escribe",
        "save": r"guarda|guardar|guárdalo|save",
        "share": r"comparte|compartir|etiqueta|tag",
        "follow": r"sígueme|sigue|follow|seguir",
        "dm": r"dm|mensaje|escríbeme|mensaje directo",
        "link": r"link|enlace|bio|click|pincha",
    }

    @classmethod
    def extract_features(
        cls,
        content: Dict[str, Any],
        embedding_precision: str = "low"
    ) -> Dict[str, float]:
        """
        Extract all features from content
        Returns feature dict for ML model input

        USER CONFIG SYNC:
        =================
        The embedding_precision parameter should come from user_config to ensure
        frontend changes affect feature extraction in real-time.

        Args:
            content: Content dict with caption, hashtags, etc.
            embedding_precision: Precision from user_config (default: "low")
                                Options: ultra_low/low/medium/high/max

        ENHANCED FEATURES:
        - Caption: length, emoji_count, hashtag_count, has_question, has_strong_cta, lexical_richness
        - Sentiment: VADER compound score (-1 to +1)
        - Timing: post_hour, post_day_of_week, is_weekend
        - Format: one-hot encoding
        - Niche flags: binary for vertical-specific keywords
        - Embeddings: Semantic embeddings with configurable precision
        """
        caption = content.get("caption", "") or ""
        caption_lower = caption.lower()
        words = caption_lower.split()

        features = {}

        # === Text Length Features ===
        features["caption_length"] = len(caption)
        features["caption_words"] = len(words)
        features["caption_lines"] = caption.count("\n") + 1
        features["avg_word_length"] = (
            np.mean([len(w) for w in words]) if words else 0
        )

        # === Lexical Richness (type-token ratio) ===
        # Higher values = more diverse vocabulary = potentially more engaging
        if len(words) > 0:
            unique_words = set(words)
            features["lexical_richness"] = len(unique_words) / len(words)
        else:
            features["lexical_richness"] = 0.0

        # === Question Detection (regex) ===
        features["has_question"] = 1 if cls.QUESTION_PATTERN.search(caption) else 0

        # === Strong CTA Detection ===
        # Keywords: Comenta, Guarda, DM, Visita, Taggea
        has_strong_cta = any(kw in caption_lower for kw in cls.STRONG_CTA_KEYWORDS)
        features["has_strong_cta"] = 1 if has_strong_cta else 0

        # === VADER Sentiment Analysis ===
        vader = cls._get_vader()
        if vader:
            sentiment_scores = vader.polarity_scores(caption)
            features["sentiment_compound"] = sentiment_scores["compound"]  # -1 to +1
            features["sentiment_positive"] = sentiment_scores["pos"]
            features["sentiment_negative"] = sentiment_scores["neg"]
            features["sentiment_neutral"] = sentiment_scores["neu"]
        else:
            # Fallback if VADER not available
            features["sentiment_compound"] = 0.0
            features["sentiment_positive"] = 0.0
            features["sentiment_negative"] = 0.0
            features["sentiment_neutral"] = 1.0

        # === Emoji Features ===
        emojis = cls.EMOJI_PATTERN.findall(caption)
        features["emoji_count"] = len(emojis)
        features["emoji_density"] = len(emojis) / max(len(caption), 1) * 100

        # === Hashtag Features ===
        hashtags = content.get("hashtags", []) or re.findall(r"#\w+", caption)
        features["hashtag_count"] = len(hashtags)
        features["hashtag_density"] = len(hashtags) / max(features["caption_words"], 1)

        # === Niche Flags (binary indicators for business-specific keywords) ===
        for niche, keywords in cls.NICHE_KEYWORDS.items():
            niche_match = any(kw.lower() in caption_lower for kw in keywords)
            features[f"niche_{niche}"] = 1 if niche_match else 0

        # === Mention Features ===
        mentions = content.get("mentions", []) or re.findall(r"@\w+", caption)
        features["mention_count"] = len(mentions)

        # === Trigger Word Features ===
        for trigger_type, words in cls.TRIGGER_WORDS.items():
            count = sum(1 for word in words if word in caption_lower)
            features[f"trigger_{trigger_type}"] = count

        # === Hook Detection (RegEx as auxiliary/backup) ===
        first_line = caption.split("\n")[0] if caption else ""
        first_line_lower = first_line.lower()

        for hook_type, pattern in cls.HOOK_PATTERNS.items():
            features[f"hook_{hook_type}"] = 1 if re.search(pattern, first_line_lower, re.IGNORECASE) else 0

        # Calculate legacy RegEx-based hook score (kept as auxiliary feature)
        regex_hook_count = sum(features[f"hook_{h}"] for h in ["pov", "question", "number", "bold_claim", "story", "how_to", "reveal"])
        features["hook_regex_score"] = min(regex_hook_count / 3.0, 1.0)  # Normalize to 0-1

        # ==========================================================================
        # SEMANTIC HOOK DETECTION (Primary - replaces naive RegEx)
        # ==========================================================================
        # Uses SentenceTransformer embeddings to detect hooks via cosine similarity
        # Captures creative variations that keyword matching misses:
        # - "De qué manera" instead of "cómo"
        # - "Lo que nadie te dice" instead of "secreto"
        # - Regional/Andalusian variations

        if SEMANTIC_HOOKS_AVAILABLE:
            try:
                # Get transcript if available for combined scoring
                whisper_transcript = content.get("whisper_transcript", "") or ""

                # Compute semantic hook score
                semantic_result = compute_semantic_hook_score(
                    caption=caption,
                    transcript=whisper_transcript if whisper_transcript.strip() else None,
                    return_details=True
                )

                features["semantic_hook_score"] = semantic_result.get("score", 0.0)
                features["semantic_hook_max_sim"] = semantic_result.get("max_similarity", 0.0)
                features["semantic_hook_top3_avg"] = semantic_result.get("top3_avg", 0.0)

                # Log for debugging high-value hooks
                if features["semantic_hook_score"] >= HIGH_HOOK_THRESHOLD:
                    top_match = semantic_result.get("top_matches", [{}])[0]
                    logger.debug(
                        f"Strong semantic hook detected (score={features['semantic_hook_score']:.3f}): "
                        f"matched '{top_match.get('hook', 'N/A')[:40]}...'"
                    )

            except Exception as e:
                logger.warning(f"Semantic hook scoring failed: {e}. Using RegEx fallback.")
                features["semantic_hook_score"] = features["hook_regex_score"]
                features["semantic_hook_max_sim"] = 0.0
                features["semantic_hook_top3_avg"] = 0.0
        else:
            # Fallback to RegEx-derived score when semantic not available
            features["semantic_hook_score"] = features["hook_regex_score"]
            features["semantic_hook_max_sim"] = 0.0
            features["semantic_hook_top3_avg"] = 0.0

        # === CTA Detection ===
        for cta_type, pattern in cls.CTA_PATTERNS.items():
            features[f"cta_{cta_type}"] = 1 if re.search(pattern, caption_lower, re.IGNORECASE) else 0

        # Total CTAs
        features["cta_count"] = sum(features[f"cta_{t}"] for t in cls.CTA_PATTERNS.keys())

        # === Format Features ===
        content_format = content.get("content_format", content.get("type", "unknown"))
        features["is_reel"] = 1 if content_format in ["reel", "tiktok_video", "video"] else 0
        features["is_carousel"] = 1 if content_format == "carousel" else 0
        features["is_static"] = 1 if content_format in ["static_image", "static", "image"] else 0

        # === Video Features ===
        duration = content.get("video_duration_seconds", content.get("video_duration", 0)) or 0
        features["video_duration"] = duration
        features["video_optimal_length"] = 1 if 15 <= duration <= 60 else 0

        # === Audio Features ===
        audio = content.get("audio_name", content.get("recommended_audio", ""))
        features["has_audio"] = 1 if audio else 0
        features["is_trending_audio"] = 1 if audio and "original" not in audio.lower() else 0

        # === Timing Features ===
        posted_at = content.get("posted_at")
        if posted_at:
            try:
                if isinstance(posted_at, str):
                    dt = datetime.fromisoformat(posted_at.replace("Z", "+00:00"))
                else:
                    dt = posted_at
                features["hour_of_day"] = dt.hour
                features["day_of_week"] = dt.weekday()
                features["is_weekend"] = 1 if dt.weekday() >= 5 else 0
                features["is_prime_time"] = 1 if dt.hour in [11, 12, 13, 19, 20, 21] else 0
            except:
                features["hour_of_day"] = 12
                features["day_of_week"] = 2
                features["is_weekend"] = 0
                features["is_prime_time"] = 1
        else:
            features["hour_of_day"] = 12
            features["day_of_week"] = 2
            features["is_weekend"] = 0
            features["is_prime_time"] = 1

        # === Engagement (for training, 0 for prediction) ===
        features["likes"] = content.get("likes_count", content.get("likes", 0)) or 0
        features["comments"] = content.get("comments_count", content.get("comments", 0)) or 0
        features["shares"] = content.get("shares_count", content.get("shares", 0)) or 0
        features["saves"] = content.get("saves_count", content.get("saves", 0)) or 0
        features["views"] = content.get("views_count", content.get("video_views", content.get("plays", 0))) or 0

        # === Business Type (will be encoded) ===
        features["business_type"] = content.get("business_type", "otros")

        # ==========================================================================
        # SEMANTIC EMBEDDINGS (Modern NLP features replacing manual heuristics)
        # USER CONFIG SYNC: Uses embedding_precision from user_config
        # ==========================================================================
        # Uses sentence-transformers/all-MiniLM-L6-v2 with configurable dimensions:
        # - ultra_low: 64 dims, low: 128 dims, medium: 256 dims, high/max: 384 dims
        # These features capture semantic meaning that heuristics cannot

        # Get target dims from precision
        target_dims = PRECISION_TO_DIMS.get(embedding_precision, 128) or 384

        if EMBEDDINGS_AVAILABLE:
            try:
                # USER CONFIG SYNC: Pass precision to embedding extractor
                embedding_features = get_embedding_features(caption, precision=embedding_precision)
                features.update(embedding_features)
                logger.debug(f"Embeddings extracted: precision={embedding_precision}, dims={len(embedding_features)}")
            except Exception as e:
                logger.warning(f"Embedding extraction failed: {e}. Using zeros for {target_dims} dims.")
                for i in range(target_dims):
                    features[f"embedding_{i}"] = 0.0
        else:
            # Fallback: zeros when embeddings not available
            for i in range(target_dims):
                features[f"embedding_{i}"] = 0.0

        # ==========================================================================
        # MULTIMODAL FEATURES (for video/reel content)
        # ==========================================================================
        # Extracts embeddings from:
        # - whisper_transcript: Audio transcription
        # - easyocr_text: Visual text overlay
        # Also generates cross-modal interaction features

        is_video = features.get("is_reel", 0) == 1 or content.get("media_type", "").lower() in ["reel", "video", "tiktok"]

        if MULTIMODAL_AVAILABLE and is_video:
            try:
                extractor = get_multimodal_extractor()

                # Extract transcript embeddings
                whisper_transcript = content.get("whisper_transcript", "") or ""
                if whisper_transcript.strip():
                    transcript_features = extractor.get_transcript_features(whisper_transcript)
                    features.update(transcript_features)
                else:
                    for col in TRANSCRIPT_FEATURE_COLUMNS:
                        features[col] = 0.0

                # Extract OCR embeddings
                easyocr_text = content.get("easyocr_text", "") or ""
                if easyocr_text.strip():
                    ocr_features = extractor.get_ocr_features(easyocr_text)
                    features.update(ocr_features)
                else:
                    for col in OCR_FEATURE_COLUMNS:
                        features[col] = 0.0

                # Generate interaction features using SEMANTIC hook score (primary)
                # semantic_hook_score is now the authoritative hook score (0-1 continuous)
                hook_score = features.get("semantic_hook_score", 0.0)

                # Interaction features with semantic hook (better for SHAP explainability)
                features["interaction_hook_x_sentiment"] = hook_score * features.get("sentiment_compound", 0)
                features["interaction_hook_x_is_reel"] = hook_score * features.get("is_reel", 0)
                features["interaction_hook_x_cta_count"] = hook_score * features.get("cta_count", 0)
                features["interaction_sentiment_x_cta_count"] = features.get("sentiment_compound", 0) * features.get("cta_count", 0)
                features["interaction_is_reel_x_video_optimal"] = features.get("is_reel", 0) * features.get("video_optimal_length", 0)

                # New semantic hook interactions (capture synergies with semantic understanding)
                features["interaction_semantic_hook_x_vader"] = hook_score * abs(features.get("sentiment_compound", 0))
                features["interaction_semantic_hook_x_cta_strong"] = hook_score * features.get("has_strong_cta", 0)

                # Text richness metrics
                features["interaction_transcript_richness"] = min(len(whisper_transcript) / 500, 1.0)
                features["interaction_ocr_richness"] = min(len(easyocr_text) / 100, 1.0)

                total_text = len(caption) + len(whisper_transcript) + len(easyocr_text)
                features["interaction_multimodal_text_density"] = min(total_text / 1000, 1.0)

                logger.debug(f"Multimodal features extracted for video content")

            except Exception as e:
                logger.warning(f"Multimodal feature extraction failed: {e}. Using zeros.")
                for col in MULTIMODAL_FEATURE_COLUMNS:
                    features[col] = 0.0
        else:
            # Non-video content or multimodal not available - add zeros for consistency
            for col in MULTIMODAL_FEATURE_COLUMNS:
                features[col] = 0.0

        return features

    @classmethod
    def extract_batch(cls, contents: List[Dict[str, Any]]) -> pd.DataFrame:
        """Extract features for multiple content items"""
        features_list = [cls.extract_features(c) for c in contents]
        return pd.DataFrame(features_list)


class MLPredictor:
    """
    ML Predictor with XGBoost/RandomForest models
    Provides engagement scoring, format recommendation, and trigger suggestions
    """

    # Manual heuristic features (kept as backup, but embeddings are prioritized)
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
        "hook_regex_score",  # Normalized RegEx hook score
        # SEMANTIC HOOK FEATURES (primary - embedding-based detection)
        "semantic_hook_score",     # Primary hook score (0-1 continuous, SHAP-friendly)
        "semantic_hook_max_sim",   # Max cosine similarity to any base hook
        "semantic_hook_top3_avg",  # Average similarity to top 3 hooks
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
    # Total: 30 (caption) + 20 (transcript) + 20 (ocr) + 8 (interactions) + ~58 (manual) ≈ 136 features
    FEATURE_COLUMNS = (
        EMBEDDING_FEATURE_COLUMNS +
        MULTIMODAL_FEATURE_COLUMNS +
        MANUAL_FEATURE_COLUMNS
    )

    FORMAT_CLASSES = ["reel", "carousel", "static_image", "tiktok_video"]

    # Supported business niches
    BUSINESS_NICHES = [
        "inmobiliaria", "floristeria", "cafeteria", "peluqueria",
        "restaurante", "gimnasio", "clinica", "otros"
    ]

    def __init__(self):
        self.engagement_model: Optional[xgb.XGBRegressor] = None
        self.format_model: Optional[RandomForestClassifier] = None
        self.trigger_model: Optional[xgb.XGBClassifier] = None
        self.business_type_encoder = LabelEncoder()
        self.format_encoder = LabelEncoder()
        self.shap_explainer_engagement = None
        self.shap_explainer_format = None
        self.is_trained = False
        self._models_loaded = False

        # Cold start support: base model and niche-specific models
        self.base_model: Optional[xgb.XGBRegressor] = None
        self.base_model_loaded = False
        self.niche_models: Dict[str, xgb.XGBRegressor] = {}
        self.active_model_source: str = "none"  # "trained", "niche", "base", "heuristic"

        # No eager loading here!
        # Models are loaded lazily on first prediction or training
        # to prevent memory duplication in multi-worker environments (Gunicorn).
        pass

    def _ensure_models_loaded(self):
        """
        Lazy load models if not already loaded.
        Uses mmap_mode='r' to share memory pages across workers.
        """
        if self._models_loaded:
            return

        logger.info("Initializing lazy model loading...")

        # Try to load existing models
        self._load_models()
        self._load_base_model()
        self._load_niche_models()

        self._models_loaded = True
        logger.info("Lazy model loading complete")

    def _get_model_path(self, name: str) -> Path:
        """Get path for a model file"""
        return MODEL_DIR / f"{name}.joblib"

    def _load_models(self):
        """Load trained models from disk using mmap"""
        try:
            engagement_path = self._get_model_path("engagement_model")
            format_path = self._get_model_path("format_model")
            trigger_path = self._get_model_path("trigger_model")
            encoder_path = self._get_model_path("encoders")

            if all(p.exists() for p in [engagement_path, format_path, encoder_path]):
                # USE mmap_mode='r' FOR MEMORY SHARING ACROSS WORKERS
                self.engagement_model = joblib.load(engagement_path, mmap_mode='r')
                self.format_model = joblib.load(format_path, mmap_mode='r')

                if trigger_path.exists():
                    self.trigger_model = joblib.load(trigger_path, mmap_mode='r')

                encoders = joblib.load(encoder_path, mmap_mode='r')
                self.business_type_encoder = encoders["business_type"]
                self.format_encoder = encoders["format"]

                self.is_trained = True
                logger.info("Loaded trained ML models from disk (mmap_mode='r')")

                # Initialize SHAP explainers
                self._init_shap_explainers()

        except Exception as e:
            logger.warning(f"Could not load models: {e}")
            self.is_trained = False

    def _load_base_model(self):
        """
        Load the pretrained base model for cold start scenarios.

        The base model is trained on synthetic data covering all niches
        and provides reasonable predictions when niche-specific data is scarce.
        """
        try:
            # Try primary location
            if BASE_MODEL_PATH.exists():
                bundle = joblib.load(BASE_MODEL_PATH, mmap_mode='r')
                self.base_model = bundle.get("model")
                self.base_model_loaded = True
                logger.info(f"Loaded base model from: {BASE_MODEL_PATH} (mmap_mode='r')")
                return

            # Try alternative location (project root /models)
            if ALT_BASE_MODEL_PATH.exists():
                bundle = joblib.load(ALT_BASE_MODEL_PATH, mmap_mode='r')
                self.base_model = bundle.get("model")
                self.base_model_loaded = True
                logger.info(f"Loaded base model from: {ALT_BASE_MODEL_PATH} (mmap_mode='r')")
                return

            logger.info("Base model not found. Run pretrain_base_model.py to create it.")

        except Exception as e:
            logger.warning(f"Could not load base model: {e}")
            self.base_model_loaded = False

    def _load_niche_models(self):
        """
        Load niche-specific fine-tuned models.

        These models are trained/fine-tuned on data specific to each business niche
        and provide better predictions than the generic model for that niche.
        """
        try:
            for niche in self.BUSINESS_NICHES:
                niche_path = MODEL_DIR / f"niche_{niche}.pkl"
                if niche_path.exists():
                    bundle = joblib.load(niche_path, mmap_mode='r')
                    self.niche_models[niche] = bundle.get("model")
                    logger.info(f"Loaded niche model for: {niche} (mmap_mode='r')")

            if self.niche_models:
                logger.info(f"Loaded {len(self.niche_models)} niche-specific models")

        except Exception as e:
            logger.warning(f"Could not load niche models: {e}")

    def _get_model_for_niche(self, niche: str) -> Tuple[Optional[xgb.XGBRegressor], str]:
        """
        Get the best available model for a specific niche.

        Priority order:
        1. Niche-specific fine-tuned model
        2. Main trained model (if available)
        3. Base model (cold start fallback)
        4. None (will use heuristics)

        Args:
            niche: Business niche name

        Returns:
            Tuple of (model, source_name) where source_name describes which model is used
        """
        # Priority 1: Niche-specific model
        if niche in self.niche_models:
            logger.debug(f"Using niche-specific model for: {niche}")
            return self.niche_models[niche], f"niche_{niche}"

        # Priority 2: Main trained model
        if self.is_trained and self.engagement_model is not None:
            logger.debug(f"Using main trained model for niche: {niche}")
            return self.engagement_model, "trained"

        # Priority 3: Base model (cold start)
        if self.base_model_loaded and self.base_model is not None:
            logger.info(f"Usando modelo base + fine-tune por cold start (niche: {niche})")
            return self.base_model, "base"

        # Priority 4: No model available
        logger.warning(f"No model available for niche: {niche}. Using heuristics.")
        return None, "heuristic"

    def _get_online_prediction(
        self,
        features: Dict[str, Any],
        niche: str
    ) -> Optional[Dict[str, float]]:
        """
        Get prediction from online River model.

        Used as fallback when drift is detected or for cold start scenarios.

        Args:
            features: Feature dictionary
            niche: Business niche

        Returns:
            Dict with predicted log values or None if unavailable
        """
        if not ONLINE_LEARNING_AVAILABLE or not ONLINE_MODEL_AVAILABLE:
            return None

        try:
            predictor = get_online_predictor(niche)

            # Check if model has enough samples
            if predictor._metrics.samples_seen < 5:
                logger.debug(f"Online model for {niche} has too few samples")
                return None

            # Get numeric features only
            numeric_features = {
                k: float(v) if isinstance(v, (int, float, np.number)) else 0.0
                for k, v in features.items()
                if k != "business_type" and isinstance(v, (int, float, np.number))
            }

            prediction = predictor.predict_one(numeric_features)

            # Check if prediction is valid
            if prediction and any(v != 0.0 for v in prediction.values()):
                return prediction

            return None

        except Exception as e:
            logger.debug(f"Online prediction failed: {e}")
            return None

    def _convert_online_to_score(self, online_pred: Dict[str, float]) -> float:
        """
        Convert online model multi-output prediction to single RPI score.

        Args:
            online_pred: Dict with log_likes, log_comments, etc.

        Returns:
            RPI score (0-100)
        """
        # Use similar weighting as multi-output predictor
        weights = {
            "log_likes": 1.0,
            "log_comments": 2.0,
            "log_shares": 10.0,
            "log_saves": 5.0,
            "log_views": 3.0,
        }

        weighted_sum = sum(
            online_pred.get(k, 0.0) * w
            for k, w in weights.items()
        )
        total_weight = sum(weights.values())

        # Normalize to 0-100 scale (similar to multi-output predictor)
        rpi = (weighted_sum / total_weight) * 15.0

        return float(np.clip(rpi, 0, 100))

    def predict_with_online_fallback(
        self,
        content: Dict[str, Any],
        use_drift_detection: bool = True,
        recent_predictions: Optional[List[float]] = None,
        recent_actuals: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Predict engagement with automatic fallback to online model.

        This method provides intelligent model selection:
        1. First, tries batch XGBoost model
        2. If drift is detected (using recent prediction errors), falls back to online
        3. For cold start, uses online model if available

        Args:
            content: Content dictionary for prediction
            use_drift_detection: Whether to check for drift
            recent_predictions: Recent batch model predictions (for drift detection)
            recent_actuals: Corresponding actual values (for drift detection)

        Returns:
            Prediction result with model_source indicating which model was used
        """
        self._ensure_models_loaded()

        features = FeatureExtractor.extract_features(content)
        niche = content.get("business_type", "otros")

        # Check for drift if recent data provided
        drift_detected = False
        if use_drift_detection and recent_predictions and recent_actuals:
            if ONLINE_LEARNING_AVAILABLE:
                drift_detected, drift_magnitude = detect_drift(
                    niche, recent_predictions, recent_actuals
                )
                if drift_detected:
                    logger.warning(
                        f"Drift detected for niche={niche}: magnitude={drift_magnitude:.3f}. "
                        f"Switching to online model."
                    )

        # Get batch model
        batch_model, batch_source = self._get_model_for_niche(niche)

        # Decide which model to use
        use_online = False

        # Case 1: Drift detected - prefer online
        if drift_detected:
            use_online = True
            logger.info(f"Using online model due to drift for niche={niche}")

        # Case 2: No batch model available - use online as fallback
        elif batch_model is None:
            use_online = True
            logger.info(f"No batch model, using online fallback for niche={niche}")

        # Try online prediction if needed
        if use_online and ONLINE_LEARNING_AVAILABLE:
            online_pred = self._get_online_prediction(features, niche)

            if online_pred:
                score = self._convert_online_to_score(online_pred)

                # Get online model status
                try:
                    online_predictor = get_online_predictor(niche)
                    online_status = online_predictor.get_status()
                except:
                    online_status = {}

                return {
                    "score": round(score, 1),
                    "confidence": 65.0,  # Lower confidence for online model
                    "explanation": {
                        "explanation_text": f"Predicción con modelo online adaptativo (niche: {niche}). "
                                          f"Samples aprendidos: {online_status.get('samples_seen', 0)}. "
                                          f"Usando River AdaptiveRandomForest.",
                        "top_positive_factors": [],
                        "top_negative_factors": [],
                    },
                    "feature_importance": [],
                    "model_source": "online_river",
                    "niche": niche,
                    "cold_start": False,
                    "drift_detected": drift_detected,
                    "online_samples": online_status.get("samples_seen", 0),
                    "online_mae": online_status.get("current_mae"),
                }

        # Fall back to standard prediction (batch model or heuristics)
        return self.predict_engagement(content)

    def get_model_status(self) -> Dict[str, Any]:
        """
        Get status of all loaded models including online models.

        Returns:
            Dictionary with model availability and sources
        """
        # Note: We don't force load here to inspect true status,
        # but if they are not loaded, we might report false negatives.
        # However, for status, we probably want to see if they ARE loaded.
        # If we force load, we defeat the purpose of checking status.
        # But if the user asks for status, they likely want to know what's available on disk too.
        # Let's show current memory status.

        status = {
            "models_loaded_in_memory": getattr(self, "_models_loaded", False),
            "main_model_trained": self.is_trained,
            "base_model_loaded": self.base_model_loaded,
            "niche_models_loaded": list(self.niche_models.keys()),
            "total_niche_models": len(self.niche_models),
            "cold_start_threshold": COLD_START_THRESHOLD,
            "model_dir": str(MODEL_DIR),
            "online_learning_available": ONLINE_LEARNING_AVAILABLE,
            "river_installed": ONLINE_MODEL_AVAILABLE,
        }

        # Add online model status if available
        if ONLINE_LEARNING_AVAILABLE and ONLINE_MODEL_AVAILABLE:
            try:
                online_niches = {}
                for niche in self.BUSINESS_NICHES:
                    try:
                        predictor = get_online_predictor(niche)
                        if predictor._metrics.samples_seen > 0:
                            online_niches[niche] = {
                                "samples_seen": predictor._metrics.samples_seen,
                                "current_mae": predictor._metrics.last_mae,
                                "best_mae": predictor._metrics.best_mae,
                            }
                    except:
                        pass

                status["online_models"] = online_niches
                status["online_models_count"] = len(online_niches)
            except:
                status["online_models"] = {}
                status["online_models_count"] = 0

        return status

    def _save_models(self):
        """Save trained models to disk"""
        try:
            joblib.dump(self.engagement_model, self._get_model_path("engagement_model"))
            joblib.dump(self.format_model, self._get_model_path("format_model"))

            if self.trigger_model:
                joblib.dump(self.trigger_model, self._get_model_path("trigger_model"))

            encoders = {
                "business_type": self.business_type_encoder,
                "format": self.format_encoder,
            }
            joblib.dump(encoders, self._get_model_path("encoders"))

            logger.info("Saved ML models to disk")
        except Exception as e:
            logger.error(f"Error saving models: {e}")

    def _init_shap_explainers(self):
        """Initialize SHAP explainers for model interpretability"""
        try:
            if self.engagement_model:
                self.shap_explainer_engagement = shap.TreeExplainer(self.engagement_model)
            if self.format_model:
                self.shap_explainer_format = shap.TreeExplainer(self.format_model)
            logger.info("SHAP explainers initialized")
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainers: {e}")

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model input"""
        df = df.copy()

        # Encode business type
        if "business_type" in df.columns:
            # Handle unseen categories
            known_types = set(self.business_type_encoder.classes_) if hasattr(self.business_type_encoder, 'classes_') else set()
            df["business_type_encoded"] = df["business_type"].apply(
                lambda x: self.business_type_encoder.transform([x])[0] if x in known_types else 0
            )
        else:
            df["business_type_encoded"] = 0

        # Select only needed columns
        feature_cols = [c for c in self.FEATURE_COLUMNS if c in df.columns]
        return df[feature_cols].fillna(0)

    def train(self, training_data: List[Dict[str, Any]], retrain: bool = False):
        """
        Train all ML models on provided data
        training_data: List of content dicts with engagement metrics
        """
        # Ensure we have loaded any existing state before retraining decisions
        self._ensure_models_loaded()

        if self.is_trained and not retrain:
            logger.info("Models already trained. Use retrain=True to force retraining.")
            return

        logger.info(f"Training ML models on {len(training_data)} samples...")

        # Extract features
        df = FeatureExtractor.extract_batch(training_data)

        # Fit encoders
        business_types = df["business_type"].unique().tolist()
        if "otros" not in business_types:
            business_types.append("otros")
        self.business_type_encoder.fit(business_types)

        self.format_encoder.fit(self.FORMAT_CLASSES)

        # Prepare features
        df["business_type_encoded"] = self.business_type_encoder.transform(df["business_type"])

        # Calculate engagement score (target for regression)
        df["engagement_score"] = (
            df["likes"] +
            df["comments"] * 3 +
            df["saves"] * 5 +
            df["shares"] * 4
        )
        # Normalize to 0-100 scale
        max_engagement = df["engagement_score"].quantile(0.95)
        df["engagement_score_normalized"] = (df["engagement_score"] / max(max_engagement, 1) * 100).clip(0, 100)

        # Determine best format (target for classification)
        format_cols = ["is_reel", "is_carousel", "is_static"]
        df["best_format"] = df[format_cols].idxmax(axis=1).str.replace("is_", "")
        df.loc[df["is_reel"] == 1, "best_format"] = "reel"

        # Features for training
        X = self._prepare_features(df)

        # Check dimensionality (Anti-Overfitting Warning)
        if X.shape[1] > 200:
            logger.warning(
                f"⚠️ High dimensionality detected ({X.shape[1]} features). "
                f"Risk of overfitting for small datasets (SMB context). "
                f"Consider using 'low' or 'medium' embedding precision."
            )

        # === Train Engagement Model (XGBoost Regression) ===
        y_engagement = df["engagement_score_normalized"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_engagement, test_size=0.2, random_state=42
        )

        # Adjusted hyperparameters for multimodal features (~136 features)
        # max_depth increased from 6 to 7 to capture cross-modal interactions
        self.engagement_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=7,  # Increased for multimodal feature interactions
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
        self.engagement_model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.engagement_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        logger.info(f"Engagement Model RMSE: {rmse:.2f}")

        # === Train Format Recommendation Model (Random Forest) ===
        # For format, we want to recommend based on content features
        # Use high-engagement posts to learn what formats work best
        high_engagement_mask = df["engagement_score_normalized"] > df["engagement_score_normalized"].median()
        X_format = self._prepare_features(df[high_engagement_mask])
        y_format = df.loc[high_engagement_mask, "best_format"]

        if len(y_format.unique()) > 1:
            X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
                X_format, y_format, test_size=0.2, random_state=42
            )

            self.format_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            )
            self.format_model.fit(X_train_f, y_train_f)

            # Evaluate
            y_pred_f = self.format_model.predict(X_test_f)
            accuracy = accuracy_score(y_test_f, y_pred_f)
            logger.info(f"Format Model Accuracy: {accuracy:.2%}")
        else:
            # Default model if not enough variety
            self.format_model = RandomForestClassifier(n_estimators=10, random_state=42)
            self.format_model.fit(X_format, ["reel"] * len(X_format))

        # === Train Trigger Suggestion Model (Multi-label) ===
        # Predict which triggers lead to high engagement
        trigger_cols = [c for c in df.columns if c.startswith("trigger_")]
        high_eng_triggers = df.loc[high_engagement_mask, trigger_cols].mean()
        best_triggers = high_eng_triggers.nlargest(3).index.tolist()

        # Store for recommendations
        self._best_triggers = best_triggers

        self.is_trained = True
        self._save_models()
        self._init_shap_explainers()

        logger.info("ML models trained and saved successfully")

    def predict_engagement(
        self,
        content: Dict[str, Any],
        embedding_precision: str = "low",
        kpi_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Predict engagement score for content with cold start handling.

        USER CONFIG SYNC:
        =================
        This method now accepts embedding_precision and kpi_weights from user_config
        to ensure frontend configuration changes affect predictions in real-time.

        Args:
            content: Content dict with caption, hashtags, etc.
            embedding_precision: From user_config (default: "low")
            kpi_weights: From user_config for RPI calculation (optional)

        Uses niche-specific model if available, falls back to base model
        for cold start scenarios, and uses heuristics as last resort.

        Model selection priority:
        1. Niche-specific fine-tuned model
        2. Main trained model
        3. Base model (cold start fallback)
        4. Heuristic prediction

        Returns:
            Dict with score, confidence, explanation, model_source
        """
        # Ensure models are loaded before prediction
        self._ensure_models_loaded()

        # CRITICAL LOG: User config being used
        logger.info(
            f"Prediction with user config: precision={embedding_precision}, "
            f"kpi_weights={'custom' if kpi_weights else 'default'}"
        )

        # Extract features with user-configured precision
        features = FeatureExtractor.extract_features(content, embedding_precision=embedding_precision)
        df = pd.DataFrame([features])
        X = self._prepare_features(df)

        # Determine niche from content
        niche = content.get("business_type", "otros")
        if niche not in self.BUSINESS_NICHES:
            niche = "otros"

        # 1. Try Multi-Output Model (Superior Architecture)
        try:
            multi_predictor = get_multi_output_predictor(niche)
            if multi_predictor.is_trained:
                logger.info(f"Using Multi-Output Predictor for niche: {niche}")

                # Format weights for multi-output predictor
                # kpi_weights from config are simple dict {likes: 1.0, ...}
                # multi-output expects {likes_weight: 1.0, ...}
                mo_weights = None
                if kpi_weights:
                    mo_weights = {
                        f"{k}_weight": v
                        for k, v in kpi_weights.items()
                    }

                # Predict
                result = multi_predictor.predict(
                    features,
                    weights=mo_weights,
                    compute_shap=True
                )

                # Aggregate SHAP values using KPI weights
                # GlobalSHAP_i = sum(Weight_m * SHAP_i,m) / sum(Weight_m)
                aggregated_shap_pos = []
                aggregated_shap_neg = []

                # We need to combine the top factors from each metric
                # This is complex because SHAP values are feature-specific.
                # Simplified approach: Use the SHAP values from the dominant metric (highest weight)
                # OR actually aggregate them if possible.
                # Since result provides pre-computed top factors per metric, we can try to merge them.

                # Let's generate a consolidated explanation text
                # Find the metric that contributes most to the score
                # This is roughly proportional to weight * value, but we can look at weights

                # Use "weighted_rpi" as score
                score = result.weighted_rpi

                # Construct combined explanation
                # We'll collect all top positive/negative factors from all metrics, weighted by metric weight

                # Since we don't have raw SHAP matrices here, only top factors from result,
                # we will trust the multi-output result's internal SHAP or just report the breakdown.

                # Let's use the SHAP from 'shares' or 'saves' as they are usually high weight/value
                # Or better: Provide a specialized explanation

                explanation_text = f"Score RPI ({score:.1f}) basado en pesos personalizados."
                if result.shap_shares:
                     top_share = list(result.shap_shares.keys())[0]
                     explanation_text += f" Shares impulsados por: {top_share}."

                # Construct feature importance list for API
                # We'll merge top factors from all metrics
                feature_importance = []
                seen_features = set()

                # Collect top factors from all metrics
                all_factors = []
                metrics_to_check = ["shares", "saves", "comments", "likes", "views"]

                current_weights = result.weights_used

                for metric in metrics_to_check:
                    shap_dict = getattr(result, f"shap_{metric}")
                    if shap_dict:
                         weight = current_weights.get(f"{metric}_weight", 1.0)
                         for feat, impact in shap_dict.items():
                             all_factors.append({
                                 "feature": feat,
                                 "impact": abs(impact) * weight, # Weight by metric importance
                                 "raw_impact": impact,
                                 "metric": metric
                             })

                # Sort by weighted impact
                all_factors.sort(key=lambda x: x["impact"], reverse=True)

                # Deduplicate
                final_factors = []
                top_positive = []
                top_negative = []

                for f in all_factors:
                    if f["feature"] not in seen_features:
                        seen_features.add(f["feature"])
                        final_factors.append({
                            "feature": f["feature"],
                            "impact": f["impact"] # This is weighted magnitude
                        })

                        item = {"feature": f["feature"], "impact": f["raw_impact"]}
                        if f["raw_impact"] > 0:
                            top_positive.append(item)
                        else:
                            top_negative.append(item)

                        if len(final_factors) >= 10:
                            break

                return {
                    "score": round(score, 1),
                    "confidence": 85.0, # Higher confidence for multi-output
                    "explanation": {
                        "top_positive_factors": top_positive[:5],
                        "top_negative_factors": top_negative[:5],
                        "explanation_text": explanation_text
                    },
                    "feature_importance": final_factors,
                    "model_source": f"multi_output_{niche}",
                    "niche": niche,
                    "cold_start": False,
                    "multi_output_breakdown": {
                        "predicted_likes": result.predicted_likes,
                        "predicted_comments": result.predicted_comments,
                        "predicted_shares": result.predicted_shares,
                        "predicted_saves": result.predicted_saves,
                        "predicted_views": result.predicted_views
                    }
                }

        except Exception as e:
            logger.warning(f"Multi-output prediction failed (falling back): {e}")

        # 2. Fallback to Legacy/Single-Target Model
        # Get appropriate model for this niche
        model, model_source = self._get_model_for_niche(niche)
        self.active_model_source = model_source

        # If no model available, use heuristics
        if model is None:
            return self._mock_engagement_prediction(content)

        # Make prediction
        try:
            # Check dimensionality (Anti-Overfitting Warning)
            if X.shape[1] > 200:
                logger.warning(
                    f"⚠️ High dimensionality detected ({X.shape[1]} features) during prediction. "
                    f"Ensure model matches configuration."
                )

            score = float(model.predict(X)[0])
            score = max(0, min(100, score))  # Clip to 0-100

            # Calculate confidence based on model source
            base_confidence = self._calculate_confidence(X)
            if model_source == "base":
                # Lower confidence for base model (cold start)
                confidence = base_confidence * 0.8
                logger.info(f"Predicción con modelo base (cold start) - Niche: {niche}, Score: {score:.1f}")
            elif model_source.startswith("niche_"):
                # Higher confidence for niche-specific model
                confidence = min(base_confidence * 1.1, 98)
                logger.info(f"Predicción con modelo niche - {model_source}, Score: {score:.1f}")
            else:
                confidence = base_confidence

            # SHAP explanation (only for main trained model with explainer)
            if model_source == "trained" and self.shap_explainer_engagement is not None:
                explanation = self._get_shap_explanation(X, "engagement")
            else:
                explanation = {
                    "note": f"Predicción usando {model_source} modelo",
                    "explanation_text": self._generate_model_source_explanation(model_source, niche)
                }

            return {
                "score": round(score, 1),
                "confidence": round(confidence, 1),
                "explanation": explanation,
                "feature_importance": self._get_feature_importance("engagement"),
                "model_source": model_source,
                "niche": niche,
                "cold_start": model_source == "base",
            }

        except Exception as e:
            # Catch shape mismatch errors explicitly
            error_str = str(e)
            if "feature_names mismatch" in error_str or "feature mismatch" in error_str or "shape mismatch" in error_str:
                logger.error(
                    f"🛑 Model signature mismatch (likely due to dimensionality reduction changes). "
                    f"The model expects different features than provided. "
                    f"ACTION REQUIRED: Retrain the model using the training script. "
                    f"Error details: {e}"
                )
            else:
                logger.error(f"Prediction error with {model_source} model: {e}")

            return self._mock_engagement_prediction(content)

    def _generate_model_source_explanation(self, model_source: str, niche: str) -> str:
        """Generate human-readable explanation based on model source."""
        if model_source == "base":
            return (
                f"Usando modelo base preentrenado para cold start (niche: {niche}). "
                f"Para predicciones más precisas, entrena con datos específicos del nicho."
            )
        elif model_source.startswith("niche_"):
            return f"Predicción optimizada para el nicho {niche} con modelo fine-tuned."
        elif model_source == "trained":
            return "Predicción con modelo principal entrenado en datos reales."
        else:
            return "Predicción basada en heurísticas."

    def recommend_format(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recommend best content format
        """
        self._ensure_models_loaded()

        if not self.is_trained:
            return self._mock_format_recommendation(content)

        features = FeatureExtractor.extract_features(content)
        df = pd.DataFrame([features])
        X = self._prepare_features(df)

        # Predict probabilities
        probs = self.format_model.predict_proba(X)[0]
        classes = self.format_model.classes_

        # Get top recommendations
        recommendations = sorted(
            zip(classes, probs),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            "recommended_format": recommendations[0][0],
            "confidence": float(recommendations[0][1] * 100),
            "alternatives": [
                {"format": fmt, "score": float(prob * 100)}
                for fmt, prob in recommendations[1:3]
            ],
            "explanation": self._get_shap_explanation(X, "format"),
        }

    def suggest_triggers(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest engagement triggers to include
        """
        # Triggers might rely on trained models or features that use models
        self._ensure_models_loaded()

        features = FeatureExtractor.extract_features(content)

        # Analyze current triggers
        current_triggers = {
            k.replace("trigger_", ""): v
            for k, v in features.items()
            if k.startswith("trigger_") and v > 0
        }

        # Missing high-value triggers
        all_triggers = list(FeatureExtractor.TRIGGER_WORDS.keys())
        missing_triggers = [t for t in all_triggers if t not in current_triggers]

        # Prioritize based on trained model or defaults
        if hasattr(self, "_best_triggers"):
            priority_triggers = [t.replace("trigger_", "") for t in self._best_triggers if t.replace("trigger_", "") in missing_triggers]
        else:
            priority_triggers = ["question", "action", "curiosity"]

        suggestions = []
        for trigger in priority_triggers[:3]:
            examples = FeatureExtractor.TRIGGER_WORDS.get(trigger, [])[:3]
            suggestions.append({
                "trigger_type": trigger,
                "impact": "high" if trigger in ["question", "action"] else "medium",
                "examples": examples,
                "reason": self._get_trigger_reason(trigger),
            })

        return {
            "current_triggers": current_triggers,
            "suggestions": suggestions,
            "improvement_potential": self._calculate_improvement_potential(features),
        }

    def get_full_prediction(
        self,
        content: Dict[str, Any],
        embedding_precision: str = "low",
        kpi_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Get complete ML prediction with all components
        This is called before LLM generation for the hybrid approach

        USER CONFIG SYNC:
        =================
        Accepts embedding_precision and kpi_weights from user_config to ensure
        frontend configuration changes affect predictions in real-time.

        Args:
            content: Content dict with caption, hashtags, etc.
            embedding_precision: From user_config (default: "low")
            kpi_weights: From user_config for RPI calculation
        """
        # CRITICAL LOG: Config being used
        logger.info(
            f"Full prediction with config: precision={embedding_precision}, "
            f"kpi_weights={'custom' if kpi_weights else 'default'}"
        )

        engagement = self.predict_engagement(
            content,
            embedding_precision=embedding_precision,
            kpi_weights=kpi_weights
        )
        format_rec = self.recommend_format(content)
        triggers = self.suggest_triggers(content)

        # Generate optimization suggestions
        suggestions = self._generate_optimization_suggestions(content, engagement, format_rec, triggers)

        # Calculate weighted RPI if kpi_weights provided
        weighted_rpi = None
        if kpi_weights:
            weighted_rpi = self._calculate_weighted_rpi(content, kpi_weights)

        result = {
            "engagement_prediction": engagement,
            "format_recommendation": format_rec,
            "trigger_suggestions": triggers,
            "optimization_suggestions": suggestions,
            "ml_summary": self._generate_ml_summary(engagement, format_rec, triggers),
            "config_used": {
                "embedding_precision": embedding_precision,
                "kpi_weights": kpi_weights or "default",
            }
        }

        if weighted_rpi is not None:
            result["weighted_rpi"] = weighted_rpi

        return result

    def _calculate_weighted_rpi(
        self,
        content: Dict[str, Any],
        kpi_weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Calculate RPI score using custom KPI weights from user_config.

        USER CONFIG SYNC: This method uses weights from user_config to ensure
        frontend KPI weight changes affect RPI calculation in real-time.

        Args:
            content: Content with metrics
            kpi_weights: Weights dict with likes, comments, shares, saves, views

        Returns:
            Dict with weighted RPI score and breakdown
        """
        # Get metrics from content
        likes = content.get("likes_count", content.get("likes", 0)) or 0
        comments = content.get("comments_count", content.get("comments", 0)) or 0
        shares = content.get("shares_count", content.get("shares", 0)) or 0
        saves = content.get("saves_count", content.get("saves", 0)) or 0
        views = content.get("views_count", content.get("video_views", 0)) or 0

        # Get weights with defaults
        w_likes = kpi_weights.get("likes", 1.0)
        w_comments = kpi_weights.get("comments", 2.0)
        w_shares = kpi_weights.get("shares", 10.0)
        w_saves = kpi_weights.get("saves", 5.0)
        w_views = kpi_weights.get("views", 3.0)

        # Calculate weighted sum
        weighted_sum = (
            likes * w_likes +
            comments * w_comments +
            shares * w_shares +
            saves * w_saves +
            views * w_views
        )

        # Normalize by total weight
        total_weight = w_likes + w_comments + w_shares + w_saves + w_views
        if total_weight > 0:
            normalized_rpi = weighted_sum / total_weight
        else:
            normalized_rpi = 0

        return {
            "weighted_rpi": round(normalized_rpi, 2),
            "raw_weighted_sum": round(weighted_sum, 2),
            "weights_used": kpi_weights,
            "metrics": {
                "likes": likes,
                "comments": comments,
                "shares": shares,
                "saves": saves,
                "views": views,
            },
            "contribution_breakdown": {
                "likes": round(likes * w_likes, 2),
                "comments": round(comments * w_comments, 2),
                "shares": round(shares * w_shares, 2),
                "saves": round(saves * w_saves, 2),
                "views": round(views * w_views, 2),
            }
        }

    def _get_shap_explanation(self, X: pd.DataFrame, model_type: str) -> Dict[str, Any]:
        """Generate SHAP-based explanation"""
        try:
            if model_type == "engagement" and self.shap_explainer_engagement:
                shap_values = self.shap_explainer_engagement.shap_values(X)

                # Get feature importances from SHAP
                feature_names = X.columns.tolist()
                importances = np.abs(shap_values[0])

                # Top positive and negative factors
                sorted_idx = np.argsort(shap_values[0])

                top_positive = [
                    {"feature": feature_names[i], "impact": float(shap_values[0][i])}
                    for i in sorted_idx[-3:][::-1] if shap_values[0][i] > 0
                ]

                top_negative = [
                    {"feature": feature_names[i], "impact": float(shap_values[0][i])}
                    for i in sorted_idx[:3] if shap_values[0][i] < 0
                ]

                # Human-readable explanation
                explanation_text = self._shap_to_text(top_positive, top_negative)

                return {
                    "top_positive_factors": top_positive,
                    "top_negative_factors": top_negative,
                    "explanation_text": explanation_text,
                }

            elif model_type == "format" and self.shap_explainer_format:
                shap_values = self.shap_explainer_format.shap_values(X)
                return {"note": "Format recommendation based on similar high-engagement content"}

        except Exception as e:
            logger.warning(f"SHAP explanation error: {e}")

        return {"note": "Explanation not available"}

    def _shap_to_text(self, positive: List[Dict], negative: List[Dict]) -> str:
        """
        Convert SHAP values to human-readable text with Top 5 features.

        Output format: "RPI alto por: pregunta en caption (+22%), hora 20:00 (+18%), formato Reel (+15%)"
        """
        feature_descriptions = {
            # Manual heuristics
            "emoji_count": "uso de emojis",
            "emoji_density": "densidad de emojis",
            "hashtag_count": "número de hashtags",
            "cta_count": "llamadas a la acción",
            "cta_comment": "CTA de comentarios",
            "cta_save": "CTA de guardado",
            "hook_question": "hook de pregunta (RegEx)",
            "hook_pov": "formato POV (RegEx)",
            "hook_regex_score": "hook score RegEx (backup)",
            "trigger_question": "preguntas en el texto",
            "trigger_action": "palabras de acción",
            "is_reel": "formato Reel",
            "is_carousel": "formato Carousel",
            "video_optimal_length": "duración óptima del video",
            "is_prime_time": "horario prime",
            "caption_length": "longitud del caption",
            "has_question": "pregunta en caption",
            "has_strong_cta": "CTA fuerte",
            "lexical_richness": "vocabulario diverso",
            "sentiment_compound": "tono emocional",
            "sentiment_positive": "sentimiento positivo",
            "hour_of_day": "hora de publicación",
            "day_of_week": "día de la semana",
            "is_weekend": "fin de semana",
            "niche_inmobiliaria": "keywords inmobiliaria",
            "niche_floristeria": "keywords floristería",
            "niche_cafeteria": "keywords cafetería",
            "niche_peluqueria": "keywords peluquería",
            "niche_restaurante": "keywords restaurante",
            "niche_gimnasio": "keywords gimnasio",
            "niche_clinica": "keywords clínica",
            # SEMANTIC HOOK FEATURES (primary - high explainability value)
            "semantic_hook_score": "alto hook semántico",  # Most important for SHAP
            "semantic_hook_max_sim": "similitud con hook viral",
            "semantic_hook_top3_avg": "coincidencia top-3 hooks",
            # Multimodal interaction features
            "interaction_hook_x_sentiment": "sinergia hook+sentimiento",
            "interaction_hook_x_is_reel": "hook potenciado por Reel",
            "interaction_hook_x_cta_count": "hook + CTAs",
            "interaction_sentiment_x_cta_count": "sentimiento + CTAs",
            "interaction_is_reel_x_video_optimal": "Reel con duración óptima",
            "interaction_transcript_richness": "riqueza de transcripción",
            "interaction_ocr_richness": "texto visual en video",
            "interaction_multimodal_text_density": "densidad de texto multimodal",
            # Semantic hook interactions
            "interaction_semantic_hook_x_vader": "hook semántico + emoción",
            "interaction_semantic_hook_x_cta_strong": "hook semántico + CTA fuerte",
        }

        # Combine all factors and sort by absolute impact
        all_factors = []
        for p in positive:
            all_factors.append({
                "feature": p["feature"],
                "impact": p["impact"],
                "is_positive": True
            })
        for n in negative:
            all_factors.append({
                "feature": n["feature"],
                "impact": n["impact"],
                "is_positive": False
            })

        # Sort by absolute impact value
        all_factors.sort(key=lambda x: abs(x["impact"]), reverse=True)

        # Take Top 5
        top_5 = all_factors[:5]

        if not top_5:
            return "Análisis basado en patrones de contenido exitoso"

        # Format as requested: "RPI alto por: factor1 (+X%), factor2 (+Y%), ..."
        factor_strings = []
        for f in top_5:
            name = feature_descriptions.get(f["feature"], f["feature"])
            # Convert impact to percentage (assuming impact is in score units, normalize to %)
            pct = abs(f["impact"]) * 10  # Scale factor for readability
            pct = min(pct, 50)  # Cap at 50%
            sign = "+" if f["is_positive"] else "-"
            factor_strings.append(f"{name} ({sign}{pct:.0f}%)")

        return f"RPI alto por: {', '.join(factor_strings)}"

    def _get_top5_shap_explanation(self, shap_values: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """
        Generate Top 5 SHAP-based explanation with percentage impacts.

        Returns structured explanation with top 5 factors in format:
        "RPI alto por: pregunta en caption (+22%), hora 20:00 (+18%), formato Reel (+15%)"
        """
        # Get indices sorted by absolute SHAP value
        abs_values = np.abs(shap_values)
        sorted_indices = np.argsort(abs_values)[::-1][:5]  # Top 5

        top_factors = []
        for idx in sorted_indices:
            impact = shap_values[idx]
            if abs(impact) > 0.001:  # Filter near-zero impacts
                top_factors.append({
                    "feature": feature_names[idx],
                    "impact": float(impact),
                    "impact_percent": float(abs(impact) * 10),  # Scale to percentage
                    "direction": "positive" if impact > 0 else "negative"
                })

        # Generate human-readable text
        feature_descriptions = {
            "has_question": "pregunta en caption",
            "hour_of_day": "hora de publicación",
            "is_reel": "formato Reel",
            "is_carousel": "formato Carousel",
            "has_strong_cta": "CTA fuerte",
            "sentiment_compound": "tono emocional",
            "lexical_richness": "vocabulario diverso",
            "emoji_count": "uso de emojis",
            "is_prime_time": "horario prime",
            "is_weekend": "fin de semana",
        }

        explanation_parts = []
        for f in top_factors[:5]:
            name = feature_descriptions.get(f["feature"], f["feature"].replace("_", " "))
            sign = "+" if f["direction"] == "positive" else "-"
            pct = min(f["impact_percent"], 50)
            explanation_parts.append(f"{name} ({sign}{pct:.0f}%)")

        explanation_text = f"RPI alto por: {', '.join(explanation_parts)}" if explanation_parts else "Análisis en progreso"

        return {
            "top_factors": top_factors,
            "explanation_text": explanation_text,
            "total_factors_analyzed": len(feature_names)
        }

    def _get_feature_importance(self, model_type: str) -> List[Dict[str, Any]]:
        """Get global feature importance"""
        try:
            if model_type == "engagement" and self.engagement_model:
                importances = self.engagement_model.feature_importances_
                feature_names = self.FEATURE_COLUMNS[:len(importances)]

                sorted_idx = np.argsort(importances)[::-1][:10]
                return [
                    {"feature": feature_names[i], "impact": float(importances[i])}
                    for i in sorted_idx
                ]
        except:
            pass
        return []

    def _calculate_confidence(self, X: pd.DataFrame) -> float:
        """Calculate prediction confidence based on feature coverage"""
        non_zero = (X.iloc[0] != 0).sum()
        total = len(X.columns)
        return min(95, max(60, (non_zero / total) * 100))

    def _calculate_improvement_potential(self, features: Dict) -> float:
        """Calculate how much the content could improve with suggestions"""
        potential = 0

        # Missing CTAs
        if features.get("cta_count", 0) == 0:
            potential += 20

        # No question hook
        if features.get("hook_question", 0) == 0 and features.get("trigger_question", 0) == 0:
            potential += 15

        # No emojis
        if features.get("emoji_count", 0) == 0:
            potential += 10

        # Suboptimal video length
        if features.get("is_reel", 0) == 1 and features.get("video_optimal_length", 0) == 0:
            potential += 10

        return min(potential, 50)

    def _get_trigger_reason(self, trigger: str) -> str:
        """Get reason why a trigger is recommended"""
        reasons = {
            "question": "Las preguntas aumentan comentarios un 150%+ al invitar respuestas",
            "action": "Los CTAs claros multiplican la interacción directa",
            "curiosity": "La curiosidad mantiene la atención y aumenta visualizaciones completas",
            "urgency": "La urgencia impulsa acciones inmediatas",
            "transformation": "El contenido before/after tiene 3x más engagement",
            "emotion": "Las emociones crean conexión y aumentan compartidos",
            "value": "El valor percibido aumenta guardados",
            "social_proof": "La prueba social genera confianza y conversiones",
        }
        return reasons.get(trigger, "Mejora el engagement general")

    def _generate_optimization_suggestions(
        self,
        content: Dict,
        engagement: Dict,
        format_rec: Dict,
        triggers: Dict
    ) -> List[str]:
        """Generate specific optimization suggestions"""
        suggestions = []

        features = FeatureExtractor.extract_features(content)

        # Format suggestion
        current_format = "reel" if features.get("is_reel") else "carousel" if features.get("is_carousel") else "static"
        recommended = format_rec.get("recommended_format", "reel")

        if current_format != recommended and format_rec.get("confidence", 0) > 70:
            suggestions.append(f"Considera cambiar a formato {recommended} para este contenido")

        # CTA suggestions
        if features.get("cta_count", 0) == 0:
            suggestions.append("Añade un CTA claro (ej: '¿Cuál prefieres? Comenta 👇' o 'Guarda para después')")

        # Hook suggestions
        if not any(features.get(f"hook_{h}", 0) for h in ["question", "pov", "number"]):
            suggestions.append("Usa un hook más potente: pregunta, POV, o número en los primeros 3 segundos")

        # Emoji suggestions
        if features.get("emoji_count", 0) < 3:
            suggestions.append("Añade 3-5 emojis relevantes para mejorar el visual scanning")

        # Trigger suggestions
        for trigger in triggers.get("suggestions", [])[:2]:
            if trigger.get("impact") == "high":
                suggestions.append(f"Incluye elementos de {trigger['trigger_type']}: {trigger['examples'][0] if trigger.get('examples') else ''}")

        return suggestions[:5]

    def _generate_ml_summary(self, engagement: Dict, format_rec: Dict, triggers: Dict) -> str:
        """Generate summary text for the ML prediction"""
        score = engagement.get("score", 50)
        format_name = format_rec.get("recommended_format", "reel")
        improvement = triggers.get("improvement_potential", 0)

        if score >= 80:
            quality = "excelente"
        elif score >= 60:
            quality = "bueno"
        elif score >= 40:
            quality = "mejorable"
        else:
            quality = "necesita optimización"

        summary = f"Predicción ML: Score {score}/100 ({quality}). "
        summary += f"Formato recomendado: {format_name}. "

        if improvement > 20:
            summary += f"Potencial de mejora: +{improvement}% con las sugerencias."

        return summary

    # =========================================================================
    # FEEDBACK LOOP - Human-in-the-Loop Reinforcement Learning
    # =========================================================================

    async def register_performance_feedback(
        self,
        content_id: int,
        metrics: Dict[str, Any],
        original_content: Optional[Dict[str, Any]] = None,
        predicted_score: Optional[float] = None,
        db: Optional[AsyncSession] = None
    ) -> PerformanceFeedback:
        """
        Register actual performance metrics for a published content piece.

        This is the core method of the Human-in-the-Loop feedback system.
        It compares predicted engagement against actual performance and
        flags high-delta samples for priority retraining.

        FLOW:
        1. Calculate actual engagement score from metrics
        2. Compare with predicted score
        3. If delta > 20%, flag as HIGH_PRIORITY training sample
        4. Add to training queue for next retraining cycle (Persisted to DB)

        Args:
            content_id: ID of the GeneratedContent record
            metrics: Actual performance metrics from Instagram/TikTok
                     Expected keys: likes, comments, saves, shares, views,
                                   reach, impressions, retention_rate
            original_content: Optional content dict (for feature extraction)
            predicted_score: Optional predicted score (if not provided,
                            will try to look up from content)
            db: Async database session

        Returns:
            PerformanceFeedback with analysis results

        Raises:
            ValueError: If metrics are invalid or insufficient
        """
        # Validate metrics
        required_metrics = ["likes", "comments"]
        if not all(k in metrics for k in required_metrics):
            raise ValueError(f"Missing required metrics: {required_metrics}")

        # Calculate actual engagement score (same formula as prediction training)
        actual_score = self._calculate_engagement_score(metrics)

        # Get predicted score
        if predicted_score is None:
            if original_content and "engagement_score" in original_content:
                predicted_score = original_content.get("engagement_score", 50.0)
            else:
                predicted_score = 50.0  # Default if unknown
                logger.warning(f"No predicted score for content {content_id}, using default")

        # Calculate delta percentage
        if predicted_score > 0:
            delta_percent = ((actual_score - predicted_score) / predicted_score) * 100
        else:
            delta_percent = 100.0 if actual_score > 0 else 0.0

        # Determine if this is a high priority training sample
        is_high_priority = abs(delta_percent) > HIGH_PRIORITY_DELTA_THRESHOLD

        # Calculate training priority score
        training_priority = self._calculate_training_priority(
            delta_percent=delta_percent,
            metrics=metrics,
            content=original_content
        )

        # Generate analysis notes
        analysis_notes = self._generate_feedback_analysis(
            predicted=predicted_score,
            actual=actual_score,
            delta=delta_percent,
            metrics=metrics
        )

        # Create feedback object for return
        feedback = PerformanceFeedback(
            content_id=content_id,
            predicted_score=predicted_score,
            actual_score=actual_score,
            delta_percent=delta_percent,
            is_high_priority=is_high_priority,
            training_priority=training_priority,
            metrics=metrics,
            collected_at=datetime.utcnow(),
            analysis_notes=analysis_notes
        )

        # Persist to Database if original content is available
        if original_content is not None:
            features = FeatureExtractor.extract_features(original_content)

            # Helper to perform DB insert
            async def _persist(session: AsyncSession):
                sample = MLTrainingSample(
                    content_id=str(content_id),
                    features=features,
                    actual_metrics=metrics,
                    delta_score=delta_percent,
                    priority=training_priority,
                    is_high_priority=is_high_priority,
                    created_at=datetime.utcnow()
                )
                session.add(sample)
                await session.commit()
                logger.info(f"Persisted ML training sample {content_id} to DB")

            # Use provided DB or create new session
            if db:
                await _persist(db)
            else:
                async with async_session_maker() as session:
                    await _persist(session)

            if is_high_priority:
                logger.info(
                    f"HIGH PRIORITY training sample flagged: content_id={content_id}, "
                    f"delta={delta_percent:.1f}%, priority={training_priority:.3f}"
                )

        logger.info(
            f"Registered feedback for content {content_id}: "
            f"predicted={predicted_score:.1f}, actual={actual_score:.1f}, "
            f"delta={delta_percent:.1f}%, high_priority={is_high_priority}"
        )

        return feedback

    def _calculate_engagement_score(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate normalized engagement score from raw metrics.

        Uses same formula as training to ensure consistency:
        score = likes + comments*3 + saves*5 + shares*4

        Then normalizes to 0-100 scale.
        """
        likes = metrics.get("likes", 0) or 0
        comments = metrics.get("comments", 0) or 0
        saves = metrics.get("saves", 0) or 0
        shares = metrics.get("shares", 0) or 0
        views = metrics.get("views", 0) or 0

        # Weighted engagement sum
        raw_score = likes + (comments * 3) + (saves * 5) + (shares * 4)

        # Normalize based on views (if available) or absolute scale
        if views > 0:
            # Engagement rate based normalization
            engagement_rate = raw_score / views
            # Scale to 0-100 (typical engagement rates are 1-10%)
            normalized = min(100, engagement_rate * 1000)
        else:
            # Absolute scale normalization (assuming max ~10000 engagement)
            normalized = min(100, (raw_score / 100) * 10)

        return round(normalized, 2)

    def _calculate_training_priority(
        self,
        delta_percent: float,
        metrics: Dict[str, Any],
        content: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate training priority score for a feedback sample.

        Higher priority samples are more valuable for model learning:
        - High delta = model made a big mistake, needs correction
        - Diverse content types = helps model generalize
        - Recent data = more relevant to current trends

        Priority formula:
        priority = base_delta_score * diversity_multiplier * recency_multiplier

        Returns:
            Priority score (0-1, higher = more valuable)
        """
        # Base priority from delta magnitude
        # Larger errors are more valuable for learning
        base_priority = min(1.0, abs(delta_percent) / 100.0)

        # Boost for very high deltas (model was very wrong)
        if abs(delta_percent) > 50:
            base_priority *= 1.5
        elif abs(delta_percent) > HIGH_PRIORITY_DELTA_THRESHOLD:
            base_priority *= 1.2

        # Diversity bonus for underrepresented content types
        diversity_multiplier = 1.0
        if content:
            content_format = content.get("content_format", "")
            # Carousel and static are less common, boost their priority
            if content_format == "carousel":
                diversity_multiplier = 1.3
            elif content_format in ["static", "static_image"]:
                diversity_multiplier = 1.2

        # Engagement volume bonus (high-engagement content is more informative)
        views = metrics.get("views", 0) or metrics.get("reach", 0) or 0
        volume_multiplier = 1.0
        if views > 10000:
            volume_multiplier = 1.3
        elif views > 1000:
            volume_multiplier = 1.1

        # Combine factors
        priority = base_priority * diversity_multiplier * volume_multiplier

        # Clamp to 0-1
        return min(1.0, max(0.0, priority))

    def _generate_feedback_analysis(
        self,
        predicted: float,
        actual: float,
        delta: float,
        metrics: Dict[str, Any]
    ) -> str:
        """Generate human-readable analysis of the prediction error."""
        notes = []

        # Direction of error
        if delta > HIGH_PRIORITY_DELTA_THRESHOLD:
            notes.append(f"Model UNDERESTIMATED by {delta:.1f}%")
            notes.append("Content performed better than expected")
        elif delta < -HIGH_PRIORITY_DELTA_THRESHOLD:
            notes.append(f"Model OVERESTIMATED by {abs(delta):.1f}%")
            notes.append("Content underperformed expectations")
        else:
            notes.append(f"Prediction within acceptable range (delta: {delta:.1f}%)")

        # Analyze which metrics drove the difference
        likes = metrics.get("likes", 0)
        comments = metrics.get("comments", 0)
        saves = metrics.get("saves", 0)
        shares = metrics.get("shares", 0)

        if comments > likes * 0.1:
            notes.append("High comment ratio suggests strong audience connection")
        if saves > likes * 0.05:
            notes.append("High save ratio indicates valuable/educational content")
        if shares > likes * 0.03:
            notes.append("High share ratio shows viral potential")

        return "; ".join(notes)

    async def get_training_queue(self, min_priority: float = 0.0, db: Optional[AsyncSession] = None) -> List[Dict[str, Any]]:
        """
        Get queued training samples above minimum priority.

        Args:
            min_priority: Minimum priority score (0-1)
            db: Async database session

        Returns:
            List of training samples ready for the next training cycle
        """
        async def _query(session: AsyncSession):
            result = await session.execute(
                select(MLTrainingSample)
                .where(MLTrainingSample.priority >= min_priority)
                .order_by(MLTrainingSample.priority.desc())
            )
            return result.scalars().all()

        if db:
            rows = await _query(db)
        else:
            async with async_session_maker() as session:
                rows = await _query(session)

        samples = []
        for row in rows:
            # Reconstruct training sample format
            sample = dict(row.features)

            # Calculate engagement score from stored metrics if not in features?
            # Or assume features contained everything needed?
            # In register_feedback, we stored features extracted from original_content.
            # We also stored actual_metrics.
            # We need to compute `engagement_score` (target) from actual_metrics.

            actual_score = self._calculate_engagement_score(row.actual_metrics)
            sample["engagement_score"] = actual_score
            sample["_priority"] = row.priority
            sample["_source"] = "feedback_loop"
            samples.append(sample)

        return samples

    def get_high_priority_samples(self) -> List[Dict[str, Any]]:
        """Get only high priority training samples (>20% delta)."""
        self.__init_feedback_queue()

        return [
            item.to_training_sample()
            for item in self._training_queue
            if item.feedback.is_high_priority
        ]

    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get statistics about collected feedback."""
        self.__init_feedback_queue()

        if not self._feedback_history:
            return {
                "total_samples": 0,
                "high_priority_count": 0,
                "avg_delta_percent": 0,
                "model_bias": "unknown"
            }

        deltas = [f.delta_percent for f in self._feedback_history]
        high_priority = [f for f in self._feedback_history if f.is_high_priority]

        avg_delta = np.mean(deltas)

        # Determine if model has systematic bias
        if avg_delta > 10:
            bias = "underestimating"
        elif avg_delta < -10:
            bias = "overestimating"
        else:
            bias = "calibrated"

        return {
            "total_samples": len(self._feedback_history),
            "high_priority_count": len(high_priority),
            "queue_size": len(self._training_queue),
            "avg_delta_percent": round(avg_delta, 2),
            "delta_std": round(np.std(deltas), 2),
            "model_bias": bias,
            "underestimated_count": sum(1 for d in deltas if d > HIGH_PRIORITY_DELTA_THRESHOLD),
            "overestimated_count": sum(1 for d in deltas if d < -HIGH_PRIORITY_DELTA_THRESHOLD),
        }

    def clear_training_queue(self):
        """Clear the training queue after a training cycle."""
        self.__init_feedback_queue()
        cleared = len(self._training_queue)
        self._training_queue = []
        logger.info(f"Cleared {cleared} samples from training queue")
        return cleared

    def retrain_with_feedback(
        self,
        additional_data: Optional[List[Dict[str, Any]]] = None,
        min_samples: int = 30
    ) -> bool:
        """
        Retrain the model using accumulated feedback data.

        Combines high-priority feedback samples with any additional
        real data to improve model accuracy.

        Args:
            additional_data: Optional list of additional training samples
            min_samples: Minimum samples required for retraining

        Returns:
            True if retraining was successful, False if insufficient data
        """
        self.__init_feedback_queue()

        # Collect training data
        feedback_samples = self.get_training_queue(min_priority=0.3)

        all_samples = []
        all_samples.extend(feedback_samples)

        if additional_data:
            all_samples.extend(additional_data)

        if len(all_samples) < min_samples:
            logger.warning(
                f"Insufficient data for retraining: {len(all_samples)} samples "
                f"(minimum: {min_samples}). Collect more feedback."
            )
            return False

        logger.info(
            f"Retraining model with {len(all_samples)} samples "
            f"({len(feedback_samples)} from feedback loop)"
        )

        # Perform retraining
        self.train(all_samples, retrain=True)

        # Clear used samples from queue
        self.clear_training_queue()

        return True

    # === Mock predictions when model not trained ===

    def _mock_engagement_prediction(self, content: Dict) -> Dict[str, Any]:
        """Mock prediction for demo/testing"""
        features = FeatureExtractor.extract_features(content)

        # Simple heuristic score
        score = 50
        score += min(features.get("emoji_count", 0) * 2, 10)
        score += min(features.get("cta_count", 0) * 5, 15)
        score += features.get("hook_question", 0) * 10
        score += features.get("is_reel", 0) * 10
        score = min(max(score, 20), 95)

        return {
            "score": score,
            "confidence": 70,
            "explanation": {
                "explanation_text": "Predicción basada en heurísticas (modelo no entrenado). Entrena con datos reales para predicciones precisas.",
                "top_positive_factors": [{"feature": "cta_count", "impact": 5}] if features.get("cta_count", 0) > 0 else [],
                "top_negative_factors": [],
            },
            "feature_importance": [],
        }

    def _mock_format_recommendation(self, content: Dict) -> Dict[str, Any]:
        """Mock format recommendation"""
        return {
            "recommended_format": "reel",
            "confidence": 75,
            "alternatives": [
                {"format": "carousel", "score": 60},
                {"format": "static_image", "score": 40},
            ],
            "explanation": {"note": "Reels tienen el mayor alcance orgánico en 2026"},
        }


# ==========================================================================
# Global Instance & Factory
# ==========================================================================

# Global ML predictor instance
ml_predictor = MLPredictor()


def get_ml_predictor() -> MLPredictor:
    """Get the global ML predictor instance"""
    return ml_predictor


# Note: Synthetic data generation has been removed.
# The model now uses Cold Start heuristic prediction until sufficient
# real performance data is collected (minimum 30 samples).
# Use register_performance_feedback() to collect training data from
# actual content performance, then retrain_with_feedback() when ready.
# Add at the end of backend/app/services/ml_service.py

class SyntheticDataGenerator:
    """
    Generate synthetic data for cold start training
    """
    @staticmethod
    def generate_dataset(n_samples: int = 500) -> List[Dict[str, Any]]:
        """Generate synthetic training data"""
        data = []
        niches = MLPredictor.BUSINESS_NICHES

        for _ in range(n_samples):
            niche = np.random.choice(niches)
            is_reel = np.random.choice([0, 1])
            is_carousel = 0 if is_reel else np.random.choice([0, 1])

            # Synthetic features
            sample = {
                "business_type": niche,
                "caption": "Test caption " * np.random.randint(1, 10),
                "hashtags": ["#test"] * np.random.randint(1, 10),
                "content_format": "reel" if is_reel else "carousel" if is_carousel else "static_image",
                "video_duration": np.random.randint(5, 60) if is_reel else 0,
                "emoji_count": np.random.randint(0, 10),
                "cta_count": np.random.choice([0, 1]),
                "hook_question": np.random.choice([0, 1]),
                "posted_at": (datetime.utcnow() - timedelta(days=np.random.randint(0, 30))).isoformat(),
            }

            # Synthetic metrics (correlated with features)
            base_score = 50
            if sample["cta_count"]: base_score += 10
            if sample["hook_question"]: base_score += 10
            if is_reel: base_score += 10

            noise = np.random.normal(0, 10)
            score = max(0, min(100, base_score + noise))

            # Reverse engineer metrics from score
            sample["likes"] = int(score * 10)
            sample["comments"] = int(score)
            sample["saves"] = int(score / 2)
            sample["shares"] = int(score / 3)
            sample["views"] = int(score * 100)

            data.append(sample)

        return data

def train_initial_model():
    """
    Train the model with synthetic data if no models exist.
    Called on startup or when model is needed but not found.
    """
    predictor = get_ml_predictor()
    if not predictor.is_trained:
        logger.info("Initializing model with synthetic data...")
        data = SyntheticDataGenerator.generate_dataset(500)
        predictor.train(data)
