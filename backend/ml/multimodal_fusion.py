#!/usr/bin/env python3
"""
Multimodal Late Fusion for Engagement Prediction
=================================================

Implements late fusion strategy for combining multiple modalities:
- Caption embeddings (text from post caption)
- Transcript embeddings (audio transcription via Whisper)
- OCR embeddings (visual text overlay via EasyOCR)
- Hook scores (visual attention in first 3 seconds)
- Heuristic features (timing, format, sentiment, etc.)

ARCHITECTURE:
=============
Late Fusion approach - each modality is encoded independently,
then concatenated before the final XGBoost predictor:

    Caption Text  → MiniLM → PCA(30) →
    Transcript    → MiniLM → PCA(20) →  [CONCAT] → XGBoost → Engagement Score
    OCR Text      → MiniLM → PCA(20) →            + SHAP Explainability
    Hook Score    → scalar           →
    Heuristics    → ~50 features     →

Total feature dimensionality: 30 + 20 + 20 + 1 + ~50 + interactions ≈ 130 features

WHY LATE FUSION?
================
- Compatible with XGBoost (tree-based, handles heterogeneous features well)
- SHAP explainability preserved (can see which modality contributes most)
- Modular: each modality can be missing (graceful degradation to zeros)
- Simple to train: no complex attention mechanisms needed

INTERACTION FEATURES:
====================
Cross-modal interactions capture synergies:
- hook_x_sentiment: Strong hook + positive sentiment = viral potential
- reel_format_x_hook: Reels benefit more from hook quality
- transcript_x_caption_similarity: Aligned audio/text = coherent message

Author: BrandPulse AI
"""

import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
BACKEND_MODELS_DIR = PROJECT_ROOT / "backend" / "ml_models"

# Ensure paths are in sys.path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# PCA dimensions for each modality
CAPTION_PCA_DIM = 30   # Caption embeddings (primary text)
TRANSCRIPT_PCA_DIM = 20  # Whisper transcription embeddings
OCR_PCA_DIM = 20       # EasyOCR text overlay embeddings

# Embedding model (reuse from features_embeddings.py)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384    # Raw embedding dimension

# PCA model paths for multimodal components
TRANSCRIPT_PCA_PATH = MODELS_DIR / "transcript_pca_20.pkl"
OCR_PCA_PATH = MODELS_DIR / "ocr_pca_20.pkl"

# Feature name prefixes
TRANSCRIPT_PREFIX = "transcript_emb_"
OCR_PREFIX = "ocr_emb_"
INTERACTION_PREFIX = "interaction_"


# =============================================================================
# Multimodal Embedding Extractor
# =============================================================================

class MultimodalEmbeddingExtractor:
    """
    Extract and reduce embeddings for multiple text modalities.

    Handles:
    - Transcript text (from Whisper ASR)
    - OCR text (from EasyOCR visual text detection)

    Uses the same sentence-transformers model as caption embeddings,
    but with separate PCA models for each modality.
    """

    # Class-level singleton for embedding model (shared with features_embeddings)
    _model = None
    _model_loaded = False

    # Separate PCA models for each modality
    _transcript_pca = None
    _transcript_pca_fitted = False
    _ocr_pca = None
    _ocr_pca_fitted = False

    def __init__(
        self,
        transcript_pca_components: int = TRANSCRIPT_PCA_DIM,
        ocr_pca_components: int = OCR_PCA_DIM,
        auto_load_pca: bool = True
    ):
        """
        Initialize multimodal embedding extractor.

        Args:
            transcript_pca_components: PCA dimensions for transcript (default: 20)
            ocr_pca_components: PCA dimensions for OCR text (default: 20)
            auto_load_pca: Whether to auto-load PCA models if they exist
        """
        self.transcript_pca_components = transcript_pca_components
        self.ocr_pca_components = ocr_pca_components

        self._available = self._check_availability()

        # Auto-load PCA models
        if auto_load_pca:
            self._load_pca_models()

    def _check_availability(self) -> bool:
        """Check if sentence-transformers is available."""
        try:
            from sentence_transformers import SentenceTransformer
            return True
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. Multimodal embeddings will return zeros. "
                "Install with: pip install sentence-transformers"
            )
            return False

    def _load_model(self):
        """Lazy load the embedding model (shared singleton)."""
        if MultimodalEmbeddingExtractor._model is not None:
            return

        if not self._available:
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model for multimodal fusion: {EMBEDDING_MODEL_NAME}")
            MultimodalEmbeddingExtractor._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            MultimodalEmbeddingExtractor._model_loaded = True
            logger.info("Multimodal embedding model loaded")

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            MultimodalEmbeddingExtractor._model_loaded = False

    def _load_pca_models(self):
        """Load pre-fitted PCA models for transcript and OCR."""
        # Load transcript PCA
        if TRANSCRIPT_PCA_PATH.exists():
            try:
                with open(TRANSCRIPT_PCA_PATH, 'rb') as f:
                    MultimodalEmbeddingExtractor._transcript_pca = pickle.load(f)
                MultimodalEmbeddingExtractor._transcript_pca_fitted = True
                logger.info(f"Loaded transcript PCA from: {TRANSCRIPT_PCA_PATH}")
            except Exception as e:
                logger.warning(f"Could not load transcript PCA: {e}")

        # Load OCR PCA
        if OCR_PCA_PATH.exists():
            try:
                with open(OCR_PCA_PATH, 'rb') as f:
                    MultimodalEmbeddingExtractor._ocr_pca = pickle.load(f)
                MultimodalEmbeddingExtractor._ocr_pca_fitted = True
                logger.info(f"Loaded OCR PCA from: {OCR_PCA_PATH}")
            except Exception as e:
                logger.warning(f"Could not load OCR PCA: {e}")

    def _init_pca(self, modality: str):
        """Initialize PCA model for a modality."""
        from sklearn.decomposition import PCA

        if modality == "transcript":
            if MultimodalEmbeddingExtractor._transcript_pca is None:
                MultimodalEmbeddingExtractor._transcript_pca = PCA(
                    n_components=self.transcript_pca_components
                )
                logger.info(f"Initialized transcript PCA: {self.transcript_pca_components} components")
        elif modality == "ocr":
            if MultimodalEmbeddingExtractor._ocr_pca is None:
                MultimodalEmbeddingExtractor._ocr_pca = PCA(
                    n_components=self.ocr_pca_components
                )
                logger.info(f"Initialized OCR PCA: {self.ocr_pca_components} components")

    # =========================================================================
    # Raw Embedding Generation
    # =========================================================================

    def get_raw_embedding(self, text: str) -> np.ndarray:
        """
        Get 384-dim raw embedding for text.

        Args:
            text: Input text

        Returns:
            numpy array of shape (384,)
        """
        if not self._available:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        if not text or not isinstance(text, str) or not text.strip():
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        self._load_model()

        if MultimodalEmbeddingExtractor._model is None:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        try:
            text = text.strip()[:2000]  # Truncate very long texts
            embedding = MultimodalEmbeddingExtractor._model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    def get_raw_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Get raw embeddings for multiple texts."""
        if not self._available or not texts:
            return np.zeros((len(texts) if texts else 0, EMBEDDING_DIM), dtype=np.float32)

        self._load_model()

        if MultimodalEmbeddingExtractor._model is None:
            return np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)

        try:
            cleaned = [t.strip()[:2000] if t and isinstance(t, str) else "" for t in texts]
            embeddings = MultimodalEmbeddingExtractor._model.encode(
                cleaned,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 100,
                batch_size=32
            )
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            return np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)

    # =========================================================================
    # PCA Fitting and Transformation
    # =========================================================================

    def fit_transcript_pca(self, texts: List[str], save: bool = True) -> "MultimodalEmbeddingExtractor":
        """Fit PCA on transcript corpus."""
        logger.info(f"Fitting transcript PCA on {len(texts)} texts...")

        embeddings = self.get_raw_embeddings_batch(texts)
        valid_mask = np.any(embeddings != 0, axis=1)
        valid_embeddings = embeddings[valid_mask]

        if len(valid_embeddings) < self.transcript_pca_components:
            logger.warning(f"Not enough valid transcripts ({len(valid_embeddings)}) for PCA")
            return self

        self._init_pca("transcript")
        MultimodalEmbeddingExtractor._transcript_pca.fit(valid_embeddings)
        MultimodalEmbeddingExtractor._transcript_pca_fitted = True

        explained_var = sum(MultimodalEmbeddingExtractor._transcript_pca.explained_variance_ratio_) * 100
        logger.info(f"Transcript PCA fitted: {explained_var:.1f}% variance explained")

        if save:
            self._save_pca("transcript")

        return self

    def fit_ocr_pca(self, texts: List[str], save: bool = True) -> "MultimodalEmbeddingExtractor":
        """Fit PCA on OCR text corpus."""
        logger.info(f"Fitting OCR PCA on {len(texts)} texts...")

        embeddings = self.get_raw_embeddings_batch(texts)
        valid_mask = np.any(embeddings != 0, axis=1)
        valid_embeddings = embeddings[valid_mask]

        if len(valid_embeddings) < self.ocr_pca_components:
            logger.warning(f"Not enough valid OCR texts ({len(valid_embeddings)}) for PCA")
            return self

        self._init_pca("ocr")
        MultimodalEmbeddingExtractor._ocr_pca.fit(valid_embeddings)
        MultimodalEmbeddingExtractor._ocr_pca_fitted = True

        explained_var = sum(MultimodalEmbeddingExtractor._ocr_pca.explained_variance_ratio_) * 100
        logger.info(f"OCR PCA fitted: {explained_var:.1f}% variance explained")

        if save:
            self._save_pca("ocr")

        return self

    def _save_pca(self, modality: str):
        """Save PCA model to disk."""
        MODELS_DIR.mkdir(exist_ok=True)

        if modality == "transcript" and MultimodalEmbeddingExtractor._transcript_pca_fitted:
            with open(TRANSCRIPT_PCA_PATH, 'wb') as f:
                pickle.dump(MultimodalEmbeddingExtractor._transcript_pca, f)
            logger.info(f"Saved transcript PCA to: {TRANSCRIPT_PCA_PATH}")
        elif modality == "ocr" and MultimodalEmbeddingExtractor._ocr_pca_fitted:
            with open(OCR_PCA_PATH, 'wb') as f:
                pickle.dump(MultimodalEmbeddingExtractor._ocr_pca, f)
            logger.info(f"Saved OCR PCA to: {OCR_PCA_PATH}")

    def transform_transcript(self, embedding: np.ndarray) -> np.ndarray:
        """Transform embedding to transcript PCA space."""
        if not MultimodalEmbeddingExtractor._transcript_pca_fitted:
            return np.zeros(self.transcript_pca_components, dtype=np.float32)

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        try:
            reduced = MultimodalEmbeddingExtractor._transcript_pca.transform(embedding)
            return reduced.squeeze().astype(np.float32)
        except Exception as e:
            logger.error(f"Transcript PCA transform failed: {e}")
            return np.zeros(self.transcript_pca_components, dtype=np.float32)

    def transform_ocr(self, embedding: np.ndarray) -> np.ndarray:
        """Transform embedding to OCR PCA space."""
        if not MultimodalEmbeddingExtractor._ocr_pca_fitted:
            return np.zeros(self.ocr_pca_components, dtype=np.float32)

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        try:
            reduced = MultimodalEmbeddingExtractor._ocr_pca.transform(embedding)
            return reduced.squeeze().astype(np.float32)
        except Exception as e:
            logger.error(f"OCR PCA transform failed: {e}")
            return np.zeros(self.ocr_pca_components, dtype=np.float32)

    # =========================================================================
    # Feature Extraction
    # =========================================================================

    def get_transcript_features(self, text: str) -> Dict[str, float]:
        """Get PCA-reduced transcript embedding features."""
        raw_emb = self.get_raw_embedding(text)
        reduced = self.transform_transcript(raw_emb)

        features = {}
        for i, val in enumerate(reduced):
            features[f"{TRANSCRIPT_PREFIX}{i+1}"] = round(float(val), 6)

        # Pad if needed
        for i in range(len(reduced), self.transcript_pca_components):
            features[f"{TRANSCRIPT_PREFIX}{i+1}"] = 0.0

        return features

    def get_ocr_features(self, text: str) -> Dict[str, float]:
        """Get PCA-reduced OCR embedding features."""
        raw_emb = self.get_raw_embedding(text)
        reduced = self.transform_ocr(raw_emb)

        features = {}
        for i, val in enumerate(reduced):
            features[f"{OCR_PREFIX}{i+1}"] = round(float(val), 6)

        # Pad if needed
        for i in range(len(reduced), self.ocr_pca_components):
            features[f"{OCR_PREFIX}{i+1}"] = 0.0

        return features

    def get_transcript_features_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Get transcript features for batch of texts."""
        raw_embs = self.get_raw_embeddings_batch(texts)

        results = []
        for i in range(len(texts)):
            reduced = self.transform_transcript(raw_embs[i])
            features = {}
            for j, val in enumerate(reduced):
                features[f"{TRANSCRIPT_PREFIX}{j+1}"] = round(float(val), 6)
            for j in range(len(reduced), self.transcript_pca_components):
                features[f"{TRANSCRIPT_PREFIX}{j+1}"] = 0.0
            results.append(features)

        return results

    def get_ocr_features_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Get OCR features for batch of texts."""
        raw_embs = self.get_raw_embeddings_batch(texts)

        results = []
        for i in range(len(texts)):
            reduced = self.transform_ocr(raw_embs[i])
            features = {}
            for j, val in enumerate(reduced):
                features[f"{OCR_PREFIX}{j+1}"] = round(float(val), 6)
            for j in range(len(reduced), self.ocr_pca_components):
                features[f"{OCR_PREFIX}{j+1}"] = 0.0
            results.append(features)

        return results

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def is_transcript_pca_fitted(self) -> bool:
        return MultimodalEmbeddingExtractor._transcript_pca_fitted

    @property
    def is_ocr_pca_fitted(self) -> bool:
        return MultimodalEmbeddingExtractor._ocr_pca_fitted


# =============================================================================
# Global Instance (Singleton)
# =============================================================================

_multimodal_extractor: Optional[MultimodalEmbeddingExtractor] = None


def get_multimodal_extractor() -> MultimodalEmbeddingExtractor:
    """Get global multimodal embedding extractor instance."""
    global _multimodal_extractor
    if _multimodal_extractor is None:
        _multimodal_extractor = MultimodalEmbeddingExtractor()
    return _multimodal_extractor


# =============================================================================
# Feature Column Names
# =============================================================================

def get_transcript_feature_names() -> List[str]:
    """Get list of transcript embedding feature names."""
    return [f"{TRANSCRIPT_PREFIX}{i+1}" for i in range(TRANSCRIPT_PCA_DIM)]


def get_ocr_feature_names() -> List[str]:
    """Get list of OCR embedding feature names."""
    return [f"{OCR_PREFIX}{i+1}" for i in range(OCR_PCA_DIM)]


def get_interaction_feature_names() -> List[str]:
    """Get list of cross-modal interaction feature names."""
    return [
        f"{INTERACTION_PREFIX}hook_x_sentiment",
        f"{INTERACTION_PREFIX}hook_x_is_reel",
        f"{INTERACTION_PREFIX}hook_x_cta_count",
        f"{INTERACTION_PREFIX}sentiment_x_cta_count",
        f"{INTERACTION_PREFIX}is_reel_x_video_optimal",
        f"{INTERACTION_PREFIX}transcript_richness",
        f"{INTERACTION_PREFIX}ocr_richness",
        f"{INTERACTION_PREFIX}multimodal_text_density",
        # Semantic hook interactions (added for enhanced engagement prediction)
        f"{INTERACTION_PREFIX}semantic_hook_x_vader",
        f"{INTERACTION_PREFIX}semantic_hook_x_cta_strong",
    ]


def get_multimodal_feature_names() -> List[str]:
    """Get all multimodal feature names (transcript + OCR + interactions)."""
    return (
        get_transcript_feature_names() +
        get_ocr_feature_names() +
        get_interaction_feature_names()
    )


TRANSCRIPT_FEATURE_COLUMNS = get_transcript_feature_names()
OCR_FEATURE_COLUMNS = get_ocr_feature_names()
INTERACTION_FEATURE_COLUMNS = get_interaction_feature_names()
MULTIMODAL_FEATURE_COLUMNS = get_multimodal_feature_names()


# =============================================================================
# Main Fusion Function
# =============================================================================

def fuse_multimodal_features(
    df: pd.DataFrame,
    fit_pca_if_needed: bool = True,
    save_pca: bool = True
) -> pd.DataFrame:
    """
    Fuse multimodal features for engagement prediction.

    Takes a DataFrame with existing features and adds:
    - Transcript embeddings (20 dims) from whisper_transcript column
    - OCR embeddings (20 dims) from easyocr_text column
    - Cross-modal interaction features

    Args:
        df: DataFrame with columns:
            - Existing heuristic features (caption_length, emoji_count, etc.)
            - Caption embeddings (embedding_1 to embedding_30)
            - hook_score (or hook_* features)
            - whisper_transcript (optional, for video/reel)
            - easyocr_text (optional, for video/reel)
            - media_type (optional, to detect video content)
            - sentiment_compound, is_reel, cta_count (for interactions)
        fit_pca_if_needed: If True, fit PCA on the data if not already fitted
        save_pca: If True, save fitted PCA models

    Returns:
        DataFrame with additional multimodal features:
        - transcript_emb_1 to transcript_emb_20
        - ocr_emb_1 to ocr_emb_20
        - interaction_* features

    Example:
        >>> df = pd.DataFrame({
        ...     'caption': ['Amazing video!'],
        ...     'whisper_transcript': ['Here is what I want to show you...'],
        ...     'easyocr_text': ['SALE 50% OFF'],
        ...     'hook_score': [0.85],
        ...     'sentiment_compound': [0.6],
        ...     'is_reel': [1],
        ...     'cta_count': [2],
        ...     # ... other features
        ... })
        >>> df_fused = fuse_multimodal_features(df)
        >>> print(df_fused.columns.tolist()[-10:])
    """
    df = df.copy()
    n_samples = len(df)

    logger.info("=" * 60)
    logger.info("MULTIMODAL LATE FUSION")
    logger.info("=" * 60)
    logger.info(f"Input shape: {df.shape}")

    # Get extractor
    extractor = get_multimodal_extractor()

    # =========================================================================
    # 1. Extract Transcript Embeddings
    # =========================================================================

    if 'whisper_transcript' in df.columns:
        transcripts = df['whisper_transcript'].fillna("").astype(str).tolist()
        non_empty_transcripts = [t for t in transcripts if t.strip()]

        logger.info(f"Processing {len(non_empty_transcripts)} non-empty transcripts...")

        # Fit PCA if needed
        if fit_pca_if_needed and not extractor.is_transcript_pca_fitted:
            if len(non_empty_transcripts) >= TRANSCRIPT_PCA_DIM:
                logger.info("Fitting transcript PCA on current corpus...")
                extractor.fit_transcript_pca(non_empty_transcripts, save=save_pca)
            else:
                logger.warning(
                    f"Not enough transcripts ({len(non_empty_transcripts)}) "
                    f"to fit PCA ({TRANSCRIPT_PCA_DIM} components needed). Using zeros."
                )

        # Extract features
        transcript_features = extractor.get_transcript_features_batch(transcripts)

        for col in TRANSCRIPT_FEATURE_COLUMNS:
            df[col] = [f.get(col, 0.0) for f in transcript_features]

        logger.info(f"Added {len(TRANSCRIPT_FEATURE_COLUMNS)} transcript embedding features")
    else:
        # No transcript column - add zeros
        logger.info("No whisper_transcript column found. Adding zero transcript features.")
        for col in TRANSCRIPT_FEATURE_COLUMNS:
            df[col] = 0.0

    # =========================================================================
    # 2. Extract OCR Embeddings
    # =========================================================================

    if 'easyocr_text' in df.columns:
        ocr_texts = df['easyocr_text'].fillna("").astype(str).tolist()
        non_empty_ocr = [t for t in ocr_texts if t.strip()]

        logger.info(f"Processing {len(non_empty_ocr)} non-empty OCR texts...")

        # Fit PCA if needed
        if fit_pca_if_needed and not extractor.is_ocr_pca_fitted:
            if len(non_empty_ocr) >= OCR_PCA_DIM:
                logger.info("Fitting OCR PCA on current corpus...")
                extractor.fit_ocr_pca(non_empty_ocr, save=save_pca)
            else:
                logger.warning(
                    f"Not enough OCR texts ({len(non_empty_ocr)}) "
                    f"to fit PCA ({OCR_PCA_DIM} components needed). Using zeros."
                )

        # Extract features
        ocr_features = extractor.get_ocr_features_batch(ocr_texts)

        for col in OCR_FEATURE_COLUMNS:
            df[col] = [f.get(col, 0.0) for f in ocr_features]

        logger.info(f"Added {len(OCR_FEATURE_COLUMNS)} OCR embedding features")
    else:
        # No OCR column - add zeros
        logger.info("No easyocr_text column found. Adding zero OCR features.")
        for col in OCR_FEATURE_COLUMNS:
            df[col] = 0.0

    # =========================================================================
    # 3. Generate Cross-Modal Interaction Features
    # =========================================================================

    logger.info("Generating cross-modal interaction features...")

    # Prefer semantic_hook_score (primary method) over RegEx-based hook_score
    if 'semantic_hook_score' in df.columns:
        hook_score = df['semantic_hook_score'].fillna(0)
        logger.info("  Using semantic_hook_score for interactions (embedding-based)")
    elif 'hook_score' in df.columns:
        hook_score = df['hook_score'].fillna(0)
        logger.info("  Using legacy hook_score for interactions")
    else:
        # Calculate from individual hook features (fallback)
        hook_cols = [c for c in df.columns if c.startswith('hook_') and c not in ['hook_score', 'hook_regex_score']]
        if hook_cols:
            hook_score = df[hook_cols].sum(axis=1) / max(len(hook_cols), 1)
        else:
            hook_score = pd.Series([0.0] * n_samples)
        logger.info("  Using calculated hook score from individual features")

    # Base features for interactions (with safe defaults)
    sentiment = df.get('sentiment_compound', pd.Series([0.0] * n_samples)).fillna(0)
    is_reel = df.get('is_reel', pd.Series([0] * n_samples)).fillna(0)
    cta_count = df.get('cta_count', pd.Series([0] * n_samples)).fillna(0)
    video_optimal = df.get('video_optimal_length', pd.Series([0] * n_samples)).fillna(0)

    # Interaction features
    df[f'{INTERACTION_PREFIX}hook_x_sentiment'] = hook_score * sentiment
    df[f'{INTERACTION_PREFIX}hook_x_is_reel'] = hook_score * is_reel
    df[f'{INTERACTION_PREFIX}hook_x_cta_count'] = hook_score * cta_count
    df[f'{INTERACTION_PREFIX}sentiment_x_cta_count'] = sentiment * cta_count
    df[f'{INTERACTION_PREFIX}is_reel_x_video_optimal'] = is_reel * video_optimal

    # Text richness from multimodal sources
    transcript_len = df.get('whisper_transcript', pd.Series([""] * n_samples)).fillna("").str.len()
    ocr_len = df.get('easyocr_text', pd.Series([""] * n_samples)).fillna("").str.len()
    caption_len = df.get('caption_length', pd.Series([0] * n_samples)).fillna(0)

    # Normalize to 0-1 range
    df[f'{INTERACTION_PREFIX}transcript_richness'] = np.minimum(transcript_len / 500, 1.0)
    df[f'{INTERACTION_PREFIX}ocr_richness'] = np.minimum(ocr_len / 100, 1.0)

    # Combined text density (normalized)
    total_text_len = caption_len + transcript_len + ocr_len
    df[f'{INTERACTION_PREFIX}multimodal_text_density'] = np.minimum(total_text_len / 1000, 1.0)

    # Semantic hook interactions (enhanced synergies with embedding-based hook detection)
    has_strong_cta = df.get('has_strong_cta', pd.Series([0] * n_samples)).fillna(0)
    df[f'{INTERACTION_PREFIX}semantic_hook_x_vader'] = hook_score * abs(sentiment)
    df[f'{INTERACTION_PREFIX}semantic_hook_x_cta_strong'] = hook_score * has_strong_cta

    logger.info(f"Added {len(INTERACTION_FEATURE_COLUMNS)} interaction features")

    # =========================================================================
    # 4. Handle Missing Values
    # =========================================================================

    # Fill NaN with 0 for all multimodal features
    multimodal_cols = TRANSCRIPT_FEATURE_COLUMNS + OCR_FEATURE_COLUMNS + INTERACTION_FEATURE_COLUMNS
    for col in multimodal_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # =========================================================================
    # 5. Summary Logging
    # =========================================================================

    n_new_features = len(multimodal_cols)
    logger.info("=" * 60)
    logger.info("MULTIMODAL FUSION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output shape: {df.shape}")
    logger.info(f"New multimodal features added: {n_new_features}")
    logger.info(f"  - Transcript embeddings: {len(TRANSCRIPT_FEATURE_COLUMNS)}")
    logger.info(f"  - OCR embeddings: {len(OCR_FEATURE_COLUMNS)}")
    logger.info(f"  - Interaction features: {len(INTERACTION_FEATURE_COLUMNS)}")
    logger.info(f"Total features: {df.shape[1]}")

    return df


def is_video_content(row: pd.Series) -> bool:
    """
    Check if a row represents video content (Reel/TikTok/Video).

    Args:
        row: DataFrame row

    Returns:
        True if video content, False otherwise
    """
    # Check media_type column
    if 'media_type' in row.index:
        media_type = str(row['media_type']).lower()
        if media_type in ['reel', 'video', 'tiktok', 'tiktok_video']:
            return True

    # Check is_reel column
    if 'is_reel' in row.index and row['is_reel'] == 1:
        return True

    # Check content_format column
    if 'content_format' in row.index:
        fmt = str(row['content_format']).lower()
        if fmt in ['reel', 'video', 'tiktok', 'tiktok_video']:
            return True

    # Check video_duration
    if 'video_duration' in row.index and row['video_duration'] > 0:
        return True

    return False


def add_multimodal_features_conditional(
    df: pd.DataFrame,
    fit_pca_if_needed: bool = True
) -> pd.DataFrame:
    """
    Add multimodal features conditionally based on content type.

    For video content (Reel/TikTok): Full multimodal fusion
    For image/text content: Only interaction features (others zeroed)

    Args:
        df: DataFrame with content data
        fit_pca_if_needed: Whether to fit PCA if not already fitted

    Returns:
        DataFrame with multimodal features added
    """
    df = df.copy()

    # Check if we have any video content
    has_video_content = False
    if 'media_type' in df.columns:
        has_video_content = df['media_type'].str.lower().isin(
            ['reel', 'video', 'tiktok', 'tiktok_video']
        ).any()
    elif 'is_reel' in df.columns:
        has_video_content = df['is_reel'].sum() > 0

    if has_video_content:
        # Full multimodal fusion for datasets with video
        logger.info("Video content detected - applying full multimodal fusion")
        return fuse_multimodal_features(df, fit_pca_if_needed=fit_pca_if_needed)
    else:
        # Just add zero multimodal columns for consistency
        logger.info("No video content - adding zero multimodal features for consistency")
        for col in MULTIMODAL_FEATURE_COLUMNS:
            df[col] = 0.0
        return df


# =============================================================================
# CLI Testing
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("\n" + "=" * 70)
    print("MULTIMODAL LATE FUSION TEST")
    print("=" * 70 + "\n")

    # Create test DataFrame
    test_data = {
        'caption': [
            "Nuevo Reel! Mira este truco increíble 🔥",
            "POV: cuando tu café está perfecto ☕",
            "3 errores que cometes al grabar Reels",
        ],
        'whisper_transcript': [
            "Hola a todos, hoy les voy a mostrar un truco que cambió mi vida...",
            "El secreto de un buen café está en la temperatura del agua...",
            "Error número uno, no usar buena iluminación...",
        ],
        'easyocr_text': [
            "TRUCO #1",
            "CAFÉ PERFECTO",
            "3 ERRORES",
        ],
        'hook_score': [0.85, 0.72, 0.91],
        'sentiment_compound': [0.6, 0.4, -0.1],
        'is_reel': [1, 1, 1],
        'cta_count': [2, 1, 3],
        'video_optimal_length': [1, 1, 0],
        'caption_length': [45, 38, 42],
        'media_type': ['reel', 'reel', 'reel'],
    }

    df = pd.DataFrame(test_data)

    print("[1] Test DataFrame created:")
    print(f"    Shape: {df.shape}")
    print(f"    Columns: {list(df.columns)}\n")

    # Apply multimodal fusion
    print("[2] Applying multimodal fusion...\n")
    df_fused = fuse_multimodal_features(df, fit_pca_if_needed=True, save_pca=False)

    print("\n[3] Result:")
    print(f"    Output shape: {df_fused.shape}")
    print(f"    New columns added: {df_fused.shape[1] - df.shape[1]}")

    # Show sample of new features
    print("\n[4] Sample multimodal features:")
    sample_cols = (
        ['transcript_emb_1', 'transcript_emb_2', 'ocr_emb_1', 'ocr_emb_2'] +
        INTERACTION_FEATURE_COLUMNS[:3]
    )
    existing_cols = [c for c in sample_cols if c in df_fused.columns]
    print(df_fused[existing_cols].round(4).to_string())

    print("\n" + "=" * 70)
    print("TEST COMPLETE!")
    print("=" * 70 + "\n")
