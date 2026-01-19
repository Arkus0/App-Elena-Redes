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

CONFIGURABLE EMBEDDING DIMENSIONS:
==================================
Now supports configurable embedding precision for all modalities:
- "max" (full 384 dims): Best precision, recommended for SMB volumes
- "high" (384 dims): Same as max
- "medium" (256 dims): Balanced precision/speed
- "low" (128 dims): Ultra fast, lower precision

DEFAULT: Full 384 dims for all modalities (safe for typical SMB data volumes)

ARCHITECTURE (with full dims):
==============================
Late Fusion approach - each modality is encoded independently,
then concatenated before the final XGBoost predictor:

    Caption Text  → MiniLM → [384 dims]  →
    Transcript    → MiniLM → [384 dims]  →  [CONCAT] → XGBoost → Engagement Score
    OCR Text      → MiniLM → [384 dims]  →            + SHAP Explainability
    Hook Score    → scalar               →
    Heuristics    → ~50 features         →

Total feature dimensionality (full): 384 + 384 + 384 + 1 + ~50 + 10 ≈ 1200+ features
(XGBoost handles this efficiently with early stopping)

WHY LATE FUSION?
================
- Compatible with XGBoost (tree-based, handles heterogeneous features well)
- SHAP explainability preserved (can see which modality contributes most)
- Modular: each modality can be missing (graceful degradation to zeros)
- Simple to train: no complex attention mechanisms needed

Author: BrandPulse AI
"""

import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

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
# Configuration - Now supports configurable dimensions
# =============================================================================

# Default: Full 384 dims for all modalities (recommended for SMB volumes)
CAPTION_DEFAULT_DIM = 384    # Caption embeddings (primary text)
TRANSCRIPT_DEFAULT_DIM = 384  # Whisper transcription embeddings
OCR_DEFAULT_DIM = 384        # EasyOCR text overlay embeddings

# Precision level mappings
PRECISION_TO_DIMS = {
    "low": 128,
    "medium": 256,
    "high": 384,
    "max": 384,
}

# Embedding model (reuse from features_embeddings.py)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384    # Raw embedding dimension

# TruncatedSVD model paths for multimodal components (when reduction is needed)
TRANSCRIPT_SVD_PATH = MODELS_DIR / "transcript_svd_{dims}.pkl"
OCR_SVD_PATH = MODELS_DIR / "ocr_svd_{dims}.pkl"

# Feature name prefixes
TRANSCRIPT_PREFIX = "transcript_emb_"
OCR_PREFIX = "ocr_emb_"
INTERACTION_PREFIX = "interaction_"


# =============================================================================
# Helper Functions for Configurable Dimensions
# =============================================================================

def get_dims_for_precision(precision: str) -> int:
    """Get dimensions for a precision level."""
    return PRECISION_TO_DIMS.get(precision.lower(), EMBEDDING_DIM)


def get_transcript_feature_names(dims: int = TRANSCRIPT_DEFAULT_DIM) -> List[str]:
    """Get list of transcript embedding feature names."""
    return [f"{TRANSCRIPT_PREFIX}{i+1}" for i in range(dims)]


def get_ocr_feature_names(dims: int = OCR_DEFAULT_DIM) -> List[str]:
    """Get list of OCR embedding feature names."""
    return [f"{OCR_PREFIX}{i+1}" for i in range(dims)]


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
        # Semantic hook interactions
        f"{INTERACTION_PREFIX}semantic_hook_x_vader",
        f"{INTERACTION_PREFIX}semantic_hook_x_cta_strong",
    ]


def get_multimodal_feature_names(
    transcript_dims: int = TRANSCRIPT_DEFAULT_DIM,
    ocr_dims: int = OCR_DEFAULT_DIM
) -> List[str]:
    """Get all multimodal feature names (transcript + OCR + interactions)."""
    return (
        get_transcript_feature_names(transcript_dims) +
        get_ocr_feature_names(ocr_dims) +
        get_interaction_feature_names()
    )


# Default feature columns (full dims)
TRANSCRIPT_FEATURE_COLUMNS = get_transcript_feature_names(TRANSCRIPT_DEFAULT_DIM)
OCR_FEATURE_COLUMNS = get_ocr_feature_names(OCR_DEFAULT_DIM)
INTERACTION_FEATURE_COLUMNS = get_interaction_feature_names()
MULTIMODAL_FEATURE_COLUMNS = get_multimodal_feature_names()

# Legacy compatibility (reduced dims)
TRANSCRIPT_PCA_DIM = TRANSCRIPT_DEFAULT_DIM  # Now full by default
OCR_PCA_DIM = OCR_DEFAULT_DIM  # Now full by default
CAPTION_PCA_DIM = CAPTION_DEFAULT_DIM  # Alias for backward compatibility


# =============================================================================
# Multimodal Embedding Extractor - Configurable Precision
# =============================================================================

class MultimodalEmbeddingExtractor:
    """
    Extract and optionally reduce embeddings for multiple text modalities.

    NEW: Supports configurable embedding precision (full 384 vs reduced dims)

    Handles:
    - Transcript text (from Whisper ASR)
    - OCR text (from EasyOCR visual text detection)

    Uses the same sentence-transformers model as caption embeddings,
    with optional TruncatedSVD reduction for each modality.

    Example:
        # Full 384 dims (default - recommended)
        extractor = MultimodalEmbeddingExtractor(precision="max")

        # Reduced dims for very large datasets
        extractor = MultimodalEmbeddingExtractor(precision="medium")  # 256 dims
    """

    # Class-level singleton for embedding model (shared with features_embeddings)
    _model = None
    _model_loaded = False

    # TruncatedSVD reducers for each dimension level
    _transcript_reducers: Dict[int, Any] = {}
    _transcript_reducers_fitted: Dict[int, bool] = {}
    _ocr_reducers: Dict[int, Any] = {}
    _ocr_reducers_fitted: Dict[int, bool] = {}

    def __init__(
        self,
        precision: str = "max",
        transcript_dims: Optional[int] = None,
        ocr_dims: Optional[int] = None,
        auto_load_reducers: bool = True
    ):
        """
        Initialize multimodal embedding extractor.

        Args:
            precision: Embedding precision level ("low", "medium", "high", "max")
                      Default is "max" (full 384 dims)
            transcript_dims: Override transcript dimensions (optional)
            ocr_dims: Override OCR dimensions (optional)
            auto_load_reducers: Whether to auto-load reducers if they exist
        """
        self._precision = precision.lower()

        # Get dimensions from precision or overrides
        base_dims = get_dims_for_precision(self._precision)
        self.transcript_dims = transcript_dims if transcript_dims else base_dims
        self.ocr_dims = ocr_dims if ocr_dims else base_dims

        # Check if reduction is needed
        self._uses_reduction = (self.transcript_dims < EMBEDDING_DIM or
                                self.ocr_dims < EMBEDDING_DIM)

        self._available = self._check_availability()

        # Auto-load reducers if needed
        if auto_load_reducers and self._uses_reduction:
            self._load_reducers()

        logger.info(
            f"MultimodalEmbeddingExtractor initialized: precision={self._precision}, "
            f"transcript_dims={self.transcript_dims}, ocr_dims={self.ocr_dims}"
        )

    @property
    def precision(self) -> str:
        """Current embedding precision."""
        return self._precision

    @property
    def uses_reduction(self) -> bool:
        """Whether dimensionality reduction is applied."""
        return self._uses_reduction

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

    def _load_reducers(self):
        """Load pre-fitted reducers for transcript and OCR."""
        # Load transcript reducer
        if self.transcript_dims < EMBEDDING_DIM:
            path = MODELS_DIR / f"transcript_svd_{self.transcript_dims}.pkl"
            if path.exists():
                try:
                    with open(path, 'rb') as f:
                        MultimodalEmbeddingExtractor._transcript_reducers[self.transcript_dims] = pickle.load(f)
                    MultimodalEmbeddingExtractor._transcript_reducers_fitted[self.transcript_dims] = True
                    logger.info(f"Loaded transcript TruncatedSVD from: {path}")
                except Exception as e:
                    logger.warning(f"Could not load transcript reducer: {e}")

        # Load OCR reducer
        if self.ocr_dims < EMBEDDING_DIM:
            path = MODELS_DIR / f"ocr_svd_{self.ocr_dims}.pkl"
            if path.exists():
                try:
                    with open(path, 'rb') as f:
                        MultimodalEmbeddingExtractor._ocr_reducers[self.ocr_dims] = pickle.load(f)
                    MultimodalEmbeddingExtractor._ocr_reducers_fitted[self.ocr_dims] = True
                    logger.info(f"Loaded OCR TruncatedSVD from: {path}")
                except Exception as e:
                    logger.warning(f"Could not load OCR reducer: {e}")

    def _init_reducer(self, modality: str, dims: int):
        """Initialize TruncatedSVD reducer for a modality."""
        from sklearn.decomposition import TruncatedSVD

        if modality == "transcript":
            if dims not in MultimodalEmbeddingExtractor._transcript_reducers:
                MultimodalEmbeddingExtractor._transcript_reducers[dims] = TruncatedSVD(
                    n_components=dims, random_state=42
                )
                logger.info(f"Initialized transcript TruncatedSVD: {dims} components")
        elif modality == "ocr":
            if dims not in MultimodalEmbeddingExtractor._ocr_reducers:
                MultimodalEmbeddingExtractor._ocr_reducers[dims] = TruncatedSVD(
                    n_components=dims, random_state=42
                )
                logger.info(f"Initialized OCR TruncatedSVD: {dims} components")

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
    # Reducer Fitting and Transformation
    # =========================================================================

    def fit_transcript_reducer(self, texts: List[str], save: bool = True) -> "MultimodalEmbeddingExtractor":
        """Fit TruncatedSVD on transcript corpus."""
        if self.transcript_dims >= EMBEDDING_DIM:
            logger.info("No reduction needed for transcripts (full dims)")
            return self

        logger.info(f"Fitting transcript TruncatedSVD on {len(texts)} texts to {self.transcript_dims} dims...")

        embeddings = self.get_raw_embeddings_batch(texts)
        valid_mask = np.any(embeddings != 0, axis=1)
        valid_embeddings = embeddings[valid_mask]

        if len(valid_embeddings) < self.transcript_dims:
            logger.warning(f"Not enough valid transcripts ({len(valid_embeddings)}) for reduction")
            return self

        self._init_reducer("transcript", self.transcript_dims)
        reducer = MultimodalEmbeddingExtractor._transcript_reducers[self.transcript_dims]
        reducer.fit(valid_embeddings)
        MultimodalEmbeddingExtractor._transcript_reducers_fitted[self.transcript_dims] = True

        explained_var = sum(reducer.explained_variance_ratio_) * 100
        logger.info(f"Transcript TruncatedSVD fitted: {explained_var:.1f}% variance explained")

        if save:
            self._save_reducer("transcript")

        return self

    def fit_ocr_reducer(self, texts: List[str], save: bool = True) -> "MultimodalEmbeddingExtractor":
        """Fit TruncatedSVD on OCR text corpus."""
        if self.ocr_dims >= EMBEDDING_DIM:
            logger.info("No reduction needed for OCR (full dims)")
            return self

        logger.info(f"Fitting OCR TruncatedSVD on {len(texts)} texts to {self.ocr_dims} dims...")

        embeddings = self.get_raw_embeddings_batch(texts)
        valid_mask = np.any(embeddings != 0, axis=1)
        valid_embeddings = embeddings[valid_mask]

        if len(valid_embeddings) < self.ocr_dims:
            logger.warning(f"Not enough valid OCR texts ({len(valid_embeddings)}) for reduction")
            return self

        self._init_reducer("ocr", self.ocr_dims)
        reducer = MultimodalEmbeddingExtractor._ocr_reducers[self.ocr_dims]
        reducer.fit(valid_embeddings)
        MultimodalEmbeddingExtractor._ocr_reducers_fitted[self.ocr_dims] = True

        explained_var = sum(reducer.explained_variance_ratio_) * 100
        logger.info(f"OCR TruncatedSVD fitted: {explained_var:.1f}% variance explained")

        if save:
            self._save_reducer("ocr")

        return self

    def _save_reducer(self, modality: str):
        """Save TruncatedSVD reducer to disk."""
        MODELS_DIR.mkdir(exist_ok=True)

        if modality == "transcript":
            dims = self.transcript_dims
            reducer = MultimodalEmbeddingExtractor._transcript_reducers.get(dims)
            is_fitted = MultimodalEmbeddingExtractor._transcript_reducers_fitted.get(dims, False)
            if reducer and is_fitted:
                path = MODELS_DIR / f"transcript_svd_{dims}.pkl"
                with open(path, 'wb') as f:
                    pickle.dump(reducer, f)
                logger.info(f"Saved transcript TruncatedSVD to: {path}")
        elif modality == "ocr":
            dims = self.ocr_dims
            reducer = MultimodalEmbeddingExtractor._ocr_reducers.get(dims)
            is_fitted = MultimodalEmbeddingExtractor._ocr_reducers_fitted.get(dims, False)
            if reducer and is_fitted:
                path = MODELS_DIR / f"ocr_svd_{dims}.pkl"
                with open(path, 'wb') as f:
                    pickle.dump(reducer, f)
                logger.info(f"Saved OCR TruncatedSVD to: {path}")

    def transform_transcript(self, embedding: np.ndarray) -> np.ndarray:
        """Transform embedding to transcript target dimensions."""
        # No reduction needed - return full embedding
        if self.transcript_dims >= EMBEDDING_DIM:
            return embedding

        # Reduction needed
        reducer = MultimodalEmbeddingExtractor._transcript_reducers.get(self.transcript_dims)
        is_fitted = MultimodalEmbeddingExtractor._transcript_reducers_fitted.get(self.transcript_dims, False)

        if not is_fitted or reducer is None:
            return np.zeros(self.transcript_dims, dtype=np.float32)

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        try:
            reduced = reducer.transform(embedding)
            return reduced.squeeze().astype(np.float32)
        except Exception as e:
            logger.error(f"Transcript transform failed: {e}")
            return np.zeros(self.transcript_dims, dtype=np.float32)

    def transform_ocr(self, embedding: np.ndarray) -> np.ndarray:
        """Transform embedding to OCR target dimensions."""
        # No reduction needed - return full embedding
        if self.ocr_dims >= EMBEDDING_DIM:
            return embedding

        # Reduction needed
        reducer = MultimodalEmbeddingExtractor._ocr_reducers.get(self.ocr_dims)
        is_fitted = MultimodalEmbeddingExtractor._ocr_reducers_fitted.get(self.ocr_dims, False)

        if not is_fitted or reducer is None:
            return np.zeros(self.ocr_dims, dtype=np.float32)

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        try:
            reduced = reducer.transform(embedding)
            return reduced.squeeze().astype(np.float32)
        except Exception as e:
            logger.error(f"OCR transform failed: {e}")
            return np.zeros(self.ocr_dims, dtype=np.float32)

    # =========================================================================
    # Feature Extraction
    # =========================================================================

    def get_transcript_features(self, text: str) -> Dict[str, float]:
        """Get transcript embedding features."""
        raw_emb = self.get_raw_embedding(text)

        # Apply reduction if needed
        if self.transcript_dims < EMBEDDING_DIM:
            final_emb = self.transform_transcript(raw_emb)
        else:
            final_emb = raw_emb

        features = {}
        for i, val in enumerate(final_emb):
            features[f"{TRANSCRIPT_PREFIX}{i+1}"] = round(float(val), 6)

        return features

    def get_ocr_features(self, text: str) -> Dict[str, float]:
        """Get OCR embedding features."""
        raw_emb = self.get_raw_embedding(text)

        # Apply reduction if needed
        if self.ocr_dims < EMBEDDING_DIM:
            final_emb = self.transform_ocr(raw_emb)
        else:
            final_emb = raw_emb

        features = {}
        for i, val in enumerate(final_emb):
            features[f"{OCR_PREFIX}{i+1}"] = round(float(val), 6)

        return features

    def get_transcript_features_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Get transcript features for batch of texts."""
        raw_embs = self.get_raw_embeddings_batch(texts)

        results = []
        for i in range(len(texts)):
            if self.transcript_dims < EMBEDDING_DIM:
                final_emb = self.transform_transcript(raw_embs[i])
            else:
                final_emb = raw_embs[i]

            features = {}
            for j, val in enumerate(final_emb):
                features[f"{TRANSCRIPT_PREFIX}{j+1}"] = round(float(val), 6)
            results.append(features)

        return results

    def get_ocr_features_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Get OCR features for batch of texts."""
        raw_embs = self.get_raw_embeddings_batch(texts)

        results = []
        for i in range(len(texts)):
            if self.ocr_dims < EMBEDDING_DIM:
                final_emb = self.transform_ocr(raw_embs[i])
            else:
                final_emb = raw_embs[i]

            features = {}
            for j, val in enumerate(final_emb):
                features[f"{OCR_PREFIX}{j+1}"] = round(float(val), 6)
            results.append(features)

        return results

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def is_transcript_reducer_fitted(self) -> bool:
        if self.transcript_dims >= EMBEDDING_DIM:
            return True  # No reduction needed
        return MultimodalEmbeddingExtractor._transcript_reducers_fitted.get(self.transcript_dims, False)

    @property
    def is_ocr_reducer_fitted(self) -> bool:
        if self.ocr_dims >= EMBEDDING_DIM:
            return True  # No reduction needed
        return MultimodalEmbeddingExtractor._ocr_reducers_fitted.get(self.ocr_dims, False)

    # Backward compatibility aliases
    @property
    def transcript_pca_components(self) -> int:
        return self.transcript_dims

    @property
    def ocr_pca_components(self) -> int:
        return self.ocr_dims

    @property
    def is_transcript_pca_fitted(self) -> bool:
        return self.is_transcript_reducer_fitted

    @property
    def is_ocr_pca_fitted(self) -> bool:
        return self.is_ocr_reducer_fitted

    def fit_transcript_pca(self, texts: List[str], save: bool = True):
        """Backward compatible alias."""
        return self.fit_transcript_reducer(texts, save)

    def fit_ocr_pca(self, texts: List[str], save: bool = True):
        """Backward compatible alias."""
        return self.fit_ocr_reducer(texts, save)

    def get_status(self) -> Dict[str, Any]:
        """Get status of the multimodal extractor."""
        return {
            "precision": self._precision,
            "transcript_dims": self.transcript_dims,
            "ocr_dims": self.ocr_dims,
            "uses_reduction": self._uses_reduction,
            "model_loaded": MultimodalEmbeddingExtractor._model_loaded,
            "transcript_reducer_fitted": self.is_transcript_reducer_fitted,
            "ocr_reducer_fitted": self.is_ocr_reducer_fitted,
        }


# =============================================================================
# Global Instance (Singleton)
# =============================================================================

_multimodal_extractors: Dict[str, MultimodalEmbeddingExtractor] = {}


def get_multimodal_extractor(precision: str = "max") -> MultimodalEmbeddingExtractor:
    """Get global multimodal embedding extractor instance."""
    key = precision.lower()
    if key not in _multimodal_extractors:
        _multimodal_extractors[key] = MultimodalEmbeddingExtractor(precision=key)
    return _multimodal_extractors[key]


# =============================================================================
# Main Fusion Function
# =============================================================================

def fuse_multimodal_features(
    df: pd.DataFrame,
    fit_reducers_if_needed: bool = True,
    save_reducers: bool = True,
    precision: str = "max"
) -> pd.DataFrame:
    """
    Fuse multimodal features for engagement prediction.

    Takes a DataFrame with existing features and adds:
    - Transcript embeddings from whisper_transcript column
    - OCR embeddings from easyocr_text column
    - Cross-modal interaction features

    Args:
        df: DataFrame with columns:
            - Existing heuristic features (caption_length, emoji_count, etc.)
            - Caption embeddings (embedding_1 to embedding_N)
            - hook_score (or hook_* features)
            - whisper_transcript (optional, for video/reel)
            - easyocr_text (optional, for video/reel)
            - media_type (optional, to detect video content)
            - sentiment_compound, is_reel, cta_count (for interactions)
        fit_reducers_if_needed: If True, fit reducers if not already fitted
        save_reducers: If True, save fitted reducers
        precision: Embedding precision ("low", "medium", "high", "max")

    Returns:
        DataFrame with additional multimodal features
    """
    df = df.copy()
    n_samples = len(df)

    # Get extractor with specified precision
    extractor = get_multimodal_extractor(precision)

    logger.info("=" * 60)
    logger.info(f"MULTIMODAL LATE FUSION (precision={precision})")
    logger.info("=" * 60)
    logger.info(f"Input shape: {df.shape}")
    logger.info(f"Transcript dims: {extractor.transcript_dims}, OCR dims: {extractor.ocr_dims}")

    # Get feature column names for this precision
    transcript_cols = get_transcript_feature_names(extractor.transcript_dims)
    ocr_cols = get_ocr_feature_names(extractor.ocr_dims)
    interaction_cols = get_interaction_feature_names()

    # =========================================================================
    # 1. Extract Transcript Embeddings
    # =========================================================================

    if 'whisper_transcript' in df.columns:
        transcripts = df['whisper_transcript'].fillna("").astype(str).tolist()
        non_empty_transcripts = [t for t in transcripts if t.strip()]

        logger.info(f"Processing {len(non_empty_transcripts)} non-empty transcripts...")

        # Fit reducer if needed (only for reduced precision)
        if fit_reducers_if_needed and extractor.transcript_dims < EMBEDDING_DIM:
            if not extractor.is_transcript_reducer_fitted:
                if len(non_empty_transcripts) >= extractor.transcript_dims:
                    logger.info("Fitting transcript TruncatedSVD on current corpus...")
                    extractor.fit_transcript_reducer(non_empty_transcripts, save=save_reducers)
                else:
                    logger.warning(
                        f"Not enough transcripts ({len(non_empty_transcripts)}) "
                        f"to fit reducer ({extractor.transcript_dims} components needed). Using zeros."
                    )

        # Extract features
        transcript_features = extractor.get_transcript_features_batch(transcripts)

        for col in transcript_cols:
            df[col] = [f.get(col, 0.0) for f in transcript_features]

        logger.info(f"Added {len(transcript_cols)} transcript embedding features")
    else:
        # No transcript column - add zeros
        logger.info("No whisper_transcript column found. Adding zero transcript features.")
        for col in transcript_cols:
            df[col] = 0.0

    # =========================================================================
    # 2. Extract OCR Embeddings
    # =========================================================================

    if 'easyocr_text' in df.columns:
        ocr_texts = df['easyocr_text'].fillna("").astype(str).tolist()
        non_empty_ocr = [t for t in ocr_texts if t.strip()]

        logger.info(f"Processing {len(non_empty_ocr)} non-empty OCR texts...")

        # Fit reducer if needed (only for reduced precision)
        if fit_reducers_if_needed and extractor.ocr_dims < EMBEDDING_DIM:
            if not extractor.is_ocr_reducer_fitted:
                if len(non_empty_ocr) >= extractor.ocr_dims:
                    logger.info("Fitting OCR TruncatedSVD on current corpus...")
                    extractor.fit_ocr_reducer(non_empty_ocr, save=save_reducers)
                else:
                    logger.warning(
                        f"Not enough OCR texts ({len(non_empty_ocr)}) "
                        f"to fit reducer ({extractor.ocr_dims} components needed). Using zeros."
                    )

        # Extract features
        ocr_features = extractor.get_ocr_features_batch(ocr_texts)

        for col in ocr_cols:
            df[col] = [f.get(col, 0.0) for f in ocr_features]

        logger.info(f"Added {len(ocr_cols)} OCR embedding features")
    else:
        # No OCR column - add zeros
        logger.info("No easyocr_text column found. Adding zero OCR features.")
        for col in ocr_cols:
            df[col] = 0.0

    # =========================================================================
    # 3. Generate Cross-Modal Interaction Features
    # =========================================================================

    logger.info("Generating cross-modal interaction features...")

    # Prefer semantic_hook_score (primary method) over RegEx-based hook_score
    if 'semantic_hook_score' in df.columns:
        hook_score = df['semantic_hook_score'].fillna(0)
        logger.info("  Using semantic_hook_score for interactions")
    elif 'hook_score' in df.columns:
        hook_score = df['hook_score'].fillna(0)
        logger.info("  Using legacy hook_score for interactions")
    else:
        hook_cols = [c for c in df.columns if c.startswith('hook_') and c not in ['hook_score', 'hook_regex_score']]
        if hook_cols:
            hook_score = df[hook_cols].sum(axis=1) / max(len(hook_cols), 1)
        else:
            hook_score = pd.Series([0.0] * n_samples)

    # Base features for interactions
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

    df[f'{INTERACTION_PREFIX}transcript_richness'] = np.minimum(transcript_len / 500, 1.0)
    df[f'{INTERACTION_PREFIX}ocr_richness'] = np.minimum(ocr_len / 100, 1.0)

    total_text_len = caption_len + transcript_len + ocr_len
    df[f'{INTERACTION_PREFIX}multimodal_text_density'] = np.minimum(total_text_len / 1000, 1.0)

    # Semantic hook interactions
    has_strong_cta = df.get('has_strong_cta', pd.Series([0] * n_samples)).fillna(0)
    df[f'{INTERACTION_PREFIX}semantic_hook_x_vader'] = hook_score * abs(sentiment)
    df[f'{INTERACTION_PREFIX}semantic_hook_x_cta_strong'] = hook_score * has_strong_cta

    logger.info(f"Added {len(interaction_cols)} interaction features")

    # =========================================================================
    # 4. Handle Missing Values
    # =========================================================================

    all_multimodal_cols = transcript_cols + ocr_cols + interaction_cols
    for col in all_multimodal_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # =========================================================================
    # 5. Summary
    # =========================================================================

    logger.info("=" * 60)
    logger.info("MULTIMODAL FUSION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output shape: {df.shape}")
    logger.info(f"New multimodal features added: {len(all_multimodal_cols)}")
    logger.info(f"  - Transcript embeddings: {len(transcript_cols)}")
    logger.info(f"  - OCR embeddings: {len(ocr_cols)}")
    logger.info(f"  - Interaction features: {len(interaction_cols)}")

    return df


def is_video_content(row: pd.Series) -> bool:
    """Check if a row represents video content."""
    if 'media_type' in row.index:
        media_type = str(row['media_type']).lower()
        if media_type in ['reel', 'video', 'tiktok', 'tiktok_video']:
            return True

    if 'is_reel' in row.index and row['is_reel'] == 1:
        return True

    if 'content_format' in row.index:
        fmt = str(row['content_format']).lower()
        if fmt in ['reel', 'video', 'tiktok', 'tiktok_video']:
            return True

    if 'video_duration' in row.index and row['video_duration'] > 0:
        return True

    return False


def add_multimodal_features_conditional(
    df: pd.DataFrame,
    fit_reducers_if_needed: bool = True,
    precision: str = "max"
) -> pd.DataFrame:
    """
    Add multimodal features conditionally based on content type.

    For video content: Full multimodal fusion
    For image/text content: Only interaction features
    """
    df = df.copy()

    has_video_content = False
    if 'media_type' in df.columns:
        has_video_content = df['media_type'].str.lower().isin(
            ['reel', 'video', 'tiktok', 'tiktok_video']
        ).any()
    elif 'is_reel' in df.columns:
        has_video_content = df['is_reel'].sum() > 0

    if has_video_content:
        logger.info("Video content detected - applying full multimodal fusion")
        return fuse_multimodal_features(
            df,
            fit_reducers_if_needed=fit_reducers_if_needed,
            precision=precision
        )
    else:
        logger.info("No video content - adding zero multimodal features")
        extractor = get_multimodal_extractor(precision)
        all_cols = get_multimodal_feature_names(extractor.transcript_dims, extractor.ocr_dims)
        for col in all_cols:
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
    print("MULTIMODAL LATE FUSION TEST - CONFIGURABLE PRECISION")
    print("=" * 70 + "\n")

    # Create test DataFrame
    test_data = {
        'caption': [
            "Nuevo Reel! Mira este truco increíble",
            "POV: cuando tu café está perfecto",
            "3 errores que cometes al grabar Reels",
        ],
        'whisper_transcript': [
            "Hola a todos, hoy les voy a mostrar un truco que cambió mi vida...",
            "El secreto de un buen café está en la temperatura del agua...",
            "Error número uno, no usar buena iluminación...",
        ],
        'easyocr_text': ["TRUCO #1", "CAFÉ PERFECTO", "3 ERRORES"],
        'hook_score': [0.85, 0.72, 0.91],
        'sentiment_compound': [0.6, 0.4, -0.1],
        'is_reel': [1, 1, 1],
        'cta_count': [2, 1, 3],
        'video_optimal_length': [1, 1, 0],
        'caption_length': [45, 38, 42],
        'media_type': ['reel', 'reel', 'reel'],
    }

    df = pd.DataFrame(test_data)

    # Test each precision level
    for precision in ["max", "medium", "low"]:
        print(f"\n{'='*60}")
        print(f"TESTING PRECISION: {precision.upper()}")
        print(f"{'='*60}")

        start = time.time()
        df_fused = fuse_multimodal_features(
            df.copy(),
            fit_reducers_if_needed=True,
            save_reducers=False,
            precision=precision
        )
        elapsed = time.time() - start

        print(f"\nResult:")
        print(f"  Output shape: {df_fused.shape}")
        print(f"  New columns: {df_fused.shape[1] - df.shape[1]}")
        print(f"  Time: {elapsed:.3f}s")

    print("\n" + "=" * 70)
    print("TEST COMPLETE!")
    print("Default precision is 'max' (full dims) - safe for SMB volumes")
    print("=" * 70 + "\n")
