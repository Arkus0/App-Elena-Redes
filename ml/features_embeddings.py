#!/usr/bin/env python3
"""
Semantic Embeddings Feature Module for BrandPulse AI
=====================================================

Modernizes the feature engineering pipeline by adding semantic embeddings
from sentence-transformers with CONFIGURABLE dimensionality.

MODEL: sentence-transformers/all-MiniLM-L6-v2
- Lightweight: ~80MB model, fast inference
- Multilingual: Works well with Spanish and English
- Output: 384-dimensional dense vectors
- Normalized: L2 normalized embeddings

EMBEDDING PRECISION MODES (precision es REQUERIDO):
===================================================
- "ultra_low" (64 dims): Ultra rapido para PC modesto/sobremesa Almeria
- "low" (128 dims): Recomendado sobremesa normal - balance seguro
- "medium" (256 dims): Balance precision/velocidad
- "high" (384 dims): Full dims con TruncatedSVD (preserva varianza)
- "max" (None): Full raw 384 dims - sin reduccion

IMPORTANTE: No hay default - debes especificar precision explicitamente.
Recomendado para sobremesa normal Almeria: "low" (128 dims)

PIPELINE:
1. Load model (auto-downloads if not cached)
2. Generate 384-dim embedding from caption text
3. Apply TruncatedSVD if precision != "max" (configurable)
4. Return features for XGBoost

Usage:
    from ml.features_embeddings import EmbeddingExtractor, EmbeddingPrecision

    # IMPORTANTE: precision es REQUERIDO
    # Para sobremesa normal - recomendado:
    extractor = EmbeddingExtractor(precision="low")  # 128 dims
    features = extractor.get_embedding_features("Tu texto aqui")

    # Para PC modesto (muy rapido):
    extractor = EmbeddingExtractor(precision="ultra_low")  # 64 dims

    # Full raw sin reduccion (si tienes buen hardware):
    extractor = EmbeddingExtractor(precision="max")  # 384 dims raw

Author: BrandPulse AI
"""

import gc
import logging
import pickle
import time
import functools
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import warnings

import numpy as np

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration & Constants
# =============================================================================

# Model configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # Output dimension of all-MiniLM-L6-v2
MAX_SEQ_LENGTH = 256  # Max tokens for embedding

# File paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)


# =============================================================================
# Embedding Precision Enum
# =============================================================================

class EmbeddingPrecision(str, Enum):
    """
    Embedding precision levels for BrandPulse AI.

    IMPORTANT: No default - precision must be explicitly passed.
    Recommended for typical SMB (100-2000 posts): "low" (128 dims) for safety.

    Levels:
    - ultra_low (16 dims): For extreme noise contexts/scraping
    - low (32 dims): Recommended for Auxiliary/SMB contexts
    - medium (64 dims): Balanced
    - high (128 dims): High precision
    - max (None): Full raw 384 dims (Raw BERT/MiniLM output)
    """
    ULTRA_LOW = "ultra_low"  # 16 dims
    LOW = "low"              # 32 dims
    MEDIUM = "medium"        # 64 dims
    HIGH = "high"            # 128 dims
    MAX = "max"              # None = full raw 384 dims

    @property
    def dimensions(self) -> int:
        """Get number of dimensions for this precision level."""
        dims = PRECISION_TO_DIMS.get(self.value)
        # max returns None, which means full EMBEDDING_DIM
        return dims if dims is not None else EMBEDDING_DIM

    @property
    def uses_reduction(self) -> bool:
        """Check if this precision uses dimensionality reduction."""
        # max = full raw embeddings, no reduction
        return self.value in ("ultra_low", "low", "medium", "high")


# Precision to dimensions mapping
# IMPORTANT: max=None means full raw 384 dims (no reduction applied)
PRECISION_TO_DIMS: Dict[str, Optional[int]] = {
    "ultra_low": 16,   # Extreme noise context
    "low": 32,         # Recommended for auxiliary signals (Transcript/OCR)
    "medium": 64,      # Balanced for SMB
    "high": 128,       # High precision
    "max": None,       # Full raw 384 dims - no reduction
}

# REMOVED: No default precision - must be explicitly passed
# This forces callers to consciously choose precision based on their hardware


# =============================================================================
# Embedding Feature Names
# =============================================================================

def get_embedding_feature_names(dims: int = EMBEDDING_DIM, zero_based: bool = True) -> List[str]:
    """
    Get list of embedding feature column names.

    Args:
        dims: Number of dimensions
        zero_based: If True (default), returns embedding_0 to embedding_N-1
                   If False (legacy), returns embedding_1 to embedding_N

    Returns:
        List of feature names: ["embedding_0", ..., "embedding_N-1"] (0-based)
        or ["embedding_1", ..., "embedding_N"] (1-based legacy)

    Note:
        0-based indexing is now the default for consistency with XGBoost
        and the refactored GrowthPredictionEngine.
    """
    if zero_based:
        return [f"embedding_{i}" for i in range(dims)]
    else:
        # Legacy 1-based indexing (deprecated)
        return [f"embedding_{i+1}" for i in range(dims)]


# Default feature columns (full 384 dims)
EMBEDDING_FEATURE_COLUMNS = get_embedding_feature_names(EMBEDDING_DIM)
PCA_COMPONENTS = EMBEDDING_DIM  # Backward compatible - now represents full dims


# =============================================================================
# Memory & Resource Estimation
# =============================================================================

def estimate_memory_usage(n_samples: int, dims: int) -> Dict[str, Any]:
    """
    Estimate memory usage for embedding operations.

    Args:
        n_samples: Number of samples
        dims: Embedding dimensions

    Returns:
        Dict with memory estimates
    """
    # Float32 = 4 bytes per value
    bytes_per_embedding = dims * 4
    total_bytes = n_samples * bytes_per_embedding

    # Convert to human readable
    if total_bytes < 1024:
        size_str = f"{total_bytes} B"
    elif total_bytes < 1024 * 1024:
        size_str = f"{total_bytes / 1024:.2f} KB"
    elif total_bytes < 1024 * 1024 * 1024:
        size_str = f"{total_bytes / (1024 * 1024):.2f} MB"
    else:
        size_str = f"{total_bytes / (1024 * 1024 * 1024):.2f} GB"

    return {
        "n_samples": n_samples,
        "dimensions": dims,
        "bytes_per_embedding": bytes_per_embedding,
        "total_bytes": total_bytes,
        "human_readable": size_str,
        "is_lightweight": total_bytes < 2 * 1024 * 1024 * 1024,  # < 2GB
    }


def get_ram_usage_mb() -> Optional[float]:
    """
    Get current process RAM usage in MB.

    Returns:
        RAM usage in MB or None if psutil not available
    """
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return None


# =============================================================================
# Dimensionality Reduction Functions
# =============================================================================

def get_reduced_embeddings(
    embeddings: np.ndarray,
    dims: Optional[int] = None,
    fit_reducer: bool = False,
    reducer: Optional[Any] = None
) -> Tuple[np.ndarray, Optional[Any], Dict[str, Any]]:
    """
    Get embeddings with optional TruncatedSVD reduction.

    This is the core function for configurable embedding dimensionality.

    Args:
        embeddings: Raw embeddings array of shape (n_samples, 384)
        dims: Target dimensions. If None, returns full 384 dims (no reduction)
        fit_reducer: If True, fit a new TruncatedSVD. If False, use provided reducer
        reducer: Pre-fitted TruncatedSVD reducer (optional)

    Returns:
        Tuple of:
        - Reduced/full embeddings array
        - Fitted reducer (or None if no reduction)
        - Stats dict with timing, variance explained, etc.

    Example:
        # Full 384 dims (no reduction)
        full_embs, _, stats = get_reduced_embeddings(raw_embs, dims=None)
        print(stats)  # {"mode": "full", "dims": 384, ...}

        # Reduced to 256 dims
        reduced_embs, svd, stats = get_reduced_embeddings(raw_embs, dims=256, fit_reducer=True)
        print(stats)  # {"mode": "reduced", "dims": 256, "variance_explained": 0.95, ...}
    """
    start_time = time.time()
    stats = {
        "input_shape": embeddings.shape,
        "input_dims": embeddings.shape[1] if embeddings.ndim > 1 else 1,
    }

    # Get RAM before
    ram_before = get_ram_usage_mb()

    # Case 1: No reduction requested (full dims)
    if dims is None or dims >= EMBEDDING_DIM:
        elapsed = time.time() - start_time
        ram_after = get_ram_usage_mb()

        stats.update({
            "mode": "full",
            "output_dims": EMBEDDING_DIM,
            "reduction_applied": False,
            "time_seconds": round(elapsed, 4),
            "ram_before_mb": round(ram_before, 2) if ram_before else None,
            "ram_after_mb": round(ram_after, 2) if ram_after else None,
        })

        logger.info(f"Embeddings: full {EMBEDDING_DIM} dims (no reduction), took {elapsed:.3f}s")
        return embeddings, None, stats

    # Case 2: Reduction requested
    from sklearn.decomposition import TruncatedSVD

    # Validate dimensions
    dims = min(dims, EMBEDDING_DIM)
    dims = max(dims, 1)

    if embeddings.shape[0] < dims:
        logger.warning(
            f"Not enough samples ({embeddings.shape[0]}) for {dims} components. "
            f"Using {embeddings.shape[0]} components."
        )
        dims = embeddings.shape[0]

    # Fit or transform
    if fit_reducer or reducer is None:
        reducer = TruncatedSVD(n_components=dims, random_state=42)
        reduced = reducer.fit_transform(embeddings)
        variance_explained = float(sum(reducer.explained_variance_ratio_))
        logger.info(
            f"Embeddings: reducidos a {dims} dims (varianza {variance_explained*100:.1f}%)"
        )
    else:
        reduced = reducer.transform(embeddings)
        variance_explained = float(sum(reducer.explained_variance_ratio_)) if hasattr(reducer, 'explained_variance_ratio_') else None

    elapsed = time.time() - start_time
    ram_after = get_ram_usage_mb()

    stats.update({
        "mode": "reduced",
        "output_dims": dims,
        "reduction_applied": True,
        "variance_explained": variance_explained,
        "variance_percent": round(variance_explained * 100, 2) if variance_explained else None,
        "time_seconds": round(elapsed, 4),
        "ram_before_mb": round(ram_before, 2) if ram_before else None,
        "ram_after_mb": round(ram_after, 2) if ram_after else None,
        "ram_delta_mb": round(ram_after - ram_before, 2) if (ram_before and ram_after) else None,
    })

    logger.info(
        f"Embeddings: reducidos a {dims} dims (varianza {variance_explained*100:.1f}%), "
        f"took {elapsed:.3f}s"
    )

    return reduced.astype(np.float32), reducer, stats


# =============================================================================
# Main Embedding Extractor Class
# =============================================================================

class EmbeddingExtractor:
    """
    Extract semantic embeddings from text using sentence-transformers.

    NEW: Configurable embedding precision (full 384 vs reduced dims)

    Features:
    - Lazy loading: Model only loaded when first used
    - Auto-download: Downloads model from HuggingFace if not cached
    - Configurable precision: full 384, 256, or 128 dims
    - TruncatedSVD reduction: Better than PCA for sparse/text data
    - Memory efficient: Estimates and logs RAM usage

    Example:
        # Default: Full 384 dims (recommended for SMB volumes)
        extractor = EmbeddingExtractor()
        features = extractor.get_embedding_features("Nuevo apartamento en Triana!")

        # Reduced dims for very large datasets
        extractor = EmbeddingExtractor(precision="medium")  # 256 dims
    """

    # Class-level singleton for model (shared across instances)
    _model = None
    _model_loaded = False

    # Class-level reducers for each precision level
    _reducers: Dict[int, Any] = {}
    _reducers_fitted: Dict[int, bool] = {}

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        precision: Union[str, EmbeddingPrecision] = None,
        auto_load_reducer: bool = True
    ):
        """
        Initialize the embedding extractor.

        Args:
            model_name: HuggingFace model name for sentence-transformers
            precision: Embedding precision level (REQUIRED - no default)
                      Options: 'ultra_low' (64), 'low' (128), 'medium' (256),
                               'high' (384), 'max' (full raw 384)
            auto_load_reducer: Whether to auto-load saved reducer if exists

        Raises:
            ValueError: If precision is not provided or invalid
        """
        self.model_name = model_name

        # Validate precision is provided
        if precision is None:
            raise ValueError(
                "precision es REQUERIDO. Opciones: 'ultra_low', 'low', 'medium', 'high', 'max'. "
                "Recomendado para sobremesa normal: 'low' (32 dims)."
            )

        # Parse precision
        if isinstance(precision, str):
            precision = precision.lower()
            valid_precisions = [p.value for p in EmbeddingPrecision]
            if precision in valid_precisions:
                self._precision = EmbeddingPrecision(precision)
            else:
                raise ValueError(
                    f"Precision '{precision}' no valida. Opciones: {valid_precisions}. "
                    f"Recomendado para sobremesa normal: 'low' (32 dims)."
                )
        else:
            self._precision = precision

        self._target_dims = self._precision.dimensions
        self._available = self._check_availability()

        # Auto-load reducer if needed and exists
        if auto_load_reducer and self._precision.uses_reduction:
            self._load_reducer()

        logger.info(
            f"EmbeddingExtractor initialized: precision={self._precision.value}, "
            f"dims={self._target_dims}, uses_reduction={self._precision.uses_reduction}"
        )

    @property
    def precision(self) -> EmbeddingPrecision:
        """Current embedding precision."""
        return self._precision

    @property
    def target_dims(self) -> int:
        """Target output dimensions."""
        return self._target_dims

    @property
    def uses_reduction(self) -> bool:
        """Whether dimensionality reduction is applied."""
        return self._precision.uses_reduction

    def _check_availability(self) -> bool:
        """Check if sentence-transformers is available."""
        try:
            from sentence_transformers import SentenceTransformer
            return True
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. Embeddings will return zeros. "
                "Install with: pip install sentence-transformers"
            )
            return False

    def _load_model(self):
        """
        Lazy load the sentence-transformers model.

        Model is downloaded automatically from HuggingFace Hub if not cached.
        Uses class-level singleton to share model across instances.
        """
        if EmbeddingExtractor._model is not None:
            return

        if not self._available:
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name}")
            logger.info("(First run will download ~80MB from HuggingFace)")

            EmbeddingExtractor._model = SentenceTransformer(self.model_name)
            EmbeddingExtractor._model.max_seq_length = MAX_SEQ_LENGTH
            EmbeddingExtractor._model_loaded = True

            logger.info(f"Embedding model loaded successfully: {self.model_name}")
            logger.info(f"  Output dimension: {EMBEDDING_DIM}")
            logger.info(f"  Max sequence length: {MAX_SEQ_LENGTH}")

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            EmbeddingExtractor._model_loaded = False

    def _get_reducer_path(self) -> Path:
        """Get path for saving/loading reducer."""
        return MODELS_DIR / f"embedding_svd_{self._target_dims}.pkl"

    def _load_reducer(self):
        """Load pre-fitted reducer from disk."""
        if self._target_dims in EmbeddingExtractor._reducers:
            return

        reducer_path = self._get_reducer_path()
        if reducer_path.exists():
            try:
                with open(reducer_path, 'rb') as f:
                    EmbeddingExtractor._reducers[self._target_dims] = pickle.load(f)
                EmbeddingExtractor._reducers_fitted[self._target_dims] = True
                logger.info(f"Loaded TruncatedSVD reducer from: {reducer_path}")
            except Exception as e:
                logger.warning(f"Could not load reducer: {e}")

    def _save_reducer(self):
        """Save fitted reducer to disk."""
        if self._target_dims not in EmbeddingExtractor._reducers:
            return

        reducer_path = self._get_reducer_path()
        try:
            with open(reducer_path, 'wb') as f:
                pickle.dump(EmbeddingExtractor._reducers[self._target_dims], f)
            logger.info(f"Saved TruncatedSVD reducer to: {reducer_path}")
        except Exception as e:
            logger.warning(f"Could not save reducer: {e}")

    # =========================================================================
    # Core Embedding Methods
    # =========================================================================

    @functools.lru_cache(maxsize=1024)
    def get_raw_embedding(self, text: str) -> np.ndarray:
        """
        Generate raw 384-dimensional embedding for text.

        Args:
            text: Input text (caption, description, etc.)

        Returns:
            numpy array of shape (384,) with L2-normalized embedding
        """
        if not self._available:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        # Handle empty or invalid text
        if not text or not isinstance(text, str) or not text.strip():
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        # Load model if needed
        self._load_model()

        if EmbeddingExtractor._model is None:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        try:
            # Clean and truncate text
            text = text.strip()
            words = text.split()
            if len(words) > MAX_SEQ_LENGTH:
                text = " ".join(words[:MAX_SEQ_LENGTH])

            # Generate embedding
            embedding = EmbeddingExtractor._model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2 normalize
                show_progress_bar=False
            )

            return embedding.astype(np.float32)

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    def get_raw_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate raw embeddings for multiple texts efficiently.

        Args:
            texts: List of text strings

        Returns:
            numpy array of shape (n_texts, 384)
        """
        if not self._available:
            return np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)

        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        # Load model if needed
        self._load_model()

        if EmbeddingExtractor._model is None:
            return np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)

        try:
            # Clean texts
            cleaned_texts = []
            for text in texts:
                if text and isinstance(text, str) and text.strip():
                    words = text.strip().split()
                    if len(words) > MAX_SEQ_LENGTH:
                        text = " ".join(words[:MAX_SEQ_LENGTH])
                    cleaned_texts.append(text)
                else:
                    cleaned_texts.append("")

            # Batch encode
            embeddings = EmbeddingExtractor._model.encode(
                cleaned_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 100,
                batch_size=32
            )

            return embeddings.astype(np.float32)

        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            return np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)

    # Backward compatible alias
    def get_caption_embedding(self, text: str) -> np.ndarray:
        """Alias for get_raw_embedding (backward compatible)."""
        return self.get_raw_embedding(text)

    def get_caption_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Alias for get_raw_embeddings_batch (backward compatible)."""
        return self.get_raw_embeddings_batch(texts)

    # =========================================================================
    # Configurable Reduction Methods
    # =========================================================================

    def fit_reducer(
        self,
        embeddings: np.ndarray,
        save: bool = True
    ) -> "EmbeddingExtractor":
        """
        Fit TruncatedSVD reducer on a corpus of embeddings.

        Only needed if using precision != "max" (i.e., reduced dims).

        Args:
            embeddings: Array of shape (n_samples, 384)
            save: Whether to save the fitted reducer to disk

        Returns:
            self for chaining
        """
        if not self.uses_reduction:
            logger.info(f"No reduction needed for precision={self._precision.value}")
            return self

        # Fit reducer
        _, reducer, stats = get_reduced_embeddings(
            embeddings,
            dims=self._target_dims,
            fit_reducer=True
        )

        EmbeddingExtractor._reducers[self._target_dims] = reducer
        EmbeddingExtractor._reducers_fitted[self._target_dims] = True

        logger.info(
            f"TruncatedSVD fitted on {embeddings.shape[0]} samples: "
            f"{self._target_dims} components, "
            f"{stats.get('variance_percent', 0):.1f}% variance explained"
        )

        if save:
            self._save_reducer()

        return self

    def fit_reducer_from_texts(
        self,
        texts: List[str],
        save: bool = True
    ) -> "EmbeddingExtractor":
        """
        Convenience method: generate embeddings and fit reducer in one step.

        Args:
            texts: List of caption texts
            save: Whether to save the fitted reducer

        Returns:
            self for chaining
        """
        if not self.uses_reduction:
            logger.info(f"No reduction needed for precision={self._precision.value}")
            return self

        logger.info(f"Fitting TruncatedSVD from {len(texts)} texts...")

        # Generate embeddings
        embeddings = self.get_raw_embeddings_batch(texts)

        # Filter out zero embeddings
        valid_mask = np.any(embeddings != 0, axis=1)
        valid_embeddings = embeddings[valid_mask]

        logger.info(f"Generated {len(valid_embeddings)} valid embeddings")

        if len(valid_embeddings) < 10:
            logger.warning("Too few valid embeddings for reducer fitting")
            return self

        return self.fit_reducer(valid_embeddings, save=save)

    def transform(self, embedding: np.ndarray) -> np.ndarray:
        """
        Transform embedding to target dimensions.

        If precision is "max"/"high", returns unchanged.
        If precision is "medium"/"low", applies TruncatedSVD.

        Args:
            embedding: Array of shape (384,) or (n, 384)

        Returns:
            Array of shape (target_dims,) or (n, target_dims)
        """
        # No reduction needed
        if not self.uses_reduction:
            return embedding

        # Check if reducer is fitted
        reducer = EmbeddingExtractor._reducers.get(self._target_dims)
        is_fitted = EmbeddingExtractor._reducers_fitted.get(self._target_dims, False)

        if not is_fitted or reducer is None:
            logger.warning(
                f"Reducer not fitted for {self._target_dims} dims. "
                f"Call fit_reducer() first or returning zeros."
            )
            if embedding.ndim == 1:
                return np.zeros(self._target_dims, dtype=np.float32)
            else:
                return np.zeros((embedding.shape[0], self._target_dims), dtype=np.float32)

        # Ensure 2D
        squeeze = False
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
            squeeze = True

        # Transform
        try:
            reduced = reducer.transform(embedding)
            if squeeze:
                reduced = reduced.squeeze()
            return reduced.astype(np.float32)
        except Exception as e:
            logger.error(f"Transform failed: {e}")
            if squeeze:
                return np.zeros(self._target_dims, dtype=np.float32)
            else:
                return np.zeros((embedding.shape[0], self._target_dims), dtype=np.float32)

    # Backward compatible alias
    def transform_to_pca(self, embedding: np.ndarray) -> np.ndarray:
        """Alias for transform (backward compatible)."""
        return self.transform(embedding)

    # =========================================================================
    # Feature Extraction for ML Pipeline
    # =========================================================================

    def get_embedding_features(self, text: str) -> Dict[str, float]:
        """
        Get embedding features as a dictionary for ML pipeline.

        Returns features named embedding_0 to embedding_N-1 (0-based indexing).

        Args:
            text: Input caption/text

        Returns:
            Dict with keys "embedding_0" through "embedding_N-1"

        Note:
            Uses 0-based indexing for consistency with XGBoost and
            GrowthPredictionEngine's dynamic embedding detection.
        """
        # Get raw embedding
        raw_embedding = self.get_raw_embedding(text)

        # Apply reduction if needed
        if self.uses_reduction:
            final_embedding = self.transform(raw_embedding)
        else:
            final_embedding = raw_embedding

        # Convert to dict (0-based indexing)
        features = {}
        for i in range(len(final_embedding)):
            features[f"embedding_{i}"] = round(float(final_embedding[i]), 6)

        # Log active dimensions
        logger.info(f"Embeddings activos: {len(final_embedding)} dims")

        return features

    def get_embedding_features_batch(
        self,
        texts: List[str]
    ) -> List[Dict[str, float]]:
        """
        Get embedding features for multiple texts efficiently.

        Args:
            texts: List of text strings

        Returns:
            List of feature dictionaries with 0-based embedding keys

        Note:
            Uses 0-based indexing (embedding_0 to embedding_N-1) for consistency
            with GrowthPredictionEngine's dynamic embedding detection.
        """
        # Get batch raw embeddings
        raw_embeddings = self.get_raw_embeddings_batch(texts)

        # Apply reduction if needed
        if self.uses_reduction:
            final_embeddings = self.transform(raw_embeddings)
        else:
            final_embeddings = raw_embeddings

        # Convert to list of dicts (0-based indexing)
        features_list = []
        n_dims = final_embeddings.shape[1] if final_embeddings.ndim > 1 else len(final_embeddings)

        for i in range(len(texts)):
            features = {}
            for j in range(n_dims):
                val = final_embeddings[i, j] if final_embeddings.ndim > 1 else final_embeddings[j]
                features[f"embedding_{j}"] = round(float(val), 6)  # 0-based
            features_list.append(features)

        # Log active dimensions (only once for batch)
        if len(texts) > 0:
            logger.info(f"Embeddings activos: {n_dims} dims (batch de {len(texts)} textos)")

        return features_list

    def get_zero_features(self) -> Dict[str, float]:
        """
        Get zero-valued embedding features (for fallback/error cases).

        Returns:
            Dict with all embedding features set to 0.0 (0-based indexing)
        """
        return {f"embedding_{i}": 0.0 for i in range(self._target_dims)}

    # =========================================================================
    # Backward Compatibility (PCA methods now use TruncatedSVD internally)
    # =========================================================================

    def fit_pca(self, embeddings: np.ndarray, save: bool = True) -> "EmbeddingExtractor":
        """
        Backward compatible: Now uses TruncatedSVD internally.
        Only applies if using reduced precision.
        """
        return self.fit_reducer(embeddings, save=save)

    def fit_pca_from_texts(self, texts: List[str], save: bool = True) -> "EmbeddingExtractor":
        """Backward compatible: Now uses TruncatedSVD internally."""
        return self.fit_reducer_from_texts(texts, save=save)

    def save_pca_model(self, path: Optional[str] = None):
        """Backward compatible: Save reducer."""
        self._save_reducer()

    def load_pca_model(self, path: Optional[str] = None) -> bool:
        """Backward compatible: Load reducer."""
        self._load_reducer()
        return self._target_dims in EmbeddingExtractor._reducers

    @property
    def is_pca_fitted(self) -> bool:
        """Check if reducer is fitted (or if no reduction needed)."""
        if not self.uses_reduction:
            return True  # No reduction needed = always "fitted"
        return EmbeddingExtractor._reducers_fitted.get(self._target_dims, False)

    @property
    def pca_components(self) -> int:
        """Backward compatible: Returns target dims."""
        return self._target_dims

    # =========================================================================
    # Resource Management
    # =========================================================================

    def unload_model(self):
        """Unload the embedding model to free memory."""
        if EmbeddingExtractor._model is not None:
            del EmbeddingExtractor._model
            EmbeddingExtractor._model = None
            EmbeddingExtractor._model_loaded = False
            gc.collect()
            logger.info("Embedding model unloaded")

    @classmethod
    def clear_all(cls):
        """Clear all class-level cached models and reducers."""
        if cls._model is not None:
            del cls._model
            cls._model = None
            cls._model_loaded = False

        cls._reducers = {}
        cls._reducers_fitted = {}

        gc.collect()
        logger.info("All embedding models and reducers cleared")

    # =========================================================================
    # Status and Info
    # =========================================================================

    @property
    def is_available(self) -> bool:
        """Check if sentence-transformers is available."""
        return self._available

    @property
    def is_model_loaded(self) -> bool:
        """Check if embedding model is loaded."""
        return EmbeddingExtractor._model_loaded

    def get_status(self) -> Dict[str, Any]:
        """Get status of the embedding extractor."""
        return {
            "sentence_transformers_available": self._available,
            "model_name": self.model_name,
            "model_loaded": EmbeddingExtractor._model_loaded,
            "embedding_dim_raw": EMBEDDING_DIM,
            "precision": self._precision.value,
            "target_dims": self._target_dims,
            "uses_reduction": self.uses_reduction,
            "reducer_fitted": self.is_pca_fitted,
            "feature_names": get_embedding_feature_names(self._target_dims),
        }


# =============================================================================
# Global Instance (Singleton Pattern) - Full Precision by Default
# =============================================================================

# Global instances for each precision level
_embedding_extractors: Dict[str, EmbeddingExtractor] = {}


def get_embedding_extractor(
    precision: Union[str, EmbeddingPrecision]
) -> EmbeddingExtractor:
    """
    Get the global embedding extractor instance for a precision level.

    Creates the instance on first call (lazy initialization).

    IMPORTANT: precision is REQUIRED - no default. Caller must explicitly
    choose based on their hardware:
    - "ultra_low" (64 dims): Ultra rapido, PC modesto
    - "low" (128 dims): Recomendado sobremesa normal Almeria
    - "medium" (256 dims): Balance precision/velocidad
    - "high" (384 dims): Full dims con TruncatedSVD
    - "max": Full raw 384 dims sin reduccion

    Args:
        precision: Precision level (REQUIRED - no default)

    Returns:
        EmbeddingExtractor instance

    Raises:
        ValueError: If precision is not provided or invalid
    """
    if precision is None:
        raise ValueError(
            "precision es REQUERIDO. Opciones: 'ultra_low', 'low', 'medium', 'high', 'max'. "
            "Recomendado para sobremesa normal: 'low' (128 dims)."
        )

    if isinstance(precision, EmbeddingPrecision):
        key = precision.value
    else:
        key = str(precision).lower()

    # Validate precision value
    valid_precisions = [p.value for p in EmbeddingPrecision]
    if key not in valid_precisions:
        raise ValueError(
            f"Precision '{key}' no valida. Opciones: {valid_precisions}. "
            f"Recomendado para sobremesa normal: 'low' (128 dims)."
        )

    if key not in _embedding_extractors:
        _embedding_extractors[key] = EmbeddingExtractor(precision=key)

    return _embedding_extractors[key]


def get_caption_embedding(text: str, precision: str) -> np.ndarray:
    """
    Convenience function: Get raw embedding for text.

    Args:
        text: Input text
        precision: Precision level (REQUIRED - no default)
                  Options: 'ultra_low', 'low', 'medium', 'high', 'max'

    Returns:
        numpy array of shape (target_dims,) based on precision
    """
    extractor = get_embedding_extractor(precision)
    raw = extractor.get_raw_embedding(text)

    if extractor.uses_reduction:
        return extractor.transform(raw)
    return raw


def get_embedding_features(text: str, precision: str) -> Dict[str, float]:
    """
    Convenience function: Get embedding features for ML pipeline.

    Args:
        text: Input text
        precision: Precision level (REQUIRED - no default)
                  Options: 'ultra_low', 'low', 'medium', 'high', 'max'

    Returns:
        Dict with embedding features
    """
    return get_embedding_extractor(precision).get_embedding_features(text)


# =============================================================================
# CLI for Testing
# =============================================================================

if __name__ == "__main__":
    import sys
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("\n" + "=" * 70)
    print("Semantic Embeddings Feature Extractor - BrandPulse AI")
    print("PRECISION LEVELS: ultra_low=16 / low=32 / medium=64 / high=128 / max=384 raw")
    print("RECOMENDADO SOBREMESA NORMAL: 'low' (32 dims)")
    print("=" * 70 + "\n")

    # Test texts (Spanish captions typical for local SMBs)
    test_texts = [
        "Nuevo apartamento en Triana! 3 habitaciones, terraza con vistas. DM para info!",
        "Delicioso cafe recien hecho. Ven a probar nuestra especialidad del dia!",
        "Corte y peinado profesional. Reserva tu cita hoy! Link en bio",
        "Plato del dia: paella valenciana. Menu completo por solo 12 euros!",
        "Ramo de rosas frescas. Perfecto para cualquier ocasion. Envio gratis!",
    ]

    # Test each precision level (including new ultra_low)
    for precision in ["ultra_low", "low", "medium", "high", "max"]:
        print(f"\n{'='*60}")
        print(f"TESTING PRECISION: {precision.upper()}")
        print(f"{'='*60}")

        extractor = EmbeddingExtractor(precision=precision)
        print(f"Status: {json.dumps(extractor.get_status(), indent=2)}")

        # Generate raw embeddings
        print(f"\n[1] Generating Raw Embeddings...")
        start = time.time()
        raw_embeddings = extractor.get_raw_embeddings_batch(test_texts)
        raw_time = time.time() - start
        print(f"    Raw shape: {raw_embeddings.shape}, time: {raw_time:.3f}s")

        # Fit reducer if needed (not for max)
        if extractor.uses_reduction:
            print(f"\n[2] Fitting TruncatedSVD reducer (precision={precision})...")
            start = time.time()
            extractor.fit_reducer(raw_embeddings, save=False)
            fit_time = time.time() - start
            print(f"    Fit time: {fit_time:.3f}s")
        else:
            print(f"\n[2] Skipping TruncatedSVD (max = full raw embeddings)")

        # Get features
        print(f"\n[3] Getting embedding features...")
        start = time.time()
        features_list = extractor.get_embedding_features_batch(test_texts)
        feat_time = time.time() - start
        print(f"    Features per sample: {len(features_list[0])}")
        print(f"    Total time: {feat_time:.3f}s")

        # Memory estimate
        mem = estimate_memory_usage(len(test_texts), extractor.target_dims)
        print(f"\n[4] Memory estimate for {len(test_texts)} samples:")
        print(f"    {mem['human_readable']} (lightweight: {mem['is_lightweight']})")

        # Sample features (0-based indexing)
        print(f"\n[5] Sample features for first text (0-based indexing):")
        first_features = features_list[0]
        print(f"    embedding_0 = {first_features.get('embedding_0', 0):.6f}")
        print(f"    embedding_1 = {first_features.get('embedding_1', 0):.6f}")
        last_key = f"embedding_{extractor.target_dims - 1}"  # 0-based: last is N-1
        print(f"    {last_key} = {first_features.get(last_key, 0):.6f}")

    print("\n" + "=" * 70)
    print("Test Complete!")
    print("IMPORTANTE: precision es REQUERIDO - no hay default")
    print("Recomendado sobremesa normal Almeria: 'low' (32 dims)")
    print("=" * 70 + "\n")
