#!/usr/bin/env python3
"""
Semantic Embeddings Feature Module for BrandPulse AI
=====================================================

Modernizes the feature engineering pipeline by adding semantic embeddings
from sentence-transformers, replacing/augmenting manual heuristic features.

MODEL: sentence-transformers/all-MiniLM-L6-v2
- Lightweight: ~80MB model, fast inference
- Multilingual: Works well with Spanish and English
- Output: 384-dimensional dense vectors
- Normalized: L2 normalized embeddings

PIPELINE:
1. Load model (auto-downloads if not cached)
2. Generate 384-dim embedding from caption text
3. Apply PCA to reduce to 30 dimensions (embedding_1 to embedding_30)
4. Concatenate with existing manual features for XGBoost

Why 30 dimensions?
- 384 dims would dominate XGBoost feature space
- 30 dims captures ~85-90% of semantic variance
- Better balance with ~58 existing manual features

Usage:
    from ml.features_embeddings import EmbeddingExtractor

    extractor = EmbeddingExtractor()

    # Get 384-dim raw embedding
    embedding = extractor.get_caption_embedding("Tu texto aqui")

    # Get 30-dim PCA-reduced features for ML
    features = extractor.get_embedding_features("Tu texto aqui")
    # Returns: {"embedding_1": 0.123, "embedding_2": -0.456, ..., "embedding_30": 0.789}

Author: BrandPulse AI
"""

import gc
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union
import warnings

import numpy as np

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Model configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # Output dimension of all-MiniLM-L6-v2
PCA_COMPONENTS = 30  # Reduce to 30 dimensions for XGBoost
MAX_SEQ_LENGTH = 256  # Max tokens for embedding

# File paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PCA_MODEL_PATH = MODELS_DIR / "embedding_pca_30.pkl"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)


# =============================================================================
# Embedding Feature Names (for ML pipeline)
# =============================================================================

def get_embedding_feature_names() -> List[str]:
    """
    Get list of embedding feature column names.

    Returns:
        List of feature names: ["embedding_1", "embedding_2", ..., "embedding_30"]
    """
    return [f"embedding_{i+1}" for i in range(PCA_COMPONENTS)]


EMBEDDING_FEATURE_COLUMNS = get_embedding_feature_names()


# =============================================================================
# Main Embedding Extractor Class
# =============================================================================

class EmbeddingExtractor:
    """
    Extract semantic embeddings from text using sentence-transformers.

    Features:
    - Lazy loading: Model only loaded when first used
    - Auto-download: Downloads model from HuggingFace if not cached
    - PCA reduction: 384-dim -> 30-dim for XGBoost compatibility
    - Caching: PCA model can be pre-fitted and saved/loaded

    Example:
        extractor = EmbeddingExtractor()

        # Raw 384-dim embedding
        raw = extractor.get_caption_embedding("Nuevo apartamento en Triana!")

        # 30-dim features for ML
        features = extractor.get_embedding_features("Nuevo apartamento en Triana!")
        print(features)  # {"embedding_1": 0.1, "embedding_2": -0.2, ...}
    """

    # Class-level singleton for model (shared across instances)
    _model = None
    _model_loaded = False
    _pca = None
    _pca_fitted = False

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        pca_components: int = PCA_COMPONENTS,
        pca_model_path: Optional[str] = None,
        auto_load_pca: bool = True
    ):
        """
        Initialize the embedding extractor.

        Args:
            model_name: HuggingFace model name for sentence-transformers
            pca_components: Number of PCA components (default: 30)
            pca_model_path: Path to pre-fitted PCA model (optional)
            auto_load_pca: Whether to auto-load PCA model if exists
        """
        self.model_name = model_name
        self.pca_components = pca_components
        self.pca_model_path = Path(pca_model_path) if pca_model_path else PCA_MODEL_PATH

        self._available = self._check_availability()

        # Auto-load PCA if exists
        if auto_load_pca and self.pca_model_path.exists():
            self._load_pca_model()

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

    def _load_pca_model(self):
        """Load pre-fitted PCA model from disk."""
        if EmbeddingExtractor._pca is not None:
            return

        try:
            if self.pca_model_path.exists():
                with open(self.pca_model_path, 'rb') as f:
                    EmbeddingExtractor._pca = pickle.load(f)
                EmbeddingExtractor._pca_fitted = True
                logger.info(f"Loaded PCA model from: {self.pca_model_path}")
            else:
                logger.debug(f"No PCA model found at: {self.pca_model_path}")
        except Exception as e:
            logger.warning(f"Could not load PCA model: {e}")
            EmbeddingExtractor._pca_fitted = False

    def _init_pca(self):
        """Initialize PCA model (but don't fit yet)."""
        if EmbeddingExtractor._pca is None:
            from sklearn.decomposition import PCA
            EmbeddingExtractor._pca = PCA(n_components=self.pca_components)
            EmbeddingExtractor._pca_fitted = False
            logger.info(f"Initialized new PCA model: {self.pca_components} components")

    # =========================================================================
    # Core Embedding Methods
    # =========================================================================

    def get_caption_embedding(self, text: str) -> np.ndarray:
        """
        Generate 384-dimensional embedding for text.

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

    def get_caption_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently.

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

    # =========================================================================
    # PCA Reduction Methods
    # =========================================================================

    def fit_pca(self, embeddings: np.ndarray, save: bool = True) -> "EmbeddingExtractor":
        """
        Fit PCA model on a corpus of embeddings.

        Should be called with embeddings from your training data before
        using get_embedding_features() for consistent dimensionality reduction.

        Args:
            embeddings: Array of shape (n_samples, 384)
            save: Whether to save the fitted PCA model to disk

        Returns:
            self for chaining
        """
        self._init_pca()

        if embeddings.shape[0] < self.pca_components:
            logger.warning(
                f"Not enough samples ({embeddings.shape[0]}) for "
                f"PCA with {self.pca_components} components. "
                f"Using {embeddings.shape[0]} components instead."
            )
            from sklearn.decomposition import PCA
            EmbeddingExtractor._pca = PCA(n_components=embeddings.shape[0])

        # Fit PCA
        EmbeddingExtractor._pca.fit(embeddings)
        EmbeddingExtractor._pca_fitted = True

        # Calculate explained variance
        explained_var = sum(EmbeddingExtractor._pca.explained_variance_ratio_) * 100
        logger.info(
            f"PCA fitted on {embeddings.shape[0]} samples: "
            f"{EmbeddingExtractor._pca.n_components_} components, "
            f"{explained_var:.1f}% variance explained"
        )

        # Save if requested
        if save:
            self.save_pca_model()

        return self

    def fit_pca_from_texts(
        self,
        texts: List[str],
        save: bool = True
    ) -> "EmbeddingExtractor":
        """
        Convenience method: generate embeddings and fit PCA in one step.

        Args:
            texts: List of caption texts
            save: Whether to save the fitted PCA model

        Returns:
            self for chaining
        """
        logger.info(f"Fitting PCA from {len(texts)} texts...")

        # Generate embeddings
        embeddings = self.get_caption_embeddings_batch(texts)

        # Filter out zero embeddings (failed encodings)
        valid_mask = np.any(embeddings != 0, axis=1)
        valid_embeddings = embeddings[valid_mask]

        logger.info(f"Generated {len(valid_embeddings)} valid embeddings")

        if len(valid_embeddings) < 10:
            logger.warning("Too few valid embeddings for PCA fitting")
            return self

        return self.fit_pca(valid_embeddings, save=save)

    def transform_to_pca(self, embedding: np.ndarray) -> np.ndarray:
        """
        Transform 384-dim embedding to PCA-reduced dimensions.

        Args:
            embedding: Array of shape (384,) or (n, 384)

        Returns:
            Array of shape (30,) or (n, 30)
        """
        # Initialize PCA if needed
        if EmbeddingExtractor._pca is None:
            self._init_pca()

        # If PCA not fitted, return zeros
        if not EmbeddingExtractor._pca_fitted:
            if embedding.ndim == 1:
                return np.zeros(self.pca_components, dtype=np.float32)
            else:
                return np.zeros((embedding.shape[0], self.pca_components), dtype=np.float32)

        # Ensure 2D
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False

        # Transform
        try:
            reduced = EmbeddingExtractor._pca.transform(embedding)
            if squeeze:
                reduced = reduced.squeeze()
            return reduced.astype(np.float32)
        except Exception as e:
            logger.error(f"PCA transform failed: {e}")
            if squeeze:
                return np.zeros(self.pca_components, dtype=np.float32)
            else:
                return np.zeros((embedding.shape[0], self.pca_components), dtype=np.float32)

    # =========================================================================
    # Feature Extraction for ML Pipeline
    # =========================================================================

    def get_embedding_features(self, text: str) -> Dict[str, float]:
        """
        Get PCA-reduced embedding features as a dictionary.

        This is the main method for ML feature extraction.
        Returns features named embedding_1 to embedding_30.

        Args:
            text: Input caption/text

        Returns:
            Dict with keys "embedding_1" through "embedding_30"
        """
        # Get raw embedding
        embedding = self.get_caption_embedding(text)

        # Apply PCA
        reduced = self.transform_to_pca(embedding)

        # Convert to dict
        features = {}
        for i in range(len(reduced)):
            features[f"embedding_{i+1}"] = round(float(reduced[i]), 6)

        # Pad with zeros if PCA has fewer components
        for i in range(len(reduced), self.pca_components):
            features[f"embedding_{i+1}"] = 0.0

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
            List of feature dictionaries
        """
        # Get batch embeddings
        embeddings = self.get_caption_embeddings_batch(texts)

        # Apply PCA
        reduced = self.transform_to_pca(embeddings)

        # Convert to list of dicts
        features_list = []
        for i in range(len(texts)):
            features = {}
            for j in range(reduced.shape[1] if reduced.ndim > 1 else len(reduced)):
                val = reduced[i, j] if reduced.ndim > 1 else reduced[j]
                features[f"embedding_{j+1}"] = round(float(val), 6)

            # Pad with zeros if needed
            for j in range(reduced.shape[1] if reduced.ndim > 1 else len(reduced), self.pca_components):
                features[f"embedding_{j+1}"] = 0.0

            features_list.append(features)

        return features_list

    def get_zero_features(self) -> Dict[str, float]:
        """
        Get zero-valued embedding features (for fallback/error cases).

        Returns:
            Dict with all embedding features set to 0.0
        """
        return {f"embedding_{i+1}": 0.0 for i in range(self.pca_components)}

    # =========================================================================
    # Model Persistence
    # =========================================================================

    def save_pca_model(self, path: Optional[str] = None):
        """Save fitted PCA model to disk."""
        save_path = Path(path) if path else self.pca_model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if EmbeddingExtractor._pca is not None and EmbeddingExtractor._pca_fitted:
            with open(save_path, 'wb') as f:
                pickle.dump(EmbeddingExtractor._pca, f)
            logger.info(f"PCA model saved to: {save_path}")
        else:
            logger.warning("No fitted PCA model to save")

    def load_pca_model(self, path: Optional[str] = None) -> bool:
        """
        Load a pre-fitted PCA model from disk.

        Returns:
            True if loaded successfully, False otherwise
        """
        load_path = Path(path) if path else self.pca_model_path

        if not load_path.exists():
            logger.warning(f"PCA model not found: {load_path}")
            return False

        try:
            with open(load_path, 'rb') as f:
                EmbeddingExtractor._pca = pickle.load(f)
            EmbeddingExtractor._pca_fitted = True
            logger.info(f"PCA model loaded from: {load_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load PCA model: {e}")
            return False

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
        """Clear all class-level cached models."""
        if cls._model is not None:
            del cls._model
            cls._model = None
            cls._model_loaded = False

        if cls._pca is not None:
            del cls._pca
            cls._pca = None
            cls._pca_fitted = False

        gc.collect()
        logger.info("All embedding models cleared")

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

    @property
    def is_pca_fitted(self) -> bool:
        """Check if PCA model is fitted."""
        return EmbeddingExtractor._pca_fitted

    def get_status(self) -> Dict[str, any]:
        """Get status of the embedding extractor."""
        return {
            "sentence_transformers_available": self._available,
            "model_name": self.model_name,
            "model_loaded": EmbeddingExtractor._model_loaded,
            "embedding_dim": EMBEDDING_DIM,
            "pca_components": self.pca_components,
            "pca_fitted": EmbeddingExtractor._pca_fitted,
            "pca_model_path": str(self.pca_model_path),
            "feature_names": EMBEDDING_FEATURE_COLUMNS,
        }


# =============================================================================
# Global Instance (Singleton Pattern)
# =============================================================================

# Global instance for easy access
_embedding_extractor: Optional[EmbeddingExtractor] = None


def get_embedding_extractor() -> EmbeddingExtractor:
    """
    Get the global embedding extractor instance.

    Creates the instance on first call (lazy initialization).

    Returns:
        EmbeddingExtractor instance
    """
    global _embedding_extractor
    if _embedding_extractor is None:
        _embedding_extractor = EmbeddingExtractor()
    return _embedding_extractor


def get_caption_embedding(text: str) -> np.ndarray:
    """
    Convenience function: Get 384-dim embedding for text.

    Args:
        text: Input text

    Returns:
        numpy array of shape (384,)
    """
    return get_embedding_extractor().get_caption_embedding(text)


def get_embedding_features(text: str) -> Dict[str, float]:
    """
    Convenience function: Get PCA-reduced embedding features.

    Args:
        text: Input text

    Returns:
        Dict with embedding_1 to embedding_30
    """
    return get_embedding_extractor().get_embedding_features(text)


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
    print("=" * 70 + "\n")

    # Test texts (Spanish captions typical for local SMBs)
    test_texts = [
        "Nuevo apartamento en Triana! 3 habitaciones, terraza con vistas. DM para info!",
        "Delicioso cafe recien hecho. Ven a probar nuestra especialidad del dia!",
        "Corte y peinado profesional. Reserva tu cita hoy! Link en bio",
        "Plato del dia: paella valenciana. Menu completo por solo 12 euros!",
        "Ramo de rosas frescas. Perfecto para cualquier ocasion. Envio gratis!",
    ]

    # Initialize extractor
    print("[1] Initializing Embedding Extractor...")
    extractor = EmbeddingExtractor()
    print(f"    Status: {json.dumps(extractor.get_status(), indent=2)}")

    # Test raw embeddings
    print("\n[2] Generating Raw Embeddings (384-dim)...")
    for text in test_texts[:2]:
        embedding = extractor.get_caption_embedding(text)
        print(f"    Text: '{text[:50]}...'")
        print(f"    Embedding shape: {embedding.shape}")
        print(f"    First 5 values: {embedding[:5].round(4)}")
        print()

    # Fit PCA on test data
    print("[3] Fitting PCA on test corpus...")
    extractor.fit_pca_from_texts(test_texts, save=True)

    # Test PCA-reduced features
    print("\n[4] Generating PCA-Reduced Features (30-dim)...")
    for text in test_texts:
        features = extractor.get_embedding_features(text)
        print(f"    Text: '{text[:40]}...'")
        print(f"    Features: embedding_1={features['embedding_1']:.4f}, "
              f"embedding_2={features['embedding_2']:.4f}, "
              f"..., embedding_30={features['embedding_30']:.4f}")
        print()

    # Batch processing
    print("[5] Batch Processing Test...")
    batch_features = extractor.get_embedding_features_batch(test_texts)
    print(f"    Processed {len(batch_features)} texts in batch")

    # Cleanup
    extractor.unload_model()

    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70 + "\n")
