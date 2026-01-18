"""
Text Intelligence Engine - NLP Pipeline for Content Analysis
=============================================================

Extracts semantic understanding from video content through:
- Speech-to-Text transcription (faster-whisper)
- OCR text detection on frames (EasyOCR)
- Semantic embeddings with dimensionality reduction (sentence-transformers + PCA)

Optimized for edge computing with memory-efficient processing.

Author: ML Engineering Team
"""

import gc
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import warnings

import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Constants
# =============================================================================

@dataclass(frozen=True)
class TextConfig:
    """
    Immutable configuration for text intelligence processing.

    QUANTIZATION NOTES:
    ===================
    Both Whisper and sentence-transformers support int8 quantization for CPU inference.

    - Whisper int8: Uses CTranslate2 backend, ~4x speedup, <1% WER degradation
    - Embeddings int8: Uses ONNX quantization, ~3-4x speedup, ~1% cosine similarity loss

    Memory savings:
    - float32: 4 bytes per weight
    - int8: 1 byte per weight (4x reduction)

    For sentence-transformers, we use dynamic quantization which quantizes weights
    but keeps activations in float32 for accuracy.
    """
    # Whisper settings
    whisper_model: str = "tiny"  # 'tiny' or 'base' for low RAM
    whisper_language: str = "es"  # Spanish default
    whisper_beam_size: int = 1  # Greedy decoding for speed
    whisper_compute_type: str = "int8"  # Quantized for low memory (~4x speedup)

    # OCR settings
    ocr_languages: Tuple[str, ...] = ("es", "en")  # Spanish + English
    ocr_gpu: bool = False  # CPU-only for edge devices
    ocr_paragraph: bool = True  # Merge text into paragraphs

    # Embedding settings
    embedding_model: str = "all-MiniLM-L6-v2"  # 384 dimensions, fast
    embedding_max_length: int = 256  # Truncate long texts

    # ==========================================================================
    # QUANTIZATION SETTINGS - Performance Optimization
    # ==========================================================================
    # Enable int8 quantization for CPU inference (~3-4x speedup, ~1% accuracy loss)
    # Uses PyTorch dynamic quantization: weights are int8, activations stay float32
    embedding_quantize: bool = True  # Enable int8 quantization for embeddings
    embedding_quantize_dtype: str = "int8"  # Quantization dtype (int8 recommended)

    # PCA settings
    pca_components: int = 10  # Reduce 384 -> 10 dimensions
    pca_model_path: Optional[str] = None  # Path to pre-fitted PCA model


@dataclass(frozen=True)
class OCRConfig:
    """Configuration for frame text density analysis."""
    min_confidence: float = 0.3  # Minimum OCR confidence threshold
    text_area_weight: float = 1.0  # Weight for text density calculation


# =============================================================================
# Whisper Transcription (Speech-to-Text)
# =============================================================================

class WhisperTranscriber:
    """
    Memory-efficient speech-to-text using faster-whisper.

    Uses CTranslate2 backend for optimized inference on CPU.
    Model 'tiny' uses ~75MB RAM, 'base' uses ~150MB RAM.
    """

    def __init__(self, config: Optional[TextConfig] = None):
        self.config = config or TextConfig()
        self._model = None
        self._available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if faster-whisper is installed."""
        try:
            from faster_whisper import WhisperModel
            return True
        except ImportError:
            logger.warning(
                "faster-whisper not installed. Transcription unavailable. "
                "Install with: pip install faster-whisper"
            )
            return False

    def _load_model(self):
        """Lazy load the Whisper model to save memory."""
        if self._model is None and self._available:
            from faster_whisper import WhisperModel

            logger.info(f"Loading Whisper model: {self.config.whisper_model}")
            self._model = WhisperModel(
                self.config.whisper_model,
                device="cpu",
                compute_type=self.config.whisper_compute_type,
                cpu_threads=2,  # Limit threads for edge devices
                num_workers=1
            )
            logger.info("Whisper model loaded successfully")

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio/video file to text.

        Args:
            audio_path: Path to audio or video file

        Returns:
            Dict with transcription text and metadata
        """
        if not self._available:
            return {
                "transcription": "",
                "language": "",
                "confidence": 0.0,
                "word_count": 0,
                "transcription_status": "whisper_not_available"
            }

        try:
            self._load_model()

            # Transcribe with optimized settings
            segments, info = self._model.transcribe(
                audio_path,
                language=self.config.whisper_language,
                beam_size=self.config.whisper_beam_size,
                word_timestamps=False,  # Disable for speed
                vad_filter=True,  # Voice Activity Detection to skip silence
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=200
                )
            )

            # Collect all text segments
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text.strip())

            full_text = " ".join(text_parts)
            word_count = len(full_text.split()) if full_text else 0

            result = {
                "transcription": full_text,
                "language": info.language,
                "confidence": round(info.language_probability, 4),
                "duration_seconds": round(info.duration, 2),
                "word_count": word_count,
                "transcription_status": "success"
            }

            logger.debug(f"Transcribed {word_count} words from {audio_path}")
            return result

        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {e}")
            return {
                "transcription": "",
                "language": "",
                "confidence": 0.0,
                "word_count": 0,
                "transcription_status": "error",
                "transcription_error": str(e)
            }

        finally:
            gc.collect()

    def unload_model(self):
        """Explicitly unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()
            logger.info("Whisper model unloaded")


# =============================================================================
# OCR Text Extraction (EasyOCR)
# =============================================================================

class OCRExtractor:
    """
    Extract text from video frames using EasyOCR.

    Calculates text_density: percentage of frame covered by text.
    Optimized for overlay text detection in TikTok/Reels.
    """

    def __init__(self, config: Optional[TextConfig] = None, ocr_config: Optional[OCRConfig] = None):
        self.config = config or TextConfig()
        self.ocr_config = ocr_config or OCRConfig()
        self._reader = None
        self._available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if EasyOCR is installed."""
        try:
            import easyocr
            return True
        except ImportError:
            logger.warning(
                "easyocr not installed. OCR unavailable. "
                "Install with: pip install easyocr"
            )
            return False

    def _load_reader(self):
        """Lazy load EasyOCR reader to save memory."""
        if self._reader is None and self._available:
            import easyocr

            logger.info(f"Loading EasyOCR with languages: {self.config.ocr_languages}")
            self._reader = easyocr.Reader(
                list(self.config.ocr_languages),
                gpu=self.config.ocr_gpu,
                verbose=False
            )
            logger.info("EasyOCR reader loaded successfully")

    def extract_text_from_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Extract text from a single frame.

        Args:
            frame: numpy array of image (BGR or grayscale)

        Returns:
            Dict with detected text and text_density
        """
        if not self._available:
            return {
                "text": "",
                "text_density": 0.0,
                "text_boxes": 0,
                "ocr_status": "easyocr_not_available"
            }

        try:
            self._load_reader()

            # Get frame dimensions
            if len(frame.shape) == 2:
                frame_height, frame_width = frame.shape
            else:
                frame_height, frame_width = frame.shape[:2]

            frame_area = frame_width * frame_height

            # Run OCR detection
            results = self._reader.readtext(
                frame,
                paragraph=self.config.ocr_paragraph,
                min_size=10,
                text_threshold=self.ocr_config.min_confidence
            )

            # Process results
            texts = []
            total_text_area = 0.0

            for detection in results:
                bbox, text, confidence = detection

                if confidence >= self.ocr_config.min_confidence:
                    texts.append(text)

                    # Calculate bounding box area
                    # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    bbox = np.array(bbox)
                    width = np.max(bbox[:, 0]) - np.min(bbox[:, 0])
                    height = np.max(bbox[:, 1]) - np.min(bbox[:, 1])
                    total_text_area += width * height

            # Calculate text density (percentage of frame covered by text)
            text_density = (total_text_area / frame_area) * 100 if frame_area > 0 else 0.0

            return {
                "text": " ".join(texts),
                "text_density": round(text_density, 4),
                "text_boxes": len(texts),
                "avg_confidence": round(
                    np.mean([d[2] for d in results]) if results else 0.0, 4
                ),
                "ocr_status": "success"
            }

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return {
                "text": "",
                "text_density": 0.0,
                "text_boxes": 0,
                "ocr_status": "error",
                "ocr_error": str(e)
            }

    def extract_text_from_frames(
        self,
        frames: List[np.ndarray],
        aggregate: bool = True
    ) -> Dict[str, Any]:
        """
        Extract text from multiple key frames.

        Args:
            frames: List of frame arrays
            aggregate: If True, combine all text and average density

        Returns:
            Dict with combined OCR results
        """
        if not frames:
            return {
                "ocr_text": "",
                "text_density": 0.0,
                "total_text_boxes": 0,
                "frames_processed": 0,
                "ocr_status": "no_frames"
            }

        all_texts = []
        all_densities = []
        total_boxes = 0

        for i, frame in enumerate(frames):
            result = self.extract_text_from_frame(frame)

            if result["ocr_status"] == "success":
                if result["text"]:
                    all_texts.append(result["text"])
                all_densities.append(result["text_density"])
                total_boxes += result["text_boxes"]

            # Clear memory between frames
            if i % 5 == 0:
                gc.collect()

        # Deduplicate similar texts (common in video frames)
        unique_texts = list(dict.fromkeys(all_texts))

        return {
            "ocr_text": " ".join(unique_texts),
            "text_density": round(np.mean(all_densities) if all_densities else 0.0, 4),
            "max_text_density": round(max(all_densities) if all_densities else 0.0, 4),
            "total_text_boxes": total_boxes,
            "frames_processed": len(frames),
            "ocr_status": "success"
        }

    def unload_reader(self):
        """Explicitly unload reader to free memory."""
        if self._reader is not None:
            del self._reader
            self._reader = None
            gc.collect()
            logger.info("EasyOCR reader unloaded")


# =============================================================================
# Semantic Embeddings with PCA Reduction
# =============================================================================

class SemanticEncoder:
    """
    Generate semantic embeddings and reduce dimensionality with PCA.

    Workflow:
    1. Concatenate: Caption + Transcription + OCR text
    2. Generate 384-dim embedding with sentence-transformers
    3. Apply PCA to reduce to 10 components (sem_pca_1 to sem_pca_10)

    XGBoost needs dense numerical features, not 384-dim sparse vectors.

    QUANTIZATION (Performance Optimization):
    ========================================
    When embedding_quantize=True, the model is quantized to int8 using
    PyTorch dynamic quantization. This provides:
    - ~3-4x inference speedup on CPU
    - ~4x memory reduction for model weights
    - ~1% loss in cosine similarity accuracy (acceptable for content matching)

    Dynamic quantization quantizes weights to int8 but keeps activations
    in float32, providing a good balance of speed and accuracy.
    """

    def __init__(self, config: Optional[TextConfig] = None):
        self.config = config or TextConfig()
        self._model = None
        self._model_quantized = False
        self._pca = None
        self._pca_fitted = False
        self._available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if sentence-transformers is installed."""
        try:
            from sentence_transformers import SentenceTransformer
            return True
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. Embeddings unavailable. "
                "Install with: pip install sentence-transformers"
            )
            return False

    def _quantize_model(self):
        """
        Apply int8 dynamic quantization to the embedding model for CPU inference.

        Dynamic quantization quantizes weights to int8 but computes activations
        in float32 at runtime. This provides ~3-4x speedup with minimal accuracy loss.

        Performance Impact:
        - Speed: ~3-4x faster inference on CPU
        - Memory: ~4x reduction in model weight memory
        - Accuracy: ~1% loss in cosine similarity (negligible for content matching)
        """
        if self._model is None or self._model_quantized:
            return

        try:
            import torch
            from torch.quantization import quantize_dynamic

            logger.info("Applying int8 dynamic quantization to embedding model...")

            # Get the underlying PyTorch model from sentence-transformers
            # The model has a _modules dict with transformer layers
            original_size = sum(
                p.numel() * p.element_size()
                for p in self._model[0].auto_model.parameters()
            )

            # Apply dynamic quantization to Linear layers
            # This quantizes weights to int8 while keeping activations float32
            quantized_model = quantize_dynamic(
                self._model[0].auto_model,
                {torch.nn.Linear},  # Quantize only Linear layers
                dtype=torch.qint8
            )

            # Replace the model's transformer with quantized version
            self._model[0].auto_model = quantized_model
            self._model_quantized = True

            # Calculate memory savings
            quantized_size = sum(
                p.numel() * (1 if p.dtype == torch.qint8 else p.element_size())
                for p in self._model[0].auto_model.parameters()
            )

            reduction_pct = (1 - quantized_size / original_size) * 100
            logger.info(
                f"Quantization complete: {original_size / 1e6:.1f}MB -> "
                f"{quantized_size / 1e6:.1f}MB ({reduction_pct:.0f}% reduction)"
            )

        except Exception as e:
            logger.warning(
                f"Could not apply quantization (falling back to float32): {e}"
            )
            self._model_quantized = False

    def _load_model(self):
        """
        Lazy load the embedding model with optional int8 quantization.

        When config.embedding_quantize is True, applies dynamic quantization
        for ~3-4x speedup on CPU inference.
        """
        if self._model is None and self._available:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.config.embedding_model}")
            self._model = SentenceTransformer(self.config.embedding_model)
            self._model.max_seq_length = self.config.embedding_max_length

            # Apply quantization if enabled
            if self.config.embedding_quantize:
                self._quantize_model()
            else:
                logger.info("Embedding model loaded (float32, no quantization)")

    def _init_pca(self):
        """Initialize PCA model (or load pre-fitted one)."""
        if self._pca is None:
            from sklearn.decomposition import PCA

            # Try to load pre-fitted PCA if path provided
            if self.config.pca_model_path and Path(self.config.pca_model_path).exists():
                with open(self.config.pca_model_path, 'rb') as f:
                    self._pca = pickle.load(f)
                self._pca_fitted = True
                logger.info(f"Loaded pre-fitted PCA from {self.config.pca_model_path}")
            else:
                self._pca = PCA(n_components=self.config.pca_components)
                self._pca_fitted = False
                logger.info("Initialized new PCA model (will need fitting)")

    def encode_text(self, text: str) -> np.ndarray:
        """
        Generate 384-dimensional embedding for text.

        Args:
            text: Input text to encode

        Returns:
            numpy array of shape (384,)
        """
        if not self._available:
            return np.zeros(384, dtype=np.float32)

        if not text or not text.strip():
            return np.zeros(384, dtype=np.float32)

        self._load_model()

        # Truncate text if too long
        words = text.split()
        if len(words) > self.config.embedding_max_length:
            text = " ".join(words[:self.config.embedding_max_length])

        embedding = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return embedding.astype(np.float32)

    def encode_combined_text(
        self,
        caption: str = "",
        transcription: str = "",
        ocr_text: str = ""
    ) -> np.ndarray:
        """
        Concatenate and encode multiple text sources.

        Args:
            caption: User caption from post
            transcription: Whisper transcription
            ocr_text: OCR extracted text

        Returns:
            384-dimensional embedding
        """
        # Concatenate with separators
        parts = []

        if caption and caption.strip():
            parts.append(f"[CAPTION] {caption.strip()}")

        if transcription and transcription.strip():
            parts.append(f"[SPEECH] {transcription.strip()}")

        if ocr_text and ocr_text.strip():
            parts.append(f"[TEXT] {ocr_text.strip()}")

        combined = " ".join(parts) if parts else ""

        return self.encode_text(combined)

    def fit_pca(self, embeddings: np.ndarray) -> "SemanticEncoder":
        """
        Fit PCA on a corpus of embeddings.

        Args:
            embeddings: Array of shape (n_samples, 384)

        Returns:
            self for chaining
        """
        self._init_pca()

        if embeddings.shape[0] < self.config.pca_components:
            logger.warning(
                f"Not enough samples ({embeddings.shape[0]}) for "
                f"PCA with {self.config.pca_components} components. "
                "Using min(samples, components)."
            )
            from sklearn.decomposition import PCA
            n_components = min(embeddings.shape[0], self.config.pca_components)
            self._pca = PCA(n_components=n_components)

        self._pca.fit(embeddings)
        self._pca_fitted = True

        explained_var = sum(self._pca.explained_variance_ratio_) * 100
        logger.info(
            f"PCA fitted: {embeddings.shape[0]} samples -> "
            f"{self._pca.n_components_} components "
            f"({explained_var:.1f}% variance explained)"
        )

        return self

    def transform_to_pca(self, embedding: np.ndarray) -> Dict[str, float]:
        """
        Transform 384-dim embedding to PCA components.

        Args:
            embedding: Array of shape (384,) or (1, 384)

        Returns:
            Dict with sem_pca_1 to sem_pca_10
        """
        self._init_pca()

        if not self._pca_fitted:
            logger.warning("PCA not fitted. Returning zeros.")
            return {
                f"sem_pca_{i+1}": 0.0
                for i in range(self.config.pca_components)
            }

        # Ensure 2D input
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # Transform
        pca_components = self._pca.transform(embedding)[0]

        # Create named dictionary
        result = {
            f"sem_pca_{i+1}": round(float(pca_components[i]), 6)
            for i in range(len(pca_components))
        }

        return result

    def encode_and_reduce(
        self,
        caption: str = "",
        transcription: str = "",
        ocr_text: str = ""
    ) -> Dict[str, Any]:
        """
        Full pipeline: encode combined text and apply PCA.

        Args:
            caption: User caption
            transcription: Whisper transcription
            ocr_text: OCR text

        Returns:
            Dict with sem_pca_1 to sem_pca_10 and metadata
        """
        # Generate embedding
        embedding = self.encode_combined_text(caption, transcription, ocr_text)

        # Check if any text was provided
        has_text = bool(caption or transcription or ocr_text)

        # Get PCA components
        pca_result = self.transform_to_pca(embedding)

        # Add metadata
        pca_result["semantic_status"] = "success" if has_text else "no_text"
        pca_result["text_sources"] = sum([
            1 if caption else 0,
            1 if transcription else 0,
            1 if ocr_text else 0
        ])

        return pca_result

    def save_pca(self, path: str):
        """Save fitted PCA model to disk."""
        if self._pca is not None and self._pca_fitted:
            with open(path, 'wb') as f:
                pickle.dump(self._pca, f)
            logger.info(f"PCA model saved to {path}")

    def unload_model(self):
        """Explicitly unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_quantized = False
            gc.collect()
            logger.info("Embedding model unloaded")

    @property
    def is_quantized(self) -> bool:
        """Check if the model is currently using int8 quantization."""
        return self._model_quantized


# =============================================================================
# Frame Extraction for OCR
# =============================================================================

class KeyFrameExtractor:
    """
    Extract key frames from video for OCR analysis.

    Samples frames at regular intervals to capture text overlays.
    """

    def __init__(self, max_frames: int = 5, target_size: Tuple[int, int] = (640, 480)):
        self.max_frames = max_frames
        self.target_size = target_size

    def extract_key_frames(self, video_path: str) -> List[np.ndarray]:
        """
        Extract evenly-spaced key frames from video.

        Args:
            video_path: Path to video file

        Returns:
            List of frame arrays (BGR format for OCR)
        """
        import cv2

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return []

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0

            if total_frames <= 0:
                return []

            # Calculate frame indices to sample
            frame_indices = np.linspace(
                0, total_frames - 1,
                min(self.max_frames, total_frames),
                dtype=int
            )

            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()

                if ret:
                    # Resize for OCR (don't go too small)
                    frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
                    frames.append(frame)

            logger.debug(f"Extracted {len(frames)} key frames from {video_path}")
            return frames

        finally:
            cap.release()
            gc.collect()


# =============================================================================
# Unified Text Intelligence Engine
# =============================================================================

class TextIntelligenceEngine:
    """
    Unified engine for text-based content understanding.

    Combines:
    - Whisper transcription (spoken words)
    - EasyOCR (text overlays)
    - Semantic embeddings + PCA (meaning compression)

    Example:
        engine = TextIntelligenceEngine()
        features = engine.extract_all_features(
            video_path="video.mp4",
            caption="Check out this amazing content!"
        )
        print(features)  # Contains transcription, OCR, and sem_pca_1 to sem_pca_10
    """

    def __init__(self, config: Optional[TextConfig] = None):
        self.config = config or TextConfig()

        # Initialize extractors (lazy loading)
        self.transcriber = WhisperTranscriber(self.config)
        self.ocr_extractor = OCRExtractor(self.config)
        self.semantic_encoder = SemanticEncoder(self.config)
        self.frame_extractor = KeyFrameExtractor()

        logger.info(
            f"TextIntelligenceEngine initialized - "
            f"Whisper: {self.config.whisper_model}, "
            f"OCR: {self.config.ocr_languages}, "
            f"Embeddings: {self.config.embedding_model}, "
            f"PCA: {self.config.pca_components} components"
        )

    def extract_transcription(self, audio_path: str) -> Dict[str, Any]:
        """Extract only transcription features."""
        return self.transcriber.transcribe(audio_path)

    def extract_ocr(
        self,
        video_path: str = None,
        frames: List[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Extract OCR features from video or pre-extracted frames.

        Args:
            video_path: Path to video file (will extract key frames)
            frames: Pre-extracted frames (if video_path not provided)

        Returns:
            Dict with OCR text and text_density
        """
        if frames is None and video_path:
            frames = self.frame_extractor.extract_key_frames(video_path)

        if not frames:
            return {
                "ocr_text": "",
                "text_density": 0.0,
                "max_text_density": 0.0,
                "total_text_boxes": 0,
                "frames_processed": 0,
                "ocr_status": "no_frames"
            }

        return self.ocr_extractor.extract_text_from_frames(frames)

    def extract_semantic_features(
        self,
        caption: str = "",
        transcription: str = "",
        ocr_text: str = ""
    ) -> Dict[str, Any]:
        """Extract semantic PCA features from text sources."""
        return self.semantic_encoder.encode_and_reduce(
            caption=caption,
            transcription=transcription,
            ocr_text=ocr_text
        )

    def extract_all_features(
        self,
        video_path: str,
        caption: str = "",
        include_transcription: bool = True,
        include_ocr: bool = True,
        include_semantics: bool = True
    ) -> Dict[str, Any]:
        """
        Extract all text intelligence features from a video.

        Args:
            video_path: Path to video file
            caption: User-provided caption for the content
            include_transcription: Whether to run Whisper
            include_ocr: Whether to run EasyOCR
            include_semantics: Whether to generate PCA features

        Returns:
            Flat dict with all text intelligence features
        """
        result: Dict[str, Any] = {
            "source": str(Path(video_path).name),
            "caption_provided": bool(caption),
            "text_status": "success"
        }

        transcription_text = ""
        ocr_text = ""

        try:
            # Step 1: Transcription (Whisper)
            if include_transcription:
                trans_result = self.transcriber.transcribe(video_path)
                transcription_text = trans_result.get("transcription", "")

                result.update({
                    "transcription": transcription_text,
                    "transcription_language": trans_result.get("language", ""),
                    "transcription_confidence": trans_result.get("confidence", 0.0),
                    "transcription_word_count": trans_result.get("word_count", 0),
                    "transcription_status": trans_result.get("transcription_status", "unknown")
                })

            # Step 2: OCR (EasyOCR)
            if include_ocr:
                ocr_result = self.extract_ocr(video_path=video_path)
                ocr_text = ocr_result.get("ocr_text", "")

                result.update({
                    "ocr_text": ocr_text,
                    "text_density": ocr_result.get("text_density", 0.0),
                    "max_text_density": ocr_result.get("max_text_density", 0.0),
                    "ocr_text_boxes": ocr_result.get("total_text_boxes", 0),
                    "ocr_frames_processed": ocr_result.get("frames_processed", 0),
                    "ocr_status": ocr_result.get("ocr_status", "unknown")
                })

            # Step 3: Semantic Embeddings + PCA
            if include_semantics:
                sem_result = self.semantic_encoder.encode_and_reduce(
                    caption=caption,
                    transcription=transcription_text,
                    ocr_text=ocr_text
                )

                # Add PCA components (sem_pca_1 to sem_pca_10)
                for key, value in sem_result.items():
                    if key.startswith("sem_pca_"):
                        result[key] = value

                result["semantic_text_sources"] = sem_result.get("text_sources", 0)
                result["semantic_status"] = sem_result.get("semantic_status", "unknown")

        except Exception as e:
            logger.error(f"Text intelligence extraction failed: {e}")
            result["text_status"] = "error"
            result["text_error"] = str(e)

        finally:
            gc.collect()

        return result

    def fit_pca_from_corpus(
        self,
        texts: List[Dict[str, str]],
        save_path: Optional[str] = None
    ) -> "TextIntelligenceEngine":
        """
        Fit PCA on a corpus of texts for dimensionality reduction.

        Args:
            texts: List of dicts with 'caption', 'transcription', 'ocr_text' keys
            save_path: Optional path to save fitted PCA model

        Returns:
            self for chaining
        """
        logger.info(f"Fitting PCA on corpus of {len(texts)} samples")

        embeddings = []
        for text_dict in texts:
            embedding = self.semantic_encoder.encode_combined_text(
                caption=text_dict.get("caption", ""),
                transcription=text_dict.get("transcription", ""),
                ocr_text=text_dict.get("ocr_text", "")
            )
            embeddings.append(embedding)

        embeddings_array = np.vstack(embeddings)
        self.semantic_encoder.fit_pca(embeddings_array)

        if save_path:
            self.semantic_encoder.save_pca(save_path)

        return self

    def unload_all_models(self):
        """Unload all models to free memory."""
        self.transcriber.unload_model()
        self.ocr_extractor.unload_reader()
        self.semantic_encoder.unload_model()
        gc.collect()
        logger.info("All text intelligence models unloaded")


# =============================================================================
# CLI Entry Point (for testing)
# =============================================================================

if __name__ == "__main__":
    import sys
    import json

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python text_intelligence.py <video_path> [caption]")
        print("\nExample:")
        print("  python text_intelligence.py video.mp4")
        print("  python text_intelligence.py video.mp4 'Check out this viral content!'")
        sys.exit(1)

    video_path = sys.argv[1]
    caption = sys.argv[2] if len(sys.argv) > 2 else ""

    print(f"\n{'='*60}")
    print("Text Intelligence Engine - NLP Feature Extraction")
    print(f"{'='*60}\n")

    # Feature extraction
    print("[1] Initializing Engine...")
    engine = TextIntelligenceEngine()

    print(f"\n[2] Extracting Text Features from: {video_path}")
    if caption:
        print(f"    Caption: {caption[:50]}...")

    features = engine.extract_all_features(
        video_path=video_path,
        caption=caption
    )

    print("\n[3] Text Intelligence Features (JSON Output):")
    print(json.dumps(features, indent=2, ensure_ascii=False))

    # Show PCA features summary
    pca_features = {k: v for k, v in features.items() if k.startswith("sem_pca_")}
    if pca_features:
        print("\n[4] Semantic PCA Components:")
        for name, value in pca_features.items():
            bar = "=" * int(abs(value) * 50)
            sign = "+" if value >= 0 else "-"
            print(f"    {name}: {value:+.4f} [{sign}{bar}]")

    # Cleanup
    engine.unload_all_models()

    print(f"\n{'='*60}")
    print("Processing Complete")
    print(f"{'='*60}\n")
