"""
Light Multimodal Processors - Cost-Optimized Whisper/EasyOCR Pipeline
=====================================================================

Optimizes multimodal processing for cost efficiency without significant quality loss:
- Whisper: faster-whisper 'tiny' or 'base' model with local cache
- EasyOCR: First 5 frames only, Spanish priority
- Hook Analysis: First 3 seconds of audio/video only
- Media Cache: Hash-based deduplication to skip already processed content
- Conditional: Skip full multimodal if not Reel/Video

Usage:
    processor = LightMultimodalProcessor(light_mode=True)
    result = processor.process(media_url, content_type="reel")

Author: ML Engineering Team
"""

import gc
import hashlib
import json
import logging
import os
import pickle
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

MODELS_DIR = Path(__file__).parent.parent / "models"
CACHE_DIR = MODELS_DIR / "light_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Light processing defaults
LIGHT_WHISPER_MODEL = "tiny"  # 'tiny' (~75MB) or 'base' (~150MB)
LIGHT_MAX_OCR_FRAMES = 5
LIGHT_HOOK_DURATION_SECONDS = 3.0
LIGHT_OCR_LANGUAGES = ("es", "en")  # Spanish priority


@dataclass
class LightProcessingConfig:
    """Configuration for light multimodal processing."""

    # Whisper settings
    whisper_model: str = LIGHT_WHISPER_MODEL
    whisper_language: str = "es"
    whisper_beam_size: int = 1  # Greedy for speed
    whisper_compute_type: str = "int8"  # Quantized
    whisper_max_duration: float = LIGHT_HOOK_DURATION_SECONDS

    # OCR settings
    ocr_max_frames: int = LIGHT_MAX_OCR_FRAMES
    ocr_languages: Tuple[str, ...] = LIGHT_OCR_LANGUAGES
    ocr_gpu: bool = False
    ocr_use_thumbnail: bool = True  # Prefer thumbnail over frame extraction

    # Hook analysis
    hook_duration_seconds: float = LIGHT_HOOK_DURATION_SECONDS

    # Cache
    cache_enabled: bool = True
    cache_ttl_hours: int = 168  # 7 days

    # Conditional processing
    skip_non_video: bool = True  # Skip multimodal for static images
    video_content_types: Tuple[str, ...] = ("reel", "video", "tiktok_video", "tiktok")


@dataclass
class LightProcessingResult:
    """Result from light multimodal processing."""

    # Transcription
    transcription: str = ""
    transcription_language: str = ""
    transcription_confidence: float = 0.0
    transcription_duration: float = 0.0

    # OCR
    ocr_text: str = ""
    text_density: float = 0.0
    ocr_frames_processed: int = 0

    # Hook analysis
    hook_score: float = 0.0
    hook_text: str = ""

    # Processing metadata
    light_mode: bool = True
    processing_time_seconds: float = 0.0
    cached: bool = False
    skipped: bool = False
    skip_reason: str = ""

    # Comparison metrics (for benchmarking)
    estimated_full_time_seconds: float = 0.0
    time_saved_seconds: float = 0.0
    time_saved_percent: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "transcription": self.transcription,
            "transcription_language": self.transcription_language,
            "transcription_confidence": self.transcription_confidence,
            "transcription_duration": self.transcription_duration,
            "ocr_text": self.ocr_text,
            "text_density": self.text_density,
            "ocr_frames_processed": self.ocr_frames_processed,
            "hook_score": self.hook_score,
            "hook_text": self.hook_text,
            "light_mode": self.light_mode,
            "processing_time_seconds": self.processing_time_seconds,
            "cached": self.cached,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "time_saved_seconds": self.time_saved_seconds,
            "time_saved_percent": self.time_saved_percent,
        }


# =============================================================================
# Media Cache - Hash-based deduplication
# =============================================================================

class MediaCache:
    """
    Hash-based cache for processed media content.

    Uses SHA256 hash of media_url to skip re-processing of already seen content.
    Cache is persisted to disk for cross-session efficiency.
    """

    def __init__(self, cache_dir: Path = CACHE_DIR, ttl_hours: int = 168):
        self.cache_dir = cache_dir
        self.cache_file = cache_dir / "media_cache.pkl"
        self.ttl_seconds = ttl_hours * 3600
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load_cache()

    def _load_cache(self):
        """Load cache from disk."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, "rb") as f:
                    self._cache = pickle.load(f)
                logger.info(f"Loaded {len(self._cache)} cached media items")
                # Clean expired entries
                self._cleanup_expired()
        except Exception as e:
            logger.warning(f"Could not load media cache: {e}")
            self._cache = {}

    def _save_cache(self):
        """Persist cache to disk."""
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self._cache, f)
        except Exception as e:
            logger.warning(f"Could not save media cache: {e}")

    def _cleanup_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        expired = [
            key for key, value in self._cache.items()
            if current_time - value.get("timestamp", 0) > self.ttl_seconds
        ]
        for key in expired:
            del self._cache[key]
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired cache entries")
            self._save_cache()

    @staticmethod
    def hash_url(media_url: str) -> str:
        """Generate SHA256 hash for media URL."""
        return hashlib.sha256(media_url.encode()).hexdigest()[:16]

    def get(self, media_url: str) -> Optional[Dict[str, Any]]:
        """Get cached result for media URL."""
        cache_key = self.hash_url(media_url)
        cached = self._cache.get(cache_key)

        if cached:
            # Check if expired
            if time.time() - cached.get("timestamp", 0) > self.ttl_seconds:
                del self._cache[cache_key]
                return None
            logger.debug(f"Cache hit for {cache_key}")
            return cached.get("result")

        return None

    def set(self, media_url: str, result: Dict[str, Any]):
        """Cache result for media URL."""
        cache_key = self.hash_url(media_url)
        self._cache[cache_key] = {
            "result": result,
            "timestamp": time.time(),
            "url_prefix": media_url[:50]  # For debugging
        }
        self._save_cache()
        logger.debug(f"Cached result for {cache_key}")

    def has(self, media_url: str) -> bool:
        """Check if media URL is cached."""
        return self.get(media_url) is not None

    def clear(self):
        """Clear all cache."""
        self._cache = {}
        self._save_cache()
        logger.info("Media cache cleared")

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "total_entries": len(self._cache),
            "cache_file": str(self.cache_file),
            "ttl_hours": self.ttl_seconds / 3600,
        }


# =============================================================================
# Light Whisper Processor
# =============================================================================

class LightWhisperProcessor:
    """
    Lightweight Whisper transcription using faster-whisper.

    Optimizations:
    - Uses 'tiny' model by default (~75MB RAM)
    - Processes only first N seconds (hook analysis)
    - int8 quantization for speed
    - Local model caching
    """

    def __init__(self, config: Optional[LightProcessingConfig] = None):
        self.config = config or LightProcessingConfig()
        self._model = None
        self._available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if faster-whisper is available."""
        try:
            from faster_whisper import WhisperModel
            return True
        except ImportError:
            logger.warning(
                "faster-whisper not installed. "
                "Install with: pip install faster-whisper"
            )
            return False

    def _load_model(self):
        """Lazy load model with caching."""
        if self._model is None and self._available:
            from faster_whisper import WhisperModel

            logger.info(f"Loading light Whisper model: {self.config.whisper_model}")

            # Model is cached locally by faster-whisper
            self._model = WhisperModel(
                self.config.whisper_model,
                device="cpu",
                compute_type=self.config.whisper_compute_type,
                cpu_threads=2,
                num_workers=1,
                download_root=str(MODELS_DIR / "whisper_cache")
            )
            logger.info(f"Light Whisper model '{self.config.whisper_model}' loaded")

    def transcribe(
        self,
        audio_path: str,
        max_duration: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio with duration limit for hook analysis.

        Args:
            audio_path: Path to audio/video file
            max_duration: Maximum seconds to transcribe (default: hook_duration)

        Returns:
            Dict with transcription and metadata
        """
        if not self._available:
            return {
                "transcription": "",
                "language": "",
                "confidence": 0.0,
                "duration_processed": 0.0,
                "status": "whisper_not_available"
            }

        max_duration = max_duration or self.config.whisper_max_duration

        try:
            self._load_model()

            # Transcribe with VAD and duration limit
            segments, info = self._model.transcribe(
                audio_path,
                language=self.config.whisper_language,
                beam_size=self.config.whisper_beam_size,
                word_timestamps=False,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=200
                )
            )

            # Collect segments up to max_duration
            text_parts = []
            total_duration = 0.0

            for segment in segments:
                # Only process up to max_duration
                if segment.start >= max_duration:
                    break

                text_parts.append(segment.text.strip())
                total_duration = min(segment.end, max_duration)

            full_text = " ".join(text_parts)

            result = {
                "transcription": full_text,
                "language": info.language,
                "confidence": round(info.language_probability, 4),
                "duration_processed": round(total_duration, 2),
                "total_duration": round(info.duration, 2),
                "word_count": len(full_text.split()) if full_text else 0,
                "status": "success",
                "light_mode": True,
                "model_used": self.config.whisper_model
            }

            logger.debug(
                f"Light transcription: {result['word_count']} words "
                f"from first {total_duration:.1f}s (total: {info.duration:.1f}s)"
            )

            return result

        except Exception as e:
            logger.error(f"Light transcription failed: {e}")
            return {
                "transcription": "",
                "language": "",
                "confidence": 0.0,
                "duration_processed": 0.0,
                "status": "error",
                "error": str(e)
            }
        finally:
            gc.collect()

    def unload(self):
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()
            logger.info("Light Whisper model unloaded")


# =============================================================================
# Light OCR Processor
# =============================================================================

class LightOCRProcessor:
    """
    Lightweight EasyOCR processing.

    Optimizations:
    - Processes only first N frames (default: 5)
    - Prefers thumbnail over full frame extraction
    - Spanish language priority
    - CPU-only operation
    """

    def __init__(self, config: Optional[LightProcessingConfig] = None):
        self.config = config or LightProcessingConfig()
        self._reader = None
        self._available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if EasyOCR is available."""
        try:
            import easyocr
            return True
        except ImportError:
            logger.warning(
                "easyocr not installed. "
                "Install with: pip install easyocr"
            )
            return False

    def _load_reader(self):
        """Lazy load EasyOCR reader."""
        if self._reader is None and self._available:
            import easyocr

            logger.info(f"Loading light EasyOCR with languages: {self.config.ocr_languages}")
            self._reader = easyocr.Reader(
                list(self.config.ocr_languages),
                gpu=self.config.ocr_gpu,
                verbose=False,
                model_storage_directory=str(MODELS_DIR / "easyocr_cache")
            )
            logger.info("Light EasyOCR reader loaded")

    def extract_from_thumbnail(
        self,
        thumbnail_url_or_path: str
    ) -> Dict[str, Any]:
        """
        Extract text from a single thumbnail image.

        Most efficient for quick OCR - single image processing.
        """
        if not self._available:
            return {
                "ocr_text": "",
                "text_density": 0.0,
                "status": "easyocr_not_available"
            }

        try:
            self._load_reader()

            # Handle URL vs local path
            import cv2
            import urllib.request

            if thumbnail_url_or_path.startswith(("http://", "https://")):
                # Download to temp file
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    urllib.request.urlretrieve(thumbnail_url_or_path, tmp.name)
                    frame = cv2.imread(tmp.name)
                    os.unlink(tmp.name)
            else:
                frame = cv2.imread(thumbnail_url_or_path)

            if frame is None:
                return {
                    "ocr_text": "",
                    "text_density": 0.0,
                    "status": "could_not_load_image"
                }

            return self._process_frame(frame)

        except Exception as e:
            logger.error(f"Thumbnail OCR failed: {e}")
            return {
                "ocr_text": "",
                "text_density": 0.0,
                "status": "error",
                "error": str(e)
            }

    def extract_from_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extract text from first N frames of video.

        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process (default: config.ocr_max_frames)
        """
        if not self._available:
            return {
                "ocr_text": "",
                "text_density": 0.0,
                "frames_processed": 0,
                "status": "easyocr_not_available"
            }

        max_frames = max_frames or self.config.ocr_max_frames

        try:
            import cv2

            self._load_reader()

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {
                    "ocr_text": "",
                    "text_density": 0.0,
                    "frames_processed": 0,
                    "status": "could_not_open_video"
                }

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30

            # Sample frames evenly from first portion of video
            # Focus on beginning for hook text
            if total_frames <= max_frames:
                frame_indices = list(range(total_frames))
            else:
                # First half of video, evenly spaced
                half_duration_frames = min(total_frames, int(fps * 10))  # First 10 seconds
                frame_indices = np.linspace(
                    0, half_duration_frames - 1,
                    min(max_frames, half_duration_frames),
                    dtype=int
                ).tolist()

            all_texts = []
            all_densities = []
            frames_processed = 0

            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()

                if not ret:
                    continue

                # Resize for faster OCR
                frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)

                result = self._process_frame(frame)
                if result["status"] == "success" and result["ocr_text"]:
                    all_texts.append(result["ocr_text"])
                    all_densities.append(result["text_density"])

                frames_processed += 1

                # Memory management
                if frames_processed % 3 == 0:
                    gc.collect()

            cap.release()

            # Deduplicate similar texts
            unique_texts = list(dict.fromkeys(all_texts))

            return {
                "ocr_text": " ".join(unique_texts),
                "text_density": np.mean(all_densities) if all_densities else 0.0,
                "max_text_density": max(all_densities) if all_densities else 0.0,
                "frames_processed": frames_processed,
                "status": "success",
                "light_mode": True
            }

        except Exception as e:
            logger.error(f"Video OCR failed: {e}")
            return {
                "ocr_text": "",
                "text_density": 0.0,
                "frames_processed": 0,
                "status": "error",
                "error": str(e)
            }
        finally:
            gc.collect()

    def _process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a single frame for OCR."""
        try:
            frame_height, frame_width = frame.shape[:2]
            frame_area = frame_width * frame_height

            results = self._reader.readtext(
                frame,
                paragraph=True,
                min_size=10,
                text_threshold=0.3
            )

            texts = []
            total_text_area = 0.0

            for detection in results:
                bbox, text, confidence = detection
                if confidence >= 0.3:
                    texts.append(text)
                    # Calculate text area
                    bbox = np.array(bbox)
                    width = np.max(bbox[:, 0]) - np.min(bbox[:, 0])
                    height = np.max(bbox[:, 1]) - np.min(bbox[:, 1])
                    total_text_area += width * height

            text_density = (total_text_area / frame_area) * 100 if frame_area > 0 else 0.0

            return {
                "ocr_text": " ".join(texts),
                "text_density": round(text_density, 4),
                "text_boxes": len(texts),
                "status": "success"
            }

        except Exception as e:
            return {
                "ocr_text": "",
                "text_density": 0.0,
                "status": "error",
                "error": str(e)
            }

    def unload(self):
        """Unload reader to free memory."""
        if self._reader is not None:
            del self._reader
            self._reader = None
            gc.collect()
            logger.info("Light EasyOCR reader unloaded")


# =============================================================================
# Light Hook Analyzer
# =============================================================================

class LightHookAnalyzer:
    """
    Lightweight hook analysis - first 3 seconds only.

    Analyzes:
    - Opening text/speech for hook patterns
    - Visual text overlay in opening frames
    - Quick semantic hook detection
    """

    # Common viral hook patterns (Spanish + English)
    HOOK_PATTERNS = [
        # Questions
        r"(?:^|\s)(?:por qu[eé]|c[oó]mo|qu[eé]|cu[aá]l|d[oó]nde|cu[aá]ndo|qui[eé]n)",
        r"(?:^|\s)(?:why|how|what|which|where|when|who)\b",
        # POV/Story hooks
        r"(?:pov|cuando|imagina|esto es|mira lo que)",
        r"(?:story time|wait for it|you won't believe)",
        # Reveal hooks
        r"(?:secreto|truco|hack|tip|dato|descubr[íi])",
        r"(?:secret|trick|hack|tip|discover)",
        # Transformation
        r"(?:antes.+despu[eé]s|transforma|cambi[oó]|de.+a\s)",
        # Urgency
        r"(?:urgente|ahora mismo|ya|no te pierdas|[úu]ltimo)",
        r"(?:urgent|right now|don't miss|last chance)",
        # Numbers
        r"(?:\d+\s*(?:tips?|trucos?|formas?|razones?|secretos?))",
        r"(?:\d+\s*(?:ways?|tips?|tricks?|reasons?|secrets?))",
    ]

    def __init__(self, config: Optional[LightProcessingConfig] = None):
        self.config = config or LightProcessingConfig()
        self._patterns_compiled = None

    def _compile_patterns(self):
        """Compile regex patterns."""
        if self._patterns_compiled is None:
            import re
            self._patterns_compiled = [
                re.compile(p, re.IGNORECASE)
                for p in self.HOOK_PATTERNS
            ]

    def analyze_hook(
        self,
        transcription: str = "",
        ocr_text: str = "",
        caption: str = ""
    ) -> Dict[str, Any]:
        """
        Analyze hook quality from first 3s content.

        Args:
            transcription: Speech from first 3 seconds
            ocr_text: Text overlay from first frames
            caption: Post caption (first line often is hook)

        Returns:
            Dict with hook score and analysis
        """
        self._compile_patterns()

        # Combine text sources, prioritize opening content
        hook_text = ""
        if caption:
            # First sentence of caption
            first_line = caption.split("\n")[0].split(".")[0]
            hook_text = first_line

        if ocr_text:
            hook_text = f"{hook_text} {ocr_text}".strip()

        if transcription:
            hook_text = f"{hook_text} {transcription}".strip()

        if not hook_text:
            return {
                "hook_score": 0.0,
                "hook_text": "",
                "patterns_matched": [],
                "status": "no_text"
            }

        # Score based on pattern matches
        patterns_matched = []
        for i, pattern in enumerate(self._patterns_compiled):
            if pattern.search(hook_text):
                patterns_matched.append(self.HOOK_PATTERNS[i][:30])

        # Base score from patterns
        base_score = min(len(patterns_matched) * 0.2, 0.6)

        # Bonuses
        bonuses = 0.0

        # Question bonus
        if "?" in hook_text:
            bonuses += 0.1

        # Emoji bonus (engagement indicator)
        import re
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "]+",
            flags=re.UNICODE
        )
        if emoji_pattern.search(hook_text):
            bonuses += 0.05

        # Ellipsis bonus (curiosity)
        if "..." in hook_text:
            bonuses += 0.05

        # Length penalty (too long = weak hook)
        word_count = len(hook_text.split())
        if word_count > 15:
            bonuses -= 0.1

        final_score = min(max(base_score + bonuses, 0.0), 1.0)

        return {
            "hook_score": round(final_score, 4),
            "hook_text": hook_text[:200],  # Truncate for storage
            "patterns_matched": patterns_matched,
            "word_count": word_count,
            "status": "success",
            "light_mode": True
        }


# =============================================================================
# Unified Light Multimodal Processor
# =============================================================================

class LightMultimodalProcessor:
    """
    Unified lightweight multimodal processor.

    Combines all optimizations:
    - Cache check first (skip if already processed)
    - Content type check (skip non-video)
    - Light Whisper (first 3s only)
    - Light OCR (5 frames max)
    - Hook analysis

    Usage:
        processor = LightMultimodalProcessor(light_mode=True)
        result = processor.process(
            media_url="https://...",
            content_type="reel",
            caption="Check this out!"
        )
    """

    def __init__(
        self,
        light_mode: bool = True,
        config: Optional[LightProcessingConfig] = None
    ):
        self.light_mode = light_mode
        self.config = config or LightProcessingConfig()

        # Initialize components lazily
        self._cache = MediaCache(ttl_hours=self.config.cache_ttl_hours) if self.config.cache_enabled else None
        self._whisper = None
        self._ocr = None
        self._hook_analyzer = None

        logger.info(
            f"LightMultimodalProcessor initialized - "
            f"light_mode={light_mode}, cache={self.config.cache_enabled}"
        )

    def _get_whisper(self) -> LightWhisperProcessor:
        """Get or create Whisper processor."""
        if self._whisper is None:
            self._whisper = LightWhisperProcessor(self.config)
        return self._whisper

    def _get_ocr(self) -> LightOCRProcessor:
        """Get or create OCR processor."""
        if self._ocr is None:
            self._ocr = LightOCRProcessor(self.config)
        return self._ocr

    def _get_hook_analyzer(self) -> LightHookAnalyzer:
        """Get or create hook analyzer."""
        if self._hook_analyzer is None:
            self._hook_analyzer = LightHookAnalyzer(self.config)
        return self._hook_analyzer

    def should_skip(self, content_type: str) -> Tuple[bool, str]:
        """
        Check if processing should be skipped for content type.

        Returns:
            Tuple of (should_skip, reason)
        """
        if not self.config.skip_non_video:
            return False, ""

        content_type_lower = content_type.lower() if content_type else ""

        if content_type_lower in self.config.video_content_types:
            return False, ""

        return True, f"Non-video content type: {content_type}"

    def process(
        self,
        media_url: Optional[str] = None,
        media_path: Optional[str] = None,
        thumbnail_url: Optional[str] = None,
        content_type: str = "reel",
        caption: str = "",
        force_reprocess: bool = False
    ) -> LightProcessingResult:
        """
        Process media with light optimizations.

        Args:
            media_url: URL of media (for caching key)
            media_path: Local path to media file
            thumbnail_url: URL of thumbnail (for quick OCR)
            content_type: Type of content (reel, video, post, etc.)
            caption: Post caption
            force_reprocess: Skip cache and reprocess

        Returns:
            LightProcessingResult with all extracted features
        """
        start_time = time.time()
        result = LightProcessingResult(light_mode=self.light_mode)

        # Generate cache key
        cache_key = media_url or media_path or thumbnail_url or ""

        # Check cache first
        if self._cache and not force_reprocess and cache_key:
            cached = self._cache.get(cache_key)
            if cached:
                result.cached = True
                result.processing_time_seconds = time.time() - start_time
                result.time_saved_seconds = cached.get("original_time", 5.0)
                result.time_saved_percent = 99.0  # Cache hit = ~99% time saved

                # Restore cached values
                for key, value in cached.items():
                    if hasattr(result, key):
                        setattr(result, key, value)

                logger.info(
                    f"Multimodal light: {result.processing_time_seconds:.2f}s (CACHED) "
                    f"vs ~{result.time_saved_seconds:.1f}s full"
                )
                return result

        # Check if should skip non-video content
        should_skip, skip_reason = self.should_skip(content_type)
        if should_skip:
            result.skipped = True
            result.skip_reason = skip_reason
            result.processing_time_seconds = time.time() - start_time

            # Still do hook analysis on caption
            if caption:
                hook_result = self._get_hook_analyzer().analyze_hook(caption=caption)
                result.hook_score = hook_result.get("hook_score", 0.0)
                result.hook_text = hook_result.get("hook_text", "")

            logger.info(f"Multimodal skipped: {skip_reason}")
            return result

        # Process with light optimizations
        try:
            # 1. Transcription (first 3s only)
            if media_path and self.light_mode:
                whisper_result = self._get_whisper().transcribe(
                    media_path,
                    max_duration=self.config.hook_duration_seconds
                )
                result.transcription = whisper_result.get("transcription", "")
                result.transcription_language = whisper_result.get("language", "")
                result.transcription_confidence = whisper_result.get("confidence", 0.0)
                result.transcription_duration = whisper_result.get("duration_processed", 0.0)

            # 2. OCR (thumbnail preferred, then first 5 frames)
            if self.light_mode:
                if thumbnail_url and self.config.ocr_use_thumbnail:
                    ocr_result = self._get_ocr().extract_from_thumbnail(thumbnail_url)
                elif media_path:
                    ocr_result = self._get_ocr().extract_from_video(
                        media_path,
                        max_frames=self.config.ocr_max_frames
                    )
                else:
                    ocr_result = {}

                result.ocr_text = ocr_result.get("ocr_text", "")
                result.text_density = ocr_result.get("text_density", 0.0)
                result.ocr_frames_processed = ocr_result.get("frames_processed", 0)

            # 3. Hook analysis
            hook_result = self._get_hook_analyzer().analyze_hook(
                transcription=result.transcription,
                ocr_text=result.ocr_text,
                caption=caption
            )
            result.hook_score = hook_result.get("hook_score", 0.0)
            result.hook_text = hook_result.get("hook_text", "")

            # Calculate timing
            result.processing_time_seconds = time.time() - start_time

            # Estimate full processing time (heuristic based on benchmarks)
            # Full Whisper tiny ~10s, base ~15s; Full OCR all frames ~8s
            result.estimated_full_time_seconds = 20.0  # Conservative estimate
            result.time_saved_seconds = max(
                0, result.estimated_full_time_seconds - result.processing_time_seconds
            )
            result.time_saved_percent = (
                (result.time_saved_seconds / result.estimated_full_time_seconds) * 100
                if result.estimated_full_time_seconds > 0 else 0
            )

            # Cache result
            if self._cache and cache_key:
                cache_data = result.to_dict()
                cache_data["original_time"] = result.processing_time_seconds
                self._cache.set(cache_key, cache_data)

            logger.info(
                f"Multimodal light: {result.processing_time_seconds:.2f}s "
                f"vs ~{result.estimated_full_time_seconds:.1f}s full "
                f"({result.time_saved_percent:.0f}% saved)"
            )

        except Exception as e:
            logger.error(f"Light multimodal processing failed: {e}")
            result.processing_time_seconds = time.time() - start_time

        finally:
            gc.collect()

        return result

    def unload_all(self):
        """Unload all models to free memory."""
        if self._whisper:
            self._whisper.unload()
        if self._ocr:
            self._ocr.unload()
        gc.collect()
        logger.info("All light processors unloaded")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self._cache:
            return self._cache.stats()
        return {"cache_enabled": False}


# =============================================================================
# Singleton Instance
# =============================================================================

_light_processor: Optional[LightMultimodalProcessor] = None


def get_light_processor(
    light_mode: bool = True,
    config: Optional[LightProcessingConfig] = None
) -> LightMultimodalProcessor:
    """
    Get singleton instance of light multimodal processor.

    Args:
        light_mode: Enable light optimizations (default: True)
        config: Custom configuration

    Returns:
        LightMultimodalProcessor instance
    """
    global _light_processor

    if _light_processor is None:
        _light_processor = LightMultimodalProcessor(
            light_mode=light_mode,
            config=config
        )

    return _light_processor


def process_media_light(
    media_url: Optional[str] = None,
    media_path: Optional[str] = None,
    thumbnail_url: Optional[str] = None,
    content_type: str = "reel",
    caption: str = "",
    light_mode: bool = True
) -> Dict[str, Any]:
    """
    Convenience function for light multimodal processing.

    Args:
        media_url: URL of media
        media_path: Local path to media
        thumbnail_url: Thumbnail URL for quick OCR
        content_type: Content type (reel, video, post)
        caption: Post caption
        light_mode: Enable light optimizations

    Returns:
        Dict with processing results
    """
    processor = get_light_processor(light_mode=light_mode)
    result = processor.process(
        media_url=media_url,
        media_path=media_path,
        thumbnail_url=thumbnail_url,
        content_type=content_type,
        caption=caption
    )
    return result.to_dict()


# =============================================================================
# Benchmark Utilities
# =============================================================================

def benchmark_light_vs_full(
    media_path: str,
    content_type: str = "reel",
    caption: str = ""
) -> Dict[str, Any]:
    """
    Benchmark light processing vs estimated full processing.

    Useful for validating optimization gains.
    """
    import time

    # Light processing
    light_config = LightProcessingConfig()
    light_processor = LightMultimodalProcessor(light_mode=True, config=light_config)

    light_start = time.time()
    light_result = light_processor.process(
        media_path=media_path,
        content_type=content_type,
        caption=caption,
        force_reprocess=True  # Skip cache for benchmark
    )
    light_time = time.time() - light_start

    # Full processing estimate (based on heuristics)
    # Could be replaced with actual full processing for comparison
    full_time_estimate = 20.0  # seconds

    # Memory estimate (heuristic)
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / (1024 * 1024)

    benchmark = {
        "light_time_seconds": round(light_time, 3),
        "full_time_estimate_seconds": full_time_estimate,
        "time_saved_seconds": round(full_time_estimate - light_time, 3),
        "time_saved_percent": round((1 - light_time / full_time_estimate) * 100, 1),
        "memory_mb": round(memory_mb, 1),
        "transcription_words": light_result.transcription.count(" ") + 1 if light_result.transcription else 0,
        "ocr_frames": light_result.ocr_frames_processed,
        "hook_score": light_result.hook_score,
        "content_type": content_type,
        "light_mode": True,
    }

    logger.info(
        f"Benchmark: Light {light_time:.2f}s vs Full ~{full_time_estimate:.1f}s "
        f"({benchmark['time_saved_percent']:.0f}% saved, {memory_mb:.0f}MB RAM)"
    )

    light_processor.unload_all()

    return benchmark


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python light_processors.py <video_path> [content_type] [caption]")
        print("\nExample:")
        print("  python light_processors.py video.mp4 reel 'Check this out!'")
        sys.exit(1)

    video_path = sys.argv[1]
    content_type = sys.argv[2] if len(sys.argv) > 2 else "reel"
    caption = sys.argv[3] if len(sys.argv) > 3 else ""

    print(f"\n{'='*60}")
    print("Light Multimodal Processor - Benchmark")
    print(f"{'='*60}\n")

    result = benchmark_light_vs_full(video_path, content_type, caption)

    print("\n[Results]")
    for key, value in result.items():
        print(f"  {key}: {value}")

    print(f"\n{'='*60}")
    print("Processing Complete")
    print(f"{'='*60}\n")
