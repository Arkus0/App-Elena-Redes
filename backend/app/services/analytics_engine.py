"""
Analytics Engine - Optimized for Edge Computing (Low RAM/CPU)
=============================================================

High-performance video and audio feature extraction with strict memory constraints.
Designed for edge devices and resource-limited environments.

Key Optimizations:
- Frame striding: Process only 1 frame per 0.5-1 second
- Downsampling: Resize frames to 224x224 before processing
- Generator-based processing: Yield frames instead of loading all in memory
- Explicit garbage collection after heavy operations
- float32 precision to reduce memory footprint
- Audio limited to first 30 seconds at 22050Hz

Extended with Text Intelligence:
- Whisper transcription (speech-to-text)
- EasyOCR (text overlay detection)
- Semantic embeddings with PCA reduction

Author: ML Engineering Team
"""

import gc
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional, Dict, Any, List, Tuple
import warnings

import cv2
import numpy as np

# Suppress librosa warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

# Lazy import for text intelligence (optional dependency)
_text_intelligence_available = None


def _check_text_intelligence() -> bool:
    """Check if text intelligence module is available."""
    global _text_intelligence_available
    if _text_intelligence_available is None:
        try:
            from app.services.text_intelligence import TextIntelligenceEngine
            _text_intelligence_available = True
        except ImportError:
            _text_intelligence_available = False
            logger.warning(
                "Text intelligence module not available. "
                "Install dependencies: pip install faster-whisper easyocr sentence-transformers"
            )
    return _text_intelligence_available


# =============================================================================
# Configuration Constants
# =============================================================================

@dataclass(frozen=True)
class VideoConfig:
    """Immutable configuration for video processing."""
    target_resolution: Tuple[int, int] = (224, 224)  # Reduce memory by ~90%
    frame_stride_seconds: float = 0.5  # Analyze 1 frame every 0.5s
    histogram_bins: int = 64  # Reduced from 256 for faster computation
    cut_threshold: float = 0.5  # Histogram correlation threshold for cuts
    dtype: np.dtype = field(default_factory=lambda: np.float32)


@dataclass(frozen=True)
class AudioConfig:
    """Immutable configuration for audio processing."""
    sample_rate: int = 22050  # Standard for music analysis
    duration_seconds: float = 30.0  # Only process first 30 seconds
    mono: bool = True  # Single channel reduces memory by 50%
    dtype: np.dtype = field(default_factory=lambda: np.float32)


# =============================================================================
# Video Feature Extractor (Memory-Optimized)
# =============================================================================

class EfficientFrameGenerator:
    """
    Memory-efficient frame generator using Python generators.

    Implements frame striding to skip unnecessary frames,
    reducing CPU and memory usage by up to 95%.
    """

    def __init__(self, video_path: str, config: VideoConfig = VideoConfig()):
        self.video_path = video_path
        self.config = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 0.0
        self._total_frames: int = 0
        self._duration: float = 0.0

    def __enter__(self) -> "EfficientFrameGenerator":
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._duration = self._total_frames / self._fps if self._fps > 0 else 0.0

        logger.debug(f"Video opened: {self._fps:.1f} FPS, {self._duration:.1f}s duration")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        gc.collect()  # Force garbage collection after releasing video

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def total_frames(self) -> int:
        return self._total_frames

    def get_stride_frames(self) -> int:
        """Calculate how many frames to skip based on stride configuration."""
        return max(1, int(self._fps * self.config.frame_stride_seconds))

    def generate_frames(self) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields downsampled frames at the configured stride.

        Memory Impact: Only ONE frame in memory at any time.

        Yields:
            np.ndarray: Downsampled grayscale frame (224x224, float32)
        """
        if self._cap is None:
            raise RuntimeError("Generator must be used within context manager")

        stride = self.get_stride_frames()
        frame_idx = 0

        while True:
            ret, frame = self._cap.read()
            if not ret:
                break

            # Only process frames at the stride interval
            if frame_idx % stride == 0:
                # Downsample to target resolution
                resized = cv2.resize(
                    frame,
                    self.config.target_resolution,
                    interpolation=cv2.INTER_AREA  # Best for downsampling
                )

                # Convert to grayscale for luminance analysis
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

                # Convert to float32 for numerical precision with lower memory
                yield gray.astype(self.config.dtype) / 255.0

            frame_idx += 1

        logger.debug(f"Processed {frame_idx // stride} frames (stride: {stride})")


class EfficientFeatureExtractor:
    """
    Memory-optimized video feature extractor for edge computing.

    Extracts DNA Features:
    - visual_energy: Motion detection via frame differencing
    - brightness_variance: Luminance variation
    - cut_density: Edit rhythm (cuts per minute)

    Memory Strategy:
    - Uses generators to process frames one-by-one
    - Accumulates only scalar metrics, not raw frames
    - Forces gc.collect() after processing
    """

    def __init__(self, config: Optional[VideoConfig] = None):
        self.config = config or VideoConfig()
        self._reset_accumulators()

    def _reset_accumulators(self) -> None:
        """Reset all metric accumulators for a new video."""
        self._frame_diffs: List[float] = []
        self._brightness_values: List[float] = []
        self._histograms: List[np.ndarray] = []
        self._prev_frame: Optional[np.ndarray] = None

    def _calculate_frame_diff(self, current: np.ndarray) -> float:
        """
        Calculate mean absolute difference between consecutive frames.

        Higher values indicate more motion/visual energy.
        """
        if self._prev_frame is None:
            return 0.0

        diff = np.abs(current - self._prev_frame)
        return float(np.mean(diff))

    def _calculate_histogram(self, frame: np.ndarray) -> np.ndarray:
        """
        Calculate normalized histogram for cut detection.

        Uses reduced bins (64 instead of 256) for speed.
        """
        hist = cv2.calcHist(
            [(frame * 255).astype(np.uint8)],
            [0],
            None,
            [self.config.histogram_bins],
            [0, 256]
        )
        # Normalize histogram
        cv2.normalize(hist, hist)
        return hist.flatten().astype(self.config.dtype)

    def _detect_cut(self, current_hist: np.ndarray) -> bool:
        """
        Detect scene cuts by comparing histogram correlation.

        A cut is detected when correlation drops below threshold.
        """
        if len(self._histograms) == 0:
            return False

        prev_hist = self._histograms[-1]
        correlation = cv2.compareHist(
            current_hist.reshape(-1, 1),
            prev_hist.reshape(-1, 1),
            cv2.HISTCMP_CORREL
        )

        return correlation < self.config.cut_threshold

    def extract_video_features(self, video_path: str) -> Dict[str, float]:
        """
        Extract all video features with memory-optimized processing.

        Args:
            video_path: Path to video file

        Returns:
            Dict with visual_energy, brightness_variance, cut_density
        """
        self._reset_accumulators()
        cut_count = 0
        duration = 0.0

        try:
            with EfficientFrameGenerator(video_path, self.config) as gen:
                duration = gen.duration

                for frame in gen.generate_frames():
                    # Accumulate brightness (mean luminance)
                    self._brightness_values.append(float(np.mean(frame)))

                    # Calculate motion energy
                    frame_diff = self._calculate_frame_diff(frame)
                    if frame_diff > 0:
                        self._frame_diffs.append(frame_diff)

                    # Histogram-based cut detection
                    hist = self._calculate_histogram(frame)
                    if self._detect_cut(hist):
                        cut_count += 1

                    # Keep only last histogram to minimize memory
                    if len(self._histograms) > 1:
                        self._histograms.pop(0)
                    self._histograms.append(hist)

                    # Update previous frame reference
                    self._prev_frame = frame

            # Calculate final metrics
            visual_energy = float(np.mean(self._frame_diffs)) if self._frame_diffs else 0.0
            brightness_variance = float(np.std(self._brightness_values)) if self._brightness_values else 0.0

            # Cuts per minute
            duration_minutes = duration / 60.0
            cut_density = float(cut_count / duration_minutes) if duration_minutes > 0 else 0.0

            return {
                "visual_energy": round(visual_energy, 6),
                "brightness_variance": round(brightness_variance, 6),
                "cut_density": round(cut_density, 2),
                "duration_seconds": round(duration, 2),
                "frames_analyzed": len(self._brightness_values)
            }

        finally:
            # Aggressive memory cleanup
            self._reset_accumulators()
            gc.collect()

    def extract_features_batch(
        self,
        video_paths: List[str],
        on_progress: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple videos with explicit garbage collection between each.

        Args:
            video_paths: List of video file paths
            on_progress: Optional callback(index, total, path) for progress

        Returns:
            List of feature dictionaries
        """
        results = []
        total = len(video_paths)

        for idx, path in enumerate(video_paths):
            try:
                if on_progress:
                    on_progress(idx, total, path)

                features = self.extract_video_features(path)
                features["path"] = path
                features["status"] = "success"
                results.append(features)

            except Exception as e:
                logger.error(f"Failed to process {path}: {e}")
                results.append({
                    "path": path,
                    "status": "error",
                    "error": str(e)
                })

            # Force garbage collection between videos
            gc.collect()

        return results


# =============================================================================
# Audio Feature Extractor (Memory-Optimized)
# =============================================================================

class EfficientAudioExtractor:
    """
    Memory-optimized audio feature extractor.

    Strategies:
    - Load only first 30 seconds
    - Resample to 22050Hz (lower than 44.1kHz standard)
    - Mono channel only
    - Extract minimal feature set (tempo, onset_strength)
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self._librosa_available = self._check_librosa()

    def _check_librosa(self) -> bool:
        """Check if librosa is available (optional dependency)."""
        try:
            import librosa
            return True
        except ImportError:
            logger.warning(
                "librosa not installed. Audio features will be unavailable. "
                "Install with: pip install librosa"
            )
            return False

    def extract_audio_features(self, audio_path: str) -> Dict[str, float]:
        """
        Extract audio features with memory optimization.

        Args:
            audio_path: Path to audio file (or video with audio track)

        Returns:
            Dict with tempo (BPM) and onset_strength
        """
        if not self._librosa_available:
            return {
                "tempo": 0.0,
                "onset_strength": 0.0,
                "audio_status": "librosa_not_available"
            }

        import librosa

        try:
            # Load ONLY first 30 seconds, mono, resampled to 22050Hz
            y, sr = librosa.load(
                audio_path,
                sr=self.config.sample_rate,
                mono=self.config.mono,
                duration=self.config.duration_seconds,
                dtype=self.config.dtype
            )

            # Calculate onset envelope (for tempo and strength)
            onset_env = librosa.onset.onset_strength(
                y=y,
                sr=sr,
                hop_length=512  # Standard hop for speed
            )

            # Extract tempo (BPM)
            tempo, _ = librosa.beat.beat_track(
                onset_envelope=onset_env,
                sr=sr,
                hop_length=512
            )

            # Handle tempo array (librosa >= 0.10 returns array)
            if isinstance(tempo, np.ndarray):
                tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
            else:
                tempo = float(tempo)

            # Calculate mean onset strength (audio energy/beat intensity)
            onset_strength = float(np.mean(onset_env))

            result = {
                "tempo": round(tempo, 2),
                "onset_strength": round(onset_strength, 6),
                "audio_duration_analyzed": min(
                    self.config.duration_seconds,
                    len(y) / sr
                ),
                "audio_status": "success"
            }

            # Cleanup
            del y, onset_env
            gc.collect()

            return result

        except Exception as e:
            logger.error(f"Audio extraction failed for {audio_path}: {e}")
            return {
                "tempo": 0.0,
                "onset_strength": 0.0,
                "audio_status": "error",
                "audio_error": str(e)
            }


# =============================================================================
# Unified Analytics Engine
# =============================================================================

class AnalyticsEngine:
    """
    Unified engine for extracting DNA features from video and audio.

    Combines video and audio extractors with memory-safe orchestration.
    Designed for edge computing environments with limited resources.

    Now includes Text Intelligence for semantic understanding:
    - Whisper transcription (spoken words)
    - EasyOCR (text overlays)
    - Semantic embeddings with PCA reduction

    Example:
        engine = AnalyticsEngine(enable_text_intelligence=True)
        features = engine.extract_complete_features("video.mp4", caption="Amazing!")
        print(json.dumps(features, indent=2))
    """

    def __init__(
        self,
        video_config: Optional[VideoConfig] = None,
        audio_config: Optional[AudioConfig] = None,
        enable_text_intelligence: bool = False,
        text_config: Optional[Any] = None
    ):
        self.video_extractor = EfficientFeatureExtractor(video_config)
        self.audio_extractor = EfficientAudioExtractor(audio_config)
        self._text_engine = None
        self._text_intelligence_enabled = enable_text_intelligence

        # Initialize text intelligence if requested
        if enable_text_intelligence and _check_text_intelligence():
            from app.services.text_intelligence import TextIntelligenceEngine, TextConfig
            config = text_config if text_config else TextConfig()
            self._text_engine = TextIntelligenceEngine(config)

        logger.info(
            f"AnalyticsEngine initialized - "
            f"Video: {self.video_extractor.config.target_resolution}, "
            f"Stride: {self.video_extractor.config.frame_stride_seconds}s | "
            f"Audio: {self.audio_extractor.config.sample_rate}Hz, "
            f"Duration: {self.audio_extractor.config.duration_seconds}s | "
            f"TextIntelligence: {'enabled' if self._text_engine else 'disabled'}"
        )

    def extract_video_features(self, video_path: str) -> Dict[str, float]:
        """Extract only video features."""
        return self.video_extractor.extract_video_features(video_path)

    def extract_audio_features(self, audio_path: str) -> Dict[str, float]:
        """Extract only audio features."""
        return self.audio_extractor.extract_audio_features(audio_path)

    def extract_text_features(
        self,
        video_path: str,
        caption: str = ""
    ) -> Dict[str, Any]:
        """
        Extract text intelligence features (transcription, OCR, semantic PCA).

        Args:
            video_path: Path to video file
            caption: User-provided caption

        Returns:
            Dict with transcription, OCR, and sem_pca_1 to sem_pca_10
        """
        if self._text_engine is None:
            return {
                "text_status": "text_intelligence_not_enabled",
                "transcription": "",
                "ocr_text": "",
                "text_density": 0.0
            }

        return self._text_engine.extract_all_features(
            video_path=video_path,
            caption=caption
        )

    def extract_all_features(
        self,
        media_path: str,
        include_video: bool = True,
        include_audio: bool = True
    ) -> Dict[str, Any]:
        """
        Extract all DNA features from a media file (video + audio only).

        Args:
            media_path: Path to video/audio file
            include_video: Whether to extract video features
            include_audio: Whether to extract audio features

        Returns:
            Flat JSON-serializable dict with all features
        """
        result: Dict[str, Any] = {
            "source": str(Path(media_path).name),
            "status": "success"
        }

        try:
            # Video features
            if include_video:
                video_features = self.video_extractor.extract_video_features(media_path)
                result.update({
                    "visual_energy": video_features.get("visual_energy", 0.0),
                    "brightness_variance": video_features.get("brightness_variance", 0.0),
                    "cut_density": video_features.get("cut_density", 0.0),
                    "duration_seconds": video_features.get("duration_seconds", 0.0),
                    "frames_analyzed": video_features.get("frames_analyzed", 0)
                })

            # Audio features
            if include_audio:
                audio_features = self.audio_extractor.extract_audio_features(media_path)
                result.update({
                    "tempo": audio_features.get("tempo", 0.0),
                    "onset_strength": audio_features.get("onset_strength", 0.0),
                    "audio_status": audio_features.get("audio_status", "unknown")
                })

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            result["status"] = "error"
            result["error"] = str(e)

        finally:
            gc.collect()

        return result

    def extract_complete_features(
        self,
        media_path: str,
        caption: str = "",
        include_video: bool = True,
        include_audio: bool = True,
        include_text: bool = True
    ) -> Dict[str, Any]:
        """
        Extract ALL features: Video DNA + Audio DNA + Text Intelligence.

        This is the comprehensive extraction method that combines:
        - Visual features (energy, brightness, cuts)
        - Audio features (tempo, onset strength)
        - Text features (transcription, OCR, semantic PCA)

        Args:
            media_path: Path to video/audio file
            caption: User-provided caption for semantic analysis
            include_video: Whether to extract video features
            include_audio: Whether to extract audio features
            include_text: Whether to extract text intelligence features

        Returns:
            Flat JSON-serializable dict with all features including sem_pca_1 to sem_pca_10

        Example output:
            {
                "source": "video.mp4",
                "status": "success",
                # Video DNA
                "visual_energy": 0.045,
                "brightness_variance": 0.12,
                "cut_density": 8.5,
                "duration_seconds": 15.2,
                # Audio DNA
                "tempo": 128.0,
                "onset_strength": 0.85,
                # Text Intelligence
                "transcription": "Hey guys check this out...",
                "transcription_word_count": 25,
                "ocr_text": "FOLLOW FOR MORE",
                "text_density": 5.2,
                # Semantic PCA (for XGBoost)
                "sem_pca_1": 0.234,
                "sem_pca_2": -0.156,
                ...
                "sem_pca_10": 0.089
            }
        """
        result: Dict[str, Any] = {
            "source": str(Path(media_path).name),
            "status": "success",
            "caption_provided": bool(caption)
        }

        try:
            # Video DNA features
            if include_video:
                video_features = self.video_extractor.extract_video_features(media_path)
                result.update({
                    "visual_energy": video_features.get("visual_energy", 0.0),
                    "brightness_variance": video_features.get("brightness_variance", 0.0),
                    "cut_density": video_features.get("cut_density", 0.0),
                    "duration_seconds": video_features.get("duration_seconds", 0.0),
                    "frames_analyzed": video_features.get("frames_analyzed", 0)
                })

            # Audio DNA features
            if include_audio:
                audio_features = self.audio_extractor.extract_audio_features(media_path)
                result.update({
                    "tempo": audio_features.get("tempo", 0.0),
                    "onset_strength": audio_features.get("onset_strength", 0.0),
                    "audio_status": audio_features.get("audio_status", "unknown")
                })

            # Text Intelligence features
            if include_text and self._text_engine is not None:
                text_features = self._text_engine.extract_all_features(
                    video_path=media_path,
                    caption=caption
                )

                # Add transcription features
                result.update({
                    "transcription": text_features.get("transcription", ""),
                    "transcription_language": text_features.get("transcription_language", ""),
                    "transcription_confidence": text_features.get("transcription_confidence", 0.0),
                    "transcription_word_count": text_features.get("transcription_word_count", 0),
                    "transcription_status": text_features.get("transcription_status", "unknown")
                })

                # Add OCR features
                result.update({
                    "ocr_text": text_features.get("ocr_text", ""),
                    "text_density": text_features.get("text_density", 0.0),
                    "max_text_density": text_features.get("max_text_density", 0.0),
                    "ocr_text_boxes": text_features.get("ocr_text_boxes", 0),
                    "ocr_status": text_features.get("ocr_status", "unknown")
                })

                # Add semantic PCA components (sem_pca_1 to sem_pca_10)
                for i in range(1, 11):
                    key = f"sem_pca_{i}"
                    result[key] = text_features.get(key, 0.0)

                result["semantic_text_sources"] = text_features.get("semantic_text_sources", 0)
                result["semantic_status"] = text_features.get("semantic_status", "unknown")
                result["text_status"] = text_features.get("text_status", "success")

            elif include_text:
                # Text intelligence requested but not available
                result["text_status"] = "text_intelligence_not_enabled"

        except Exception as e:
            logger.error(f"Complete feature extraction failed: {e}")
            result["status"] = "error"
            result["error"] = str(e)

        finally:
            gc.collect()

        return result

    def fit_semantic_pca(
        self,
        corpus: List[Dict[str, str]],
        save_path: Optional[str] = None
    ) -> "AnalyticsEngine":
        """
        Fit the PCA model on a corpus of texts.

        Must be called before extracting semantic features if no pre-fitted
        PCA model is provided.

        Args:
            corpus: List of dicts with 'caption', 'transcription', 'ocr_text' keys
            save_path: Optional path to save the fitted PCA model

        Returns:
            self for chaining
        """
        if self._text_engine is None:
            logger.warning("Text intelligence not enabled. Cannot fit PCA.")
            return self

        self._text_engine.fit_pca_from_corpus(corpus, save_path)
        return self

    def extract_batch(
        self,
        media_paths: List[str],
        include_video: bool = True,
        include_audio: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process multiple media files with optimized memory management.

        Forces garbage collection between each file to prevent memory buildup.
        """
        results = []

        for path in media_paths:
            features = self.extract_all_features(
                path,
                include_video=include_video,
                include_audio=include_audio
            )
            results.append(features)
            gc.collect()  # Aggressive cleanup between files

        return results

    def to_json(self, features: Dict[str, Any], pretty: bool = False) -> str:
        """Convert features dict to JSON string."""
        return json.dumps(features, indent=2 if pretty else None)


# =============================================================================
# Memory Profiling Utilities (Development Only)
# =============================================================================

def estimate_memory_usage(video_path: str) -> Dict[str, Any]:
    """
    Estimate memory usage for processing a video.

    Useful for capacity planning on edge devices.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {"error": "Cannot open video"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    config = VideoConfig()
    stride = int(fps * config.frame_stride_seconds)
    frames_to_process = total_frames // stride

    # Memory calculations
    original_frame_bytes = width * height * 3  # BGR
    downsampled_frame_bytes = config.target_resolution[0] * config.target_resolution[1]

    # With our approach, only 2 frames in memory at most (current + previous)
    peak_memory_bytes = downsampled_frame_bytes * 2 * 4  # float32 = 4 bytes

    # Without optimization: all frames in memory
    naive_memory_bytes = original_frame_bytes * total_frames

    memory_reduction = 1 - (peak_memory_bytes / naive_memory_bytes) if naive_memory_bytes > 0 else 0

    return {
        "original_resolution": f"{width}x{height}",
        "target_resolution": f"{config.target_resolution[0]}x{config.target_resolution[1]}",
        "total_frames": total_frames,
        "frames_to_process": frames_to_process,
        "fps": round(fps, 2),
        "stride_frames": stride,
        "peak_memory_mb": round(peak_memory_bytes / (1024 * 1024), 3),
        "naive_memory_mb": round(naive_memory_bytes / (1024 * 1024), 1),
        "memory_reduction_percent": round(memory_reduction * 100, 1)
    }


# =============================================================================
# CLI Entry Point (for testing)
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python analytics_engine.py <video_path> [--text] [caption]")
        print("\nExamples:")
        print("  python analytics_engine.py video.mp4")
        print("  python analytics_engine.py video.mp4 --text")
        print("  python analytics_engine.py video.mp4 --text 'Check this viral content!'")
        sys.exit(1)

    video_path = sys.argv[1]
    enable_text = "--text" in sys.argv
    caption = ""

    # Parse caption (last argument if not a flag)
    if len(sys.argv) > 2 and not sys.argv[-1].startswith("--"):
        caption = sys.argv[-1]

    print(f"\n{'='*60}")
    print("Analytics Engine - Complete Feature Extraction")
    print(f"{'='*60}\n")

    # Memory estimation
    print("[1] Memory Estimation:")
    mem_info = estimate_memory_usage(video_path)
    for key, value in mem_info.items():
        print(f"    {key}: {value}")

    # Feature extraction
    print(f"\n[2] Initializing Engine (Text Intelligence: {enable_text})...")
    engine = AnalyticsEngine(enable_text_intelligence=enable_text)

    if enable_text:
        print(f"\n[3] Extracting Complete Features (Video + Audio + Text)...")
        if caption:
            print(f"    Caption: {caption[:50]}...")
        features = engine.extract_complete_features(video_path, caption=caption)
    else:
        print(f"\n[3] Extracting DNA Features (Video + Audio)...")
        features = engine.extract_all_features(video_path)

    print("\n[4] Extracted Features (JSON Output):")
    print(engine.to_json(features, pretty=True))

    # Show PCA summary if available
    pca_features = {k: v for k, v in features.items() if k.startswith("sem_pca_")}
    if pca_features:
        print("\n[5] Semantic PCA Components:")
        for name, value in pca_features.items():
            bar = "=" * int(abs(value) * 50) if isinstance(value, (int, float)) else ""
            sign = "+" if isinstance(value, (int, float)) and value >= 0 else "-"
            print(f"    {name}: {value:+.4f} [{sign}{bar}]")

    print(f"\n{'='*60}")
    print("Processing Complete")
    print(f"{'='*60}\n")
