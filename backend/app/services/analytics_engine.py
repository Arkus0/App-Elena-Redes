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

HOOK THEORY (Algorithm-Aligned Feature Engineering):
=====================================================
The TikTok/IG algorithm evaluates videos as TEMPORAL SEQUENCES, not flat data.
The first 3 seconds (hook) determine 90% of video success.

Temporal Features:
- hook_energy: Visual energy in seconds 0-3 (critical for retention)
- retention_energy: Visual energy for the rest of the video
- hook_cut_rate: Cut rate in first 3s (weighted 10x more than later cuts)
- retention_cut_rate: Cut rate for the rest of the video
- face_in_hook: Binary - is there a face in the first 3 seconds?

Justification:
- If hook_energy is low, the video dies regardless of how good the rest is
- A cut in second 1 is worth 10x a cut in second 50
- Face-to-camera in first frame is prioritized by the algorithm

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

    # Hook Theory configuration
    hook_duration_seconds: float = 3.0  # First 3 seconds are the "hook"
    hook_cut_weight: float = 10.0  # Cuts in hook weighted 10x more than retention cuts

    # Face detection configuration
    face_detection_enabled: bool = True
    face_scale_factor: float = 1.1  # OpenCV cascade scale factor
    face_min_neighbors: int = 4  # Minimum neighbors for detection
    face_min_size: Tuple[int, int] = (30, 30)  # Minimum face size

    # ==========================================================================
    # QUALITY GATE CONFIGURATION - Camera Instability Detection
    # ==========================================================================
    # Distinguishes "dynamic editing" (intentional cuts/motion) from
    # "bad filming" (unintended camera shake, poor stabilization)

    # Optical flow parameters (Farneback algorithm)
    optical_flow_pyr_scale: float = 0.5  # Pyramid scale for flow computation
    optical_flow_levels: int = 3  # Number of pyramid levels
    optical_flow_winsize: int = 15  # Averaging window size
    optical_flow_iterations: int = 3  # Iterations at each pyramid level
    optical_flow_poly_n: int = 5  # Polynomial expansion neighborhood
    optical_flow_poly_sigma: float = 1.1  # Std for polynomial expansion

    # Instability thresholds
    instability_threshold_low: float = 0.02  # Below this = stable footage
    instability_threshold_high: float = 0.08  # Above this = severe shake
    instability_penalty_max: float = 0.6  # Maximum penalty (60% reduction)

    # Production quality scoring
    min_brightness_threshold: float = 0.15  # Below this = too dark
    max_brightness_threshold: float = 0.85  # Above this = overexposed
    min_contrast_threshold: float = 0.05  # Below this = washed out


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

    def generate_frames(self) -> Generator[Tuple[np.ndarray, float], None, None]:
        """
        Generator that yields downsampled frames at the configured stride.

        Memory Impact: Only ONE frame in memory at any time.

        Yields:
            Tuple[np.ndarray, float]: (Downsampled grayscale frame, timestamp in seconds)
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
                # Calculate timestamp in seconds
                timestamp = frame_idx / self._fps if self._fps > 0 else 0.0

                # Downsample to target resolution
                resized = cv2.resize(
                    frame,
                    self.config.target_resolution,
                    interpolation=cv2.INTER_AREA  # Best for downsampling
                )

                # Convert to grayscale for luminance analysis
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

                # Convert to float32 for numerical precision with lower memory
                yield gray.astype(self.config.dtype) / 255.0, timestamp

            frame_idx += 1

        logger.debug(f"Processed {frame_idx // stride} frames (stride: {stride})")

    def generate_frames_with_color(self) -> Generator[Tuple[np.ndarray, np.ndarray, float], None, None]:
        """
        Generator that yields both grayscale and color frames for face detection.

        Used specifically for hook analysis where face detection is needed.

        Yields:
            Tuple[np.ndarray, np.ndarray, float]: (grayscale frame, color frame, timestamp)
        """
        if self._cap is None:
            raise RuntimeError("Generator must be used within context manager")

        stride = self.get_stride_frames()
        frame_idx = 0

        while True:
            ret, frame = self._cap.read()
            if not ret:
                break

            if frame_idx % stride == 0:
                timestamp = frame_idx / self._fps if self._fps > 0 else 0.0

                # Downsample to target resolution
                resized = cv2.resize(
                    frame,
                    self.config.target_resolution,
                    interpolation=cv2.INTER_AREA
                )

                # Grayscale for motion analysis
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                gray_float = gray.astype(self.config.dtype) / 255.0

                yield gray_float, gray, timestamp

            frame_idx += 1


class FaceDetector:
    """
    Lightweight face detector using OpenCV Haar Cascades.

    Used only for hook analysis (first 3 seconds) to detect
    face-to-camera content which is prioritized by the algorithm.
    """

    def __init__(self, config: VideoConfig):
        self.config = config
        self._cascade = None
        self._available = False
        self._load_cascade()

    def _load_cascade(self) -> None:
        """Load the face detection cascade classifier."""
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._cascade = cv2.CascadeClassifier(cascade_path)
            if self._cascade.empty():
                logger.warning("Face cascade classifier failed to load")
                self._available = False
            else:
                self._available = True
                logger.debug("Face cascade classifier loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load face detector: {e}")
            self._available = False

    def detect_face(self, gray_frame: np.ndarray) -> bool:
        """
        Detect if a face is present in the frame.

        Args:
            gray_frame: Grayscale frame (uint8, 0-255)

        Returns:
            True if at least one face is detected
        """
        if not self._available or self._cascade is None:
            return False

        try:
            faces = self._cascade.detectMultiScale(
                gray_frame,
                scaleFactor=self.config.face_scale_factor,
                minNeighbors=self.config.face_min_neighbors,
                minSize=self.config.face_min_size
            )
            return len(faces) > 0
        except Exception as e:
            logger.debug(f"Face detection error: {e}")
            return False


class EfficientFeatureExtractor:
    """
    Memory-optimized video feature extractor implementing HOOK THEORY.

    The TikTok/IG algorithm evaluates videos as temporal sequences.
    The first 3 seconds (hook) determine 90% of video success.

    Extracts Temporal DNA Features:
    - hook_energy: Visual energy in seconds 0-3 (CRITICAL)
    - retention_energy: Visual energy for rest of video
    - hook_cut_rate: Weighted cut rate in hook (10x multiplier)
    - retention_cut_rate: Cut rate for rest of video
    - face_in_hook: Binary - face detected in first 3 seconds
    - visual_energy: Global energy (for backward compatibility)
    - brightness_variance: Luminance variation
    - cut_density: Global edit rhythm (for backward compatibility)

    Memory Strategy:
    - Uses generators to process frames one-by-one
    - Accumulates only scalar metrics, not raw frames
    - Forces gc.collect() after processing
    """

    def __init__(self, config: Optional[VideoConfig] = None):
        self.config = config or VideoConfig()
        self._face_detector = FaceDetector(self.config)
        self._reset_accumulators()

    def _reset_accumulators(self) -> None:
        """Reset all metric accumulators for a new video."""
        # Global accumulators
        self._frame_diffs: List[float] = []
        self._brightness_values: List[float] = []
        self._histograms: List[np.ndarray] = []
        self._prev_frame: Optional[np.ndarray] = None
        self._prev_hist: Optional[np.ndarray] = None

        # Hook-specific accumulators (0-3 seconds)
        self._hook_frame_diffs: List[float] = []
        self._hook_cuts: int = 0
        self._hook_face_detected: bool = False

        # Retention-specific accumulators (after 3 seconds)
        self._retention_frame_diffs: List[float] = []
        self._retention_cuts: int = 0

        # Quality Gate accumulators - Camera Instability Detection
        self._global_motion_magnitudes: List[float] = []  # Global camera translation
        self._local_motion_variances: List[float] = []  # Local motion variance (editing)
        self._contrast_values: List[float] = []  # Frame contrast values

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
        cv2.normalize(hist, hist)
        return hist.flatten().astype(self.config.dtype)

    def _detect_cut(self, current_hist: np.ndarray) -> bool:
        """
        Detect scene cuts by comparing histogram correlation.

        A cut is detected when correlation drops below threshold.
        """
        if self._prev_hist is None:
            return False

        correlation = cv2.compareHist(
            current_hist.reshape(-1, 1),
            self._prev_hist.reshape(-1, 1),
            cv2.HISTCMP_CORREL
        )

        return correlation < self.config.cut_threshold

    def _analyze_optical_flow(self, current_frame: np.ndarray) -> Tuple[float, float]:
        """
        Analyze optical flow to distinguish camera shake from intentional motion.

        Uses Farneback dense optical flow to compute:
        1. Global motion magnitude: Mean flow vector (camera translation/shake)
        2. Local motion variance: Variance of flow vectors (editing/subject motion)

        KEY INSIGHT:
        - Camera shake = HIGH global motion + LOW local variance (whole frame moves together)
        - Dynamic editing = Variable global + HIGH local variance (subjects move differently)

        Args:
            current_frame: Current grayscale frame (float32, 0-1)

        Returns:
            Tuple of (global_motion_magnitude, local_motion_variance)
        """
        if self._prev_frame is None:
            return 0.0, 0.0

        try:
            # Convert to uint8 for optical flow computation
            prev_uint8 = (self._prev_frame * 255).astype(np.uint8)
            curr_uint8 = (current_frame * 255).astype(np.uint8)

            # Compute dense optical flow using Farneback algorithm
            flow = cv2.calcOpticalFlowFarneback(
                prev_uint8,
                curr_uint8,
                None,
                pyr_scale=self.config.optical_flow_pyr_scale,
                levels=self.config.optical_flow_levels,
                winsize=self.config.optical_flow_winsize,
                iterations=self.config.optical_flow_iterations,
                poly_n=self.config.optical_flow_poly_n,
                poly_sigma=self.config.optical_flow_poly_sigma,
                flags=0
            )

            # Separate flow into x and y components
            flow_x = flow[..., 0]
            flow_y = flow[..., 1]

            # Calculate magnitude at each pixel
            magnitude = np.sqrt(flow_x**2 + flow_y**2)

            # GLOBAL MOTION: Mean flow vector magnitude
            # High value = whole frame is shifting (camera shake or pan)
            mean_flow_x = float(np.mean(flow_x))
            mean_flow_y = float(np.mean(flow_y))
            global_motion = float(np.sqrt(mean_flow_x**2 + mean_flow_y**2))

            # LOCAL MOTION VARIANCE: Standard deviation of magnitudes
            # High value = different parts moving differently (editing, subject motion)
            # Low value = uniform motion (camera shake)
            local_variance = float(np.std(magnitude))

            # Normalize by frame diagonal to make resolution-independent
            frame_diagonal = np.sqrt(current_frame.shape[0]**2 + current_frame.shape[1]**2)
            global_motion_normalized = global_motion / frame_diagonal
            local_variance_normalized = local_variance / frame_diagonal

            return global_motion_normalized, local_variance_normalized

        except Exception as e:
            logger.debug(f"Optical flow computation failed: {e}")
            return 0.0, 0.0

    def _calculate_instability_score(self) -> Tuple[float, float]:
        """
        Calculate camera instability score from accumulated motion data.

        Instability is HIGH when:
        - Global motion is high (camera moving)
        - Local variance is low (uniform movement = shake, not editing)

        Returns:
            Tuple of (instability_score 0-1, shake_ratio)
        """
        if not self._global_motion_magnitudes or not self._local_motion_variances:
            return 0.0, 0.0

        mean_global_motion = float(np.mean(self._global_motion_magnitudes))
        mean_local_variance = float(np.mean(self._local_motion_variances))

        # Shake ratio: high global motion with low local variance = camera shake
        # When local variance is high relative to global motion, it's editing
        if mean_global_motion > 0:
            # Ratio of "uniform motion" vs "differential motion"
            shake_ratio = mean_global_motion / (mean_local_variance + 0.001)
        else:
            shake_ratio = 0.0

        # Normalize instability score to 0-1
        # Higher when: high global motion AND low local variance
        raw_instability = mean_global_motion * (1.0 / (mean_local_variance + 0.01))

        # Apply sigmoid-like normalization to clamp between 0 and 1
        instability_score = min(1.0, raw_instability / 5.0)

        return instability_score, shake_ratio

    def _calculate_production_quality(
        self,
        instability_score: float,
        brightness_values: List[float],
        contrast_values: List[float]
    ) -> Dict[str, float]:
        """
        Calculate production quality score (0-1) based on filming quality metrics.

        Quality is INVERSELY proportional to:
        - Unintended camera shake (instability)
        - Poor brightness (too dark or overexposed)
        - Low contrast (washed out footage)

        Args:
            instability_score: Camera shake score (0-1)
            brightness_values: List of mean brightness values per frame
            contrast_values: List of contrast (std dev) values per frame

        Returns:
            Dict with quality metrics including production_quality_score
        """
        # === INSTABILITY PENALTY ===
        # Smoothly penalize instability, max penalty at threshold_high
        if instability_score <= self.config.instability_threshold_low:
            instability_penalty = 0.0
        elif instability_score >= self.config.instability_threshold_high:
            instability_penalty = self.config.instability_penalty_max
        else:
            # Linear interpolation between thresholds
            range_size = self.config.instability_threshold_high - self.config.instability_threshold_low
            normalized = (instability_score - self.config.instability_threshold_low) / range_size
            instability_penalty = normalized * self.config.instability_penalty_max

        # === BRIGHTNESS QUALITY ===
        if brightness_values:
            mean_brightness = float(np.mean(brightness_values))
            brightness_std = float(np.std(brightness_values))

            # Penalize if too dark or too bright
            if mean_brightness < self.config.min_brightness_threshold:
                # Too dark: penalize proportionally
                brightness_quality = mean_brightness / self.config.min_brightness_threshold
            elif mean_brightness > self.config.max_brightness_threshold:
                # Overexposed: penalize proportionally
                brightness_quality = (1.0 - mean_brightness) / (1.0 - self.config.max_brightness_threshold)
            else:
                # Good range: no penalty
                brightness_quality = 1.0

            # Penalize excessive brightness variance (flickering)
            if brightness_std > 0.15:
                brightness_quality *= 0.9
        else:
            mean_brightness = 0.0
            brightness_std = 0.0
            brightness_quality = 0.0

        # === CONTRAST QUALITY ===
        if contrast_values:
            mean_contrast = float(np.mean(contrast_values))

            # Penalize low contrast (washed out footage)
            if mean_contrast < self.config.min_contrast_threshold:
                contrast_quality = mean_contrast / self.config.min_contrast_threshold
            else:
                contrast_quality = 1.0
        else:
            mean_contrast = 0.0
            contrast_quality = 0.0

        # === COMPOSITE PRODUCTION QUALITY SCORE ===
        # Weighted combination: instability (50%), brightness (25%), contrast (25%)
        stability_score = 1.0 - instability_penalty

        production_quality_score = (
            0.50 * stability_score +
            0.25 * brightness_quality +
            0.25 * contrast_quality
        )

        # Clamp to 0-1
        production_quality_score = max(0.0, min(1.0, production_quality_score))

        return {
            "production_quality_score": round(production_quality_score, 4),
            "camera_stability_score": round(stability_score, 4),
            "brightness_quality": round(brightness_quality, 4),
            "contrast_quality": round(contrast_quality, 4),
            "mean_brightness": round(mean_brightness, 4),
            "mean_contrast": round(mean_contrast, 4),
            "instability_score": round(instability_score, 4),
            "instability_penalty": round(instability_penalty, 4),
        }

    def _apply_instability_penalty(
        self,
        raw_energy: float,
        instability_score: float
    ) -> float:
        """
        Apply instability penalty to visual energy score.

        Reduces visual energy when camera shake is detected, preventing
        shaky footage from being scored as "high energy content".

        Args:
            raw_energy: Original visual energy score
            instability_score: Camera instability score (0-1)

        Returns:
            Adjusted visual energy with penalty applied
        """
        if instability_score <= self.config.instability_threshold_low:
            # No penalty for stable footage
            return raw_energy

        if instability_score >= self.config.instability_threshold_high:
            # Maximum penalty for very shaky footage
            penalty = self.config.instability_penalty_max
        else:
            # Linear interpolation
            range_size = self.config.instability_threshold_high - self.config.instability_threshold_low
            normalized = (instability_score - self.config.instability_threshold_low) / range_size
            penalty = normalized * self.config.instability_penalty_max

        adjusted_energy = raw_energy * (1.0 - penalty)
        logger.debug(
            f"Visual energy adjusted: {raw_energy:.4f} -> {adjusted_energy:.4f} "
            f"(instability: {instability_score:.4f}, penalty: {penalty:.2%})"
        )

        return adjusted_energy

    def extract_video_features(self, video_path: str) -> Dict[str, float]:
        """
        Extract all video features with TEMPORAL HOOK THEORY segmentation
        and QUALITY GATE for camera instability detection.

        Separates features into:
        - Hook zone (0-3 seconds): Critical for algorithm retention
        - Retention zone (3+ seconds): Secondary importance

        Quality Gate:
        - Detects camera shake vs intentional editing using optical flow
        - Applies penalty to visual_energy when shake is detected
        - Calculates production_quality_score based on stability, brightness, contrast

        Args:
            video_path: Path to video file

        Returns:
            Dict with temporal features:
            - hook_energy, retention_energy
            - hook_cut_rate, retention_cut_rate (weighted)
            - face_in_hook
            - visual_energy (with instability penalty applied)
            - production_quality_score (0-1, quality gate metric)
            - Plus legacy global features for backward compatibility
        """
        self._reset_accumulators()
        total_cuts = 0
        duration = 0.0
        hook_duration = self.config.hook_duration_seconds
        frames_analyzed = 0

        try:
            with EfficientFrameGenerator(video_path, self.config) as gen:
                duration = gen.duration

                for frame, timestamp in gen.generate_frames():
                    frames_analyzed += 1
                    is_hook = timestamp <= hook_duration

                    # Accumulate brightness (mean luminance)
                    mean_brightness = float(np.mean(frame))
                    self._brightness_values.append(mean_brightness)

                    # Accumulate contrast (standard deviation = local contrast)
                    frame_contrast = float(np.std(frame))
                    self._contrast_values.append(frame_contrast)

                    # Calculate motion energy (raw frame diff)
                    frame_diff = self._calculate_frame_diff(frame)
                    if frame_diff > 0:
                        self._frame_diffs.append(frame_diff)

                        # Separate into hook vs retention
                        if is_hook:
                            self._hook_frame_diffs.append(frame_diff)
                        else:
                            self._retention_frame_diffs.append(frame_diff)

                    # === QUALITY GATE: Optical Flow Analysis ===
                    # Distinguish camera shake from intentional motion
                    global_motion, local_variance = self._analyze_optical_flow(frame)
                    if global_motion > 0 or local_variance > 0:
                        self._global_motion_magnitudes.append(global_motion)
                        self._local_motion_variances.append(local_variance)

                    # Histogram-based cut detection
                    hist = self._calculate_histogram(frame)
                    if self._detect_cut(hist):
                        total_cuts += 1
                        if is_hook:
                            self._hook_cuts += 1
                        else:
                            self._retention_cuts += 1

                    # Face detection ONLY in hook (first 3 seconds)
                    if is_hook and not self._hook_face_detected:
                        if self.config.face_detection_enabled:
                            # Convert to uint8 for face detection
                            gray_uint8 = (frame * 255).astype(np.uint8)
                            if self._face_detector.detect_face(gray_uint8):
                                self._hook_face_detected = True
                                logger.debug(f"Face detected at {timestamp:.2f}s")

                    # Update previous frame/histogram
                    self._prev_frame = frame
                    self._prev_hist = hist

            # =================================================================
            # Calculate Temporal Features
            # =================================================================

            # Hook energy (0-3s) - CRITICAL for retention
            hook_energy = (
                float(np.mean(self._hook_frame_diffs))
                if self._hook_frame_diffs else 0.0
            )

            # Retention energy (3s+)
            retention_energy = (
                float(np.mean(self._retention_frame_diffs))
                if self._retention_frame_diffs else 0.0
            )

            # Global visual energy (raw, before instability penalty)
            raw_visual_energy = (
                float(np.mean(self._frame_diffs))
                if self._frame_diffs else 0.0
            )

            # Brightness variance (global)
            brightness_variance = (
                float(np.std(self._brightness_values))
                if self._brightness_values else 0.0
            )

            # =================================================================
            # QUALITY GATE: Camera Instability Detection & Penalty
            # =================================================================
            # Distinguish "dynamic editing" (good) from "bad filming" (shake)

            instability_score, shake_ratio = self._calculate_instability_score()

            # Apply instability penalty to visual energy
            # Shaky footage should NOT be scored as "high energy content"
            visual_energy = self._apply_instability_penalty(
                raw_visual_energy,
                instability_score
            )

            # Apply same penalty to hook/retention energy for consistency
            adjusted_hook_energy = self._apply_instability_penalty(
                hook_energy,
                instability_score
            )
            adjusted_retention_energy = self._apply_instability_penalty(
                retention_energy,
                instability_score
            )

            # Calculate production quality score
            quality_metrics = self._calculate_production_quality(
                instability_score=instability_score,
                brightness_values=self._brightness_values,
                contrast_values=self._contrast_values
            )

            # =================================================================
            # Cut Rate Calculation with Hook Weighting
            # =================================================================
            # A cut in second 1 is worth 10x a cut in second 50

            # Hook cut rate (cuts per minute in hook, weighted by 10x)
            hook_minutes = min(hook_duration, duration) / 60.0
            hook_cut_rate = (
                float(self._hook_cuts / hook_minutes) * self.config.hook_cut_weight
                if hook_minutes > 0 else 0.0
            )

            # Retention cut rate (cuts per minute after hook)
            retention_duration = max(0, duration - hook_duration)
            retention_minutes = retention_duration / 60.0
            retention_cut_rate = (
                float(self._retention_cuts / retention_minutes)
                if retention_minutes > 0 else 0.0
            )

            # Global cut density (backward compatibility)
            duration_minutes = duration / 60.0
            cut_density = float(total_cuts / duration_minutes) if duration_minutes > 0 else 0.0

            # Weighted total cut score (hook cuts worth 10x)
            weighted_cut_score = (
                (self._hook_cuts * self.config.hook_cut_weight) + self._retention_cuts
            )

            return {
                # === TEMPORAL FEATURES (Hook Theory) ===
                # Note: hook_energy and retention_energy now have instability penalty applied
                "hook_energy": round(adjusted_hook_energy, 6),
                "retention_energy": round(adjusted_retention_energy, 6),
                "hook_cut_rate": round(hook_cut_rate, 2),
                "retention_cut_rate": round(retention_cut_rate, 2),
                "face_in_hook": 1 if self._hook_face_detected else 0,
                "weighted_cut_score": round(weighted_cut_score, 2),

                # === GLOBAL FEATURES (with Quality Gate) ===
                # visual_energy now has instability penalty applied
                "visual_energy": round(visual_energy, 6),
                "visual_energy_raw": round(raw_visual_energy, 6),  # Original for debugging
                "brightness_variance": round(brightness_variance, 6),
                "cut_density": round(cut_density, 2),

                # === PRODUCTION QUALITY METRICS (Quality Gate) ===
                "production_quality_score": quality_metrics["production_quality_score"],
                "camera_stability_score": quality_metrics["camera_stability_score"],
                "instability_score": quality_metrics["instability_score"],
                "instability_penalty": quality_metrics["instability_penalty"],
                "brightness_quality": quality_metrics["brightness_quality"],
                "contrast_quality": quality_metrics["contrast_quality"],
                "mean_brightness": quality_metrics["mean_brightness"],
                "mean_contrast": quality_metrics["mean_contrast"],

                # === METADATA ===
                "duration_seconds": round(duration, 2),
                "frames_analyzed": frames_analyzed,
                "hook_duration": hook_duration,
                "hook_frames": len(self._hook_frame_diffs),
                "retention_frames": len(self._retention_frame_diffs),
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
            # Video features (now with Hook Theory temporal features + Quality Gate)
            if include_video:
                video_features = self.video_extractor.extract_video_features(media_path)
                result.update({
                    # Temporal Hook Theory features
                    "hook_energy": video_features.get("hook_energy", 0.0),
                    "retention_energy": video_features.get("retention_energy", 0.0),
                    "hook_cut_rate": video_features.get("hook_cut_rate", 0.0),
                    "retention_cut_rate": video_features.get("retention_cut_rate", 0.0),
                    "face_in_hook": video_features.get("face_in_hook", 0),
                    "weighted_cut_score": video_features.get("weighted_cut_score", 0.0),

                    # Global features (with Quality Gate instability penalty)
                    "visual_energy": video_features.get("visual_energy", 0.0),
                    "brightness_variance": video_features.get("brightness_variance", 0.0),
                    "cut_density": video_features.get("cut_density", 0.0),
                    "duration_seconds": video_features.get("duration_seconds", 0.0),
                    "frames_analyzed": video_features.get("frames_analyzed", 0),

                    # Production Quality metrics (Quality Gate)
                    "production_quality_score": video_features.get("production_quality_score", 0.0),
                    "camera_stability_score": video_features.get("camera_stability_score", 0.0),
                    "instability_score": video_features.get("instability_score", 0.0),
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
        - Visual features (energy, brightness, cuts) with Quality Gate
        - Audio features (tempo, onset strength)
        - Text features (transcription, OCR, semantic PCA)
        - Production quality scoring (camera stability, brightness, contrast)

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
                # === TEMPORAL FEATURES (Hook Theory) ===
                "hook_energy": 0.065,      # Energy in first 3s (with instability penalty)
                "retention_energy": 0.042,  # Energy after 3 seconds
                "hook_cut_rate": 120.0,    # Cuts/min in hook (weighted 10x)
                "retention_cut_rate": 8.0,  # Cuts/min after hook
                "face_in_hook": 1,         # Face detected in first 3s
                "weighted_cut_score": 12.0, # Total weighted cuts
                # === GLOBAL FEATURES (with Quality Gate) ===
                "visual_energy": 0.045,     # Has instability penalty applied
                "brightness_variance": 0.12,
                "cut_density": 8.5,
                "duration_seconds": 15.2,
                # === PRODUCTION QUALITY (Quality Gate) ===
                "production_quality_score": 0.85,  # 0-1, overall quality
                "camera_stability_score": 0.92,    # 0-1, inverse of shake
                "instability_score": 0.03,         # Camera shake detected
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
            # Video DNA features (with Hook Theory + Quality Gate)
            if include_video:
                video_features = self.video_extractor.extract_video_features(media_path)
                result.update({
                    # === TEMPORAL FEATURES (Hook Theory) ===
                    # Critical for algorithm - first 3 seconds determine 90% of success
                    # Note: These now have instability penalty applied
                    "hook_energy": video_features.get("hook_energy", 0.0),
                    "retention_energy": video_features.get("retention_energy", 0.0),
                    "hook_cut_rate": video_features.get("hook_cut_rate", 0.0),
                    "retention_cut_rate": video_features.get("retention_cut_rate", 0.0),
                    "face_in_hook": video_features.get("face_in_hook", 0),
                    "weighted_cut_score": video_features.get("weighted_cut_score", 0.0),

                    # === GLOBAL FEATURES (with Quality Gate) ===
                    # visual_energy has instability penalty applied
                    "visual_energy": video_features.get("visual_energy", 0.0),
                    "brightness_variance": video_features.get("brightness_variance", 0.0),
                    "cut_density": video_features.get("cut_density", 0.0),
                    "duration_seconds": video_features.get("duration_seconds", 0.0),
                    "frames_analyzed": video_features.get("frames_analyzed", 0),

                    # === PRODUCTION QUALITY METRICS (Quality Gate) ===
                    # Distinguishes "dynamic editing" from "bad filming"
                    "production_quality_score": video_features.get("production_quality_score", 0.0),
                    "camera_stability_score": video_features.get("camera_stability_score", 0.0),
                    "instability_score": video_features.get("instability_score", 0.0),
                    "brightness_quality": video_features.get("brightness_quality", 0.0),
                    "contrast_quality": video_features.get("contrast_quality", 0.0),
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
