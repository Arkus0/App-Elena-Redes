import logging
import tempfile
import os
import time
import aiohttp
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from sklearn.cluster import KMeans
from fastapi.concurrency import run_in_threadpool

# Core Analytics Engine
from app.services.analytics_engine import AnalyticsEngine

# Reuse Hook Logic from Light Processor for consistency
# Note: Assuming ml package is in python path
from ml.light_processors import LightHookAnalyzer

logger = logging.getLogger(__name__)

class ContentProcessorService:
    """
    Orchestrates the "Full Mode" (Deep Analysis) pipeline.

    Features:
    - Downloads media from URL.
    - Uses AnalyticsEngine (Video + Audio + Text Intelligence).
    - Extracts Dominant Colors (K-Means).
    - Reuses LightHookAnalyzer for consistent scoring metric.
    """

    def __init__(self):
        # Enable Text Intelligence (Whisper + EasyOCR + Embeddings)
        self.analytics = AnalyticsEngine(enable_text_intelligence=True)
        self.hook_analyzer = LightHookAnalyzer()

    async def _download_media(self, url: str) -> str:
        """Download media to a temporary file."""
        if not url:
            raise ValueError("No URL provided")

        # Determine extension (basic check)
        ext = ".mp4"
        if ".jpg" in url or ".jpeg" in url:
            ext = ".jpg"
        elif ".png" in url:
            ext = ".png"

        fd, path = tempfile.mkstemp(suffix=ext)
        os.close(fd)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        raise ValueError(f"Failed to download media: {resp.status}")
                    with open(path, 'wb') as f:
                        f.write(await resp.read())
        except Exception as e:
            if os.path.exists(path):
                os.unlink(path)
            raise e

        return path

    def _extract_dominant_colors(self, video_path: str, k: int = 5, num_frames: int = 5) -> List[str]:
        """
        Extract dominant colors using K-Means clustering on sampled frames.
        Returns list of Hex strings (e.g. ['#FF0000', ...]).
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return []

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                return []

            pixels = []

            # Sample frames (avoid reading every frame)
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Resize to speed up (100x100 is sufficient for color)
                    frame = cv2.resize(frame, (100, 100), interpolation=cv2.INTER_AREA)
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Reshape to list of pixels
                    pixels.append(frame_rgb.reshape(-1, 3))

            cap.release()

            if not pixels:
                return []

            all_pixels = np.vstack(pixels)

            # K-Means Clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
            kmeans.fit(all_pixels)

            colors = kmeans.cluster_centers_.astype(int)

            hex_colors = []
            for r, g, b in colors:
                hex_colors.append(f"#{r:02x}{g:02x}{b:02x}".upper())

            return hex_colors

        except Exception as e:
            logger.error(f"Dominant color extraction failed: {e}")
            return []

    async def process_full(self, media_url: str, caption: str = "") -> Dict[str, Any]:
        """
        Run the FULL analysis pipeline.

        Returns a dictionary compatible with the ingest pipeline, but enriched.
        """
        start_time = time.time()
        temp_path = None

        # Default empty result structure
        result = {
            "transcription": "",
            "ocr_text": "",
            "hook_score": 0.0,
            "visual_features": {},
            "light_mode": False,
            "processing_time_seconds": 0.0,
            "status": "failed"
        }

        try:
            # 1. Download Media
            logger.info(f"FULL MODE: Downloading media from {media_url[:30]}...")
            temp_path = await self._download_media(media_url)

            # 2. Analytics Engine (Async offload)
            logger.info("FULL MODE: extracting complete features...")
            features = await self.analytics.extract_complete_features_async(
                media_path=temp_path,
                caption=caption,
                include_video=True,
                include_audio=True,
                include_text=True
            )

            # 3. Dominant Colors (Async offload)
            logger.info("FULL MODE: extracting dominant colors...")
            dominant_colors = await run_in_threadpool(
                self._extract_dominant_colors,
                temp_path
            )

            # 4. Hook Analysis (Consistent metric)
            hook_result = self.hook_analyzer.analyze_hook(
                transcription=features.get("transcription", ""),
                ocr_text=features.get("ocr_text", ""),
                caption=caption
            )

            # 5. Construct Result
            result.update({
                "status": "success",
                "transcription": features.get("transcription", ""),
                "ocr_text": features.get("ocr_text", ""),
                "hook_score": hook_result.get("hook_score", 0.0),
                "hook_text": hook_result.get("hook_text", ""),

                # Rich Visual Features
                "visual_features": {
                    "dominant_colors": dominant_colors,
                    "brightness": features.get("mean_brightness", 0.0),
                    "contrast": features.get("mean_contrast", 0.0),
                    "visual_energy": features.get("visual_energy", 0.0),
                    "hook_energy": features.get("hook_energy", 0.0),
                    "face_in_hook": features.get("face_in_hook", 0),
                    "instability_score": features.get("instability_score", 0.0),
                    "production_quality": features.get("production_quality_score", 0.0),
                    "tempo": features.get("tempo", 0.0),
                    "audio_onset": features.get("onset_strength", 0.0),
                    "text_density": features.get("text_density", 0.0)
                }
            })

        except Exception as e:
            logger.error(f"FULL MODE processing failed: {e}")
            result["error"] = str(e)

        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

        result["processing_time_seconds"] = time.time() - start_time
        logger.info(f"FULL MODE finished in {result['processing_time_seconds']:.2f}s")

        return result

# Singleton Instance
_instance = None

def get_content_processor() -> ContentProcessorService:
    global _instance
    if _instance is None:
        _instance = ContentProcessorService()
    return _instance
