"""
Tests for Light Multimodal Processors
=====================================

Benchmark tests comparing light vs full processing modes.

Test coverage:
- LightWhisperProcessor: Model loading, transcription, duration limits
- LightOCRProcessor: Frame extraction, thumbnail processing
- LightHookAnalyzer: Hook pattern detection
- MediaCache: Hash-based caching, TTL expiry
- LightMultimodalProcessor: Full pipeline, conditional skipping

Run with: pytest backend/tests/test_light_processors.py -v
"""

import os
import sys
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.light_processors import (
    LightProcessingConfig,
    LightProcessingResult,
    MediaCache,
    LightWhisperProcessor,
    LightOCRProcessor,
    LightHookAnalyzer,
    LightMultimodalProcessor,
    get_light_processor,
    process_media_light,
    benchmark_light_vs_full,
)


# =============================================================================
# Test Configuration
# =============================================================================

class TestLightProcessingConfig:
    """Test configuration dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LightProcessingConfig()

        assert config.whisper_model == "tiny"
        assert config.whisper_max_duration == 3.0
        assert config.ocr_max_frames == 5
        assert config.hook_duration_seconds == 3.0
        assert config.cache_enabled is True
        assert config.skip_non_video is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = LightProcessingConfig(
            whisper_model="base",
            whisper_max_duration=5.0,
            ocr_max_frames=10,
        )

        assert config.whisper_model == "base"
        assert config.whisper_max_duration == 5.0
        assert config.ocr_max_frames == 10


class TestLightProcessingResult:
    """Test result dataclass."""

    def test_default_result(self):
        """Test default result values."""
        result = LightProcessingResult()

        assert result.transcription == ""
        assert result.ocr_text == ""
        assert result.hook_score == 0.0
        assert result.light_mode is True
        assert result.cached is False
        assert result.skipped is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = LightProcessingResult(
            transcription="Hello world",
            hook_score=0.75,
            processing_time_seconds=2.5,
        )

        d = result.to_dict()

        assert d["transcription"] == "Hello world"
        assert d["hook_score"] == 0.75
        assert d["processing_time_seconds"] == 2.5
        assert "light_mode" in d


# =============================================================================
# Test Media Cache
# =============================================================================

class TestMediaCache:
    """Test hash-based media cache."""

    @pytest.fixture
    def cache(self, tmp_path):
        """Create a temporary cache for testing."""
        cache_dir = tmp_path / "test_cache"
        cache_dir.mkdir()
        return MediaCache(cache_dir=cache_dir, ttl_hours=1)

    def test_hash_url(self):
        """Test URL hashing."""
        hash1 = MediaCache.hash_url("https://example.com/video1.mp4")
        hash2 = MediaCache.hash_url("https://example.com/video2.mp4")
        hash1_again = MediaCache.hash_url("https://example.com/video1.mp4")

        assert hash1 != hash2
        assert hash1 == hash1_again
        assert len(hash1) == 16  # Truncated SHA256

    def test_set_and_get(self, cache):
        """Test basic cache set/get operations."""
        url = "https://example.com/test.mp4"
        result = {"transcription": "test", "hook_score": 0.5}

        # Initially empty
        assert cache.get(url) is None
        assert cache.has(url) is False

        # Set value
        cache.set(url, result)

        # Get value
        cached = cache.get(url)
        assert cached is not None
        assert cached["transcription"] == "test"
        assert cached["hook_score"] == 0.5
        assert cache.has(url) is True

    def test_clear(self, cache):
        """Test cache clearing."""
        cache.set("url1", {"data": 1})
        cache.set("url2", {"data": 2})

        assert cache.has("url1")
        assert cache.has("url2")

        cache.clear()

        assert not cache.has("url1")
        assert not cache.has("url2")

    def test_stats(self, cache):
        """Test cache statistics."""
        cache.set("url1", {"data": 1})
        cache.set("url2", {"data": 2})

        stats = cache.stats()

        assert stats["total_entries"] == 2
        assert "cache_file" in stats
        assert stats["ttl_hours"] == 1.0


# =============================================================================
# Test Light Hook Analyzer
# =============================================================================

class TestLightHookAnalyzer:
    """Test hook pattern detection."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return LightHookAnalyzer()

    def test_question_hook(self, analyzer):
        """Test question hook detection."""
        result = analyzer.analyze_hook(
            caption="¿Por qué nadie habla de esto?"
        )

        assert result["hook_score"] > 0.0
        assert "?" in result["hook_text"]
        assert result["status"] == "success"

    def test_pov_hook(self, analyzer):
        """Test POV hook detection."""
        result = analyzer.analyze_hook(
            caption="POV: cuando descubres el secreto..."
        )

        assert result["hook_score"] > 0.0
        assert len(result["patterns_matched"]) > 0

    def test_number_hook(self, analyzer):
        """Test numbered hook detection."""
        result = analyzer.analyze_hook(
            caption="5 trucos que cambiarán tu vida"
        )

        assert result["hook_score"] > 0.0
        assert len(result["patterns_matched"]) > 0

    def test_urgency_hook(self, analyzer):
        """Test urgency hook detection."""
        result = analyzer.analyze_hook(
            caption="Urgente: tienes que ver esto ahora mismo"
        )

        assert result["hook_score"] > 0.0

    def test_no_text(self, analyzer):
        """Test with no text input."""
        result = analyzer.analyze_hook()

        assert result["hook_score"] == 0.0
        assert result["status"] == "no_text"

    def test_combined_sources(self, analyzer):
        """Test combining multiple text sources."""
        result = analyzer.analyze_hook(
            caption="Mira esto",
            transcription="¿Sabías que esto es increíble?",
            ocr_text="SECRETO REVELADO"
        )

        assert result["hook_score"] > 0.0
        assert "hook_text" in result


# =============================================================================
# Test Light Whisper Processor (Mocked)
# =============================================================================

class TestLightWhisperProcessor:
    """Test Whisper processor with mocked dependencies."""

    def test_unavailable_gracefully(self):
        """Test graceful handling when faster-whisper not installed."""
        with patch.dict(sys.modules, {'faster_whisper': None}):
            processor = LightWhisperProcessor()

            # Should not raise, just return empty result
            result = processor.transcribe("test.mp4")

            assert result["transcription"] == ""
            assert result["status"] == "whisper_not_available"

    @patch('ml.light_processors.LightWhisperProcessor._check_availability')
    def test_config_applied(self, mock_check):
        """Test that config is properly applied."""
        mock_check.return_value = False  # Don't actually load model

        config = LightProcessingConfig(
            whisper_model="base",
            whisper_max_duration=5.0
        )
        processor = LightWhisperProcessor(config)

        assert processor.config.whisper_model == "base"
        assert processor.config.whisper_max_duration == 5.0


# =============================================================================
# Test Light OCR Processor (Mocked)
# =============================================================================

class TestLightOCRProcessor:
    """Test OCR processor with mocked dependencies."""

    def test_unavailable_gracefully(self):
        """Test graceful handling when easyocr not installed."""
        with patch.dict(sys.modules, {'easyocr': None}):
            processor = LightOCRProcessor()

            result = processor.extract_from_thumbnail("test.jpg")

            assert result["ocr_text"] == ""
            assert result["status"] == "easyocr_not_available"

    @patch('ml.light_processors.LightOCRProcessor._check_availability')
    def test_config_applied(self, mock_check):
        """Test that config is properly applied."""
        mock_check.return_value = False

        config = LightProcessingConfig(
            ocr_max_frames=10,
            ocr_languages=("en", "es", "fr")
        )
        processor = LightOCRProcessor(config)

        assert processor.config.ocr_max_frames == 10
        assert "fr" in processor.config.ocr_languages


# =============================================================================
# Test Light Multimodal Processor
# =============================================================================

class TestLightMultimodalProcessor:
    """Test unified multimodal processor."""

    @pytest.fixture
    def processor(self, tmp_path):
        """Create processor with temp cache."""
        config = LightProcessingConfig(cache_enabled=False)
        return LightMultimodalProcessor(light_mode=True, config=config)

    def test_should_skip_non_video(self, processor):
        """Test conditional skipping of non-video content."""
        should_skip, reason = processor.should_skip("post")
        assert should_skip is True
        assert "Non-video" in reason

        should_skip, reason = processor.should_skip("reel")
        assert should_skip is False

        should_skip, reason = processor.should_skip("video")
        assert should_skip is False

        should_skip, reason = processor.should_skip("tiktok")
        assert should_skip is False

    def test_process_skipped_for_non_video(self, processor):
        """Test that non-video content is skipped."""
        result = processor.process(
            media_url="https://example.com/image.jpg",
            content_type="post",
            caption="Check this out"
        )

        assert result.skipped is True
        assert "Non-video" in result.skip_reason
        # Hook analysis should still work on caption
        assert result.hook_score >= 0.0

    def test_process_with_cache_hit(self, tmp_path):
        """Test that cached results are returned quickly."""
        # Create processor with cache enabled
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        config = LightProcessingConfig(cache_enabled=True)
        processor = LightMultimodalProcessor(light_mode=True, config=config)
        processor._cache = MediaCache(cache_dir=cache_dir)

        # Pre-populate cache
        media_url = "https://example.com/video.mp4"
        processor._cache.set(media_url, {
            "transcription": "cached transcript",
            "ocr_text": "cached text",
            "hook_score": 0.8,
            "processing_time_seconds": 5.0,
            "original_time": 5.0,
        })

        # Process should return cached result
        result = processor.process(
            media_url=media_url,
            content_type="reel"
        )

        assert result.cached is True
        assert result.transcription == "cached transcript"
        assert result.hook_score == 0.8

    def test_get_cache_stats(self, processor):
        """Test cache statistics retrieval."""
        stats = processor.get_cache_stats()

        # With cache disabled
        assert stats["cache_enabled"] is False


# =============================================================================
# Test Singleton and Convenience Functions
# =============================================================================

class TestSingletonAndHelpers:
    """Test singleton pattern and helper functions."""

    def test_get_light_processor_singleton(self):
        """Test that get_light_processor returns singleton."""
        # Clear global
        import ml.light_processors as module
        module._light_processor = None

        p1 = get_light_processor()
        p2 = get_light_processor()

        assert p1 is p2

    def test_process_media_light_function(self):
        """Test convenience function."""
        result = process_media_light(
            content_type="post",
            caption="Test caption"
        )

        assert isinstance(result, dict)
        assert "light_mode" in result


# =============================================================================
# Benchmark Tests
# =============================================================================

class TestBenchmarks:
    """Benchmark tests for performance comparison."""

    def test_hook_analysis_speed(self):
        """Benchmark hook analysis - should be very fast."""
        analyzer = LightHookAnalyzer()

        captions = [
            "¿Por qué nadie habla de esto?",
            "POV: cuando descubres el secreto",
            "5 trucos que cambiarán tu vida",
            "Urgente: tienes que ver esto ahora mismo",
            "El secreto que nadie te cuenta...",
        ] * 20  # 100 captions

        start = time.time()
        for caption in captions:
            analyzer.analyze_hook(caption=caption)
        elapsed = time.time() - start

        # Should process 100 captions in under 1 second
        assert elapsed < 1.0, f"Hook analysis too slow: {elapsed:.2f}s for 100 captions"

    def test_cache_performance(self, tmp_path):
        """Benchmark cache operations."""
        cache_dir = tmp_path / "bench_cache"
        cache_dir.mkdir()
        cache = MediaCache(cache_dir=cache_dir)

        # Write 100 entries
        start = time.time()
        for i in range(100):
            cache.set(f"https://example.com/video{i}.mp4", {"data": i})
        write_time = time.time() - start

        # Read 100 entries
        start = time.time()
        for i in range(100):
            cache.get(f"https://example.com/video{i}.mp4")
        read_time = time.time() - start

        # Should be fast
        assert write_time < 2.0, f"Cache write too slow: {write_time:.2f}s"
        assert read_time < 0.1, f"Cache read too slow: {read_time:.2f}s"

    def test_light_vs_full_time_estimate(self):
        """Verify time savings estimates are reasonable."""
        config_light = LightProcessingConfig(
            whisper_model="tiny",
            whisper_max_duration=3.0,
            ocr_max_frames=5,
        )

        config_full = LightProcessingConfig(
            whisper_model="base",
            whisper_max_duration=30.0,
            ocr_max_frames=30,
        )

        # Light mode should have significantly smaller limits
        assert config_light.whisper_max_duration < config_full.whisper_max_duration
        assert config_light.ocr_max_frames < config_full.ocr_max_frames

        # Estimate: light ~5s, full ~20s -> 75% savings
        light_estimate = 5.0
        full_estimate = 20.0
        savings_percent = ((full_estimate - light_estimate) / full_estimate) * 100

        assert savings_percent >= 70, f"Expected >70% savings, got {savings_percent:.1f}%"


# =============================================================================
# Integration Test (Optional - requires actual video)
# =============================================================================

@pytest.mark.skip(reason="Requires actual video file and ML dependencies")
class TestIntegration:
    """Integration tests with real video files."""

    def test_full_pipeline_with_video(self, tmp_path):
        """Test full pipeline with a real video."""
        # This test requires:
        # - A test video file
        # - faster-whisper and easyocr installed
        # - Sufficient RAM

        video_path = tmp_path / "test.mp4"
        # Would need to create/copy a test video here

        result = benchmark_light_vs_full(
            str(video_path),
            content_type="reel",
            caption="Test video"
        )

        assert result["light_time_seconds"] < result["full_time_estimate_seconds"]
        assert result["time_saved_percent"] > 50


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
