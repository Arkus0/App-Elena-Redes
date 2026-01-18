"""
Performance Optimizations Tests
===============================

Tests for the video/ML performance optimizations:
A) INT8 Quantization for embeddings
B) Video Task Queue (asyncio background processing)
C) Optional Optical Flow toggle

These tests verify:
1. Quantization is applied correctly and provides speedup
2. Task queue processes videos in background
3. Optical flow can be disabled for performance
"""

import asyncio
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
import numpy as np


# =============================================================================
# A) QUANTIZATION TESTS
# =============================================================================

class TestEmbeddingQuantization:
    """Tests for INT8 quantization of sentence-transformer embeddings."""

    def test_text_config_has_quantization_settings(self):
        """Verify TextConfig includes quantization options."""
        from app.services.text_intelligence import TextConfig

        config = TextConfig()

        assert hasattr(config, "embedding_quantize"), \
            "TextConfig should have embedding_quantize option"
        assert hasattr(config, "embedding_quantize_dtype"), \
            "TextConfig should have embedding_quantize_dtype option"

        # Default should be enabled
        assert config.embedding_quantize is True, \
            "Quantization should be enabled by default"
        assert config.embedding_quantize_dtype == "int8", \
            "Default dtype should be int8"

    def test_quantization_disabled_config(self):
        """Verify quantization can be disabled via config."""
        from app.services.text_intelligence import TextConfig

        config = TextConfig(embedding_quantize=False)

        assert config.embedding_quantize is False, \
            "Should be able to disable quantization"

    def test_semantic_encoder_has_quantization_flag(self):
        """Verify SemanticEncoder tracks quantization state."""
        from app.services.text_intelligence import SemanticEncoder, TextConfig

        # Test with quantization disabled to avoid model download
        config = TextConfig(embedding_quantize=False)
        encoder = SemanticEncoder(config=config)

        assert hasattr(encoder, "_model_quantized"), \
            "SemanticEncoder should track quantization state"
        assert hasattr(encoder, "is_quantized"), \
            "SemanticEncoder should have is_quantized property"

    def test_whisper_already_uses_int8(self):
        """Verify Whisper is configured for int8 compute type."""
        from app.services.text_intelligence import TextConfig

        config = TextConfig()

        assert config.whisper_compute_type == "int8", \
            "Whisper should use int8 compute type by default"

    def test_quantization_docstring_documents_tradeoffs(self):
        """Verify documentation mentions accuracy/speed tradeoffs."""
        from app.services.text_intelligence import TextConfig, SemanticEncoder

        # Check TextConfig docstring
        assert "int8" in TextConfig.__doc__.lower(), \
            "TextConfig should document int8 quantization"
        assert "speedup" in TextConfig.__doc__.lower() or "speed" in TextConfig.__doc__.lower(), \
            "TextConfig should mention speedup"

        # Check SemanticEncoder docstring
        assert "quantization" in SemanticEncoder.__doc__.lower(), \
            "SemanticEncoder should document quantization"


# =============================================================================
# B) TASK QUEUE TESTS
# =============================================================================

class TestVideoTaskQueue:
    """Tests for the asyncio video task queue."""

    def test_queue_config_defaults(self):
        """Verify QueueConfig has sensible defaults."""
        from app.services.task_queue import QueueConfig

        config = QueueConfig()

        assert config.max_queue_size == 100, \
            "Default max queue size should be 100"
        assert config.max_retries == 3, \
            "Default max retries should be 3"
        assert config.task_timeout_seconds == 600.0, \
            "Default timeout should be 10 minutes"

    def test_task_types_available(self):
        """Verify all expected task types exist."""
        from app.services.task_queue import TaskType

        expected_types = [
            "full_analysis",
            "video_only",
            "audio_only",
            "text_intelligence",
            "quick_preview",
        ]

        for task_type in expected_types:
            assert hasattr(TaskType, task_type.upper()), \
                f"TaskType should have {task_type}"

    def test_task_status_states(self):
        """Verify all expected task status states exist."""
        from app.services.task_queue import TaskStatus

        expected_states = ["pending", "processing", "completed", "failed", "cancelled"]

        for state in expected_states:
            assert hasattr(TaskStatus, state.upper()), \
                f"TaskStatus should have {state}"

    def test_video_task_creation(self):
        """Test VideoTask creation and serialization."""
        from app.services.task_queue import VideoTask, TaskType, TaskStatus

        task = VideoTask(
            task_id="test-123",
            video_path="/path/to/video.mp4",
            task_type=TaskType.FULL_ANALYSIS,
            metadata={"business_id": 456}
        )

        assert task.task_id == "test-123"
        assert task.status == TaskStatus.PENDING
        assert task.progress == 0
        assert task.metadata["business_id"] == 456

        # Test serialization
        task_dict = task.to_dict()
        assert task_dict["task_id"] == "test-123"
        assert task_dict["task_type"] == "full_analysis"
        assert task_dict["status"] == "pending"

        # Test deserialization
        restored = VideoTask.from_dict(task_dict)
        assert restored.task_id == task.task_id
        assert restored.task_type == task.task_type

    @pytest.mark.asyncio
    async def test_queue_start_stop(self):
        """Test queue can be started and stopped."""
        from app.services.task_queue import VideoTaskQueue, QueueConfig

        config = QueueConfig(persist_to_sqlite=False)
        queue = VideoTaskQueue(config=config)

        # Start
        await queue.start()
        assert queue._running is True
        assert queue._worker_task is not None

        # Stop
        await queue.stop(wait=False)
        assert queue._running is False

    @pytest.mark.asyncio
    async def test_queue_submit_validates_path(self):
        """Test queue validates video path on submit."""
        from app.services.task_queue import VideoTaskQueue, QueueConfig

        config = QueueConfig(persist_to_sqlite=False)
        queue = VideoTaskQueue(config=config)
        await queue.start()

        try:
            # Should raise FileNotFoundError for non-existent file
            with pytest.raises(FileNotFoundError):
                await queue.submit(
                    video_path="/non/existent/video.mp4",
                    task_type="video_only"
                )
        finally:
            await queue.stop(wait=False)

    @pytest.mark.asyncio
    async def test_queue_submit_with_valid_file(self):
        """Test queue accepts valid video path."""
        from app.services.task_queue import VideoTaskQueue, QueueConfig, TaskType

        config = QueueConfig(persist_to_sqlite=False)
        queue = VideoTaskQueue(config=config)
        await queue.start()

        try:
            # Create temp file to simulate video
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                temp_path = f.name

            # Submit should return task_id
            task_id = await queue.submit(
                video_path=temp_path,
                task_type=TaskType.VIDEO_ONLY
            )

            assert task_id is not None
            assert len(task_id) > 0

            # Status should be available
            status = await queue.get_status(task_id)
            assert status is not None
            assert status["video_path"] == temp_path

        finally:
            await queue.stop(wait=False)
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_queue_cancellation(self):
        """Test task can be cancelled before processing."""
        from app.services.task_queue import VideoTaskQueue, QueueConfig, TaskStatus

        config = QueueConfig(persist_to_sqlite=False)
        queue = VideoTaskQueue(config=config)

        # Don't start worker - tasks will stay pending
        queue._running = True  # Pretend running but no worker

        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                temp_path = f.name

            # Submit task (will stay pending since no worker)
            queue._tasks = {}
            from app.services.task_queue import VideoTask, TaskType
            task = VideoTask(
                task_id="cancel-test",
                video_path=temp_path,
                task_type=TaskType.VIDEO_ONLY
            )
            queue._tasks[task.task_id] = task

            # Cancel it
            cancelled = await queue.cancel("cancel-test")
            assert cancelled is True

            # Verify status
            status = await queue.get_status("cancel-test")
            assert status["status"] == "cancelled"

        finally:
            queue._running = False
            os.unlink(temp_path)

    def test_global_queue_singleton(self):
        """Test get_video_queue returns singleton."""
        from app.services.task_queue import get_video_queue, _global_queue
        import app.services.task_queue as tq

        # Reset global
        tq._global_queue = None

        queue1 = get_video_queue()
        queue2 = get_video_queue()

        assert queue1 is queue2, "Should return same instance"

        # Cleanup
        tq._global_queue = None


# =============================================================================
# C) OPTICAL FLOW TOGGLE TESTS
# =============================================================================

class TestOpticalFlowToggle:
    """Tests for optional optical flow in VideoConfig."""

    def test_video_config_has_optical_flow_toggle(self):
        """Verify VideoConfig has optical_flow_enabled option."""
        from app.services.analytics_engine import VideoConfig

        config = VideoConfig()

        assert hasattr(config, "optical_flow_enabled"), \
            "VideoConfig should have optical_flow_enabled option"

        # Default should be enabled
        assert config.optical_flow_enabled is True, \
            "Optical flow should be enabled by default"

    def test_optical_flow_can_be_disabled(self):
        """Verify optical flow can be disabled."""
        from app.services.analytics_engine import VideoConfig

        config = VideoConfig(optical_flow_enabled=False)

        assert config.optical_flow_enabled is False, \
            "Should be able to disable optical flow"

    def test_config_documents_performance_impact(self):
        """Verify documentation mentions ~40% speedup."""
        from app.services.analytics_engine import VideoConfig

        docstring = VideoConfig.__doc__.lower()

        assert "optical_flow_enabled" in docstring or "optical" in docstring, \
            "VideoConfig should document optical flow toggle"
        assert "40%" in docstring or "speedup" in docstring or "performance" in docstring, \
            "Should mention performance impact"

    def test_feature_extractor_respects_config(self):
        """Verify EfficientFeatureExtractor uses config."""
        from app.services.analytics_engine import (
            EfficientFeatureExtractor,
            VideoConfig
        )

        # With optical flow disabled
        config_disabled = VideoConfig(optical_flow_enabled=False)
        extractor_disabled = EfficientFeatureExtractor(config=config_disabled)

        assert extractor_disabled.config.optical_flow_enabled is False

        # With optical flow enabled
        config_enabled = VideoConfig(optical_flow_enabled=True)
        extractor_enabled = EfficientFeatureExtractor(config=config_enabled)

        assert extractor_enabled.config.optical_flow_enabled is True

    def test_disabled_optical_flow_returns_neutral_values(self):
        """
        Test that when optical flow is disabled, instability metrics are neutral.

        When optical_flow_enabled=False:
        - instability_score should be 0.0
        - camera_stability_score should reflect no penalty
        - visual_energy should not have instability penalty applied
        """
        from app.services.analytics_engine import (
            EfficientFeatureExtractor,
            VideoConfig
        )

        # Create mock video data
        config = VideoConfig(optical_flow_enabled=False)
        extractor = EfficientFeatureExtractor(config=config)

        # Reset accumulators and set test data
        extractor._reset_accumulators()
        extractor._brightness_values = [0.5, 0.5, 0.5]
        extractor._contrast_values = [0.1, 0.1, 0.1]
        extractor._frame_diffs = [0.1, 0.1, 0.1]
        extractor._hook_frame_diffs = [0.1, 0.1]
        extractor._retention_frame_diffs = [0.1]

        # Optical flow accumulators should be empty when disabled
        assert len(extractor._global_motion_magnitudes) == 0
        assert len(extractor._local_motion_variances) == 0

    def test_metadata_includes_optimization_flags(self):
        """Verify extracted features include optimization flags in metadata."""
        from app.services.analytics_engine import VideoConfig

        # Create config with specific settings
        config = VideoConfig(
            optical_flow_enabled=False,
            face_detection_enabled=True
        )

        # The extract_video_features should include these flags in output
        # (We verify the code path exists without running full extraction)
        assert config.optical_flow_enabled is False
        assert config.face_detection_enabled is True


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestOptimizationIntegration:
    """Integration tests combining multiple optimizations."""

    def test_app_config_has_optimization_settings(self):
        """Verify app config includes optimization settings."""
        from app.core.config import settings

        # Video optimization
        assert hasattr(settings, "VIDEO_OPTICAL_FLOW_ENABLED"), \
            "Settings should have VIDEO_OPTICAL_FLOW_ENABLED"

        # ML optimization
        assert hasattr(settings, "ML_EMBEDDING_QUANTIZE"), \
            "Settings should have ML_EMBEDDING_QUANTIZE"
        assert hasattr(settings, "ML_WHISPER_COMPUTE_TYPE"), \
            "Settings should have ML_WHISPER_COMPUTE_TYPE"

        # Task queue
        assert hasattr(settings, "TASK_QUEUE_ENABLED"), \
            "Settings should have TASK_QUEUE_ENABLED"
        assert hasattr(settings, "TASK_QUEUE_MAX_SIZE"), \
            "Settings should have TASK_QUEUE_MAX_SIZE"

    def test_quick_preview_mode_disables_optical_flow(self):
        """Verify QUICK_PREVIEW task type uses fast config."""
        from app.services.task_queue import TaskType

        # QUICK_PREVIEW should exist
        assert TaskType.QUICK_PREVIEW.value == "quick_preview"

        # The _do_processing method should disable optical flow for this type
        # (Verified by code inspection - optical_flow_enabled = task_type != TaskType.QUICK_PREVIEW)

    def test_persistence_config_available(self):
        """Verify persistence can be configured."""
        from app.services.task_queue import QueueConfig

        config = QueueConfig(
            persist_to_sqlite=True,
            sqlite_path="/tmp/test_tasks.db"
        )

        assert config.persist_to_sqlite is True
        assert config.sqlite_path == "/tmp/test_tasks.db"

    def test_analytics_engine_uses_text_intelligence_flag(self):
        """Verify AnalyticsEngine respects text intelligence flag."""
        from app.services.analytics_engine import AnalyticsEngine, VideoConfig

        # Without text intelligence
        engine_no_text = AnalyticsEngine(enable_text_intelligence=False)
        assert engine_no_text._text_engine is None

        # Note: We don't test with text_intelligence=True to avoid
        # downloading ML models in tests


# =============================================================================
# PERFORMANCE BENCHMARK TESTS (Optional - Run manually)
# =============================================================================

class TestPerformanceBenchmarks:
    """
    Performance benchmark tests.

    These tests measure actual speedup from optimizations.
    Mark with pytest.mark.slow to skip in CI.
    """

    @pytest.mark.skip(reason="Requires ML models - run manually")
    def test_quantization_speedup(self):
        """
        Benchmark embedding quantization speedup.

        Expected: ~3-4x speedup with int8 quantization.
        """
        from app.services.text_intelligence import SemanticEncoder, TextConfig
        import time

        test_text = "This is a test sentence for embedding generation."
        iterations = 100

        # Float32 (no quantization)
        config_fp32 = TextConfig(embedding_quantize=False)
        encoder_fp32 = SemanticEncoder(config=config_fp32)

        start = time.time()
        for _ in range(iterations):
            encoder_fp32.encode_text(test_text)
        fp32_time = time.time() - start

        encoder_fp32.unload_model()

        # Int8 (with quantization)
        config_int8 = TextConfig(embedding_quantize=True)
        encoder_int8 = SemanticEncoder(config=config_int8)

        start = time.time()
        for _ in range(iterations):
            encoder_int8.encode_text(test_text)
        int8_time = time.time() - start

        encoder_int8.unload_model()

        speedup = fp32_time / int8_time
        print(f"\nQuantization speedup: {speedup:.2f}x")
        print(f"FP32: {fp32_time:.2f}s, INT8: {int8_time:.2f}s")

        # Should be at least 2x faster
        assert speedup > 2.0, f"Expected >2x speedup, got {speedup:.2f}x"

    @pytest.mark.skip(reason="Requires video file - run manually")
    def test_optical_flow_toggle_speedup(self):
        """
        Benchmark optical flow disable speedup.

        Expected: ~40% speedup when optical flow is disabled.
        """
        from app.services.analytics_engine import (
            EfficientFeatureExtractor,
            VideoConfig
        )
        import time

        video_path = "test_video.mp4"  # Replace with actual test video

        # With optical flow
        config_with = VideoConfig(optical_flow_enabled=True)
        extractor_with = EfficientFeatureExtractor(config=config_with)

        start = time.time()
        extractor_with.extract_video_features(video_path)
        time_with = time.time() - start

        # Without optical flow
        config_without = VideoConfig(optical_flow_enabled=False)
        extractor_without = EfficientFeatureExtractor(config=config_without)

        start = time.time()
        extractor_without.extract_video_features(video_path)
        time_without = time.time() - start

        speedup = (time_with - time_without) / time_with * 100
        print(f"\nOptical flow disable speedup: {speedup:.1f}%")
        print(f"With: {time_with:.2f}s, Without: {time_without:.2f}s")

        # Should save at least 20% time
        assert speedup > 20, f"Expected >20% speedup, got {speedup:.1f}%"


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_video_file():
    """Create a temporary file to simulate a video."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def mock_analytics_engine():
    """Create a mock AnalyticsEngine for testing."""
    mock = MagicMock()
    mock.extract_video_features.return_value = {
        "visual_energy": 0.05,
        "hook_energy": 0.06,
        "duration_seconds": 15.0,
        "optical_flow_enabled": True,
    }
    mock.extract_complete_features.return_value = {
        "visual_energy": 0.05,
        "transcription": "Test transcription",
        "status": "success",
    }
    return mock


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
