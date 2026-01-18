"""
Video Processing Task Queue - Asyncio Background Worker
========================================================

Provides a lightweight task queue for processing videos in the background,
keeping the API responsive. Videos are processed one-at-a-time to avoid
resource contention on edge devices.

ARCHITECTURE:
=============
- Uses asyncio.Queue for in-memory task management
- Single worker coroutine processes tasks sequentially
- Optional SQLite persistence for crash recovery
- Callbacks for task completion/error notification

PERFORMANCE BENEFITS:
=====================
- API endpoints return immediately with task_id
- Video processing doesn't block HTTP responses
- Sequential processing prevents CPU/memory contention
- Graceful shutdown waits for current task to complete

USAGE:
======
    # Initialize queue (usually in app startup)
    queue = VideoTaskQueue()
    await queue.start()

    # Submit task (from API endpoint)
    task_id = await queue.submit(
        video_path="/path/to/video.mp4",
        task_type="full_analysis",
        metadata={"business_id": 123}
    )

    # Check status
    status = await queue.get_status(task_id)
    # Returns: {"status": "processing", "progress": 50, ...}

    # Shutdown (in app shutdown)
    await queue.stop()

Author: ML Engineering Team
"""

import asyncio
import gc
import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import threading

logger = logging.getLogger(__name__)


# =============================================================================
# Task Status and Models
# =============================================================================

class TaskStatus(str, Enum):
    """Status of a video processing task."""
    PENDING = "pending"          # In queue, waiting to be processed
    PROCESSING = "processing"    # Currently being processed
    COMPLETED = "completed"      # Successfully completed
    FAILED = "failed"            # Failed with error
    CANCELLED = "cancelled"      # Cancelled by user


class TaskType(str, Enum):
    """Type of video processing task."""
    FULL_ANALYSIS = "full_analysis"        # Video + Audio + Text Intelligence
    VIDEO_ONLY = "video_only"              # Only video DNA features
    AUDIO_ONLY = "audio_only"              # Only audio features
    TEXT_INTELLIGENCE = "text_intelligence" # Only transcription + OCR + embeddings
    QUICK_PREVIEW = "quick_preview"        # Fast mode (no optical flow)


@dataclass
class VideoTask:
    """
    Represents a video processing task in the queue.

    Attributes:
        task_id: Unique identifier for the task
        video_path: Path to the video file
        task_type: Type of analysis to perform
        status: Current status of the task
        progress: Progress percentage (0-100)
        result: Processing result when completed
        error: Error message if failed
        metadata: Additional metadata (business_id, user_id, etc.)
        created_at: When the task was created
        started_at: When processing started
        completed_at: When processing finished
        retry_count: Number of retry attempts
    """
    task_id: str
    video_path: str
    task_type: TaskType = TaskType.FULL_ANALYSIS
    status: TaskStatus = TaskStatus.PENDING
    progress: int = 0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    retry_count: int = 0
    caption: str = ""  # Optional caption for text analysis

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "video_path": self.video_path,
            "task_type": self.task_type.value,
            "status": self.status.value,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "retry_count": self.retry_count,
            "caption": self.caption,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoTask":
        """Create task from dictionary."""
        return cls(
            task_id=data["task_id"],
            video_path=data["video_path"],
            task_type=TaskType(data.get("task_type", "full_analysis")),
            status=TaskStatus(data.get("status", "pending")),
            progress=data.get("progress", 0),
            result=data.get("result"),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", time.time()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            retry_count=data.get("retry_count", 0),
            caption=data.get("caption", ""),
        )


# =============================================================================
# Queue Configuration
# =============================================================================

@dataclass(frozen=True)
class QueueConfig:
    """
    Configuration for the video processing queue.

    Attributes:
        max_queue_size: Maximum number of pending tasks (0 = unlimited)
        max_retries: Maximum retry attempts for failed tasks
        retry_delay_seconds: Delay between retries
        task_timeout_seconds: Maximum time for a single task
        persist_to_sqlite: Enable SQLite persistence for crash recovery
        sqlite_path: Path to SQLite database file
        cleanup_completed_after_hours: Auto-cleanup completed tasks after N hours
    """
    max_queue_size: int = 100
    max_retries: int = 3
    retry_delay_seconds: float = 5.0
    task_timeout_seconds: float = 600.0  # 10 minutes max per task
    persist_to_sqlite: bool = False
    sqlite_path: str = "./video_tasks.db"
    cleanup_completed_after_hours: int = 24


# =============================================================================
# SQLite Persistence Layer (Optional)
# =============================================================================

class TaskPersistence:
    """
    Optional SQLite persistence for task queue.

    Enables crash recovery by persisting pending/processing tasks.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()

    def initialize(self):
        """Create database and tables if they don't exist."""
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                video_path TEXT NOT NULL,
                task_type TEXT NOT NULL,
                status TEXT NOT NULL,
                progress INTEGER DEFAULT 0,
                result TEXT,
                error TEXT,
                metadata TEXT,
                created_at REAL NOT NULL,
                started_at REAL,
                completed_at REAL,
                retry_count INTEGER DEFAULT 0,
                caption TEXT DEFAULT ''
            )
        """)
        self._conn.commit()
        logger.info(f"Task persistence initialized: {self.db_path}")

    def save_task(self, task: VideoTask):
        """Save or update a task in the database."""
        if not self._conn:
            return

        with self._lock:
            self._conn.execute("""
                INSERT OR REPLACE INTO tasks
                (task_id, video_path, task_type, status, progress, result,
                 error, metadata, created_at, started_at, completed_at,
                 retry_count, caption)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task.task_id,
                task.video_path,
                task.task_type.value,
                task.status.value,
                task.progress,
                json.dumps(task.result) if task.result else None,
                task.error,
                json.dumps(task.metadata),
                task.created_at,
                task.started_at,
                task.completed_at,
                task.retry_count,
                task.caption,
            ))
            self._conn.commit()

    def load_pending_tasks(self) -> List[VideoTask]:
        """Load all pending and processing tasks (for crash recovery)."""
        if not self._conn:
            return []

        with self._lock:
            cursor = self._conn.execute("""
                SELECT * FROM tasks
                WHERE status IN ('pending', 'processing')
                ORDER BY created_at ASC
            """)
            rows = cursor.fetchall()

        tasks = []
        for row in rows:
            task = VideoTask(
                task_id=row[0],
                video_path=row[1],
                task_type=TaskType(row[2]),
                status=TaskStatus.PENDING,  # Reset processing to pending
                progress=0,
                result=json.loads(row[5]) if row[5] else None,
                error=row[6],
                metadata=json.loads(row[7]) if row[7] else {},
                created_at=row[8],
                started_at=None,
                completed_at=None,
                retry_count=row[11] + 1,  # Increment retry count
                caption=row[12] if len(row) > 12 else "",
            )
            tasks.append(task)

        return tasks

    def get_task(self, task_id: str) -> Optional[VideoTask]:
        """Get a task by ID."""
        if not self._conn:
            return None

        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM tasks WHERE task_id = ?",
                (task_id,)
            )
            row = cursor.fetchone()

        if not row:
            return None

        return VideoTask(
            task_id=row[0],
            video_path=row[1],
            task_type=TaskType(row[2]),
            status=TaskStatus(row[3]),
            progress=row[4],
            result=json.loads(row[5]) if row[5] else None,
            error=row[6],
            metadata=json.loads(row[7]) if row[7] else {},
            created_at=row[8],
            started_at=row[9],
            completed_at=row[10],
            retry_count=row[11],
            caption=row[12] if len(row) > 12 else "",
        )

    def cleanup_old_tasks(self, hours: int):
        """Remove completed/failed tasks older than N hours."""
        if not self._conn:
            return

        cutoff = time.time() - (hours * 3600)
        with self._lock:
            self._conn.execute("""
                DELETE FROM tasks
                WHERE status IN ('completed', 'failed', 'cancelled')
                AND completed_at < ?
            """, (cutoff,))
            self._conn.commit()

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


# =============================================================================
# Video Task Queue
# =============================================================================

class VideoTaskQueue:
    """
    Asyncio-based task queue for video processing.

    Processes videos one-at-a-time in the background to avoid blocking
    the API and prevent resource contention.

    Features:
    - Async submit/status APIs
    - Sequential processing (one video at a time)
    - Optional SQLite persistence for crash recovery
    - Progress callbacks
    - Graceful shutdown

    Example:
        queue = VideoTaskQueue()
        await queue.start()

        # Submit from API endpoint
        task_id = await queue.submit(
            video_path="video.mp4",
            task_type="full_analysis"
        )

        # Check status
        status = await queue.get_status(task_id)
    """

    def __init__(
        self,
        config: Optional[QueueConfig] = None,
        on_complete: Optional[Callable[[VideoTask], None]] = None,
        on_error: Optional[Callable[[VideoTask, Exception], None]] = None,
    ):
        """
        Initialize the video task queue.

        Args:
            config: Queue configuration
            on_complete: Callback when task completes successfully
            on_error: Callback when task fails
        """
        self.config = config or QueueConfig()
        self.on_complete = on_complete
        self.on_error = on_error

        # Asyncio queue for pending tasks
        self._queue: asyncio.Queue[VideoTask] = asyncio.Queue(
            maxsize=self.config.max_queue_size if self.config.max_queue_size > 0 else 0
        )

        # In-memory task registry for status lookups
        self._tasks: Dict[str, VideoTask] = {}
        self._lock = asyncio.Lock()

        # Worker state
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
        self._current_task: Optional[VideoTask] = None

        # Optional persistence
        self._persistence: Optional[TaskPersistence] = None
        if self.config.persist_to_sqlite:
            self._persistence = TaskPersistence(self.config.sqlite_path)

        # Analytics engine (lazy loaded)
        self._analytics_engine = None

    def _get_analytics_engine(self):
        """Lazy load the analytics engine."""
        if self._analytics_engine is None:
            from app.services.analytics_engine import AnalyticsEngine
            self._analytics_engine = AnalyticsEngine(enable_text_intelligence=True)
        return self._analytics_engine

    async def start(self):
        """
        Start the queue worker.

        Call this during application startup.
        """
        if self._running:
            logger.warning("Queue already running")
            return

        # Initialize persistence if enabled
        if self._persistence:
            self._persistence.initialize()

            # Recover pending tasks from previous session
            pending_tasks = self._persistence.load_pending_tasks()
            for task in pending_tasks:
                if task.retry_count <= self.config.max_retries:
                    logger.info(f"Recovering task {task.task_id} (retry {task.retry_count})")
                    await self._queue.put(task)
                    self._tasks[task.task_id] = task
                else:
                    logger.warning(f"Dropping task {task.task_id} (exceeded max retries)")
                    task.status = TaskStatus.FAILED
                    task.error = "Exceeded maximum retry attempts"
                    self._persistence.save_task(task)

        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("Video task queue started")

    async def stop(self, wait: bool = True):
        """
        Stop the queue worker.

        Args:
            wait: If True, wait for current task to complete
        """
        self._running = False

        if self._worker_task:
            if wait and self._current_task:
                logger.info("Waiting for current task to complete...")
                # Give current task time to finish
                try:
                    await asyncio.wait_for(
                        self._worker_task,
                        timeout=self.config.task_timeout_seconds
                    )
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for task, cancelling...")
                    self._worker_task.cancel()
            else:
                self._worker_task.cancel()

            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        if self._persistence:
            self._persistence.close()

        logger.info("Video task queue stopped")

    async def submit(
        self,
        video_path: str,
        task_type: Union[TaskType, str] = TaskType.FULL_ANALYSIS,
        metadata: Optional[Dict[str, Any]] = None,
        caption: str = "",
    ) -> str:
        """
        Submit a video for processing.

        Args:
            video_path: Path to the video file
            task_type: Type of analysis to perform
            metadata: Additional metadata (business_id, user_id, etc.)
            caption: Optional caption for text intelligence

        Returns:
            task_id: Unique identifier for tracking the task

        Raises:
            asyncio.QueueFull: If queue is at capacity
            FileNotFoundError: If video file doesn't exist
        """
        # Validate video path
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Convert string to enum if needed
        if isinstance(task_type, str):
            task_type = TaskType(task_type)

        # Create task
        task = VideoTask(
            task_id=str(uuid.uuid4()),
            video_path=video_path,
            task_type=task_type,
            metadata=metadata or {},
            caption=caption,
        )

        # Add to queue
        async with self._lock:
            self._tasks[task.task_id] = task

        await self._queue.put(task)

        # Persist if enabled
        if self._persistence:
            self._persistence.save_task(task)

        logger.info(f"Task submitted: {task.task_id} ({task_type.value})")
        return task.task_id

    async def get_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current status of a task.

        Args:
            task_id: The task identifier

        Returns:
            Status dictionary or None if task not found
        """
        async with self._lock:
            task = self._tasks.get(task_id)

        if task:
            return task.to_dict()

        # Check persistence if not in memory
        if self._persistence:
            task = self._persistence.get_task(task_id)
            if task:
                return task.to_dict()

        return None

    async def get_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the result of a completed task.

        Args:
            task_id: The task identifier

        Returns:
            Result dictionary or None if not completed
        """
        status = await self.get_status(task_id)
        if status and status["status"] == "completed":
            return status["result"]
        return None

    async def cancel(self, task_id: str) -> bool:
        """
        Cancel a pending task.

        Args:
            task_id: The task identifier

        Returns:
            True if cancelled, False if task not found or already processing
        """
        async with self._lock:
            task = self._tasks.get(task_id)
            if task and task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
                task.completed_at = time.time()

                if self._persistence:
                    self._persistence.save_task(task)

                logger.info(f"Task cancelled: {task_id}")
                return True

        return False

    def get_queue_length(self) -> int:
        """Get number of pending tasks."""
        return self._queue.qsize()

    def is_processing(self) -> bool:
        """Check if a task is currently being processed."""
        return self._current_task is not None

    async def _worker_loop(self):
        """
        Main worker loop - processes tasks sequentially.

        Runs continuously until stop() is called.
        """
        logger.info("Worker loop started")

        while self._running:
            try:
                # Wait for next task with timeout
                try:
                    task = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0  # Check running flag every second
                    )
                except asyncio.TimeoutError:
                    continue

                # Skip cancelled tasks
                if task.status == TaskStatus.CANCELLED:
                    self._queue.task_done()
                    continue

                # Process the task
                self._current_task = task
                await self._process_task(task)
                self._current_task = None
                self._queue.task_done()

                # Force garbage collection between tasks
                gc.collect()

            except asyncio.CancelledError:
                logger.info("Worker loop cancelled")
                break
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                await asyncio.sleep(1)  # Prevent tight error loop

        logger.info("Worker loop ended")

    async def _process_task(self, task: VideoTask):
        """
        Process a single video task.

        Args:
            task: The task to process
        """
        logger.info(f"Processing task {task.task_id}: {task.video_path}")

        # Update status
        task.status = TaskStatus.PROCESSING
        task.started_at = time.time()
        task.progress = 0

        if self._persistence:
            self._persistence.save_task(task)

        try:
            # Run processing in executor to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._do_processing,
                task
            )

            # Update task with result
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.progress = 100
            task.completed_at = time.time()

            logger.info(
                f"Task completed: {task.task_id} "
                f"({task.completed_at - task.started_at:.1f}s)"
            )

            # Callback
            if self.on_complete:
                try:
                    self.on_complete(task)
                except Exception as e:
                    logger.error(f"on_complete callback error: {e}")

        except Exception as e:
            logger.error(f"Task failed: {task.task_id} - {e}")

            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()

            # Retry if under limit
            if task.retry_count < self.config.max_retries:
                logger.info(f"Scheduling retry for {task.task_id}")
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                task.started_at = None
                task.completed_at = None
                task.progress = 0

                await asyncio.sleep(self.config.retry_delay_seconds)
                await self._queue.put(task)

            # Callback
            if self.on_error:
                try:
                    self.on_error(task, e)
                except Exception as cb_error:
                    logger.error(f"on_error callback error: {cb_error}")

        finally:
            # Always persist final state
            if self._persistence:
                self._persistence.save_task(task)

            # Update in-memory registry
            async with self._lock:
                self._tasks[task.task_id] = task

    def _do_processing(self, task: VideoTask) -> Dict[str, Any]:
        """
        Synchronous processing function (runs in executor).

        This is where the actual video analysis happens.
        """
        from app.services.analytics_engine import AnalyticsEngine, VideoConfig

        # Configure based on task type
        enable_text = task.task_type in [
            TaskType.FULL_ANALYSIS,
            TaskType.TEXT_INTELLIGENCE
        ]

        # Use fast mode for quick preview (disable optical flow)
        optical_flow_enabled = task.task_type != TaskType.QUICK_PREVIEW

        video_config = VideoConfig(optical_flow_enabled=optical_flow_enabled)

        # Create engine for this task
        engine = AnalyticsEngine(
            video_config=video_config,
            enable_text_intelligence=enable_text
        )

        try:
            if task.task_type == TaskType.VIDEO_ONLY:
                return engine.extract_video_features(task.video_path)

            elif task.task_type == TaskType.AUDIO_ONLY:
                return engine.extract_audio_features(task.video_path)

            elif task.task_type == TaskType.TEXT_INTELLIGENCE:
                return engine.extract_text_features(
                    video_path=task.video_path,
                    caption=task.caption
                )

            else:  # FULL_ANALYSIS or QUICK_PREVIEW
                return engine.extract_complete_features(
                    media_path=task.video_path,
                    caption=task.caption,
                    include_video=True,
                    include_audio=True,
                    include_text=enable_text
                )

        finally:
            # Cleanup
            if hasattr(engine, '_text_engine') and engine._text_engine:
                engine._text_engine.unload_all_models()
            gc.collect()


# =============================================================================
# Global Queue Instance (Singleton)
# =============================================================================

_global_queue: Optional[VideoTaskQueue] = None


def get_video_queue(
    config: Optional[QueueConfig] = None,
    on_complete: Optional[Callable] = None,
    on_error: Optional[Callable] = None,
) -> VideoTaskQueue:
    """
    Get or create the global video task queue.

    Use this to get a singleton instance of the queue.

    Args:
        config: Queue configuration (only used on first call)
        on_complete: Completion callback
        on_error: Error callback

    Returns:
        The global VideoTaskQueue instance
    """
    global _global_queue

    if _global_queue is None:
        _global_queue = VideoTaskQueue(
            config=config,
            on_complete=on_complete,
            on_error=on_error,
        )

    return _global_queue


async def init_queue(config: Optional[QueueConfig] = None) -> VideoTaskQueue:
    """
    Initialize and start the global video task queue.

    Call this during application startup.

    Args:
        config: Queue configuration

    Returns:
        The started VideoTaskQueue instance
    """
    queue = get_video_queue(config=config)
    await queue.start()
    return queue


async def shutdown_queue(wait: bool = True):
    """
    Shutdown the global video task queue.

    Call this during application shutdown.

    Args:
        wait: If True, wait for current task to complete
    """
    global _global_queue

    if _global_queue:
        await _global_queue.stop(wait=wait)
        _global_queue = None


# =============================================================================
# CLI Entry Point (for testing)
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    async def test_queue():
        print("\n" + "=" * 60)
        print("Video Task Queue - Test")
        print("=" * 60 + "\n")

        if len(sys.argv) < 2:
            print("Usage: python task_queue.py <video_path>")
            return

        video_path = sys.argv[1]

        # Create queue with persistence
        config = QueueConfig(persist_to_sqlite=True)
        queue = VideoTaskQueue(config=config)
        await queue.start()

        try:
            # Submit task
            print(f"[1] Submitting task for: {video_path}")
            task_id = await queue.submit(
                video_path=video_path,
                task_type=TaskType.QUICK_PREVIEW,  # Fast mode for testing
                metadata={"test": True}
            )
            print(f"    Task ID: {task_id}")

            # Poll status
            print("\n[2] Waiting for completion...")
            while True:
                status = await queue.get_status(task_id)
                if status:
                    print(f"    Status: {status['status']} ({status['progress']}%)")

                    if status["status"] in ["completed", "failed", "cancelled"]:
                        break

                await asyncio.sleep(1)

            # Show result
            print("\n[3] Result:")
            result = await queue.get_result(task_id)
            if result:
                for key, value in result.items():
                    if not isinstance(value, (dict, list)):
                        print(f"    {key}: {value}")

        finally:
            await queue.stop()

        print("\n" + "=" * 60)
        print("Test Complete")
        print("=" * 60 + "\n")

    asyncio.run(test_queue())
