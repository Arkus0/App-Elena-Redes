"""
Elena Bridge - Raw Data Ingestion API
Receives content and profile data extracted from Instagram/TikTok by the Chrome extension
Supports both individual content (posts, reels, videos) and full profile analysis

Human-in-the-Loop Integration:
When isOwnProfile=True, the backend registers real performance metrics for ML feedback loop.
This closes the loop between predictions and actual performance, enabling continuous learning.

Light Mode Multimodal Processing:
When light_mode=True (default), uses optimized Whisper/EasyOCR processing:
- Whisper: 'tiny' model, first 3 seconds only (hook analysis)
- EasyOCR: First 5 frames or thumbnail only
- Cache: Hash-based deduplication to skip already processed media
- Skip: Non-video content skips multimodal processing entirely
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Header
from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Any, Dict
from datetime import datetime
import logging
import uuid
import hashlib
import math
import time

# Import ML service for feedback loop
from app.services.ml_service import get_ml_predictor, FeatureExtractor
from app.core.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Pydantic Schemas for Profile Data
# ============================================================================

class ProfileStats(BaseModel):
    """Statistics from a social media profile"""
    followers: Optional[int] = None
    following: Optional[int] = None
    posts: Optional[int] = None
    likes: Optional[int] = None  # TikTok specific


class ProfileData(BaseModel):
    """Profile data extracted from social media"""
    platform: Literal["instagram", "tiktok", "unknown"]
    username: str
    displayName: Optional[str] = None
    bio: Optional[str] = None
    profilePicUrl: Optional[str] = None
    isVerified: bool = False
    isPrivate: bool = False
    stats: ProfileStats
    extractedAt: str
    sourceUrl: str
    extractionMethod: Literal["hydrated_data", "dom_scraping"]
    rawHtml: Optional[str] = None


class ExtractedPost(BaseModel):
    """A post/video extracted from the profile grid"""
    id: str
    type: Literal["image", "video", "carousel", "reel"]
    thumbnailUrl: Optional[str] = None
    caption: Optional[str] = None
    likes: Optional[int] = None
    comments: Optional[int] = None
    shares: Optional[int] = None
    views: Optional[int] = None
    timestamp: Optional[str] = None


# ============================================================================
# Pydantic Schemas for Individual Content
# ============================================================================

class ContentAuthor(BaseModel):
    """Author information for individual content"""
    username: str
    displayName: Optional[str] = None
    profilePicUrl: Optional[str] = None
    isVerified: bool = False


class MediaItem(BaseModel):
    """Individual media item (image/video)"""
    type: Literal["image", "video"]
    url: Optional[str] = None
    thumbnailUrl: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    duration: Optional[float] = None
    altText: Optional[str] = None


class ContentMetrics(BaseModel):
    """Engagement metrics for content"""
    likes: Optional[int] = None
    comments: Optional[int] = None
    shares: Optional[int] = None
    saves: Optional[int] = None
    views: Optional[int] = None
    plays: Optional[int] = None


class AudioInfo(BaseModel):
    """Audio information for reels/videos"""
    title: Optional[str] = None
    artist: Optional[str] = None
    isOriginal: bool = False
    audioUrl: Optional[str] = None


class ContentData(BaseModel):
    """Individual content data (post, reel, video)"""
    platform: Literal["instagram", "tiktok", "unknown"]
    contentType: Literal["post", "reel", "video", "carousel", "story"]
    contentId: str
    contentUrl: str
    author: ContentAuthor
    media: List[MediaItem] = Field(default_factory=list)
    caption: Optional[str] = None
    hashtags: List[str] = Field(default_factory=list)
    mentions: List[str] = Field(default_factory=list)
    metrics: ContentMetrics
    audio: Optional[AudioInfo] = None
    postedAt: Optional[str] = None
    extractedAt: str
    sourceUrl: str
    extractionMethod: Literal["hydrated_data", "dom_scraping"]


# ============================================================================
# Unified Payload Schema
# ============================================================================

class IngestMetadata(BaseModel):
    """Metadata about the extraction"""
    extractionMethod: Optional[str] = None
    platform: Optional[str] = None
    sourceUrl: Optional[str] = None
    contentType: Optional[str] = None
    contentId: Optional[str] = None
    author: Optional[str] = None
    username: Optional[str] = None


class RawIngestPayload(BaseModel):
    """
    Unified payload for both content and profile ingestion

    Human-in-the-Loop:
    When isOwnProfile=True, the content belongs to the client's own account.
    This triggers the ML feedback loop to register real performance metrics.
    """
    source: str = Field(default="elena_bridge_extension", description="Source identifier")
    version: str = Field(default="1.0.0", description="Extension version")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    pageType: Optional[str] = None
    type: Optional[Literal["content", "profile", "unknown"]] = None

    # For individual content (posts, reels, videos)
    content: Optional[ContentData] = None

    # For profile analysis
    profile: Optional[ProfileData] = None
    recentPosts: List[ExtractedPost] = Field(default_factory=list)

    # Shared
    metadata: Optional[IngestMetadata] = None

    # Human-in-the-Loop: indicates content is from client's own account
    # When True, backend registers real metrics for ML feedback loop
    isOwnProfile: bool = Field(default=False, description="True if content is from own profile for feedback loop")

    # Light Mode: enables optimized multimodal processing (default: True)
    # Light mode: ~5s vs ~20s full processing, 75% time saved
    lightMode: bool = Field(default=True, description="Use light multimodal processing (recommended)")

    class Config:
        extra = "allow"


class IngestResponse(BaseModel):
    """Response after ingesting data"""
    success: bool
    message: str
    task_id: Optional[str] = None
    data_type: Optional[str] = None
    identifier: Optional[str] = None  # content_id or username
    items_count: int = 0


# ============================================================================
# Light Mode Multimodal Processing Helper
# ============================================================================

def _process_multimodal_light(
    content: ContentData,
    light_mode: bool = True
) -> Dict[str, Any]:
    """
    Process multimodal content with light optimizations.

    Args:
        content: ContentData with media URLs
        light_mode: Enable light processing (default: True)

    Returns:
        Dict with multimodal features (transcription, OCR, hook_score)
    """
    settings = get_settings()
    start_time = time.time()

    # Default result
    result = {
        "transcription": "",
        "ocr_text": "",
        "hook_score": 0.0,
        "light_mode": light_mode,
        "processing_time_seconds": 0.0,
        "time_saved_seconds": 0.0,
        "cached": False,
        "skipped": False,
    }

    # Check if light mode is enabled globally
    if not settings.LIGHT_MODE_ENABLED:
        light_mode = False

    # Get media URL/thumbnail
    media_url = None
    thumbnail_url = None

    for media in content.media:
        if media.type == "video" and media.url:
            media_url = media.url
        if media.thumbnailUrl:
            thumbnail_url = media.thumbnailUrl

    # Skip if no media
    if not media_url and not thumbnail_url:
        result["skipped"] = True
        result["skip_reason"] = "No media URL available"
        return result

    try:
        from ml.light_processors import get_light_processor, LightProcessingConfig

        # Create config from settings
        config = LightProcessingConfig(
            whisper_model=settings.LIGHT_WHISPER_MODEL,
            whisper_max_duration=settings.LIGHT_WHISPER_MAX_DURATION,
            ocr_max_frames=settings.LIGHT_OCR_MAX_FRAMES,
            ocr_use_thumbnail=settings.LIGHT_OCR_USE_THUMBNAIL,
            hook_duration_seconds=settings.LIGHT_HOOK_DURATION,
            cache_enabled=settings.LIGHT_CACHE_ENABLED,
            cache_ttl_hours=settings.LIGHT_CACHE_TTL_HOURS,
            skip_non_video=settings.LIGHT_SKIP_NON_VIDEO,
        )

        processor = get_light_processor(light_mode=light_mode, config=config)

        # Process with light mode
        light_result = processor.process(
            media_url=media_url,
            thumbnail_url=thumbnail_url,
            content_type=content.contentType,
            caption=content.caption or "",
        )

        result.update({
            "transcription": light_result.transcription,
            "ocr_text": light_result.ocr_text,
            "hook_score": light_result.hook_score,
            "text_density": light_result.text_density,
            "processing_time_seconds": light_result.processing_time_seconds,
            "time_saved_seconds": light_result.time_saved_seconds,
            "time_saved_percent": light_result.time_saved_percent,
            "cached": light_result.cached,
            "skipped": light_result.skipped,
            "skip_reason": light_result.skip_reason,
        })

        # Log time savings
        if light_result.time_saved_seconds > 0:
            logger.info(
                f"Multimodal light: {light_result.processing_time_seconds:.2f}s "
                f"vs ~{light_result.estimated_full_time_seconds:.1f}s full "
                f"({light_result.time_saved_percent:.0f}% saved)"
            )

    except ImportError:
        logger.warning("light_processors not available, skipping multimodal")
        result["skipped"] = True
        result["skip_reason"] = "light_processors module not available"
    except Exception as e:
        logger.error(f"Light multimodal processing failed: {e}")
        result["error"] = str(e)

    result["processing_time_seconds"] = time.time() - start_time
    return result


# ============================================================================
# Background Tasks for Processing
# ============================================================================

def _generate_content_hash(content: ContentData) -> str:
    """
    Generate a unique hash for content identification.
    Uses platform + contentId + author for uniqueness.
    """
    unique_str = f"{content.platform}:{content.contentId}:{content.author.username}"
    return hashlib.sha256(unique_str.encode()).hexdigest()[:16]


def _calculate_real_targets(content: ContentData) -> Dict[str, float]:
    """
    Calculate real target metrics from extracted content for ML feedback.

    Multi-objetivo targets:
    - engagement_rate: (likes + comments*2 + saves*3 + shares*4) / views (if available)
    - log_likes, log_comments, log_shares, log_saves, log_views: log1p transforms

    Returns dict compatible with ml_service.register_performance_feedback
    """
    metrics = content.metrics

    likes = metrics.likes or 0
    comments = metrics.comments or 0
    shares = metrics.shares or 0
    saves = metrics.saves or 0
    views = metrics.views or metrics.plays or 0

    # Log transforms (log1p to handle zeros)
    targets = {
        "likes": likes,
        "comments": comments,
        "shares": shares,
        "saves": saves,
        "views": views,
        "log_likes": math.log1p(likes),
        "log_comments": math.log1p(comments),
        "log_shares": math.log1p(shares),
        "log_saves": math.log1p(saves),
        "log_views": math.log1p(views),
    }

    # Calculate engagement rate
    if views > 0:
        # Weighted engagement rate
        weighted_engagement = likes + (comments * 2) + (saves * 3) + (shares * 4)
        targets["engagement_rate"] = (weighted_engagement / views) * 100
    else:
        # Fallback: use total interactions as proxy
        targets["engagement_rate"] = min(100, (likes + comments * 2 + saves * 3 + shares * 4) / 10)

    return targets


def _content_to_features_dict(content: ContentData) -> Dict[str, Any]:
    """
    Convert ContentData to a dict suitable for FeatureExtractor.

    Maps extension payload fields to ML feature extraction format.
    """
    # Determine content format
    content_format = "static"
    if content.contentType == "reel":
        content_format = "reel"
    elif content.contentType == "video":
        content_format = "tiktok_video"
    elif content.contentType == "carousel":
        content_format = "carousel"

    # Get video duration if available
    video_duration = 0
    for media in content.media:
        if media.type == "video" and media.duration:
            video_duration = media.duration
            break

    # Audio info
    audio_name = None
    if content.audio:
        audio_name = content.audio.title or content.audio.artist

    return {
        "caption": content.caption or "",
        "hashtags": content.hashtags,
        "mentions": content.mentions,
        "content_format": content_format,
        "type": content_format,
        "video_duration_seconds": video_duration,
        "video_duration": video_duration,
        "audio_name": audio_name,
        "recommended_audio": audio_name,
        "posted_at": content.postedAt,
        # Metrics for training (will be used as actual values)
        "likes_count": content.metrics.likes or 0,
        "likes": content.metrics.likes or 0,
        "comments_count": content.metrics.comments or 0,
        "comments": content.metrics.comments or 0,
        "shares_count": content.metrics.shares or 0,
        "shares": content.metrics.shares or 0,
        "saves_count": content.metrics.saves or 0,
        "saves": content.metrics.saves or 0,
        "views_count": content.metrics.views or content.metrics.plays or 0,
        "video_views": content.metrics.views or content.metrics.plays or 0,
        # Business type will be set by caller if known
        "business_type": "otros",
    }


async def process_content_data(
    task_id: str,
    content: ContentData,
    is_own_profile: bool = False,
    light_mode: bool = True
):
    """
    Background task to process individual content (post, reel, video).

    Human-in-the-Loop:
    When is_own_profile=True, registers real performance metrics for ML feedback loop.
    This enables the model to learn from actual post performance.

    Light Mode:
    When light_mode=True (default), uses optimized multimodal processing:
    - Whisper: tiny model, first 3s only
    - EasyOCR: 5 frames max or thumbnail
    - Cache: Skip already processed media
    """
    logger.info(
        f"[Task {task_id}] Processing {content.contentType} from {content.platform} "
        f"(light_mode={light_mode})"
    )

    try:
        # Log content details
        logger.info(
            f"[Task {task_id}] Content ID: {content.contentId}, "
            f"Author: @{content.author.username}"
        )

        # Log engagement metrics
        metrics = content.metrics
        logger.info(
            f"[Task {task_id}] Metrics - Likes: {metrics.likes}, "
            f"Comments: {metrics.comments}, Views: {metrics.views}, "
            f"Shares: {metrics.shares}, Saves: {metrics.saves}"
        )

        # Log hashtags
        if content.hashtags:
            logger.info(f"[Task {task_id}] Hashtags: {', '.join(content.hashtags[:10])}")

        # Log media info
        media_count = len(content.media)
        if media_count > 0:
            media_types = [m.type for m in content.media]
            logger.info(f"[Task {task_id}] Media items: {media_count} ({', '.join(media_types)})")

        # Log audio info for reels/videos
        if content.audio and content.audio.title:
            logger.info(
                f"[Task {task_id}] Audio: '{content.audio.title}' by {content.audio.artist or 'Unknown'}"
            )

        # =====================================================================
        # LIGHT MODE MULTIMODAL PROCESSING
        # =====================================================================
        multimodal_result = {}
        if content.contentType in ("reel", "video") or any(m.type == "video" for m in content.media):
            logger.info(
                f"[Task {task_id}] Processing multimodal content "
                f"({'light' if light_mode else 'full'} mode)"
            )
            multimodal_result = _process_multimodal_light(content, light_mode=light_mode)

            if multimodal_result.get("cached"):
                logger.info(
                    f"[Task {task_id}] Multimodal CACHED - instant retrieval"
                )
            elif multimodal_result.get("skipped"):
                logger.info(
                    f"[Task {task_id}] Multimodal SKIPPED: {multimodal_result.get('skip_reason')}"
                )
            else:
                logger.info(
                    f"[Task {task_id}] Multimodal processed in "
                    f"{multimodal_result.get('processing_time_seconds', 0):.2f}s "
                    f"(saved {multimodal_result.get('time_saved_percent', 0):.0f}%)"
                )

            # Log extracted content
            if multimodal_result.get("transcription"):
                word_count = len(multimodal_result["transcription"].split())
                logger.info(f"[Task {task_id}] Transcription: {word_count} words (hook)")

            if multimodal_result.get("ocr_text"):
                logger.info(
                    f"[Task {task_id}] OCR text: {len(multimodal_result['ocr_text'])} chars, "
                    f"density: {multimodal_result.get('text_density', 0):.2f}%"
                )

            if multimodal_result.get("hook_score", 0) > 0:
                logger.info(
                    f"[Task {task_id}] Hook score: {multimodal_result['hook_score']:.2f}"
                )

        # =====================================================================
        # HUMAN-IN-THE-LOOP: Register ML Feedback for Own Profile Content
        # =====================================================================
        if is_own_profile:
            logger.info(
                f"[Task {task_id}] 🎯 PERFIL PROPIO detectado - Registrando feedback ML"
            )

            try:
                # Generate unique content hash
                content_hash = _generate_content_hash(content)

                # Calculate real targets from metrics
                real_targets = _calculate_real_targets(content)

                # Convert content to features dict for ML
                content_dict = _content_to_features_dict(content)

                # Get ML predictor instance
                ml_predictor = get_ml_predictor()

                # Register feedback with ML service
                # Note: We use content_hash as ID since this is external content
                feedback = ml_predictor.register_performance_feedback(
                    content_id=int(content_hash, 16) % (10**9),  # Convert hash to numeric ID
                    metrics=real_targets,
                    original_content=content_dict,
                    predicted_score=None  # Will use default; actual vs predicted tracked over time
                )

                logger.info(
                    f"[Task {task_id}] ✅ Feedback real registrado de post propio – modelo aprenderá. "
                    f"Hash: {content_hash}, Engagement Rate: {real_targets['engagement_rate']:.2f}%, "
                    f"Delta: {feedback.delta_percent:.1f}%, High Priority: {feedback.is_high_priority}"
                )

                # Log training queue status
                stats = ml_predictor.get_feedback_statistics()
                logger.info(
                    f"[Task {task_id}] 📊 Queue ML: {stats['queue_size']} samples, "
                    f"{stats['high_priority_count']} high priority, "
                    f"Model bias: {stats['model_bias']}"
                )

                # Check if we should trigger online update (threshold > 10)
                if stats['queue_size'] >= 10:
                    logger.info(
                        f"[Task {task_id}] 🔄 Threshold alcanzado ({stats['queue_size']} >= 10) - "
                        f"Considerar retrain o online update en próximo ciclo"
                    )

            except Exception as e:
                logger.error(f"[Task {task_id}] Error registering ML feedback: {e}")
                # Don't fail the whole task, just log the error

        else:
            # Competitor content - ingest for baseline training
            logger.info(
                f"[Task {task_id}] Contenido de competidor - almacenado para baseline training"
            )

        logger.info(f"[Task {task_id}] Content processing completed successfully")

    except Exception as e:
        logger.error(f"[Task {task_id}] Error processing content: {e}")


async def process_profile_data(
    task_id: str,
    profile: ProfileData,
    posts: List[ExtractedPost],
    is_own_profile: bool = False
):
    """
    Background task to process profile data.

    Human-in-the-Loop:
    When is_own_profile=True, registers all recent posts for ML feedback loop.
    This is useful for bulk sync of the client's own content metrics.
    """
    logger.info(f"[Task {task_id}] Processing {profile.platform} profile: @{profile.username}")

    try:
        # Log profile stats
        logger.info(
            f"[Task {task_id}] Stats - Followers: {profile.stats.followers}, "
            f"Following: {profile.stats.following}, Posts: {profile.stats.posts}"
        )

        # Process recent posts
        posts_count = len(posts)
        if posts_count > 0:
            logger.info(f"[Task {task_id}] Processing {posts_count} recent posts")

            total_likes = sum(p.likes or 0 for p in posts)
            total_comments = sum(p.comments or 0 for p in posts)

            if posts_count > 0:
                avg_likes = total_likes / posts_count
                avg_comments = total_comments / posts_count
                logger.info(
                    f"[Task {task_id}] Avg engagement - Likes: {avg_likes:.0f}, "
                    f"Comments: {avg_comments:.0f}"
                )

        # =====================================================================
        # HUMAN-IN-THE-LOOP: Register ML Feedback for Own Profile Posts
        # =====================================================================
        if is_own_profile and posts_count > 0:
            logger.info(
                f"[Task {task_id}] 🎯 PERFIL PROPIO - Registrando {posts_count} posts para feedback ML"
            )

            try:
                ml_predictor = get_ml_predictor()
                feedback_count = 0
                high_priority_count = 0

                for post in posts:
                    # Generate unique hash for this post
                    unique_str = f"{profile.platform}:{post.id}:{profile.username}"
                    post_hash = hashlib.sha256(unique_str.encode()).hexdigest()[:16]

                    # Calculate real targets
                    likes = post.likes or 0
                    comments = post.comments or 0
                    shares = post.shares or 0
                    views = post.views or 0

                    real_targets = {
                        "likes": likes,
                        "comments": comments,
                        "shares": shares,
                        "saves": 0,  # Not available in grid view
                        "views": views,
                        "log_likes": math.log1p(likes),
                        "log_comments": math.log1p(comments),
                        "log_shares": math.log1p(shares),
                        "log_saves": 0,
                        "log_views": math.log1p(views),
                    }

                    # Calculate engagement rate
                    if views > 0:
                        weighted_engagement = likes + (comments * 2) + (shares * 4)
                        real_targets["engagement_rate"] = (weighted_engagement / views) * 100
                    else:
                        real_targets["engagement_rate"] = min(100, (likes + comments * 2 + shares * 4) / 10)

                    # Create minimal content dict
                    content_dict = {
                        "caption": post.caption or "",
                        "content_format": post.type if post.type in ["reel", "carousel"] else "static",
                        "type": post.type,
                        "posted_at": post.timestamp,
                        "likes": likes,
                        "comments": comments,
                        "shares": shares,
                        "views": views,
                        "business_type": "otros",
                    }

                    # Register feedback
                    try:
                        feedback = ml_predictor.register_performance_feedback(
                            content_id=int(post_hash, 16) % (10**9),
                            metrics=real_targets,
                            original_content=content_dict,
                            predicted_score=None
                        )
                        feedback_count += 1
                        if feedback.is_high_priority:
                            high_priority_count += 1
                    except Exception as e:
                        logger.warning(f"[Task {task_id}] Error registering feedback for post {post.id}: {e}")

                logger.info(
                    f"[Task {task_id}] ✅ Feedback registrado para {feedback_count}/{posts_count} posts, "
                    f"{high_priority_count} high priority"
                )

                # Log training queue status
                stats = ml_predictor.get_feedback_statistics()
                logger.info(
                    f"[Task {task_id}] 📊 Queue ML total: {stats['queue_size']} samples"
                )

            except Exception as e:
                logger.error(f"[Task {task_id}] Error en bulk feedback registration: {e}")

        elif not is_own_profile:
            # Competitor profile - store for baseline analysis
            logger.info(
                f"[Task {task_id}] Perfil de competidor - almacenado para análisis baseline"
            )

        logger.info(f"[Task {task_id}] Profile processing completed successfully")

    except Exception as e:
        logger.error(f"[Task {task_id}] Error processing profile: {e}")


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/raw", response_model=IngestResponse)
async def ingest_raw_data(
    payload: RawIngestPayload,
    background_tasks: BackgroundTasks,
    x_elena_bridge_version: Optional[str] = Header(None),
    x_extension_id: Optional[str] = Header(None),
    x_content_type: Optional[str] = Header(None),
):
    """
    Ingest raw data from Elena Bridge Chrome extension.

    Supports two types of data:
    - **content**: Individual posts, reels, or videos selected by the user
    - **profile**: Full profile analysis with recent posts

    The extraction is "stealth" - it only reads what's visible in the DOM,
    no API calls are made to the social networks.
    """
    logger.info(
        f"Received ingest request from extension v{x_elena_bridge_version or 'unknown'} "
        f"(type: {payload.type or x_content_type or 'unknown'})"
    )

    # Generate task ID for tracking
    task_id = str(uuid.uuid4())[:8]

    # Handle individual content (posts, reels, videos)
    if payload.content:
        content = payload.content
        is_own = payload.isOwnProfile

        light_mode = payload.lightMode
        logger.info(
            f"[{task_id}] Ingesting {content.contentType} from @{content.author.username} "
            f"({content.platform}, method: {content.extractionMethod})"
            f"{' [PERFIL PROPIO - Feedback Loop]' if is_own else ''}"
            f" [{'LIGHT' if light_mode else 'FULL'} mode]"
        )

        # Queue background processing with own profile flag and light mode
        background_tasks.add_task(process_content_data, task_id, content, is_own, light_mode)

        return IngestResponse(
            success=True,
            message=f"{content.contentType.capitalize()} from @{content.author.username} queued for processing"
                    + (" (feedback loop activado)" if is_own else "")
                    + (f" [modo {'ligero' if light_mode else 'completo'}]"),
            task_id=task_id,
            data_type=content.contentType,
            identifier=content.contentId,
            items_count=len(content.media)
        )

    # Handle profile data
    if payload.profile:
        profile = payload.profile
        is_own = payload.isOwnProfile

        logger.info(
            f"[{task_id}] Ingesting profile @{profile.username} from {profile.platform} "
            f"(method: {profile.extractionMethod})"
            f"{' [PERFIL PROPIO - Bulk Feedback Loop]' if is_own else ''}"
        )

        # Queue background processing with own profile flag
        background_tasks.add_task(
            process_profile_data, task_id, profile, payload.recentPosts, is_own
        )

        return IngestResponse(
            success=True,
            message=f"Profile @{profile.username} queued for processing"
                    + (f" ({len(payload.recentPosts)} posts para feedback)" if is_own else ""),
            task_id=task_id,
            data_type="profile",
            identifier=profile.username,
            items_count=len(payload.recentPosts)
        )

    # No valid data found
    raise HTTPException(
        status_code=400,
        detail="No content or profile data in payload. Make sure you're on a supported page."
    )


@router.get("/status")
async def ingest_status():
    """
    Check the status of the ingest endpoint.
    Used by the extension to verify connectivity.
    """
    return {
        "status": "online",
        "endpoint": "/api/ingest/raw",
        "supported_platforms": ["instagram", "tiktok"],
        "supported_content_types": ["post", "reel", "video", "carousel", "profile"],
        "version": "1.0.0"
    }
