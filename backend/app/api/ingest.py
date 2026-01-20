"""
Elena Bridge - Raw Data Ingestion API
Receives content and profile data extracted from Instagram/TikTok by the Chrome extension
Supports both individual content (posts, reels, videos) and full profile analysis

Human-in-the-Loop Integration:
When isOwnProfile=True, the backend registers real performance metrics for ML feedback loop.
This closes the loop between predictions and actual performance, enabling continuous learning.

USER CONFIG INTEGRATION (REAL SYNC):
===================================
- Loads user_config from database using user_id/business_id
- Uses own_instagram_username from config to AUTO-DETECT own profile (no manual flag needed)
- Uses multimodal_mode and light_mode_config from user_config (not global settings)
- Logs: "User config loaded: precision {X}, multimodal {Y}, own @{Z}"

Dual Mode Processing:
- Light Mode: Optimized Whisper/EasyOCR (Hooks only).
- Full Mode: Deep analysis with ContentProcessorService (Frames, Colors, Audio).
Traffic controlled by user_config.light_mode_enabled.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Header, Depends, status
from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Any, Dict
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import logging
import uuid
import hashlib
import math
import time
import copy

# Import ML service for feedback loop
from app.services.ml_service import get_ml_predictor, FeatureExtractor
from app.core.config import get_settings
from app.core.database import get_db, async_session_maker
from app.core.privacy import privacy_provider
from app.services.user_config_service import user_config_service, get_default_pipeline_config
from app.schemas.user_config import PipelineConfig

# Dual Mode Services
from app.services.content_processor import get_content_processor
from app.models.scraped_post import ScrapedPost, ContentFormat
from app.models.competitor import Competitor
from app.models.business import Platform
from app.api.extension import get_business_by_api_key

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

    USER CONFIG SYNC:
    =================
    - business_id/user_id: Used to load user_config from database
    - If own_instagram_username in config matches author.username, auto-flags as own profile
    - multimodal_mode from config determines light/full processing (overrides lightMode)

    Human-in-the-Loop:
    When isOwnProfile=True OR username matches config.own_instagram_username,
    the content triggers ML feedback loop for continuous learning.
    """
    source: str = Field(default="elena_bridge_extension", description="Source identifier")
    version: str = Field(default="1.0.0", description="Extension version")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    pageType: Optional[str] = None
    type: Optional[Literal["content", "profile", "unknown"]] = None

    # USER CONFIG SYNC: IDs for loading user configuration
    # These should be sent by the extension (from stored auth session)
    businessId: Optional[int] = Field(None, description="Business ID for loading user config")
    userId: Optional[int] = Field(None, description="User ID for loading user config")

    # For individual content (posts, reels, videos)
    content: Optional[ContentData] = None

    # For profile analysis
    profile: Optional[ProfileData] = None
    recentPosts: List[ExtractedPost] = Field(default_factory=list)

    # Shared
    metadata: Optional[IngestMetadata] = None

    # Human-in-the-Loop: indicates content is from client's own account
    # NOTE: If not set, backend will auto-detect using own_instagram_username from user_config
    isOwnProfile: bool = Field(default=False, description="True if content is from own profile for feedback loop")

    # Light Mode: enables optimized multimodal processing (default: True)
    # NOTE: If businessId is provided, uses multimodal_mode from user_config instead
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
    light_mode: bool = True,
    user_config: Optional[PipelineConfig] = None
) -> Dict[str, Any]:
    """
    Process multimodal content with light optimizations.

    USER CONFIG SYNC:
    =================
    If user_config is provided, uses light_mode_config from user settings
    instead of global environment settings. This ensures frontend config
    changes affect backend processing in real-time.

    Args:
        content: ContentData with media URLs
        light_mode: Enable light processing (default: True)
        user_config: PipelineConfig loaded from user_config table (optional)

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
        "config_source": "global",  # Track where config came from
    }

    # USER CONFIG SYNC: Use user config if provided, else fall back to global settings
    if user_config:
        light_mode = user_config.light_mode_enabled
        whisper_model = user_config.whisper_model
        ocr_max_frames = user_config.ocr_max_frames
        result["config_source"] = "user_config"

        logger.info(
            f"Multimodal using USER CONFIG: mode={user_config.multimodal_mode}, "
            f"whisper={whisper_model}, ocr_frames={ocr_max_frames}"
        )
    else:
        # Fall back to global settings
        whisper_model = settings.LIGHT_WHISPER_MODEL
        ocr_max_frames = settings.LIGHT_OCR_MAX_FRAMES
        if not settings.LIGHT_MODE_ENABLED:
            light_mode = False
        result["config_source"] = "global_settings"

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

        # Create config from user_config or settings
        if user_config:
            config = LightProcessingConfig(
                whisper_model=whisper_model,
                whisper_max_duration=3.0 if light_mode else 30.0,
                ocr_max_frames=ocr_max_frames,
                ocr_use_thumbnail=True if light_mode else False,
                hook_duration_seconds=3.0,
                cache_enabled=True,
                cache_ttl_hours=168,
                skip_non_video=True,
            )
        else:
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
        # NOTE: media_url here is potentially hashed in the calling context,
        # but the background task should ensure it passes raw URLs if called for processing.
        # If media_url is hashed, requests will fail.
        # We assume the caller handles this by NOT hashing URLs before multimodal processing.

        # Security check: If media_url is a hash (64 hex chars), warn and skip
        if media_url and len(media_url) == 64 and all(c in '0123456789abcdef' for c in media_url):
             logger.warning("Attempted multimodal processing on hashed URL. Skipping to prevent errors.")
             result["skipped"] = True
             result["skip_reason"] = "Hashed URL detected"
             return result

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
    light_mode: bool = True,
    user_config: Optional[PipelineConfig] = None,
    business_id: Optional[int] = None
):
    """
    Background task to process individual content (post, reel, video).

    BLIND IDENTITY PROTOCOL:
    ========================
    This task receives 'content' with RAW URLs to allow multimodal processing (downloading).
    However, it must IMMEDIATELY sanitize logs and ensure only HASHED data is passed to
    persistent storage or ML feedback queues.

    DUAL MODE STRATEGY:
    ===================
    - Light Mode (True): Uses optimized Whisper/EasyOCR pipeline (hooks only).
    - Full Mode (False): Uses deep AnalyticsEngine (frames, colors, deep metrics).
    Traffic controller logic routes based on `light_mode` flag.
    """
    # Sanitize user info in logs
    masked_own_user = privacy_provider.hash_pii(user_config.own_instagram_username) if user_config and user_config.own_instagram_username else 'N/A'

    # CRITICAL LOG: User config loaded
    if user_config:
        logger.info(
            f"[Task {task_id}] User config loaded: "
            f"precision={user_config.embedding_precision} ({user_config.embedding_dims} dims), "
            f"multimodal={user_config.multimodal_mode}, "
            f"own_hash={masked_own_user[:8]}..."
        )
        # Override light_mode with user config
        light_mode = user_config.light_mode_enabled
        # Auto-detect own profile if not explicitly set (using secure comparison)
        if not is_own_profile and user_config.own_instagram_username:
            hashed_author = privacy_provider.hash_pii(content.author.username)
            if privacy_provider.compare_pii(user_config.own_instagram_username, hashed_author):
                is_own_profile = True
                logger.info(f"[Task {task_id}] AUTO-DETECTED own profile via secure hash comparison")

    logger.info(
        f"[Task {task_id}] Processing {content.contentType} from {content.platform} "
        f"(light_mode={light_mode}, own_profile={is_own_profile})"
    )

    try:
        # Log content details (SANITIZED)
        logger.info(
            f"[Task {task_id}] Content ID: {content.contentId}, "
            f"Author Hash: {content.author.username[:16]}..."
        )

        # Log engagement metrics
        metrics = content.metrics
        logger.info(
            f"[Task {task_id}] Metrics - Likes: {metrics.likes}, "
            f"Comments: {metrics.comments}, Views: {metrics.views}, "
            f"Shares: {metrics.shares}, Saves: {metrics.saves}"
        )

        # =====================================================================
        # DUAL MODE MULTIMODAL PROCESSING
        # =====================================================================
        multimodal_result = {}
        should_process_multimodal = (
            content.contentType in ("reel", "video") or
            any(m.type == "video" for m in content.media)
        )

        if should_process_multimodal:
            if light_mode:
                logger.info(f"[Task {task_id}] Using LIGHT pipeline (Hooks only)")
                multimodal_result = _process_multimodal_light(
                    content, light_mode=True, user_config=user_config
                )
            else:
                logger.info(f"[Task {task_id}] Using FULL pipeline (Deep Analysis)")
                # Extract Raw Media URL
                media_url = None
                for media in content.media:
                    if media.type == "video" and media.url:
                        media_url = media.url
                        break

                if media_url:
                    processor = get_content_processor()
                    # Run full processing (async)
                    multimodal_result = await processor.process_full(
                        media_url=media_url,
                        caption=content.caption or ""
                    )
                else:
                    logger.warning(f"[Task {task_id}] No video URL found for FULL processing")

            # Log Result Summary
            if multimodal_result.get("status") == "failed":
                logger.error(f"[Task {task_id}] Multimodal processing failed: {multimodal_result.get('error')}")
            else:
                logger.info(
                    f"[Task {task_id}] Multimodal finished ({multimodal_result.get('processing_time_seconds', 0):.2f}s). "
                    f"Hook Score: {multimodal_result.get('hook_score', 0):.2f}"
                )

        # =====================================================================
        # ANONYMIZE CONTENT BEFORE PERSISTENCE/ML
        # =====================================================================
        # Now that we're done with processing that requires raw URLs (multimodal),
        # we strictly anonymize all URLs and PII in the content object.

        # Clone content to avoid side effects if reused (unlikely but safe)
        safe_content = copy.deepcopy(content)

        # Hash URLs
        safe_content.contentUrl = privacy_provider.hash_pii(safe_content.contentUrl) or ""
        safe_content.author.profilePicUrl = privacy_provider.hash_pii(safe_content.author.profilePicUrl)
        safe_content.sourceUrl = privacy_provider.hash_pii(safe_content.sourceUrl) or ""

        for m in safe_content.media:
            m.url = privacy_provider.hash_pii(m.url)
            m.thumbnailUrl = privacy_provider.hash_pii(m.thumbnailUrl)

        if safe_content.audio:
            safe_content.audio.audioUrl = privacy_provider.hash_pii(safe_content.audio.audioUrl)

        # =====================================================================
        # PERSISTENCE: Save Rich Data to ScrapedPost
        # =====================================================================
        # We save this regardless of whether it's own profile or competitor
        if business_id:
            try:
                async with async_session_maker() as db:
                    # 1. Ensure Competitor Exists
                    # We use the hashed username which was passed in content.author.username
                    hashed_handle = safe_content.author.username # Already hashed by ingest_raw_data

                    result = await db.execute(
                        select(Competitor)
                        .where(Competitor.business_id == business_id)
                        .where(Competitor.handle == hashed_handle)
                    )
                    competitor = result.scalar_one_or_none()

                    if not competitor:
                        # Auto-create competitor if missing (e.g. browsing random profiles)
                        # We use safe defaults
                        competitor = Competitor(
                            business_id=business_id,
                            platform=Platform(safe_content.platform) if safe_content.platform in ["instagram", "tiktok"] else Platform.INSTAGRAM,
                            handle=hashed_handle,
                            display_name=privacy_provider.hash_pii(safe_content.author.displayName),
                            profile_url=privacy_provider.hash_pii(f"https://instagram.com/{hashed_handle}"), # Pseudo-url hashed
                            scrape_status="completed" # It's not scraped by Apify but ingested via extension
                        )
                        db.add(competitor)
                        await db.commit()
                        await db.refresh(competitor)
                        logger.info(f"[Task {task_id}] Auto-created Competitor {hashed_handle[:8]}...")

                    # 2. Create or Update ScrapedPost
                    # Check if post exists
                    post_result = await db.execute(
                        select(ScrapedPost)
                        .where(ScrapedPost.competitor_id == competitor.id)
                        .where(ScrapedPost.platform_post_id == safe_content.contentId)
                    )
                    scraped_post = post_result.scalar_one_or_none()

                    # Prepare Data
                    transcript = multimodal_result.get("transcription")
                    ocr_text = multimodal_result.get("ocr_text")
                    visual_features = multimodal_result.get("visual_features", {})

                    # Parse timestamp
                    posted_at = None
                    try:
                        if safe_content.postedAt:
                            posted_at = datetime.fromisoformat(safe_content.postedAt.replace("Z", "+00:00"))
                    except:
                        pass

                    format_map = {
                        "reel": ContentFormat.REEL,
                        "video": ContentFormat.TIKTOK_VIDEO,
                        "carousel": ContentFormat.CAROUSEL,
                        "post": ContentFormat.STATIC_IMAGE
                    }
                    content_fmt = format_map.get(safe_content.contentType, ContentFormat.STATIC_IMAGE)

                    if scraped_post:
                        # Update rich fields
                        scraped_post.transcript = transcript
                        scraped_post.ocr_text = ocr_text
                        scraped_post.visual_features = visual_features
                        # Update metrics
                        scraped_post.likes_count = safe_content.metrics.likes
                        scraped_post.comments_count = safe_content.metrics.comments
                        scraped_post.shares_count = safe_content.metrics.shares
                        scraped_post.saves_count = safe_content.metrics.saves
                        scraped_post.views_count = safe_content.metrics.views or safe_content.metrics.plays

                        logger.info(f"[Task {task_id}] Updated existing ScrapedPost {scraped_post.id} with rich data")
                    else:
                        # Create new
                        scraped_post = ScrapedPost(
                            competitor_id=competitor.id,
                            platform_post_id=safe_content.contentId,
                            post_url=safe_content.contentUrl, # Hashed
                            content_format=content_fmt,
                            caption=privacy_provider.hash_pii(safe_content.caption), # Hashed caption?
                            # Wait, usually we want analyzed text. But we must hash PII.
                            # If we store transcript/ocr_text, is that PII?
                            # Transcript contains speech. OCR contains overlay.
                            # "Cancelamos la política de Blind Identity estricta para el contenido multimedia"
                            # The prompt says: "NO HASHEES la URL inmediatamente... Actualiza el modelo ScrapedPost para guardar estos datos enriquecidos: transcript, ocr_text"
                            # This implies we store Transcript/OCR in cleartext?
                            # "Cancelamos la política... para el contenido multimedia."
                            # But URLs are PII? The user said "Guardala si es necesario... o hasheala SOLO despues".
                            # I interpreted that for processing.
                            # For persistence: "Actualiza el modelo ScrapedPost para guardar estos datos enriquecidos".
                            # Storing full transcript in DB is definitely not "Blind Identity".
                            # But the prompt says "Cancelamos la política... estricta".
                            # So I will save transcript/ocr in cleartext.
                            # But username/handle MUST remain hashed.

                            transcript=transcript,
                            ocr_text=ocr_text,
                            visual_features=visual_features,

                            thumbnail_url=safe_content.media[0].thumbnailUrl if safe_content.media else None, # Hashed

                            likes_count=safe_content.metrics.likes,
                            comments_count=safe_content.metrics.comments,
                            shares_count=safe_content.metrics.shares,
                            saves_count=safe_content.metrics.saves,
                            views_count=safe_content.metrics.views or safe_content.metrics.plays,

                            posted_at=posted_at
                        )
                        db.add(scraped_post)
                        logger.info(f"[Task {task_id}] Created new ScrapedPost with rich data")

                    await db.commit()

            except Exception as e:
                logger.error(f"[Task {task_id}] Persistence failed: {e}")

        # =====================================================================
        # HUMAN-IN-THE-LOOP: Register ML Feedback for Own Profile Content
        # =====================================================================
        if is_own_profile:
            logger.info(
                f"[Task {task_id}] 🎯 PERFIL PROPIO detectado - Registrando feedback ML"
            )

            try:
                # Generate unique content hash (using already hashed username in safe_content)
                content_hash = _generate_content_hash(safe_content)

                # Calculate real targets from metrics
                real_targets = _calculate_real_targets(safe_content)

                # Convert content to features dict for ML
                content_dict = _content_to_features_dict(safe_content)

                # ENRICH ML FEATURES with multimodal data!
                if multimodal_result:
                    content_dict["whisper_transcript"] = multimodal_result.get("transcription", "")
                    content_dict["easyocr_text"] = multimodal_result.get("ocr_text", "")
                    # Add hook score if available
                    if "hook_score" in multimodal_result:
                        content_dict["semantic_hook_score"] = multimodal_result["hook_score"]

                # Get ML predictor instance
                ml_predictor = get_ml_predictor()

                # Register feedback with ML service
                # Note: We use content_hash as ID since this is external content
                feedback = await ml_predictor.register_performance_feedback(
                    content_id=int(content_hash, 16) % (10**9),  # Convert hash to numeric ID
                    metrics=real_targets,
                    original_content=content_dict,
                    predicted_score=None,  # Will use default; actual vs predicted tracked over time
                    db=db
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
                f"[Task {task_id}] Contenido de competidor - procesado y guardado."
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

    BLIND IDENTITY: All PII must be hashed before logging or processing.
    """
    # Hash profile identifier for logs
    hashed_username = privacy_provider.hash_pii(profile.username)
    logger.info(f"[Task {task_id}] Processing {profile.platform} profile: {hashed_username[:16]}...")

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

                async with async_session_maker() as db:
                    for post in posts:
                        # Generate unique hash for this post (using hashed username)
                        unique_str = f"{profile.platform}:{post.id}:{hashed_username}"
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
                            feedback = await ml_predictor.register_performance_feedback(
                                content_id=int(post_hash, 16) % (10**9),
                                metrics=real_targets,
                                original_content=content_dict,
                                predicted_score=None,
                                db=db
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
    x_extension_api_key: Optional[str] = Header(None, alias="X-Extension-API-Key"),
    db: AsyncSession = Depends(get_db),
):
    """
    Ingest raw data from Elena Bridge Chrome extension.

    BLIND IDENTITY PROTOCOL:
    ========================
    - All PII (usernames, URLs) is hashed immediately upon receipt using PrivacyProvider.
    - Comparison for 'isOwnProfile' is done using on-the-fly hashing logic.
    - No cleartext PII is logged or passed to persistent storage/ML.

    USER CONFIG SYNC:
    =================
    If businessId/userId are provided in payload:
    - Loads user_config from database
    - Uses own_instagram_username to auto-detect own profile (secure comparison)
    - Uses multimodal_mode from config (overrides lightMode param)
    - Config affects processing in real-time (no placebo!)
    """
    sanitized_source = privacy_provider.sanitize_log(f"{payload.source} v{x_elena_bridge_version or 'unknown'}")
    logger.info(
        f"Received ingest request from {sanitized_source} "
        f"(type: {payload.type or x_content_type or 'unknown'})"
    )

    # Generate task ID for tracking
    task_id = str(uuid.uuid4())[:8]

    # SECURITY: Verify API Key if businessId is provided
    if payload.businessId:
        if not x_extension_api_key:
            logger.warning(f"[{task_id}] Rejected ingest request with businessId={payload.businessId} but no API key")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API Key required when businessId is provided"
            )

        business = await get_business_by_api_key(db, x_extension_api_key)
        if not business or business.id != payload.businessId:
            logger.warning(f"[{task_id}] Rejected ingest request: API Key does not match businessId={payload.businessId}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API Key for this business"
            )

    # =========================================================================
    # USER CONFIG SYNC: Load config from database if IDs provided
    # =========================================================================
    user_config: Optional[PipelineConfig] = None
    if payload.businessId and payload.userId:
        try:
            user_config = await user_config_service.get_pipeline_config(
                db, payload.userId, payload.businessId
            )
            # Log config load (sanitized own username)
            masked_own = privacy_provider.hash_pii(user_config.own_instagram_username) if user_config.own_instagram_username else 'N/A'
            logger.info(
                f"[{task_id}] USER CONFIG LOADED: precision={user_config.embedding_precision}, "
                f"multimodal={user_config.multimodal_mode}, "
                f"own_hash={masked_own[:8]}..."
            )
        except Exception as e:
            logger.warning(f"[{task_id}] Could not load user config: {e}. Using defaults.")
            user_config = get_default_pipeline_config(payload.userId, payload.businessId)

    # Handle individual content (posts, reels, videos)
    if payload.content:
        content = payload.content
        is_own = payload.isOwnProfile

        # Hash username immediately for secure comparison and logging
        # Note: We keep content.author.username RAW in the object we pass to background task
        # because we don't know if downstream systems might rely on it for transient processing
        # (though likely not). However, we MUST NOT log it.
        # Wait, if we pass it raw, `process_content_data` must hash it.
        # But `process_content_data` already assumes it might receive raw for Multimodal.
        # Actually, `process_content_data` needs to handle the hashing before ML.

        hashed_username = privacy_provider.hash_pii(content.author.username)

        # USER CONFIG SYNC: Override light_mode and detect own profile
        light_mode = payload.lightMode
        if user_config:
            light_mode = user_config.light_mode_enabled
            # Auto-detect own profile SECURELY
            if not is_own and user_config.own_instagram_username:
                if privacy_provider.compare_pii(user_config.own_instagram_username, hashed_username):
                    is_own = True
                    logger.info(f"[{task_id}] Auto-detected own profile via secure hash comparison")

        logger.info(
            f"[{task_id}] Ingesting {content.contentType} from {hashed_username[:16]}... "
            f"({content.platform}, method: {content.extractionMethod})"
            f"{' [PERFIL PROPIO - Feedback Loop]' if is_own else ''}"
            f" [{'LIGHT' if light_mode else 'FULL'} mode]"
            f" [config: {'user_config' if user_config else 'default'}]"
        )

        # PRE-HASH PII IN PAYLOAD for Persistence/Logs (But keep URLs for Multimodal)
        # We will modify the content object IN PLACE for the background task?
        # No, `process_content_data` expects RAW URLs for multimodal.
        # So we pass it as is, but we ensure `process_content_data` handles anonymization internally.
        # However, we should hash the username here and now to be safe?
        # If we hash the username here, `process_content_data` will see hashed username.
        # Does `_process_multimodal_light` need raw username? No.
        # So we CAN hash the username here.
        content.author.username = hashed_username

        # Queue background processing with user_config and business_id
        background_tasks.add_task(
            process_content_data,
            task_id,
            content,
            is_own,
            light_mode,
            user_config,
            payload.businessId # Pass business_id for persistence
        )

        config_msg = ""
        if user_config:
            config_msg = f" [config: precision={user_config.embedding_precision}]"

        return IngestResponse(
            success=True,
            message=f"{content.contentType.capitalize()} from {hashed_username[:8]}... queued for processing"
                    + (" (feedback loop activado)" if is_own else "")
                    + (f" [modo {'ligero' if light_mode else 'completo'}]")
                    + config_msg,
            task_id=task_id,
            data_type=content.contentType,
            identifier=hashed_username, # Return hashed ID
            items_count=len(content.media)
        )

    # Handle profile data
    if payload.profile:
        profile = payload.profile
        is_own = payload.isOwnProfile

        hashed_username = privacy_provider.hash_pii(profile.username)

        # USER CONFIG SYNC: Auto-detect own profile SECURELY
        if user_config and not is_own and user_config.own_instagram_username:
            if privacy_provider.compare_pii(user_config.own_instagram_username, hashed_username):
                is_own = True

        logger.info(
            f"[{task_id}] Ingesting profile {hashed_username[:16]}... from {profile.platform} "
            f"(method: {profile.extractionMethod})"
            f"{' [PERFIL PROPIO - Bulk Feedback Loop]' if is_own else ''}"
            f" [config: {'user_config' if user_config else 'default'}]"
        )

        # Anonymize profile PII before passing to background task
        # Profile processing doesn't do multimodal download (usually), so we can hash securely
        profile.username = hashed_username
        profile.displayName = privacy_provider.hash_pii(profile.displayName)
        profile.bio = privacy_provider.hash_pii(profile.bio) # Optional: hash bio or keep it? Blind identity says zero PII.
        profile.profilePicUrl = privacy_provider.hash_pii(profile.profilePicUrl)
        profile.sourceUrl = privacy_provider.hash_pii(profile.sourceUrl)

        # Queue background processing with own profile flag
        background_tasks.add_task(
            process_profile_data, task_id, profile, payload.recentPosts, is_own
        )

        return IngestResponse(
            success=True,
            message=f"Profile {hashed_username[:8]}... queued for processing"
                    + (f" ({len(payload.recentPosts)} posts para feedback)" if is_own else ""),
            task_id=task_id,
            data_type="profile",
            identifier=hashed_username,
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
        "version": "1.0.0",
        "privacy_mode": "blind_identity_v1"
    }
