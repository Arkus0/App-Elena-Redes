"""
Elena Bridge - Raw Data Ingestion API
Receives content and profile data extracted from Instagram/TikTok by the Chrome extension
Supports both individual content (posts, reels, videos) and full profile analysis
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Header
from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Any, Dict
from datetime import datetime
import logging
import uuid

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
# Background Tasks for Processing
# ============================================================================

async def process_content_data(task_id: str, content: ContentData):
    """
    Background task to process individual content (post, reel, video).
    """
    logger.info(f"[Task {task_id}] Processing {content.contentType} from {content.platform}")

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
            f"Shares: {metrics.shares}"
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

        # TODO: Integrate with AnalyticsEngine for deeper analysis
        # Ideas:
        # - Analyze caption sentiment
        # - Detect content patterns (hooks, CTAs)
        # - Compare metrics to account averages
        # - Store for trend analysis

        logger.info(f"[Task {task_id}] Content processing completed successfully")

    except Exception as e:
        logger.error(f"[Task {task_id}] Error processing content: {e}")


async def process_profile_data(task_id: str, profile: ProfileData, posts: List[ExtractedPost]):
    """
    Background task to process profile data.
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

        # TODO: Integrate with AnalyticsEngine
        # Ideas:
        # - Calculate engagement rate
        # - Analyze bio for keywords
        # - Detect growth patterns
        # - Store competitor data

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
        logger.info(
            f"[{task_id}] Ingesting {content.contentType} from @{content.author.username} "
            f"({content.platform}, method: {content.extractionMethod})"
        )

        # Queue background processing
        background_tasks.add_task(process_content_data, task_id, content)

        return IngestResponse(
            success=True,
            message=f"{content.contentType.capitalize()} from @{content.author.username} queued for processing",
            task_id=task_id,
            data_type=content.contentType,
            identifier=content.contentId,
            items_count=len(content.media)
        )

    # Handle profile data
    if payload.profile:
        profile = payload.profile
        logger.info(
            f"[{task_id}] Ingesting profile @{profile.username} from {profile.platform} "
            f"(method: {profile.extractionMethod})"
        )

        # Queue background processing
        background_tasks.add_task(
            process_profile_data, task_id, profile, payload.recentPosts
        )

        return IngestResponse(
            success=True,
            message=f"Profile @{profile.username} queued for processing",
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
