"""
Elena Bridge - Raw Data Ingestion API
Receives profile data extracted from Instagram/TikTok by the Chrome extension
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Header
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime
import logging
import uuid

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Pydantic Schemas for Incoming Data
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
    """A post/video extracted from the profile"""
    id: str
    type: Literal["image", "video", "carousel", "reel"]
    thumbnailUrl: Optional[str] = None
    caption: Optional[str] = None
    likes: Optional[int] = None
    comments: Optional[int] = None
    shares: Optional[int] = None
    views: Optional[int] = None
    timestamp: Optional[str] = None


class IngestMetadata(BaseModel):
    """Metadata about the extraction"""
    extractionMethod: Optional[str] = None
    platform: Optional[str] = None
    sourceUrl: Optional[str] = None


class RawIngestPayload(BaseModel):
    """
    Full payload received from Elena Bridge extension
    """
    source: str = Field(default="elena_bridge_extension", description="Source identifier")
    version: str = Field(default="1.0.0", description="Extension version")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    profile: Optional[ProfileData] = None
    recentPosts: List[ExtractedPost] = Field(default_factory=list)
    metadata: Optional[IngestMetadata] = None

    class Config:
        # Allow extra fields for forward compatibility
        extra = "allow"


class IngestResponse(BaseModel):
    """Response after ingesting data"""
    success: bool
    message: str
    task_id: Optional[str] = None
    profile_username: Optional[str] = None
    posts_count: int = 0


# ============================================================================
# Background Task for Processing
# ============================================================================

async def process_ingested_data(task_id: str, payload: RawIngestPayload):
    """
    Background task to process ingested profile data.
    This is where you'd integrate with AnalyticsEngine for deeper analysis.
    """
    logger.info(f"[Task {task_id}] Starting background processing")

    try:
        profile = payload.profile
        if not profile:
            logger.warning(f"[Task {task_id}] No profile data to process")
            return

        # Log extraction details
        logger.info(
            f"[Task {task_id}] Processing {profile.platform} profile: @{profile.username}"
        )
        logger.info(
            f"[Task {task_id}] Stats - Followers: {profile.stats.followers}, "
            f"Posts: {profile.stats.posts}, Method: {profile.extractionMethod}"
        )

        # Process recent posts
        posts_count = len(payload.recentPosts)
        if posts_count > 0:
            logger.info(f"[Task {task_id}] Processing {posts_count} recent posts")

            # Calculate average engagement if we have data
            total_likes = sum(p.likes or 0 for p in payload.recentPosts)
            total_comments = sum(p.comments or 0 for p in payload.recentPosts)

            if posts_count > 0:
                avg_likes = total_likes / posts_count
                avg_comments = total_comments / posts_count
                logger.info(
                    f"[Task {task_id}] Avg engagement - Likes: {avg_likes:.0f}, "
                    f"Comments: {avg_comments:.0f}"
                )

        # TODO: Integrate with AnalyticsEngine for deeper analysis
        # Example:
        # from app.services.analytics_engine import AnalyticsEngine
        # engine = AnalyticsEngine()
        # await engine.analyze_profile(profile, payload.recentPosts)

        logger.info(f"[Task {task_id}] Background processing completed successfully")

    except Exception as e:
        logger.error(f"[Task {task_id}] Error in background processing: {e}")


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/raw", response_model=IngestResponse)
async def ingest_raw_data(
    payload: RawIngestPayload,
    background_tasks: BackgroundTasks,
    x_elena_bridge_version: Optional[str] = Header(None),
    x_extension_id: Optional[str] = Header(None),
):
    """
    Ingest raw profile data from Elena Bridge Chrome extension.

    This endpoint receives DOM-extracted data from Instagram/TikTok profiles
    and queues it for asynchronous processing.

    The extraction is "stealth" - it only reads what's visible in the DOM,
    no API calls are made to the social networks.
    """
    logger.info(
        f"Received ingest request from extension v{x_elena_bridge_version or 'unknown'}"
    )

    # Validate we have profile data
    if not payload.profile:
        raise HTTPException(
            status_code=400,
            detail="No profile data in payload. Make sure you're on a profile page."
        )

    # Generate task ID for tracking
    task_id = str(uuid.uuid4())[:8]

    # Log the incoming data
    profile = payload.profile
    logger.info(
        f"[{task_id}] Ingesting @{profile.username} from {profile.platform} "
        f"(method: {profile.extractionMethod})"
    )

    # Queue background processing
    background_tasks.add_task(process_ingested_data, task_id, payload)

    return IngestResponse(
        success=True,
        message=f"Profile @{profile.username} queued for processing",
        task_id=task_id,
        profile_username=profile.username,
        posts_count=len(payload.recentPosts)
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
        "version": "1.0.0"
    }
