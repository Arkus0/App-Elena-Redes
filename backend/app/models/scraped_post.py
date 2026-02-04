"""
ScrapedPost Model - Individual posts/videos scraped from competitors

SURVIVOR BIAS FIX (2026-01):
Added fields for balanced sampling to collect both viral posts AND flops:
- is_viral: Binary classification (1=viral, 0=flop, NULL=middle)
- engagement_ratio: Engagement / Followers ratio
- performance_tier: top, middle, bottom
- performance_percentile: Position in engagement ranking
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum, JSON, Text, Float, Boolean
from sqlalchemy.orm import relationship
import enum
from app.core.database import Base


class ContentFormat(str, enum.Enum):
    REEL = "reel"
    CAROUSEL = "carousel"
    STATIC_IMAGE = "static_image"
    STORY = "story"
    TIKTOK_VIDEO = "tiktok_video"
    LINKEDIN_POST = "linkedin_post"
    LINKEDIN_CAROUSEL = "linkedin_carousel"


class PerformanceTier(str, enum.Enum):
    """Performance tier based on engagement ranking"""
    TOP = "top"         # Top 20% - viral content
    MIDDLE = "middle"   # Middle 60% - average content
    BOTTOM = "bottom"   # Bottom 20% - flop content


class ScrapedPost(Base):
    __tablename__ = "scraped_posts"

    id = Column(Integer, primary_key=True, index=True)
    # Bolt Optimization: Added index=True to competitor_id for faster filtering by competitor
    competitor_id = Column(Integer, ForeignKey("competitors.id"), nullable=False, index=True)

    # Post identification
    # Bolt Optimization: Added index=True to platform_post_id for faster deduplication checks
    platform_post_id = Column(String(100), nullable=False, index=True)  # Original ID from platform
    post_url = Column(String(500), nullable=True)
    content_format = Column(Enum(ContentFormat), nullable=False)

    # Content
    caption = Column(Text, nullable=True)
    hashtags = Column(JSON, default=[])  # Extracted hashtags
    mentions = Column(JSON, default=[])  # @mentions in caption

    # Media
    thumbnail_url = Column(String(500), nullable=True)
    media_urls = Column(JSON, default=[])  # Array of image/video URLs
    video_duration_seconds = Column(Integer, nullable=True)
    audio_name = Column(String(255), nullable=True)  # For Reels/TikTok
    audio_original = Column(String(10), default="unknown")  # original, trending, custom

    # === MULTIMODAL DATA (Rich Analysis) ===
    transcript = Column(Text, nullable=True)
    ocr_text = Column(Text, nullable=True)
    visual_features = Column(JSON, default={})  # Stores dominant_colors, brightness, etc.

    # Engagement metrics (the gold!)
    likes_count = Column(Integer, default=0)
    comments_count = Column(Integer, default=0)
    shares_count = Column(Integer, default=0)
    saves_count = Column(Integer, default=0)
    views_count = Column(Integer, default=0)  # For videos
    plays_count = Column(Integer, default=0)  # TikTok specific

    # Calculated engagement score (normalized 0-100)
    # Bolt Optimization: Added index=True to engagement_score for faster sorting of top/bottom posts
    engagement_score = Column(Float, default=0.0, index=True)
    engagement_rate = Column(Float, default=0.0)  # (likes+comments) / followers * 100

    # === SURVIVOR BIAS FIX: New fields for balanced sampling ===
    # Binary viral classification for ML training
    # 1 = Top 20% (viral/success), 0 = Bottom 20% (flop/failure), NULL = Middle 60%
    is_viral = Column(Boolean, nullable=True, index=True)

    # Engagement ratio: (weighted_engagement / followers) * 100
    # This normalizes engagement across accounts of different sizes
    engagement_ratio = Column(Float, default=0.0)

    # Performance classification
    performance_tier = Column(Enum(PerformanceTier), nullable=True)
    performance_percentile = Column(Float, nullable=True)  # 0-100, higher = better

    # Sampling metadata
    sampling_strategy = Column(String(50), nullable=True)  # balanced, top_only, etc.
    is_augmented = Column(Boolean, default=False)  # True if synthetically generated
    # === END SURVIVOR BIAS FIX ===

    # Timing
    # Bolt Optimization: Added index=True to posted_at for faster time-based filtering and trend analysis
    posted_at = Column(DateTime, nullable=True, index=True)
    day_of_week = Column(String(20), nullable=True)
    hour_of_day = Column(Integer, nullable=True)

    # AI Analysis (populated by Claude)
    analyzed = Column(String(10), default="no")
    hook_type = Column(String(100), nullable=True)  # question, surprise, bold_claim, etc.
    hook_text = Column(String(500), nullable=True)  # First line or text overlay
    cta_type = Column(String(100), nullable=True)  # comment, save, share, dm, link
    emotional_triggers = Column(JSON, default=[])  # curiosity, fomo, inspiration, etc.
    content_pillars = Column(JSON, default=[])  # tips, bts, transformation, story, etc.
    visual_elements = Column(JSON, default=[])  # faces, text_overlay, before_after, etc.

    # Timestamps
    scraped_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    competitor = relationship("Competitor", back_populates="scraped_posts")

    def calculate_engagement_ratio(self, follower_count: int = 1) -> float:
        """
        Calculate engagement ratio normalized by followers.

        SURVIVOR BIAS FIX: This ratio allows comparing engagement
        across accounts of different sizes.

        Formula: (weighted_engagement / followers) * 100

        Returns:
            Engagement ratio as percentage (can exceed 100 for viral content)
        """
        weighted_engagement = (
            self.likes_count +
            (self.comments_count * 3) +
            (self.shares_count * 4) +
            (self.saves_count * 5)
        )

        if follower_count > 0:
            return (weighted_engagement / follower_count) * 100
        return 0.0

    def calculate_engagement_score(self, follower_count: int = 1) -> float:
        """Calculate normalized engagement score 0-100"""
        total_engagement = (
            self.likes_count +
            (self.comments_count * 3) +  # Comments weighted 3x
            (self.shares_count * 4) +    # Shares weighted 4x
            (self.saves_count * 5)       # Saves weighted 5x (highest intent)
        )

        # Normalize by followers (engagement rate style)
        if follower_count > 0:
            rate = (total_engagement / follower_count) * 100
            # Cap at 100 for extremely viral content
            return min(rate * 10, 100)  # Scale up for small accounts
        return 0.0
