"""
Content Schemas - Content generation and calendar
"""
from datetime import datetime, date
from typing import Optional, List
from pydantic import BaseModel, Field


class ContentGenerationRequest(BaseModel):
    """Request to generate content calendar"""
    # Time period
    month: int = Field(..., ge=1, le=12)
    year: int = Field(..., ge=2024, le=2030)

    # Volume
    posts_count: int = Field(default=15, ge=5, le=40)

    # Goals
    primary_goal: str = "engagement"  # awareness, leads, foot_traffic, sales, engagement
    secondary_goals: List[str] = []

    # Platform preferences
    platforms: List[str] = ["instagram", "tiktok"]

    # Content mix (percentages)
    content_mix: Optional[dict] = None
    # {"reels": 60, "carousels": 20, "static": 10, "stories": 10}

    # Specific requests
    include_trends: bool = True
    refresh_competitor_data: bool = True
    specific_topics: List[str] = []  # Optional specific topics to include


class FilmingGuide(BaseModel):
    """Detailed filming guide for video content"""
    setup: str
    equipment_needed: List[str]
    lighting_tips: str
    camera_angles: List[str]
    props_needed: List[str]
    location_suggestions: List[str]
    text_overlays: List[dict]  # {"text": "", "timing": "", "position": ""}
    b_roll_ideas: List[str]
    estimated_filming_time: str
    editing_tips: List[str]


class ContentPiece(BaseModel):
    """Single piece of generated content"""
    id: int
    title: str
    platform: str
    content_format: str
    status: str

    # The content
    caption: str
    hashtags: List[str]
    hook_text: Optional[str]

    # Video specific
    video_script: Optional[str]
    script_structure: Optional[dict]
    filming_guide: Optional[FilmingGuide]
    recommended_audio: Optional[str]
    audio_alternatives: List[str]

    # Image prompt
    image_prompt: Optional[str]

    # Engagement prediction
    engagement_score: float
    engagement_explanation: str
    similar_viral_posts: List[dict]

    # Metadata
    patterns_used: List[str]
    framework_used: Optional[str]
    content_goal: Optional[str]
    cta_type: Optional[str]

    # Scheduling
    scheduled_date: Optional[date]
    optimal_posting_time: Optional[str]

    # Variations
    variation_label: Optional[str]
    has_variations: bool = False

    class Config:
        from_attributes = True


class GeneratedContentResponse(BaseModel):
    """Response for single content generation"""
    content: ContentPiece
    variations: List[ContentPiece] = []


class CalendarResponse(BaseModel):
    """Full content calendar response"""
    calendar_id: int
    name: str
    month: int
    year: int
    primary_goal: str
    total_pieces: int
    status: str

    # Content breakdown
    content_by_platform: dict  # {"instagram": 10, "tiktok": 5}
    content_by_format: dict  # {"reels": 8, "carousels": 4, "static": 3}

    # Average metrics
    avg_engagement_score: float
    top_predicted_posts: List[ContentPiece]

    # All content
    content_pieces: List[ContentPiece]

    # Insights
    calendar_insights: List[str]
    key_themes: List[str]

    created_at: datetime


class ContentExport(BaseModel):
    """Export format for content"""
    export_format: str = "csv"  # csv, json, notion
    include_scripts: bool = True
    include_filming_guides: bool = True
    include_image_prompts: bool = True
    content_ids: Optional[List[int]] = None  # None = all


class EngagementPrediction(BaseModel):
    """Detailed engagement prediction"""
    content_id: int
    score: float  # 0-100
    confidence: float  # 0-100

    # Breakdown
    hook_score: float
    cta_score: float
    format_score: float
    timing_score: float
    trend_alignment_score: float

    # Explanation
    explanation: str
    strengths: List[str]
    weaknesses: List[str]
    improvement_suggestions: List[str]

    # Comparisons
    similar_competitor_posts: List[dict]
    predicted_metrics: dict  # {"likes": "500-1000", "comments": "50-100", ...}


class ViralScanRequest(BaseModel):
    """Request to scan for viral opportunities"""
    keyword: str
    platform: str = "instagram"  # instagram, tiktok
    niche: Optional[str] = None  # Override business type niche
    max_results: int = 10


class ViralOpportunity(BaseModel):
    """A viral trend opportunity"""
    trend_id: str
    trend_name: str
    platform: str
    description: str

    # Trend metrics
    total_views: int
    total_videos: int
    growth_rate: str  # "rising", "peaked", "declining"
    time_sensitive: bool

    # Your opportunity
    relevance_score: float  # 0-100 how relevant to your business
    difficulty_score: float  # 0-100 how hard to execute

    # Ready-to-use ideas
    content_ideas: List[dict]
    # {"title": "", "hook": "", "script_outline": "", "filming_tips": ""}

    # Reference videos
    top_examples: List[dict]
    # {"url": "", "views": 0, "why_it_worked": ""}

    # Timing
    best_time_to_post: str
    trend_expiry_estimate: str


class ViralScanResponse(BaseModel):
    """Response from viral opportunity scan"""
    keyword: str
    platform: str
    scan_date: datetime
    opportunities: List[ViralOpportunity]
    general_insights: List[str]
