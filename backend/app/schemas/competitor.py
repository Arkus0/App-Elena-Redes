"""
Competitor Schemas - Competitor analysis and scraping
"""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel


class CompetitorCreate(BaseModel):
    platform: str
    handle: str


class CompetitorPreviewResponse(BaseModel):
    """Lightweight profile preview"""
    platform: str
    handle: str
    profile_pic_url: Optional[str]
    full_name: Optional[str]
    biography: Optional[str]
    followers_count: int
    is_private: bool
    verified: bool = False


class CompetitorDiscoveryRequest(BaseModel):
    hashtags: List[str]
    location_keywords: Optional[List[str]] = []
    niche_keywords: Optional[List[str]] = []

    # Size Filters
    min_followers: int = 100
    max_followers: int = 100000

    # Activity Filters (New)
    require_active: bool = True           # If True, applies the filters below
    max_days_since_last_post: int = 30    # Ignore accounts inactive for > 30 days
    min_posts_last_month: int = 1         # Ignore accounts with 0 posts recently


class DiscoveredCompetitor(BaseModel):
    handle: str
    platform: str
    followers: int
    relevance_score: int
    activity_status: str  # "Active", "Inactive", "Unknown"
    last_post_date: Optional[str] = None
    match_reasons: List[str]
    profile_pic_url: Optional[str] = None


class ScrapedPostSummary(BaseModel):
    """Summary of a scraped post"""
    id: int
    platform_post_id: str
    post_url: Optional[str]
    content_format: str
    caption_preview: Optional[str]  # First 200 chars
    thumbnail_url: Optional[str]
    likes_count: int
    comments_count: int
    shares_count: int
    saves_count: int
    views_count: int
    engagement_score: float
    posted_at: Optional[datetime]
    hook_type: Optional[str]
    cta_type: Optional[str]

    class Config:
        from_attributes = True


class CompetitorResponse(BaseModel):
    id: int
    platform: str
    handle: str
    display_name: Optional[str]
    bio: Optional[str]
    followers_count: int
    posts_count: int
    avg_engagement_rate: int
    last_scraped_at: Optional[datetime]
    scrape_status: str
    top_performing_formats: List[str]
    common_hashtags: List[str]

    class Config:
        from_attributes = True


class TopPost(BaseModel):
    """Top performing post with analysis"""
    post_id: int
    post_url: Optional[str]
    thumbnail_url: Optional[str]
    content_format: str
    engagement_score: float
    likes: int
    comments: int
    saves: int
    shares: int
    caption_preview: str
    hook_type: Optional[str]
    hook_text: Optional[str]
    cta_type: Optional[str]
    emotional_triggers: List[str]
    why_it_worked: str  # AI explanation


class PatternInsight(BaseModel):
    """Extracted pattern insight"""
    pattern_type: str
    pattern_name: str
    description: str
    confidence_score: float
    examples_count: int
    avg_engagement: float


class CompetitorAnalysis(BaseModel):
    """Full competitor analysis response"""
    competitor_id: int
    handle: str
    platform: str
    followers: int
    total_posts_analyzed: int
    analysis_date: datetime

    # Key insights
    top_posts: List[TopPost]
    winning_patterns: List[PatternInsight]

    # Summarized recommendations
    best_posting_times: List[str]
    best_formats: List[str]
    best_content_pillars: List[str]
    trending_hashtags: List[str]
    recommended_hooks: List[str]
    recommended_ctas: List[str]

    # AI summary
    overall_strategy_summary: str
    key_takeaways: List[str]
    content_gaps: List[str]  # What you could do differently
