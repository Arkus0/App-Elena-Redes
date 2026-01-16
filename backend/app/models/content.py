"""
GeneratedContent & ContentCalendar Models - AI-generated content pieces
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Text, Float, Enum, Date
from sqlalchemy.orm import relationship
import enum
from app.core.database import Base


class ContentGoal(str, enum.Enum):
    AWARENESS = "awareness"
    LEADS = "leads"
    FOOT_TRAFFIC = "foot_traffic"
    SALES = "sales"
    ENGAGEMENT = "engagement"
    BRAND_BUILDING = "brand_building"


class ContentStatus(str, enum.Enum):
    DRAFT = "draft"
    READY = "ready"
    SCHEDULED = "scheduled"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class GeneratedContent(Base):
    __tablename__ = "generated_content"

    id = Column(Integer, primary_key=True, index=True)
    business_id = Column(Integer, ForeignKey("businesses.id"), nullable=False)
    calendar_id = Column(Integer, ForeignKey("content_calendars.id"), nullable=True)

    # Content identification
    title = Column(String(255), nullable=False)  # Internal reference
    platform = Column(String(50), nullable=False)  # instagram, tiktok, linkedin
    content_format = Column(String(50), nullable=False)  # reel, carousel, static, story, tiktok_video
    status = Column(Enum(ContentStatus), default=ContentStatus.DRAFT)

    # The actual content
    caption = Column(Text, nullable=False)
    hashtags = Column(JSON, default=[])
    hook_text = Column(String(500), nullable=True)  # First line / text overlay

    # Script (for videos)
    video_script = Column(Text, nullable=True)  # Full script with timestamps
    script_structure = Column(JSON, default={})
    # {"hook_0_3s": "", "value_3_20s": "", "cta_20_30s": ""}

    # Filming guide
    filming_guide = Column(Text, nullable=True)
    visual_directions = Column(JSON, default=[])  # Angles, lighting, props
    text_overlays = Column(JSON, default=[])  # Text to add in editing
    recommended_audio = Column(String(255), nullable=True)  # Trending sound suggestion
    audio_alternatives = Column(JSON, default=[])

    # AI Image generation prompt
    image_prompt = Column(Text, nullable=True)  # For DALL-E / Stability.ai

    # Engagement prediction (Hybrid ML/LLM)
    engagement_score = Column(Float, default=0.0)  # 0-100 predicted score (from ML model)
    engagement_explanation = Column(Text, nullable=True)  # "Este Reel tiene 87% porque..."
    similar_viral_posts = Column(JSON, default=[])  # References to competitor posts
    ml_prediction_data = Column(JSON, default={})  # Full ML prediction with SHAP explanation

    # Patterns used
    patterns_used = Column(JSON, default=[])  # IDs of ExtractedPattern used
    framework_used = Column(String(100), nullable=True)  # AIDA, PAS, Hook-Value-CTA

    # Goals and targeting
    content_goal = Column(Enum(ContentGoal), nullable=True)
    target_audience = Column(String(255), nullable=True)
    cta_type = Column(String(100), nullable=True)

    # A/B Variations
    variation_of = Column(Integer, ForeignKey("generated_content.id"), nullable=True)
    variation_label = Column(String(50), nullable=True)  # A, B, C

    # Scheduling
    scheduled_date = Column(Date, nullable=True)
    scheduled_time = Column(String(10), nullable=True)  # HH:MM format
    optimal_posting_time = Column(String(100), nullable=True)  # Recommendation

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    business = relationship("Business", back_populates="generated_content")
    calendar = relationship("ContentCalendar", back_populates="content_pieces")
    variations = relationship("GeneratedContent", backref="original", remote_side=[id])


class ContentCalendar(Base):
    __tablename__ = "content_calendars"

    id = Column(Integer, primary_key=True, index=True)
    business_id = Column(Integer, ForeignKey("businesses.id"), nullable=False)

    # Calendar info
    name = Column(String(255), nullable=False)  # "Enero 2026 - Floristería Rosa"
    month = Column(Integer, nullable=False)  # 1-12
    year = Column(Integer, nullable=False)
    description = Column(Text, nullable=True)

    # Goals for this period
    primary_goal = Column(Enum(ContentGoal), nullable=True)
    secondary_goals = Column(JSON, default=[])
    target_posts_count = Column(Integer, default=15)

    # Content mix strategy
    content_mix = Column(JSON, default={})
    # {"reels": 60, "carousels": 25, "static": 15} percentages

    # Analysis used
    patterns_snapshot = Column(JSON, default=[])  # Pattern IDs used for generation
    competitors_analyzed = Column(JSON, default=[])  # Competitor IDs

    # Status
    status = Column(String(50), default="draft")  # draft, active, completed
    generated_at = Column(DateTime, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    business = relationship("Business", back_populates="calendars")
    content_pieces = relationship("GeneratedContent", back_populates="calendar", cascade="all, delete-orphan")
