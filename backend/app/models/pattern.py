"""
ExtractedPattern Model - Winning patterns identified from competitor analysis
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Text, Float, Enum
from sqlalchemy.orm import relationship
import enum
from app.core.database import Base


class PatternType(str, enum.Enum):
    HOOK = "hook"
    CTA = "cta"
    VISUAL = "visual"
    AUDIO = "audio"
    CAPTION_STRUCTURE = "caption_structure"
    CONTENT_PILLAR = "content_pillar"
    POSTING_TIME = "posting_time"
    FORMAT = "format"
    EMOTIONAL_TRIGGER = "emotional_trigger"
    HASHTAG_STRATEGY = "hashtag_strategy"


class ExtractedPattern(Base):
    __tablename__ = "extracted_patterns"

    id = Column(Integer, primary_key=True, index=True)
    business_id = Column(Integer, ForeignKey("businesses.id"), nullable=False)

    # Pattern identification
    pattern_type = Column(Enum(PatternType), nullable=False)
    pattern_name = Column(String(255), nullable=False)  # e.g., "Question Hook", "Before-After Visual"

    # Pattern details
    description = Column(Text, nullable=False)  # Full description for content generation
    examples = Column(JSON, default=[])  # Real examples from scraped posts
    platform = Column(String(50), nullable=True)  # instagram, tiktok, or null for universal

    # Effectiveness metrics
    avg_engagement_score = Column(Float, default=0.0)
    usage_count = Column(Integer, default=0)  # How many top posts use this
    success_rate = Column(Float, default=0.0)  # % of posts using this that performed well

    # Specifics by pattern type
    pattern_data = Column(JSON, default={})
    # For HOOK: {"trigger_words": [], "structure": "", "length_range": []}
    # For CTA: {"action_type": "", "placement": "", "emoji_usage": ""}
    # For VISUAL: {"elements": [], "colors": [], "text_position": ""}
    # For AUDIO: {"trending_sounds": [], "audio_type": ""}
    # For CAPTION: {"length_range": [], "emoji_density": "", "line_breaks": ""}
    # For CONTENT_PILLAR: {"themes": [], "sub_topics": []}
    # For POSTING_TIME: {"best_days": [], "best_hours": [], "timezone": ""}

    # Business type relevance
    business_types = Column(JSON, default=[])  # Which business types this works for

    # Confidence score (0-100)
    confidence_score = Column(Float, default=0.0)

    # Source tracking
    source_post_ids = Column(JSON, default=[])  # IDs of posts this was extracted from
    extracted_at = Column(DateTime, default=datetime.utcnow)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    business = relationship("Business", back_populates="patterns")
