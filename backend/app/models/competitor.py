"""
Competitor Model - Track competitor accounts for analysis
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum, JSON
from sqlalchemy.orm import relationship
from app.models.business import Platform
from app.core.database import Base


class Competitor(Base):
    __tablename__ = "competitors"

    id = Column(Integer, primary_key=True, index=True)
    # Bolt Optimization: Added index=True to business_id for faster filtering by business
    business_id = Column(Integer, ForeignKey("businesses.id"), nullable=False, index=True)

    # Account info
    platform = Column(Enum(Platform), nullable=False)
    # Bolt Optimization: Added index=True to handle for faster lookups
    handle = Column(String(100), nullable=False, index=True)  # @username without @
    profile_url = Column(String(500), nullable=True)
    display_name = Column(String(255), nullable=True)
    bio = Column(String(1000), nullable=True)

    # Metrics (updated on scrape)
    followers_count = Column(Integer, default=0)
    following_count = Column(Integer, default=0)
    posts_count = Column(Integer, default=0)
    avg_engagement_rate = Column(Integer, default=0)  # As percentage * 100

    # Scraping status
    last_scraped_at = Column(DateTime, nullable=True)
    scrape_status = Column(String(50), default="pending")  # pending, in_progress, completed, failed
    scrape_error = Column(String(500), nullable=True)

    # Activity metrics (for filtering inactive competitors)
    last_post_date = Column(DateTime, nullable=True)  # Date of most recent post
    days_since_last_post = Column(Integer, nullable=True)  # Cached for queries
    activity_status = Column(String(20), default="unknown")  # active, inactive, dormant, unknown
    activity_score = Column(Integer, default=0)  # 0-100 score
    posting_frequency = Column(Integer, nullable=True)  # Posts per month (avg)

    # Analysis data (JSON)
    top_performing_formats = Column(JSON, default=[])  # reels, carousels, static
    peak_posting_times = Column(JSON, default=[])
    common_hashtags = Column(JSON, default=[])

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    business = relationship("Business", back_populates="competitors")
    scraped_posts = relationship("ScrapedPost", back_populates="competitor", cascade="all, delete-orphan")
