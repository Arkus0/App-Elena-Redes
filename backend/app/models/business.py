"""
Business Model - Core entity for local SMBs
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum, JSON, Text
from sqlalchemy.orm import relationship
import enum
from app.core.database import Base


class BusinessType(str, enum.Enum):
    INMOBILIARIA = "inmobiliaria"
    FLORISTERIA = "floristeria"
    CAFETERIA = "cafeteria"
    PELUQUERIA = "peluqueria"
    TIENDA_LOCAL = "tienda_local"
    RESTAURANTE = "restaurante"
    GIMNASIO = "gimnasio"
    CLINICA = "clinica"
    OTROS = "otros"


class Platform(str, enum.Enum):
    INSTAGRAM = "instagram"
    TIKTOK = "tiktok"
    LINKEDIN = "linkedin"


class EmbeddingPrecision(str, enum.Enum):
    """
    Embedding precision levels for ML pipeline.

    Default: MAX (full 384 dims) - recommended for typical SMB data volumes.
    Safe for 100-2000 posts, training <1min, RAM <2GB on normal desktop.

    - LOW (128 dims): Ultra fast, lower precision
    - MEDIUM (256 dims): Balanced precision/speed
    - HIGH (384 dims): Full precision
    - MAX (384 dims): Full raw embeddings - best for regional slang/nuances
    """
    LOW = "low"        # 128 dims
    MEDIUM = "medium"  # 256 dims
    HIGH = "high"      # 384 dims
    MAX = "max"        # 384 dims (default - full raw embeddings)


class Business(Base):
    __tablename__ = "businesses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Basic info
    name = Column(String(255), nullable=False)
    business_type = Column(Enum(BusinessType), nullable=False)
    description = Column(Text, nullable=True)
    location = Column(String(255), nullable=True)  # City/neighborhood

    # Social handles
    instagram_handle = Column(String(100), nullable=True)
    tiktok_handle = Column(String(100), nullable=True)
    linkedin_handle = Column(String(100), nullable=True)

    # Platform preferences (JSON array)
    active_platforms = Column(JSON, default=["instagram", "tiktok"])

    # Content preferences
    content_goals = Column(JSON, default=[])  # awareness, leads, foot_traffic, sales
    posting_frequency = Column(String(50), default="3-5_per_week")
    brand_voice = Column(String(100), default="friendly_professional")

    # ML Configuration
    # Embedding precision: "low" (128), "medium" (256), "high" (384), "max" (384 raw)
    # Default: "max" - full 384 dims, safe for SMB volumes (100-2000 posts, <1min train, <2GB RAM)
    embedding_precision = Column(
        Enum(EmbeddingPrecision),
        default=EmbeddingPrecision.MAX,
        nullable=False
    )

    # Human-in-the-Loop Configuration
    # Own username for each platform - used to identify posts from the client's own account
    # When extension detects content from these usernames, it triggers feedback loop
    # Example: "inmoalmeria" (without @)
    own_instagram_username = Column(String(100), nullable=True)
    own_tiktok_username = Column(String(100), nullable=True)

    # Onboarding status
    onboarding_completed = Column(DateTime, nullable=True)
    last_analysis_at = Column(DateTime, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    competitors = relationship("Competitor", back_populates="business", cascade="all, delete-orphan")
    patterns = relationship("ExtractedPattern", back_populates="business", cascade="all, delete-orphan")
    generated_content = relationship("GeneratedContent", back_populates="business", cascade="all, delete-orphan")
    calendars = relationship("ContentCalendar", back_populates="business", cascade="all, delete-orphan")
