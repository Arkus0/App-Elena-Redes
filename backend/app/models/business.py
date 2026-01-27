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

    IMPORTANT: Default changed to LOW (128 dims) - safe for typical sobremesa.

    - ULTRA_LOW (64 dims): Ultra rapido para PC modesto
    - LOW (128 dims): Recomendado sobremesa normal Almeria (NEW DEFAULT)
    - MEDIUM (256 dims): Balance precision/velocidad
    - HIGH (384 dims): Full dims con TruncatedSVD
    - MAX (full raw): 384 dims sin reduccion - mejor matices creativos/slang local
    """
    ULTRA_LOW = "ultra_low"  # 64 dims - ultra rapido, PC modesto
    LOW = "low"              # 128 dims - recomendado sobremesa normal
    MEDIUM = "medium"        # 256 dims - balance
    HIGH = "high"            # 384 dims - full con TruncatedSVD
    MAX = "max"              # full raw 384 dims sin reduccion


class Business(Base):
    __tablename__ = "businesses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

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
    # Embedding precision: "ultra_low" (64), "low" (128), "medium" (256), "high" (384), "max" (full raw)
    # Default: "low" - 128 dims, seguro para sobremesa normal Almeria (100-2000 posts, <1min train, <1GB RAM)
    embedding_precision = Column(
        Enum(EmbeddingPrecision),
        default=EmbeddingPrecision.LOW,
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
