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
