"""
Business Schemas - Onboarding and business management
"""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from app.models.business import BusinessType, EmbeddingPrecision


class CompetitorInput(BaseModel):
    """Competitor input for onboarding"""
    platform: str  # instagram, tiktok, linkedin
    handle: str  # @username without @


class BusinessCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=255)
    business_type: BusinessType
    description: Optional[str] = None
    location: Optional[str] = None
    instagram_handle: Optional[str] = None
    tiktok_handle: Optional[str] = None
    linkedin_handle: Optional[str] = None
    active_platforms: List[str] = ["instagram", "tiktok"]


class BusinessUpdate(BaseModel):
    name: Optional[str] = None
    business_type: Optional[BusinessType] = None
    description: Optional[str] = None
    location: Optional[str] = None
    instagram_handle: Optional[str] = None
    tiktok_handle: Optional[str] = None
    linkedin_handle: Optional[str] = None
    active_platforms: Optional[List[str]] = None
    content_goals: Optional[List[str]] = None
    posting_frequency: Optional[str] = None
    brand_voice: Optional[str] = None
    embedding_precision: Optional[EmbeddingPrecision] = None


class BusinessResponse(BaseModel):
    id: int
    name: str
    business_type: BusinessType
    description: Optional[str]
    location: Optional[str]
    instagram_handle: Optional[str]
    tiktok_handle: Optional[str]
    linkedin_handle: Optional[str]
    active_platforms: List[str]
    content_goals: List[str]
    posting_frequency: str
    brand_voice: str
    embedding_precision: EmbeddingPrecision
    onboarding_completed: Optional[datetime]
    last_analysis_at: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


class OnboardingRequest(BaseModel):
    """Complete onboarding request"""
    # Business info
    name: str = Field(..., min_length=2, max_length=255)
    business_type: BusinessType
    description: Optional[str] = None
    location: Optional[str] = None

    # Social handles
    instagram_handle: Optional[str] = None
    tiktok_handle: Optional[str] = None
    linkedin_handle: Optional[str] = None

    # Platforms to use
    active_platforms: List[str] = ["instagram", "tiktok"]

    # Competitors to analyze (3-7)
    competitors: List[CompetitorInput] = Field(..., min_length=3, max_length=7)

    # Goals
    content_goals: List[str] = ["awareness", "engagement"]
    posting_frequency: str = "3-5_per_week"
    brand_voice: str = "friendly_professional"


class OnboardingResponse(BaseModel):
    """Response after onboarding completes"""
    business_id: int
    message: str
    competitors_added: int
    scraping_status: str  # pending, in_progress, completed
    estimated_analysis_time: str
    next_steps: List[str]
