"""
BrandPulse AI - Configuration
Core settings for the application
"""
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    APP_NAME: str = "BrandPulse AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production-min-32-chars"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./brandpulse.db"

    # AI - Grok
    GROK_API_KEY: str = ""
    GROK_MODEL: str = "grok-4-1-fast-reasoning"

    # Apify - Primary data source
    APIFY_API_KEY: str = ""
    APIFY_INSTAGRAM_ACTOR: str = "apify/instagram-scraper"
    APIFY_TIKTOK_ACTOR: str = "apify/tiktok-scraper"
    APIFY_LINKEDIN_ACTOR: str = "apify/linkedin-posts-scraper"

    # Cache settings
    CACHE_TTL_HOURS: int = 48  # 48 hours cache for scraped data

    # Fallback APIs (optional)
    PHANTOMBUSTER_API_KEY: Optional[str] = None
    BUZZSUMO_API_KEY: Optional[str] = None
    USE_PHANTOMBUSTER_FALLBACK: bool = False
    USE_BUZZSUMO_FALLBACK: bool = False

    # Content Generation
    MAX_POSTS_PER_COMPETITOR: int = 30
    MIN_ENGAGEMENT_THRESHOLD: int = 100  # Minimum likes+comments to consider

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
