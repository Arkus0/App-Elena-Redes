"""
BrandPulse AI - Configuration
Core settings for the application
"""
from functools import lru_cache
from typing import Optional
from pydantic import model_validator
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

    # ==========================================================================
    # PRIVACY & SECURITY
    # ==========================================================================
    # Blind Identity settings for PII anonymization
    PRIVACY_MODE_ENABLED: bool = True
    # Salt for SHA-256 hashing. CHANGE THIS IN PRODUCTION!
    DYNAMIC_SALT: str = "dev-dynamic-salt-change-in-prod-v1"

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

    # ==========================================================================
    # VIDEO PROCESSING OPTIMIZATION SETTINGS
    # ==========================================================================
    # These settings control performance vs. accuracy trade-offs

    # Optical Flow - Camera stability analysis (~40% of processing time)
    # Set to False for faster processing when camera stability is not critical
    VIDEO_OPTICAL_FLOW_ENABLED: bool = True

    # ML Model Quantization - int8 quantization for ~3-4x speedup
    # Trades ~1% accuracy for significant speed improvement on CPU
    ML_EMBEDDING_QUANTIZE: bool = True
    ML_WHISPER_COMPUTE_TYPE: str = "int8"  # Options: int8, float16, float32

    # Video Task Queue - Background processing configuration
    TASK_QUEUE_ENABLED: bool = True
    TASK_QUEUE_MAX_SIZE: int = 100
    TASK_QUEUE_PERSIST: bool = False  # Enable SQLite persistence for crash recovery
    TASK_QUEUE_SQLITE_PATH: str = "./video_tasks.db"
    TASK_QUEUE_MAX_RETRIES: int = 3
    TASK_QUEUE_TIMEOUT_SECONDS: int = 600  # 10 minutes max per task

    # ==========================================================================
    # LIGHT MODE MULTIMODAL PROCESSING
    # ==========================================================================
    # Optimizes Whisper/EasyOCR processing for cost efficiency
    # Light mode: ~5s processing vs ~20s full (75% faster, minimal quality loss)

    # Master toggle - True for light mode (recommended for sobremesa)
    LIGHT_MODE_ENABLED: bool = True

    # Whisper light settings
    LIGHT_WHISPER_MODEL: str = "tiny"  # 'tiny' (~75MB) or 'base' (~150MB)
    LIGHT_WHISPER_MAX_DURATION: float = 3.0  # Only first 3 seconds (hook analysis)

    # OCR light settings
    LIGHT_OCR_MAX_FRAMES: int = 5  # Only first 5 frames
    LIGHT_OCR_USE_THUMBNAIL: bool = True  # Prefer thumbnail over frame extraction

    # Hook analysis
    LIGHT_HOOK_DURATION: float = 3.0  # Analyze first 3 seconds only

    # Cache settings (skip already processed media)
    LIGHT_CACHE_ENABLED: bool = True
    LIGHT_CACHE_TTL_HOURS: int = 168  # 7 days

    # Skip multimodal for non-video content
    LIGHT_SKIP_NON_VIDEO: bool = True

    @model_validator(mode="after")
    def check_secure_defaults(self) -> "Settings":
        """
        Ensure that insecure default secrets are not used in production.
        """
        if not self.DEBUG:
            if self.SECRET_KEY == "your-secret-key-change-in-production-min-32-chars":
                raise ValueError("SECRET_KEY must be changed in production!")
            if self.DYNAMIC_SALT == "dev-dynamic-salt-change-in-prod-v1":
                raise ValueError("DYNAMIC_SALT must be changed in production!")
        return self

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
