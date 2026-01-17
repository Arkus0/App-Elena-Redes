"""
BrandPulse AI - Main FastAPI Application
The ultimate co-pilot for high-engagement content for local SMBs
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.core.config import settings
from app.core.database import init_db
from app.api import api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    await init_db()
    logger.info("Database initialized")

    yield

    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="""
    ## BrandPulse AI - Content Generation for Local SMBs

    Generate HIGH-ENGAGEMENT content for Instagram, TikTok, and LinkedIn based on
    real competitor analysis and proven engagement patterns.

    ### Features:
    - **Onboarding**: Set up your business and competitors for analysis
    - **Competitor Analysis**: Deep analysis of top-performing competitor content
    - **Pattern Extraction**: AI-powered identification of winning content patterns
    - **Content Generation**: Full content calendars with scripts and filming guides
    - **Engagement Prediction**: Score predictions with detailed explanations
    - **Viral Scanner**: Scan trending content and generate reactive ideas
    - **A/B Variations**: Generate multiple versions of content for testing
    - **Export**: Export content to CSV or JSON

    ### Supported Business Types:
    - Inmobiliarias (Real Estate)
    - Floristerías (Florists)
    - Cafeterías (Cafes)
    - Peluquerías (Salons)
    - Tiendas locales (Retail)
    - Restaurantes
    - Gimnasios
    - Clínicas
    - Y más...
    """,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "AI-powered content generation for local SMBs",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "services": {
            "database": "connected",
            "apify": "configured" if settings.APIFY_API_KEY else "not_configured",
            "grok": "configured" if settings.GROK_API_KEY else "not_configured",
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
    )
