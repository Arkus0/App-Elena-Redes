"""
BrandPulse AI - API Routes
"""
from fastapi import APIRouter
from app.api import auth, business, competitors, content, viral

api_router = APIRouter()

# Include all route modules
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(business.router, prefix="/business", tags=["Business"])
api_router.include_router(competitors.router, prefix="/competitors", tags=["Competitors"])
api_router.include_router(content.router, prefix="/content", tags=["Content"])
api_router.include_router(viral.router, prefix="/viral", tags=["Viral Scanner"])
