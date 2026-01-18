"""
BrandPulse AI - API Routes
"""
from fastapi import APIRouter
from app.api import auth, business, competitors, content, viral, ml, growth, trends, abtest, kpi, multi_output

api_router = APIRouter()

# Include all route modules
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(business.router, prefix="/business", tags=["Business"])
api_router.include_router(competitors.router, prefix="/competitors", tags=["Competitors"])
api_router.include_router(content.router, prefix="/content", tags=["Content"])
api_router.include_router(viral.router, prefix="/viral", tags=["Viral Scanner"])
api_router.include_router(ml.router, prefix="/ml", tags=["ML Predictions"])
api_router.include_router(growth.router, prefix="/growth", tags=["Growth Prediction"])
api_router.include_router(trends.router, prefix="/trends", tags=["Trend Velocity"])
api_router.include_router(abtest.router, tags=["A/B Testing"])
api_router.include_router(kpi.router, prefix="/kpi", tags=["KPI Weights"])
api_router.include_router(multi_output.router, prefix="/multi-output", tags=["Multi-Output Predictions"])
