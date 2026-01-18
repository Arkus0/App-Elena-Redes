"""
Scraped Data API Route
Handles incoming data from the Chrome Extension and populates Viral Benchmarks.
"""
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from pydantic import BaseModel
import json
import logging

from app.core.database import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.viral_benchmark import ViralBenchmark
from app.services.analytics_engine import AnalyticsEngine

logger = logging.getLogger(__name__)
router = APIRouter()

class ScrapedPostInput(BaseModel):
    platform: str
    platform_post_id: str
    niche: str
    likes: int
    comments: int
    shares: int
    views: int
    posted_at: str
    media_url: str = None
    account_avg_engagement: float = 0.0 # Provided by extension or calculated

    class Config:
        extra = "ignore"

@router.post("/", status_code=202)
async def receive_scraped_data(
    posts: List[ScrapedPostInput],
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    # current_user: User = Depends(get_current_user) # Optional: secure this if extension has token
):
    """
    Receive scraped posts from Chrome Extension.
    Process them to update Viral Benchmarks.
    """
    if not posts:
        return {"message": "No posts received"}

    background_tasks.add_task(process_scraped_posts, posts)
    return {"message": f"Processing {len(posts)} posts in background"}

async def process_scraped_posts(posts: List[ScrapedPostInput]):
    """
    Background task to process scraped posts.
    1. Calculate engagement rate/score.
    2. Compare to account average to classify (Viral vs Flop).
    3. Extract features (if media_url).
    4. Save to ViralBenchmark.
    """
    # Create a new session since we are in background task
    from app.core.database import async_session_maker

    analytics_engine = AnalyticsEngine(enable_text_intelligence=True) # Use full features

    async with async_session_maker() as db:
        for post in posts:
            try:
                # 1. Calculate Score (Simplified RPI proxy)
                # Weighted: likes + 2*comments + 3*shares
                engagement_score = post.likes + (post.comments * 2) + (post.shares * 3)

                # 2. Classify
                # If we don't have account avg, we skip classification or guess
                if post.account_avg_engagement <= 0:
                    # Fallback: maybe skip or mark as neutral?
                    # For now, require avg for benchmark logic
                    continue

                is_viral = False
                is_flop = False

                ratio = engagement_score / post.account_avg_engagement

                if ratio >= 1.5: # 50% better than average -> Viral candidate
                    is_viral = True
                elif ratio <= 0.2: # 20% of average -> Flop
                    is_flop = True # We use is_viral=False for this in boolean
                else:
                    # Average content - skip to keep benchmark clean (only extremes)
                    continue

                # 3. Extract Features
                features = {}
                if post.media_url:
                    try:
                        # This might be slow if downloading video.
                        # In prod, maybe just metadata or download to temp.
                        # Assuming AnalyticsEngine handles URL or local path.
                        # Ideally, extension sends features, but here we extract.
                        # For speed, we might skip heavy video extraction if URL is remote and large.
                        # Let's assume we rely on metadata + simple extraction or skip heavy lifting here.

                        # Mocking extraction for remote URL safety in this context
                        # Real implementation would download `post.media_url` to temp file
                        features = {
                            "hook_energy": 0.5, # Placeholder
                            "production_quality_score": 0.5
                        }
                    except Exception as e:
                        logger.error(f"Feature extraction failed for {post.platform_post_id}: {e}")

                # Add metadata features
                features.update({
                    "likes": post.likes,
                    "comments": post.comments,
                    "duration": 0 # Unknown
                })

                # 4. Save Benchmark
                # Check duplicate
                existing = await db.execute(
                    select(ViralBenchmark).where(
                        ViralBenchmark.platform_post_id == post.platform_post_id
                    )
                )
                if existing.scalar_one_or_none():
                    continue

                benchmark = ViralBenchmark(
                    niche=post.niche,
                    platform=post.platform,
                    platform_post_id=post.platform_post_id,
                    engagement_real=ratio, # Use ratio as normalized score
                    is_viral=is_viral, # True if viral, False if flop (filtered above)
                    features_json=features
                )
                db.add(benchmark)

            except Exception as e:
                logger.error(f"Error processing post {post.platform_post_id}: {e}")

        await db.commit()
