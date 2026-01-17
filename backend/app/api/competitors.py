"""
Competitors API Routes
Competitor management and analysis
"""
from typing import List, Optional, Any, Dict
from datetime import datetime, timedelta
import json
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.business import Business, Platform
from app.models.competitor import Competitor
from app.models.scraped_post import ScrapedPost
from app.models.pattern import ExtractedPattern
from app.models.cached_analysis import CachedAnalysis
from app.schemas.competitor import (
    CompetitorCreate,
    CompetitorResponse,
    CompetitorAnalysis,
    TopPost,
    PatternInsight,
    ScrapedPostSummary,
)

router = APIRouter()


# =============================================================================
# Caching Helpers
# =============================================================================

async def get_cached_data(
    db: AsyncSession,
    business_id: int,
    cache_key: str
) -> Optional[Dict[str, Any]]:
    """Retrieve valid cached data if available"""
    result = await db.execute(
        select(CachedAnalysis)
        .where(CachedAnalysis.business_id == business_id)
        .where(CachedAnalysis.type == cache_key)
        .where(CachedAnalysis.expires_at > datetime.utcnow())
    )
    cache = result.scalar_one_or_none()
    return cache.data if cache else None

async def set_cached_data(
    db: AsyncSession,
    business_id: int,
    cache_key: str,
    data: Dict[str, Any],
    ttl_hours: int = 48
):
    """Save data to cache"""
    # Check if cache entry exists to update or create new
    result = await db.execute(
        select(CachedAnalysis)
        .where(CachedAnalysis.business_id == business_id)
        .where(CachedAnalysis.type == cache_key)
    )
    existing_cache = result.scalar_one_or_none()

    expires_at = datetime.utcnow() + timedelta(hours=ttl_hours)

    if existing_cache:
        existing_cache.data = data
        existing_cache.expires_at = expires_at
    else:
        new_cache = CachedAnalysis(
            business_id=business_id,
            type=cache_key,
            data=data,
            expires_at=expires_at
        )
        db.add(new_cache)

    await db.commit()


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/{business_id}", response_model=List[CompetitorResponse])
async def get_competitors(
    business_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get all competitors for a business
    """
    # Verify business ownership
    biz_result = await db.execute(
        select(Business)
        .where(Business.id == business_id)
        .where(Business.user_id == current_user.id)
    )
    if not biz_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Business not found")

    result = await db.execute(
        select(Competitor).where(Competitor.business_id == business_id)
    )
    competitors = result.scalars().all()

    return competitors


@router.post("/{business_id}/add", response_model=CompetitorResponse, status_code=201)
async def add_competitor(
    business_id: int,
    competitor_data: CompetitorCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Add a new competitor to analyze
    """
    # Verify business ownership
    biz_result = await db.execute(
        select(Business)
        .where(Business.id == business_id)
        .where(Business.user_id == current_user.id)
    )
    business = biz_result.scalar_one_or_none()
    if not business:
        raise HTTPException(status_code=404, detail="Business not found")

    # Check if competitor already exists
    existing = await db.execute(
        select(Competitor)
        .where(Competitor.business_id == business_id)
        .where(Competitor.handle == competitor_data.handle.lstrip("@"))
        .where(Competitor.platform == Platform(competitor_data.platform))
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Competitor already added")

    # Create competitor
    competitor = Competitor(
        business_id=business_id,
        platform=Platform(competitor_data.platform),
        handle=competitor_data.handle.lstrip("@"),
        scrape_status="pending",
    )
    db.add(competitor)
    await db.commit()
    await db.refresh(competitor)

    # Trigger background scraping
    from app.api.business import scrape_competitors_background
    background_tasks.add_task(
        scrape_single_competitor,
        competitor.id,
        business.business_type.value
    )

    return competitor


async def scrape_single_competitor(competitor_id: int, business_type: str):
    """Background task to scrape a single competitor"""
    from app.core.database import async_session_maker
    from app.services.apify_service import ApifyService
    from app.services.pattern_extractor import PatternExtractor
    from datetime import datetime

    apify_service = ApifyService()
    pattern_extractor = PatternExtractor()

    async with async_session_maker() as db:
        result = await db.execute(
            select(Competitor).where(Competitor.id == competitor_id)
        )
        competitor = result.scalar_one_or_none()
        if not competitor:
            return

        try:
            competitor.scrape_status = "in_progress"
            await db.commit()

            # Scrape based on platform
            if competitor.platform == Platform.INSTAGRAM:
                data = await apify_service.scrape_instagram_profile(competitor.handle)
            elif competitor.platform == Platform.TIKTOK:
                data = await apify_service.scrape_tiktok_profile(competitor.handle)
            else:
                data = await apify_service.scrape_linkedin_profile(competitor.handle)

            # Update competitor profile
            profile = data.get("profile", {})
            competitor.display_name = profile.get("full_name") or profile.get("display_name")
            competitor.bio = profile.get("bio")
            competitor.followers_count = profile.get("followers", 0)
            competitor.following_count = profile.get("following", 0)
            competitor.posts_count = profile.get("posts_count") or profile.get("videos_count", 0)

            # Save posts
            from app.models.scraped_post import ContentFormat

            for post_data in data.get("posts", []):
                format_map = {
                    "reel": ContentFormat.REEL,
                    "carousel": ContentFormat.CAROUSEL,
                    "static_image": ContentFormat.STATIC_IMAGE,
                    "tiktok_video": ContentFormat.TIKTOK_VIDEO,
                    "linkedin_post": ContentFormat.LINKEDIN_POST,
                }
                content_format = format_map.get(post_data.get("type"), ContentFormat.STATIC_IMAGE)

                post = ScrapedPost(
                    competitor_id=competitor.id,
                    platform_post_id=post_data.get("platform_id", ""),
                    post_url=post_data.get("url"),
                    content_format=content_format,
                    caption=post_data.get("caption"),
                    hashtags=post_data.get("hashtags", []),
                    thumbnail_url=post_data.get("thumbnail"),
                    video_duration_seconds=post_data.get("video_duration"),
                    audio_name=post_data.get("audio_name"),
                    likes_count=post_data.get("likes", 0),
                    comments_count=post_data.get("comments", 0),
                    shares_count=post_data.get("shares", 0),
                    saves_count=post_data.get("saves", 0),
                    views_count=post_data.get("video_views", 0) or post_data.get("plays", 0),
                    engagement_score=post_data.get("engagement_score", 0),
                )
                db.add(post)

            competitor.last_scraped_at = datetime.utcnow()
            competitor.scrape_status = "completed"
            await db.commit()

            # Extract patterns
            await pattern_extractor.extract_patterns_from_competitor(
                db, competitor.id, competitor.business_id, business_type
            )

        except Exception as e:
            competitor.scrape_status = "failed"
            competitor.scrape_error = str(e)[:500]
            await db.commit()


@router.get("/{business_id}/{competitor_id}/analysis", response_model=CompetitorAnalysis)
async def get_competitor_analysis(
    business_id: int,
    competitor_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed analysis of a competitor
    Uses caching to reduce database load and avoid re-calculating insights
    """
    # Verify business ownership
    biz_result = await db.execute(
        select(Business)
        .where(Business.id == business_id)
        .where(Business.user_id == current_user.id)
    )
    if not biz_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Business not found")

    # Check Cache
    cache_key = f"competitor_analysis_{competitor_id}"
    cached_data = await get_cached_data(db, business_id, cache_key)
    if cached_data:
        return cached_data

    # --- Cache Miss: Generate Analysis ---

    # Get competitor
    comp_result = await db.execute(
        select(Competitor)
        .where(Competitor.id == competitor_id)
        .where(Competitor.business_id == business_id)
    )
    competitor = comp_result.scalar_one_or_none()
    if not competitor:
        raise HTTPException(status_code=404, detail="Competitor not found")

    # Get top posts
    posts_result = await db.execute(
        select(ScrapedPost)
        .where(ScrapedPost.competitor_id == competitor_id)
        .order_by(ScrapedPost.engagement_score.desc())
        .limit(10)
    )
    top_posts = posts_result.scalars().all()

    # Get patterns
    patterns_result = await db.execute(
        select(ExtractedPattern)
        .where(ExtractedPattern.business_id == business_id)
        .where(ExtractedPattern.source_post_ids.contains([competitor_id]))
        .order_by(ExtractedPattern.confidence_score.desc())
    )
    patterns = patterns_result.scalars().all()

    # If no patterns found with competitor filter, get all business patterns
    if not patterns:
        patterns_result = await db.execute(
            select(ExtractedPattern)
            .where(ExtractedPattern.business_id == business_id)
            .order_by(ExtractedPattern.confidence_score.desc())
            .limit(15)
        )
        patterns = patterns_result.scalars().all()

    # Build response data objects
    top_posts_data = [
        TopPost(
            post_id=p.id,
            post_url=p.post_url,
            thumbnail_url=p.thumbnail_url,
            content_format=p.content_format.value if p.content_format else "unknown",
            engagement_score=p.engagement_score or 0,
            likes=p.likes_count,
            comments=p.comments_count,
            saves=p.saves_count,
            shares=p.shares_count,
            caption_preview=p.caption[:200] if p.caption else "",
            hook_type=p.hook_type,
            hook_text=p.hook_text,
            cta_type=p.cta_type,
            emotional_triggers=p.emotional_triggers or [],
            why_it_worked=f"Este post tuvo un engagement score de {p.engagement_score:.0f} con {p.likes_count} likes y {p.comments_count} comentarios."
        )
        for p in top_posts
    ]

    patterns_data = [
        PatternInsight(
            pattern_type=p.pattern_type.value if p.pattern_type else "unknown",
            pattern_name=p.pattern_name,
            description=p.description,
            confidence_score=p.confidence_score or 0,
            examples_count=len(p.examples) if p.examples else 0,
            avg_engagement=p.avg_engagement_score or 0,
        )
        for p in patterns[:10]
    ]

    # Extract specific insights
    best_times = []
    best_formats = []
    best_pillars = []
    trending_hashtags = []
    recommended_hooks = []
    recommended_ctas = []

    for p in patterns:
        if p.pattern_type:
            if p.pattern_type.value == "posting_time":
                best_times.extend(p.pattern_data.get("best_times", []))
            elif p.pattern_type.value == "format":
                best_formats.append(p.pattern_name)
            elif p.pattern_type.value == "content_pillar":
                best_pillars.append(p.pattern_name)
            elif p.pattern_type.value == "hashtag_strategy":
                trending_hashtags.extend(p.examples[:5] if p.examples else [])
            elif p.pattern_type.value == "hook":
                recommended_hooks.extend(p.examples[:3] if p.examples else [])
            elif p.pattern_type.value == "cta":
                recommended_ctas.extend(p.examples[:3] if p.examples else [])

    # Create response object
    analysis_result = CompetitorAnalysis(
        competitor_id=competitor.id,
        handle=competitor.handle,
        platform=competitor.platform.value,
        followers=competitor.followers_count,
        total_posts_analyzed=len(top_posts),
        analysis_date=competitor.last_scraped_at,
        top_posts=top_posts_data,
        winning_patterns=patterns_data,
        best_posting_times=list(set(best_times))[:5],
        best_formats=list(set(best_formats))[:5] or ["reels", "carousels"],
        best_content_pillars=list(set(best_pillars))[:5],
        trending_hashtags=list(set(trending_hashtags))[:10],
        recommended_hooks=list(set(recommended_hooks))[:5],
        recommended_ctas=list(set(recommended_ctas))[:5],
        overall_strategy_summary=f"@{competitor.handle} tiene {competitor.followers_count:,} seguidores con un engagement rate promedio. Sus posts más exitosos utilizan formatos visuales de transformación y hooks de preguntas.",
        key_takeaways=[
            "Los Reels con transformaciones before/after tienen el mayor engagement",
            "Las preguntas en caption aumentan significativamente los comentarios",
            "El contenido behind-the-scenes genera alta conexión con la audiencia",
            "Los hashtags locales combinados con hashtags de nicho funcionan mejor",
        ],
        content_gaps=[
            "Podrías crear más contenido educativo tipo 'tips rápidos'",
            "Hay oportunidad en videos storytime con narrativas personales",
            "El formato carousel educativo está subexplotado en este nicho",
        ]
    )

    # Convert to dict for caching (Pydantic to JSON-compatible dict)
    analysis_dict = json.loads(analysis_result.model_dump_json())

    # Save to Cache
    await set_cached_data(db, business_id, cache_key, analysis_dict)

    return analysis_result


@router.get("/{business_id}/{competitor_id}/posts", response_model=List[ScrapedPostSummary])
async def get_competitor_posts(
    business_id: int,
    competitor_id: int,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get scraped posts from a competitor
    """
    # Verify access
    biz_result = await db.execute(
        select(Business)
        .where(Business.id == business_id)
        .where(Business.user_id == current_user.id)
    )
    if not biz_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Business not found")

    # Get posts
    posts_result = await db.execute(
        select(ScrapedPost)
        .where(ScrapedPost.competitor_id == competitor_id)
        .order_by(ScrapedPost.engagement_score.desc())
        .limit(limit)
    )
    posts = posts_result.scalars().all()

    return [
        ScrapedPostSummary(
            id=p.id,
            platform_post_id=p.platform_post_id,
            post_url=p.post_url,
            content_format=p.content_format.value if p.content_format else "unknown",
            caption_preview=p.caption[:200] if p.caption else None,
            thumbnail_url=p.thumbnail_url,
            likes_count=p.likes_count,
            comments_count=p.comments_count,
            shares_count=p.shares_count,
            saves_count=p.saves_count,
            views_count=p.views_count,
            engagement_score=p.engagement_score or 0,
            posted_at=p.posted_at,
            hook_type=p.hook_type,
            cta_type=p.cta_type,
        )
        for p in posts
    ]


@router.delete("/{business_id}/{competitor_id}")
async def delete_competitor(
    business_id: int,
    competitor_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Remove a competitor
    """
    # Verify access
    biz_result = await db.execute(
        select(Business)
        .where(Business.id == business_id)
        .where(Business.user_id == current_user.id)
    )
    if not biz_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Business not found")

    # Delete competitor (cascade deletes posts)
    comp_result = await db.execute(
        select(Competitor)
        .where(Competitor.id == competitor_id)
        .where(Competitor.business_id == business_id)
    )
    competitor = comp_result.scalar_one_or_none()
    if not competitor:
        raise HTTPException(status_code=404, detail="Competitor not found")

    await db.delete(competitor)
    await db.commit()

    return {"message": "Competitor removed successfully"}
