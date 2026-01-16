"""
Business API Routes
Onboarding and business management
"""
from datetime import datetime
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.business import Business, BusinessType
from app.models.competitor import Competitor
from app.models.business import Platform
from app.schemas.business import (
    BusinessCreate,
    BusinessUpdate,
    BusinessResponse,
    OnboardingRequest,
    OnboardingResponse,
)
from app.services.apify_service import ApifyService
from app.services.pattern_extractor import PatternExtractor

router = APIRouter()


@router.post("/onboard", response_model=OnboardingResponse, status_code=status.HTTP_201_CREATED)
async def onboard_business(
    request: OnboardingRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Complete onboarding flow:
    1. Create business profile
    2. Add competitors
    3. Trigger scraping (background)
    4. Return status
    """
    # Create business
    business = Business(
        user_id=current_user.id,
        name=request.name,
        business_type=request.business_type,
        description=request.description,
        location=request.location,
        instagram_handle=request.instagram_handle,
        tiktok_handle=request.tiktok_handle,
        linkedin_handle=request.linkedin_handle,
        active_platforms=request.active_platforms,
        content_goals=request.content_goals,
        posting_frequency=request.posting_frequency,
        brand_voice=request.brand_voice,
    )
    db.add(business)
    await db.flush()  # Get business ID

    # Add competitors
    competitors_added = 0
    for comp in request.competitors:
        platform_enum = Platform(comp.platform) if comp.platform in [p.value for p in Platform] else Platform.INSTAGRAM

        competitor = Competitor(
            business_id=business.id,
            platform=platform_enum,
            handle=comp.handle.lstrip("@"),  # Remove @ if present
            scrape_status="pending",
        )
        db.add(competitor)
        competitors_added += 1

    await db.commit()

    # Trigger background scraping
    background_tasks.add_task(
        scrape_competitors_background,
        business.id,
        request.business_type.value
    )

    return OnboardingResponse(
        business_id=business.id,
        message="Negocio creado exitosamente. Análisis de competidores en progreso.",
        competitors_added=competitors_added,
        scraping_status="in_progress",
        estimated_analysis_time="5-10 minutos",
        next_steps=[
            "Espera a que se complete el análisis de competidores",
            "Revisa los patrones de engagement identificados",
            "Genera tu primer calendario de contenido",
        ]
    )


async def scrape_competitors_background(business_id: int, business_type: str):
    """
    Background task to scrape all competitors
    """
    from app.core.database import async_session_maker

    apify_service = ApifyService()
    pattern_extractor = PatternExtractor()

    async with async_session_maker() as db:
        # Get business
        result = await db.execute(
            select(Business).where(Business.id == business_id)
        )
        business = result.scalar_one_or_none()
        if not business:
            return

        # Get competitors
        result = await db.execute(
            select(Competitor).where(Competitor.business_id == business_id)
        )
        competitors = result.scalars().all()

        for competitor in competitors:
            try:
                # Update status
                competitor.scrape_status = "in_progress"
                await db.commit()

                # Scrape based on platform
                if competitor.platform == Platform.INSTAGRAM:
                    data = await apify_service.scrape_instagram_profile(competitor.handle)
                elif competitor.platform == Platform.TIKTOK:
                    data = await apify_service.scrape_tiktok_profile(competitor.handle)
                else:
                    data = await apify_service.scrape_linkedin_profile(competitor.handle)

                # Update competitor with profile data
                profile = data.get("profile", {})
                competitor.display_name = profile.get("full_name") or profile.get("display_name")
                competitor.bio = profile.get("bio")
                competitor.followers_count = profile.get("followers", 0)
                competitor.following_count = profile.get("following", 0)
                competitor.posts_count = profile.get("posts_count") or profile.get("videos_count", 0)

                # Save scraped posts
                from app.models.scraped_post import ScrapedPost, ContentFormat

                for post_data in data.get("posts", []):
                    # Map format
                    format_map = {
                        "reel": ContentFormat.REEL,
                        "carousel": ContentFormat.CAROUSEL,
                        "static_image": ContentFormat.STATIC_IMAGE,
                        "tiktok_video": ContentFormat.TIKTOK_VIDEO,
                        "linkedin_post": ContentFormat.LINKEDIN_POST,
                        "linkedin_carousel": ContentFormat.LINKEDIN_CAROUSEL,
                    }
                    content_format = format_map.get(
                        post_data.get("type", "static_image"),
                        ContentFormat.STATIC_IMAGE
                    )

                    post = ScrapedPost(
                        competitor_id=competitor.id,
                        platform_post_id=post_data.get("platform_id", ""),
                        post_url=post_data.get("url"),
                        content_format=content_format,
                        caption=post_data.get("caption"),
                        hashtags=post_data.get("hashtags", []),
                        mentions=post_data.get("mentions", []),
                        thumbnail_url=post_data.get("thumbnail"),
                        media_urls=post_data.get("media_urls", []),
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
                    db, competitor.id, business_id, business_type
                )

            except Exception as e:
                competitor.scrape_status = "failed"
                competitor.scrape_error = str(e)[:500]
                await db.commit()

        # Update business onboarding status
        business.onboarding_completed = datetime.utcnow()
        business.last_analysis_at = datetime.utcnow()
        await db.commit()


@router.get("/me", response_model=List[BusinessResponse])
async def get_my_businesses(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get all businesses for current user
    """
    result = await db.execute(
        select(Business).where(Business.user_id == current_user.id)
    )
    businesses = result.scalars().all()
    return businesses


@router.get("/{business_id}", response_model=BusinessResponse)
async def get_business(
    business_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get a specific business
    """
    result = await db.execute(
        select(Business)
        .where(Business.id == business_id)
        .where(Business.user_id == current_user.id)
    )
    business = result.scalar_one_or_none()

    if not business:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Business not found"
        )

    return business


@router.put("/{business_id}", response_model=BusinessResponse)
async def update_business(
    business_id: int,
    update_data: BusinessUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update business settings
    """
    result = await db.execute(
        select(Business)
        .where(Business.id == business_id)
        .where(Business.user_id == current_user.id)
    )
    business = result.scalar_one_or_none()

    if not business:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Business not found"
        )

    # Update fields
    update_dict = update_data.model_dump(exclude_unset=True)
    for field, value in update_dict.items():
        setattr(business, field, value)

    await db.commit()
    await db.refresh(business)

    return business


@router.get("/{business_id}/status")
async def get_business_status(
    business_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get onboarding/analysis status
    """
    result = await db.execute(
        select(Business)
        .where(Business.id == business_id)
        .where(Business.user_id == current_user.id)
    )
    business = result.scalar_one_or_none()

    if not business:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Business not found"
        )

    # Get competitor scraping status
    comp_result = await db.execute(
        select(Competitor).where(Competitor.business_id == business_id)
    )
    competitors = comp_result.scalars().all()

    comp_status = {
        "total": len(competitors),
        "completed": len([c for c in competitors if c.scrape_status == "completed"]),
        "in_progress": len([c for c in competitors if c.scrape_status == "in_progress"]),
        "failed": len([c for c in competitors if c.scrape_status == "failed"]),
        "pending": len([c for c in competitors if c.scrape_status == "pending"]),
    }

    # Get pattern count
    from app.models.pattern import ExtractedPattern
    pattern_result = await db.execute(
        select(ExtractedPattern).where(ExtractedPattern.business_id == business_id)
    )
    patterns = pattern_result.scalars().all()

    return {
        "business_id": business_id,
        "onboarding_completed": business.onboarding_completed is not None,
        "last_analysis": business.last_analysis_at,
        "competitors": comp_status,
        "patterns_extracted": len(patterns),
        "ready_for_content": comp_status["completed"] > 0 and len(patterns) > 0,
    }
