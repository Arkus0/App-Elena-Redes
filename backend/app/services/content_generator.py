"""
Content Generator Service
The magic engine that creates high-engagement content
Now with Hybrid ML/LLM architecture for cost-efficient predictions
"""
import logging
import random
from typing import List, Dict, Any, Optional
from datetime import datetime, date, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.models.business import Business
from app.models.competitor import Competitor
from app.models.scraped_post import ScrapedPost
from app.models.pattern import ExtractedPattern, PatternType
from app.models.content import GeneratedContent, ContentCalendar, ContentGoal, ContentStatus
from app.models.abtest import ABTestExperiment
from app.services.ai_service import AIService
from app.services.apify_service import ApifyService
from app.services.ml_service import get_ml_predictor

logger = logging.getLogger(__name__)


class ContentGenerator:
    """
    Generates high-engagement content based on competitor analysis
    and proven engagement patterns
    """

    def __init__(self):
        self.ai_service = AIService()
        self.apify_service = ApifyService()
        self.ml_predictor = get_ml_predictor()

    async def generate_calendar(
        self,
        db: AsyncSession,
        business_id: int,
        month: int,
        year: int,
        posts_count: int = 15,
        primary_goal: str = "engagement",
        platforms: List[str] = None,
        content_mix: Dict[str, int] = None,
        refresh_data: bool = True
    ) -> ContentCalendar:
        """
        Generate a full content calendar for a month
        """
        platforms = platforms or ["instagram", "tiktok"]

        # Get business info
        result = await db.execute(
            select(Business).where(Business.id == business_id)
        )
        business = result.scalar_one_or_none()
        if not business:
            raise ValueError(f"Business {business_id} not found")

        # Get patterns for this business
        patterns_result = await db.execute(
            select(ExtractedPattern)
            .where(ExtractedPattern.business_id == business_id)
            .order_by(ExtractedPattern.confidence_score.desc())
            .limit(20)
        )
        patterns = patterns_result.scalars().all()

        # Get top posts from competitors for reference
        top_posts = await self._get_top_competitor_posts(db, business_id, limit=15)

        # Create calendar
        calendar = ContentCalendar(
            business_id=business_id,
            name=f"Calendario {self._get_month_name(month)} {year} - {business.name}",
            month=month,
            year=year,
            primary_goal=ContentGoal(primary_goal) if primary_goal in [e.value for e in ContentGoal] else ContentGoal.ENGAGEMENT,
            target_posts_count=posts_count,
            content_mix=content_mix or self._get_default_content_mix(),
            patterns_snapshot=[p.id for p in patterns],
            status="generating",
        )
        db.add(calendar)
        await db.flush()  # Get calendar ID

        # Determine content distribution
        content_plan = self._plan_content_distribution(
            posts_count, platforms, content_mix or self._get_default_content_mix()
        )

        # === BANDIT INTEGRATION ===
        # Check for active experiment
        experiment_result = await db.execute(
            select(ABTestExperiment)
            .options(selectinload(ABTestExperiment.variants))
            .where(ABTestExperiment.business_id == business_id)
            .where(ABTestExperiment.is_active == True)
            .where(ABTestExperiment.test_name == "Format Optimization")
        )
        experiment = experiment_result.scalar_one_or_none()

        # Generate each content piece
        business_info = {
            "name": business.name,
            "business_type": business.business_type.value,
            "location": business.location,
            "brand_voice": business.brand_voice,
            "instagram_handle": business.instagram_handle,
            "tiktok_handle": business.tiktok_handle,
        }

        patterns_data = [
            {
                "type": p.pattern_type.value,
                "name": p.pattern_name,
                "description": p.description,
                "examples": p.examples,
                "data": p.pattern_data,
            }
            for p in patterns
        ]

        generated_pieces = []
        schedule_date = date(year, month, 1)

        for item in content_plan:
            platform = item["platform"]
            content_format = item["format"]
            goal = item.get("goal", primary_goal)

            # === APPLY BANDIT RECOMMENDATION ===
            bandit_recommendation = None
            if experiment:
                recommended_variant = self._perform_thompson_sampling(experiment)

                # Check compatibility
                compatible = True
                # TikTok is generally video only (reels/tiktok_video)
                if platform == "tiktok" and recommended_variant not in ["reel", "tiktok_video", "video"]:
                    compatible = False

                # If compatible, use it!
                if compatible:
                    content_format = recommended_variant
                    bandit_recommendation = recommended_variant

            # Get relevant top posts for this format
            relevant_posts = [
                p for p in top_posts
                if self._match_format(p, content_format)
            ][:3]

            # === HYBRID ML/LLM APPROACH ===
            # Step 1: Get ML predictions FIRST (fast, cost-efficient)
            ml_input = {
                "caption": "",  # Empty for pre-generation prediction
                "content_format": content_format,
                "business_type": business.business_type.value,
                "hashtags": [],
            }
            ml_prediction = self.ml_predictor.get_full_prediction(ml_input)

            # Step 2: Inject ML recommendations into LLM prompt
            ml_recommendations = {
                "recommended_format": ml_prediction.get("format_recommendation", {}).get("recommended_format", content_format),
                "trigger_suggestions": [
                    s.get("trigger_type") for s in ml_prediction.get("trigger_suggestions", {}).get("suggestions", [])
                ],
                "optimization_tips": ml_prediction.get("optimization_suggestions", []),
            }

            # Inject Bandit Message
            if bandit_recommendation:
                ml_recommendations["bandit_message"] = (
                    f"The mathematical optimizer has determined that a {bandit_recommendation.upper()} "
                    f"is the optimal format for this niche. Generate the content specifically for a {bandit_recommendation} structure."
                )

            # Step 3: Generate content with AI (LLM) using ML recommendations
            content_data = await self.ai_service.generate_content_piece(
                business_info=business_info,
                patterns=patterns_data,
                platform=platform,
                content_format=content_format,
                goal=goal,
                similar_top_posts=relevant_posts,
                ml_recommendations=ml_recommendations  # Inject ML insights
            )

            # Create content record
            content_piece = GeneratedContent(
                business_id=business_id,
                calendar_id=calendar.id,
                title=content_data.get("title", f"{content_format.title()} - {schedule_date}"),
                platform=platform,
                content_format=content_format,
                status=ContentStatus.DRAFT,
                caption=content_data.get("caption", ""),
                hashtags=content_data.get("hashtags", []),
                hook_text=content_data.get("hook_text", ""),
                video_script=self._format_script(content_data.get("video_script")),
                script_structure=content_data.get("video_script", {}),
                filming_guide=self._format_filming_guide(content_data.get("filming_guide")),
                visual_directions=content_data.get("filming_guide", {}).get("angles", []),
                text_overlays=content_data.get("filming_guide", {}).get("text_overlays", []),
                recommended_audio=content_data.get("recommended_audio", ""),
                audio_alternatives=content_data.get("audio_alternatives", []),
                image_prompt=content_data.get("image_prompt", ""),
                cta_type=content_data.get("cta_type", ""),
                content_goal=ContentGoal(goal) if goal in [e.value for e in ContentGoal] else None,
                framework_used=content_data.get("framework_used", ""),
                scheduled_date=schedule_date,
                optimal_posting_time=self._get_optimal_time(patterns_data, platform),
            )

            # === HYBRID ENGAGEMENT PREDICTION ===
            # Step 4: Use ML for fast engagement scoring (not LLM!)
            final_ml_prediction = self.ml_predictor.predict_engagement({
                "caption": content_data.get("caption", ""),
                "hashtags": content_data.get("hashtags", []),
                "content_format": content_format,
                "business_type": business.business_type.value,
                "video_duration_seconds": 30 if content_format in ["reel", "tiktok_video"] else 0,
            })

            content_piece.engagement_score = final_ml_prediction.get("score", 0)
            content_piece.engagement_explanation = final_ml_prediction.get("explanation", {}).get(
                "explanation_text",
                "Predicción ML basada en patrones de contenido exitoso"
            )

            # Store ML metadata for dashboard display
            content_piece.ml_prediction_data = {
                "score": final_ml_prediction.get("score"),
                "confidence": final_ml_prediction.get("confidence"),
                "top_factors": final_ml_prediction.get("explanation", {}).get("top_positive_factors", []),
                "format_recommendation": ml_prediction.get("format_recommendation", {}),
                "triggers_used": ml_recommendations.get("trigger_suggestions", []),
                "bandit_recommendation": bandit_recommendation,
            }
            content_piece.similar_viral_posts = [
                {"id": p.get("id"), "engagement": p.get("engagement_score")}
                for p in relevant_posts
            ]
            content_piece.patterns_used = [p.pattern_name for p in patterns[:3]]

            db.add(content_piece)
            generated_pieces.append(content_piece)

            # Move to next posting date (skip some days for natural spacing)
            schedule_date = self._get_next_posting_date(schedule_date, posts_count, month, year)

        # Update calendar status
        calendar.status = "active"
        calendar.generated_at = datetime.utcnow()
        calendar.competitors_analyzed = [c.id for c in (await self._get_competitors(db, business_id))]

        await db.commit()

        return calendar

    async def generate_single_content(
        self,
        db: AsyncSession,
        business_id: int,
        platform: str,
        content_format: str,
        goal: str = "engagement",
        specific_topic: str = None
    ) -> GeneratedContent:
        """
        Generate a single piece of content
        """
        # Get business and patterns
        result = await db.execute(
            select(Business).where(Business.id == business_id)
        )
        business = result.scalar_one_or_none()
        if not business:
            raise ValueError(f"Business {business_id} not found")

        patterns_result = await db.execute(
            select(ExtractedPattern)
            .where(ExtractedPattern.business_id == business_id)
            .order_by(ExtractedPattern.confidence_score.desc())
            .limit(10)
        )
        patterns = patterns_result.scalars().all()

        business_info = {
            "name": business.name,
            "business_type": business.business_type.value,
            "location": business.location,
            "brand_voice": business.brand_voice,
        }

        patterns_data = [
            {
                "type": p.pattern_type.value,
                "name": p.pattern_name,
                "description": p.description,
                "examples": p.examples,
            }
            for p in patterns
        ]

        # Generate content
        content_data = await self.ai_service.generate_content_piece(
            business_info=business_info,
            patterns=patterns_data,
            platform=platform,
            content_format=content_format,
            goal=goal
        )

        # Create record
        content_piece = GeneratedContent(
            business_id=business_id,
            title=content_data.get("title", f"{content_format.title()} Content"),
            platform=platform,
            content_format=content_format,
            status=ContentStatus.DRAFT,
            caption=content_data.get("caption", ""),
            hashtags=content_data.get("hashtags", []),
            hook_text=content_data.get("hook_text", ""),
            video_script=self._format_script(content_data.get("video_script")),
            script_structure=content_data.get("video_script", {}),
            filming_guide=self._format_filming_guide(content_data.get("filming_guide")),
            visual_directions=content_data.get("filming_guide", {}).get("angles", []),
            text_overlays=content_data.get("filming_guide", {}).get("text_overlays", []),
            recommended_audio=content_data.get("recommended_audio", ""),
            audio_alternatives=content_data.get("audio_alternatives", []),
            image_prompt=content_data.get("image_prompt", ""),
            cta_type=content_data.get("cta_type", ""),
            content_goal=ContentGoal(goal) if goal in [e.value for e in ContentGoal] else None,
            framework_used=content_data.get("framework_used", ""),
        )

        # Predict engagement
        prediction = await self.ai_service.predict_engagement(
            content_data, patterns_data, {"avg_likes": 2000, "avg_comments": 100}
        )
        content_piece.engagement_score = prediction.get("score", 0)
        content_piece.engagement_explanation = prediction.get("explanation", "")

        db.add(content_piece)
        await db.commit()

        return content_piece

    async def generate_variations(
        self,
        db: AsyncSession,
        content_id: int,
        num_variations: int = 2
    ) -> List[GeneratedContent]:
        """
        Generate A/B variations of existing content
        """
        # Get original content
        result = await db.execute(
            select(GeneratedContent).where(GeneratedContent.id == content_id)
        )
        original = result.scalar_one_or_none()
        if not original:
            raise ValueError(f"Content {content_id} not found")

        # Get business and patterns
        business_result = await db.execute(
            select(Business).where(Business.id == original.business_id)
        )
        business = business_result.scalar_one_or_none()

        patterns_result = await db.execute(
            select(ExtractedPattern)
            .where(ExtractedPattern.business_id == original.business_id)
            .limit(10)
        )
        patterns = patterns_result.scalars().all()

        variations = []
        variation_labels = ["B", "C", "D"]

        for i in range(min(num_variations, 3)):
            # Generate variation with different hook/CTA
            business_info = {
                "name": business.name,
                "business_type": business.business_type.value,
                "location": business.location,
                "brand_voice": business.brand_voice,
            }

            patterns_data = [
                {
                    "type": p.pattern_type.value,
                    "name": p.pattern_name,
                    "description": p.description,
                    "examples": p.examples,
                }
                for p in patterns
            ]

            # Slightly modify the goal to get variations
            variation_goals = ["engagement", "leads", "awareness"]
            goal = variation_goals[i % len(variation_goals)]

            content_data = await self.ai_service.generate_content_piece(
                business_info=business_info,
                patterns=patterns_data,
                platform=original.platform,
                content_format=original.content_format,
                goal=goal
            )

            variation = GeneratedContent(
                business_id=original.business_id,
                calendar_id=original.calendar_id,
                title=f"{original.title} - Variación {variation_labels[i]}",
                platform=original.platform,
                content_format=original.content_format,
                status=ContentStatus.DRAFT,
                caption=content_data.get("caption", ""),
                hashtags=content_data.get("hashtags", []),
                hook_text=content_data.get("hook_text", ""),
                video_script=self._format_script(content_data.get("video_script")),
                script_structure=content_data.get("video_script", {}),
                filming_guide=self._format_filming_guide(content_data.get("filming_guide")),
                recommended_audio=content_data.get("recommended_audio", ""),
                audio_alternatives=content_data.get("audio_alternatives", []),
                image_prompt=content_data.get("image_prompt", ""),
                cta_type=content_data.get("cta_type", ""),
                framework_used=content_data.get("framework_used", ""),
                variation_of=original.id,
                variation_label=variation_labels[i],
            )

            # Predict engagement for variation
            prediction = await self.ai_service.predict_engagement(
                content_data, patterns_data, {"avg_likes": 2000, "avg_comments": 100}
            )
            variation.engagement_score = prediction.get("score", 0)
            variation.engagement_explanation = prediction.get("explanation", "")

            db.add(variation)
            variations.append(variation)

        # Mark original as having variations
        original.variation_label = "A"

        await db.commit()

        return variations

    async def scan_viral_opportunities(
        self,
        db: AsyncSession,
        business_id: int,
        keyword: str,
        platform: str = "instagram"
    ) -> List[Dict[str, Any]]:
        """
        Scan for viral opportunities based on trending content
        """
        # Get business info
        result = await db.execute(
            select(Business).where(Business.id == business_id)
        )
        business = result.scalar_one_or_none()
        if not business:
            raise ValueError(f"Business {business_id} not found")

        # Search trending content via Apify
        trending_data = await self.apify_service.search_trending_content(
            keyword, platform, max_results=20
        )

        # Get AI to generate ideas based on trends
        business_info = {
            "name": business.name,
            "business_type": business.business_type.value,
            "location": business.location,
        }

        ideas = await self.ai_service.generate_viral_ideas(
            trending_data, business_info, keyword
        )

        # Enhance with relevance scores
        for idea in ideas:
            idea["relevance_score"] = self._calculate_relevance_score(
                idea, business.business_type.value
            )

        return sorted(ideas, key=lambda x: x.get("relevance_score", 0), reverse=True)

    # ============ HELPER METHODS ============

    def _perform_thompson_sampling(self, experiment: ABTestExperiment) -> str:
        """
        Thompson Sampling: Sample from Beta(alpha, beta) for each variant.
        Return the variant name with the highest sample.
        """
        best_variant = None
        max_sample = -1.0

        if not experiment.variants:
            return "reel"  # Default fallback

        for variant in experiment.variants:
            # Sample from Beta distribution
            sample = random.betavariate(variant.alpha_param, variant.beta_param)
            if sample > max_sample:
                max_sample = sample
                best_variant = variant.variant_name

        return best_variant or "reel"

    async def _get_competitors(self, db: AsyncSession, business_id: int) -> List[Competitor]:
        """Get all competitors for a business"""
        result = await db.execute(
            select(Competitor).where(Competitor.business_id == business_id)
        )
        return result.scalars().all()

    async def _get_top_competitor_posts(
        self,
        db: AsyncSession,
        business_id: int,
        limit: int = 15
    ) -> List[Dict[str, Any]]:
        """Get top performing posts from all competitors"""
        # Get competitor IDs
        competitors = await self._get_competitors(db, business_id)
        if not competitors:
            return []

        competitor_ids = [c.id for c in competitors]

        # Get top posts
        result = await db.execute(
            select(ScrapedPost)
            .where(ScrapedPost.competitor_id.in_(competitor_ids))
            .order_by(ScrapedPost.engagement_score.desc())
            .limit(limit)
        )
        posts = result.scalars().all()

        return [
            {
                "id": p.id,
                "caption": p.caption[:300] if p.caption else "",
                "format": p.content_format.value if p.content_format else "unknown",
                "engagement_score": p.engagement_score,
                "likes": p.likes_count,
                "comments": p.comments_count,
                "hook_text": p.hook_text,
                "cta_type": p.cta_type,
            }
            for p in posts
        ]

    def _get_default_content_mix(self) -> Dict[str, int]:
        """Default content mix percentages"""
        return {
            "reels": 50,
            "carousels": 25,
            "static": 15,
            "stories": 10,
        }

    def _plan_content_distribution(
        self,
        posts_count: int,
        platforms: List[str],
        content_mix: Dict[str, int]
    ) -> List[Dict[str, Any]]:
        """Plan content distribution across platforms and formats"""
        plan = []

        # Platform distribution
        platform_counts = {}
        posts_per_platform = posts_count // len(platforms)
        for i, platform in enumerate(platforms):
            if i == len(platforms) - 1:
                # Last platform gets remainder
                platform_counts[platform] = posts_count - sum(platform_counts.values())
            else:
                platform_counts[platform] = posts_per_platform

        # Format distribution per platform
        for platform, count in platform_counts.items():
            format_map = self._get_platform_formats(platform)

            for fmt, pct in content_mix.items():
                if fmt in format_map:
                    num_posts = max(1, int(count * pct / 100))
                    for _ in range(num_posts):
                        if len([p for p in plan if p["platform"] == platform]) < count:
                            plan.append({
                                "platform": platform,
                                "format": format_map[fmt],
                            })

        # Ensure we have exact count
        while len(plan) < posts_count:
            plan.append({
                "platform": random.choice(platforms),
                "format": "reel" if "instagram" in platforms else "tiktok_video",
            })

        return plan[:posts_count]

    def _get_platform_formats(self, platform: str) -> Dict[str, str]:
        """Map generic formats to platform-specific formats"""
        if platform == "instagram":
            return {
                "reels": "reel",
                "carousels": "carousel",
                "static": "static_image",
                "stories": "story",
            }
        elif platform == "tiktok":
            return {
                "reels": "tiktok_video",
                "carousels": "tiktok_video",
                "static": "tiktok_video",
                "stories": "tiktok_video",
            }
        elif platform == "linkedin":
            return {
                "reels": "linkedin_post",
                "carousels": "linkedin_carousel",
                "static": "linkedin_post",
                "stories": "linkedin_post",
            }
        return {"reels": "reel"}

    def _match_format(self, post: Dict, content_format: str) -> bool:
        """Check if post matches the target format"""
        post_format = post.get("format", "").lower()
        if content_format in ["reel", "tiktok_video"]:
            return post_format in ["reel", "tiktok_video", "video"]
        return post_format == content_format

    def _format_script(self, script_data: Dict) -> Optional[str]:
        """Format video script for display"""
        if not script_data:
            return None

        parts = []
        if script_data.get("hook_0_3s"):
            parts.append(f"[0-3s HOOK]\n{script_data['hook_0_3s']}")
        if script_data.get("content_3_20s"):
            parts.append(f"\n[3-20s CONTENIDO]\n{script_data['content_3_20s']}")
        if script_data.get("cta_20_30s"):
            parts.append(f"\n[20-30s CTA]\n{script_data['cta_20_30s']}")

        return "\n".join(parts) if parts else None

    def _format_filming_guide(self, guide_data: Dict) -> Optional[str]:
        """Format filming guide for display"""
        if not guide_data:
            return None

        sections = []

        if guide_data.get("setup"):
            sections.append(f"📍 SETUP: {guide_data['setup']}")

        if guide_data.get("equipment"):
            sections.append(f"📷 EQUIPO: {', '.join(guide_data['equipment'])}")

        if guide_data.get("angles"):
            sections.append(f"🎬 ÁNGULOS: {', '.join(guide_data['angles'])}")

        if guide_data.get("text_overlays"):
            overlays = [f"  • '{t['text']}' ({t.get('timing', 'N/A')})" for t in guide_data['text_overlays']]
            sections.append(f"📝 TEXTOS EN PANTALLA:\n" + "\n".join(overlays))

        if guide_data.get("b_roll"):
            sections.append(f"🎞️ B-ROLL: {', '.join(guide_data['b_roll'])}")

        if guide_data.get("duration"):
            sections.append(f"⏱️ DURACIÓN: {guide_data['duration']}")

        return "\n\n".join(sections) if sections else None

    def _get_optimal_time(self, patterns: List[Dict], platform: str) -> str:
        """Get optimal posting time from patterns"""
        for p in patterns:
            if p.get("type") == "posting_time":
                times = p.get("data", {}).get("best_times", [])
                if times:
                    return times[0]

        # Defaults by platform
        defaults = {
            "instagram": "11:00 AM",
            "tiktok": "7:00 PM",
            "linkedin": "9:00 AM",
        }
        return defaults.get(platform, "12:00 PM")

    def _get_next_posting_date(
        self,
        current_date: date,
        total_posts: int,
        month: int,
        year: int
    ) -> date:
        """Calculate next posting date with natural spacing"""
        # Calculate days between posts
        import calendar
        days_in_month = calendar.monthrange(year, month)[1]
        spacing = max(1, days_in_month // total_posts)

        next_date = current_date + timedelta(days=spacing)

        # Stay within the month
        if next_date.month != month:
            next_date = date(year, month, days_in_month)

        return next_date

    def _get_month_name(self, month: int) -> str:
        """Get Spanish month name"""
        months = {
            1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
            5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
            9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
        }
        return months.get(month, "Mes")

    def _calculate_relevance_score(self, idea: Dict, business_type: str) -> float:
        """Calculate how relevant a viral idea is to the business"""
        score = 50.0  # Base score

        # Difficulty bonus (easier = higher score)
        difficulty_map = {"facil": 20, "medio": 10, "dificil": 0}
        score += difficulty_map.get(idea.get("difficulty", "medio"), 10)

        # Time efficiency bonus
        time = idea.get("time_to_create", "1 hora")
        if "30" in time or "20" in time:
            score += 15
        elif "1 hora" in time or "45" in time:
            score += 10

        # Platform preference
        if idea.get("platform") in ["instagram", "tiktok"]:
            score += 10

        return min(score, 100)
