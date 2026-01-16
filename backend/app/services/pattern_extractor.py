"""
Pattern Extractor Service
Extracts winning patterns from scraped competitor posts
"""
import logging
from typing import List, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.pattern import ExtractedPattern, PatternType
from app.models.scraped_post import ScrapedPost
from app.models.competitor import Competitor
from app.services.ai_service import AIService

logger = logging.getLogger(__name__)


class PatternExtractor:
    """
    Extracts engagement patterns from competitor posts
    Uses AI to identify what makes content successful
    """

    def __init__(self):
        self.ai_service = AIService()

    async def extract_patterns_from_competitor(
        self,
        db: AsyncSession,
        competitor_id: int,
        business_id: int,
        business_type: str
    ) -> List[ExtractedPattern]:
        """
        Extract patterns from a single competitor's posts
        """
        # Get competitor's top posts
        result = await db.execute(
            select(ScrapedPost)
            .where(ScrapedPost.competitor_id == competitor_id)
            .order_by(ScrapedPost.engagement_score.desc())
            .limit(20)
        )
        posts = result.scalars().all()

        if not posts:
            logger.warning(f"No posts found for competitor {competitor_id}")
            return []

        # Get competitor info
        competitor_result = await db.execute(
            select(Competitor).where(Competitor.id == competitor_id)
        )
        competitor = competitor_result.scalar_one_or_none()

        if not competitor:
            return []

        # Convert posts to dict for AI analysis
        posts_data = [
            {
                "caption": p.caption,
                "type": p.content_format.value if p.content_format else "unknown",
                "likes": p.likes_count,
                "comments": p.comments_count,
                "saves": p.saves_count,
                "video_views": p.views_count,
                "audio_name": p.audio_name,
                "hashtags": p.hashtags or [],
                "engagement_score": p.engagement_score,
            }
            for p in posts
        ]

        # Get AI analysis
        analysis = await self.ai_service.analyze_posts_for_patterns(
            posts_data,
            business_type,
            competitor.platform.value
        )

        # Create pattern records
        patterns = []

        # Hook patterns
        for hook_data in analysis.get("hook_patterns", []):
            pattern = ExtractedPattern(
                business_id=business_id,
                pattern_type=PatternType.HOOK,
                pattern_name=hook_data.get("pattern_name", "Unknown Hook"),
                description=hook_data.get("description", ""),
                examples=hook_data.get("examples", []),
                platform=competitor.platform.value,
                avg_engagement_score=hook_data.get("avg_engagement", 0),
                usage_count=len(hook_data.get("examples", [])),
                pattern_data={
                    "best_for": hook_data.get("best_for", []),
                    "trigger_type": hook_data.get("pattern_name", "").lower(),
                },
                business_types=[business_type],
                confidence_score=80.0,
                source_post_ids=[p.id for p in posts[:5]],
            )
            patterns.append(pattern)

        # CTA patterns
        for cta_data in analysis.get("cta_patterns", []):
            effectiveness_map = {"alta": 90, "media": 70, "baja": 50}
            pattern = ExtractedPattern(
                business_id=business_id,
                pattern_type=PatternType.CTA,
                pattern_name=cta_data.get("pattern_name", "Unknown CTA"),
                description=cta_data.get("description", ""),
                examples=cta_data.get("examples", []),
                platform=competitor.platform.value,
                success_rate=effectiveness_map.get(cta_data.get("effectiveness", "media"), 70),
                pattern_data={
                    "effectiveness": cta_data.get("effectiveness", "media"),
                },
                business_types=[business_type],
                confidence_score=75.0,
                source_post_ids=[p.id for p in posts[:5]],
            )
            patterns.append(pattern)

        # Content pillars
        for pillar_data in analysis.get("content_pillars", []):
            pattern = ExtractedPattern(
                business_id=business_id,
                pattern_type=PatternType.CONTENT_PILLAR,
                pattern_name=pillar_data.get("pillar_name", "Unknown Pillar"),
                description=pillar_data.get("description", ""),
                examples=pillar_data.get("example_topics", []),
                platform=competitor.platform.value,
                pattern_data={
                    "frequency": pillar_data.get("frequency", "0%"),
                    "topics": pillar_data.get("example_topics", []),
                },
                business_types=[business_type],
                confidence_score=85.0,
                source_post_ids=[p.id for p in posts[:5]],
            )
            patterns.append(pattern)

        # Visual patterns
        for visual_data in analysis.get("visual_patterns", []):
            pattern = ExtractedPattern(
                business_id=business_id,
                pattern_type=PatternType.VISUAL,
                pattern_name=visual_data.get("pattern_name", "Unknown Visual"),
                description=visual_data.get("description", ""),
                platform=competitor.platform.value,
                pattern_data={
                    "usage_frequency": visual_data.get("usage_frequency", "media"),
                },
                business_types=[business_type],
                confidence_score=70.0,
                source_post_ids=[p.id for p in posts[:5]],
            )
            patterns.append(pattern)

        # Audio trends
        for audio_data in analysis.get("audio_trends", []):
            pattern = ExtractedPattern(
                business_id=business_id,
                pattern_type=PatternType.AUDIO,
                pattern_name=f"Audio: {audio_data.get('audio_type', 'Unknown')}",
                description=f"Tipo de audio que funciona: {audio_data.get('audio_type')}",
                examples=audio_data.get("examples", []),
                platform=competitor.platform.value,
                pattern_data={
                    "audio_type": audio_data.get("audio_type"),
                    "impact": audio_data.get("impact_on_reach", "medio"),
                },
                business_types=[business_type],
                confidence_score=75.0,
                source_post_ids=[p.id for p in posts[:5]],
            )
            patterns.append(pattern)

        # Caption structure
        caption_data = analysis.get("caption_structure", {})
        if caption_data:
            pattern = ExtractedPattern(
                business_id=business_id,
                pattern_type=PatternType.CAPTION_STRUCTURE,
                pattern_name="Estructura de Caption Óptima",
                description=f"Longitud: {caption_data.get('avg_length')}, Emojis: {caption_data.get('emoji_usage')}, Preguntas: {caption_data.get('question_usage')}",
                platform=competitor.platform.value,
                pattern_data=caption_data,
                business_types=[business_type],
                confidence_score=80.0,
                source_post_ids=[p.id for p in posts[:5]],
            )
            patterns.append(pattern)

        # Posting times
        best_times = analysis.get("best_posting_times", [])
        if best_times:
            pattern = ExtractedPattern(
                business_id=business_id,
                pattern_type=PatternType.POSTING_TIME,
                pattern_name="Mejores Horarios de Publicación",
                description=f"Horarios con mejor engagement: {', '.join(best_times)}",
                examples=best_times,
                platform=competitor.platform.value,
                pattern_data={
                    "best_times": best_times,
                },
                business_types=[business_type],
                confidence_score=85.0,
                source_post_ids=[p.id for p in posts[:5]],
            )
            patterns.append(pattern)

        # Hashtag strategy
        hashtag_data = analysis.get("hashtag_strategy", {})
        if hashtag_data:
            pattern = ExtractedPattern(
                business_id=business_id,
                pattern_type=PatternType.HASHTAG_STRATEGY,
                pattern_name="Estrategia de Hashtags",
                description=f"Media de {hashtag_data.get('avg_count', 0)} hashtags. Mix: {hashtag_data.get('mix', 'variado')}",
                examples=hashtag_data.get("top_hashtags", []),
                platform=competitor.platform.value,
                pattern_data=hashtag_data,
                business_types=[business_type],
                confidence_score=80.0,
                source_post_ids=[p.id for p in posts[:5]],
            )
            patterns.append(pattern)

        # Save all patterns
        for pattern in patterns:
            db.add(pattern)

        await db.commit()

        return patterns

    async def extract_all_patterns(
        self,
        db: AsyncSession,
        business_id: int,
        business_type: str
    ) -> Dict[str, Any]:
        """
        Extract patterns from all competitors of a business
        Returns aggregated insights
        """
        # Get all competitors
        result = await db.execute(
            select(Competitor)
            .where(Competitor.business_id == business_id)
            .where(Competitor.scrape_status == "completed")
        )
        competitors = result.scalars().all()

        all_patterns = []
        for competitor in competitors:
            patterns = await self.extract_patterns_from_competitor(
                db, competitor.id, business_id, business_type
            )
            all_patterns.extend(patterns)

        # Aggregate key insights
        return {
            "total_patterns": len(all_patterns),
            "patterns_by_type": self._count_by_type(all_patterns),
            "top_patterns": self._get_top_patterns(all_patterns),
            "key_insights": self._generate_insights(all_patterns),
        }

    def _count_by_type(self, patterns: List[ExtractedPattern]) -> Dict[str, int]:
        """Count patterns by type"""
        counts = {}
        for p in patterns:
            type_name = p.pattern_type.value if p.pattern_type else "unknown"
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts

    def _get_top_patterns(self, patterns: List[ExtractedPattern], limit: int = 10) -> List[Dict]:
        """Get top patterns by confidence score"""
        sorted_patterns = sorted(
            patterns,
            key=lambda x: (x.confidence_score or 0, x.avg_engagement_score or 0),
            reverse=True
        )
        return [
            {
                "name": p.pattern_name,
                "type": p.pattern_type.value if p.pattern_type else "unknown",
                "confidence": p.confidence_score,
                "description": p.description,
            }
            for p in sorted_patterns[:limit]
        ]

    def _generate_insights(self, patterns: List[ExtractedPattern]) -> List[str]:
        """Generate human-readable insights from patterns"""
        insights = []

        # Find most common hook type
        hook_patterns = [p for p in patterns if p.pattern_type == PatternType.HOOK]
        if hook_patterns:
            best_hook = max(hook_patterns, key=lambda x: x.avg_engagement_score or 0)
            insights.append(
                f"El hook más efectivo es '{best_hook.pattern_name}' con engagement promedio alto"
            )

        # Find best CTA
        cta_patterns = [p for p in patterns if p.pattern_type == PatternType.CTA]
        if cta_patterns:
            best_cta = max(cta_patterns, key=lambda x: x.success_rate or 0)
            insights.append(
                f"El CTA '{best_cta.pattern_name}' tiene la mayor tasa de éxito"
            )

        # Posting times
        time_patterns = [p for p in patterns if p.pattern_type == PatternType.POSTING_TIME]
        if time_patterns:
            all_times = []
            for tp in time_patterns:
                all_times.extend(tp.pattern_data.get("best_times", []))
            if all_times:
                insights.append(
                    f"Los mejores horarios para publicar son: {', '.join(set(all_times)[:3])}"
                )

        # Content pillars
        pillar_patterns = [p for p in patterns if p.pattern_type == PatternType.CONTENT_PILLAR]
        if pillar_patterns:
            pillar_names = [p.pattern_name for p in pillar_patterns[:3]]
            insights.append(
                f"Los pilares de contenido más exitosos son: {', '.join(pillar_names)}"
            )

        return insights
