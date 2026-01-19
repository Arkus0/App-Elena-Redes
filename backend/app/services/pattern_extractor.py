"""
Pattern Extractor Service
Extracts winning patterns from scraped competitor posts
"""
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, update

# Add project root to path for ml module imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.models.pattern import ExtractedPattern, PatternType
from app.models.scraped_post import ScrapedPost
from app.models.competitor import Competitor
from app.models.business import Business
from app.services.ai_service import AIService

logger = logging.getLogger(__name__)

# Import embedding extractor
try:
    from ml.features_embeddings import get_embedding_extractor
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    logger.warning("ML module not found. Semantic deduplication will be disabled.")
    EMBEDDINGS_AVAILABLE = False


class PatternExtractor:
    """
    Extracts engagement patterns from competitor posts
    Uses AI to identify what makes content successful
    """

    def __init__(self):
        self.ai_service = AIService()

    def _compute_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0.0

        try:
            v1 = np.array(vec1)
            v2 = np.array(vec2)

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(np.dot(v1, v2) / (norm1 * norm2))
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0

    def _deduplicate_batch(self, patterns: List[ExtractedPattern]) -> List[ExtractedPattern]:
        """
        Deduplicate patterns within the current batch before DB check.
        Uses greedy clustering: first pattern is kept, subsequent similar ones are merged/dropped.
        """
        if not EMBEDDINGS_AVAILABLE or not patterns:
            return patterns

        unique_patterns = []
        try:
            # Use "low" precision (32 dims) for efficiency as requested/recommended
            extractor = get_embedding_extractor(precision="low")

            # 1. Pre-compute embeddings for the batch
            for p in patterns:
                if not p.description:
                    continue

                # Generate embedding
                # get_raw_embedding returns full 384 dims
                raw = extractor.get_raw_embedding(p.description)

                # Transform to target dims (32 if precision="low")
                if extractor.uses_reduction:
                    emb = extractor.transform(raw)
                else:
                    emb = raw

                p.embedding = emb.tolist()

            # 2. Greedy Deduplication
            for p in patterns:
                if not p.embedding:
                    unique_patterns.append(p)
                    continue

                is_duplicate = False
                for existing in unique_patterns:
                    if not existing.embedding:
                        continue

                    sim = self._compute_similarity(p.embedding, existing.embedding)

                    # Threshold 0.90 for batch dedup too
                    if sim > 0.90:
                        is_duplicate = True
                        # Merge examples into the 'canonical' one
                        if p.examples:
                            existing.examples = (existing.examples or []) + (p.examples or [])
                            # We don't cap at 10 here yet, will do at DB merge

                        # Merge usage count logic (if it was >1 in extraction)
                        existing.usage_count = (existing.usage_count or 1) + (p.usage_count or 1)
                        break

                if not is_duplicate:
                    unique_patterns.append(p)

            return unique_patterns

        except Exception as e:
            logger.error(f"Error in batch deduplication: {e}")
            return patterns

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

        # === Semantic Deduplication & Storage ===

        # 1. Intra-Batch Deduplication
        patterns = self._deduplicate_batch(patterns)

        # 2. Fetch Existing Patterns for this Business
        # We need to compare against the client's existing knowledge base
        existing_patterns_result = await db.execute(
            select(ExtractedPattern)
            .where(ExtractedPattern.business_id == business_id)
        )
        existing_patterns = existing_patterns_result.scalars().all()

        # Pre-load embeddings for comparison (optimize logic)
        existing_vectors = []
        for ep in existing_patterns:
             # Ensure embedding exists and is valid
             if ep.embedding and isinstance(ep.embedding, list):
                 existing_vectors.append((ep, ep.embedding))

        # 3. Check-Merge-Insert Flow
        patterns_to_add = []
        updated_patterns = []

        for new_pattern in patterns:
            # Ensure embedding exists (might have been skipped if batch dedup failed or empty desc)
            if not new_pattern.embedding and EMBEDDINGS_AVAILABLE and new_pattern.description:
                try:
                    extractor = get_embedding_extractor(precision="low")
                    raw = extractor.get_raw_embedding(new_pattern.description)
                    if extractor.uses_reduction:
                        emb = extractor.transform(raw)
                    else:
                        emb = raw
                    new_pattern.embedding = emb.tolist()
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for pattern: {e}")

            match_found = False
            best_match_pattern = None
            best_match_score = 0.0

            # Vector comparison against DB patterns
            if new_pattern.embedding and existing_vectors:
                for existing_p, existing_vec in existing_vectors:
                    sim = self._compute_similarity(new_pattern.embedding, existing_vec)
                    if sim > 0.90 and sim > best_match_score:
                        best_match_score = sim
                        best_match_pattern = existing_p

            if best_match_pattern:
                # UPDATE Existing (The "Freshness" Logic)
                match_found = True

                # Update Timestamps
                best_match_pattern.last_active_at = datetime.utcnow()

                # Increment Count
                best_match_pattern.usage_count = (best_match_pattern.usage_count or 0) + 1

                # Weighted Score Update (EMA)
                # Formula: New Score = (Old * 0.3) + (New * 0.7)
                old_score = best_match_pattern.avg_engagement_score or 0.0
                new_score = new_pattern.avg_engagement_score or 0.0
                best_match_pattern.avg_engagement_score = (old_score * 0.3) + (new_score * 0.7)

                # Merge Examples (Append and Keep Last 10)
                current_examples = list(best_match_pattern.examples) if best_match_pattern.examples else []
                new_examples = new_pattern.examples or []

                # Simple append
                merged_examples = current_examples + new_examples

                # Dedup examples by string representation to be clean (optional but good)
                # (Simple string set dedup)
                unique_examples = []
                seen_examples = set()
                for ex in merged_examples:
                    ex_str = str(ex)
                    if ex_str not in seen_examples:
                        seen_examples.add(ex_str)
                        unique_examples.append(ex)

                # Keep last 10
                best_match_pattern.examples = unique_examples[-10:]

                updated_patterns.append(best_match_pattern)

            else:
                # INSERT New
                # new_pattern.created_at is default
                new_pattern.last_active_at = datetime.utcnow()
                new_pattern.usage_count = 1
                patterns_to_add.append(new_pattern)

        # Bulk save new patterns
        for p in patterns_to_add:
            db.add(p)

        # Existing patterns are attached to session so they will update on commit

        await db.commit()

        return patterns_to_add + updated_patterns

    async def extract_anti_patterns(
        self,
        db: AsyncSession,
        business_id: int,
        business_type: str
    ) -> List[ExtractedPattern]:
        """
        Extract Anti-Patterns (flaws) from failed posts in the niche.
        Noise Filter: Only runs if niche has > 50 posts.
        """
        # 1. Noise Filter: Check total posts in niche
        count_query = (
            select(func.count(ScrapedPost.id))
            .join(Competitor, ScrapedPost.competitor_id == Competitor.id)
            .join(Business, Competitor.business_id == Business.id)
            .where(Business.business_type == business_type)
        )
        count_result = await db.execute(count_query)
        total_posts_in_niche = count_result.scalar() or 0

        if total_posts_in_niche <= 50:
            logger.info(f"Skipping anti-pattern extraction: Only {total_posts_in_niche} posts in niche {business_type} (needs > 50)")
            return []

        # 2. Extract Bottom 20 Posts (Failed examples)
        query = (
            select(ScrapedPost)
            .join(Competitor, ScrapedPost.competitor_id == Competitor.id)
            .join(Business, Competitor.business_id == Business.id)
            .where(Business.business_type == business_type)
            .order_by(ScrapedPost.engagement_score.asc())  # Bottom performance
            .limit(20)
        )
        result = await db.execute(query)
        failed_posts = result.scalars().all()

        if not failed_posts:
            return []

        # 3. Prepare data for AI
        posts_data = [
            {
                "caption": p.caption,
                "type": p.content_format.value if p.content_format else "unknown",
                "engagement_score": p.engagement_score,
                "hashtags": p.hashtags or []
            }
            for p in failed_posts
        ]

        # 4. Get AI Analysis (Anti-Patterns)
        anti_patterns_data = await self.ai_service.analyze_anti_patterns(
            posts_data,
            business_type
        )

        # 5. Save Patterns
        saved_patterns = []
        for ap in anti_patterns_data:
            pattern = ExtractedPattern(
                business_id=business_id,  # Save for the requesting business
                pattern_type=PatternType.NEGATIVE_SIGNAL,
                pattern_name=ap.get("pattern_name", "Unknown Anti-Pattern"),
                description=ap.get("description", ""),
                examples=ap.get("examples", []),
                platform=None,  # Universal anti-pattern
                pattern_data={
                    "avoid_strategy": ap.get("avoid_strategy", ""),
                    "source": "niche_analysis"
                },
                business_types=[business_type],
                confidence_score=90.0,  # High confidence derived from failure data
                source_post_ids=[p.id for p in failed_posts[:5]]
            )
            db.add(pattern)
            saved_patterns.append(pattern)

        await db.commit()
        logger.info(f"Extracted {len(saved_patterns)} anti-patterns for {business_type}")
        return saved_patterns

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
