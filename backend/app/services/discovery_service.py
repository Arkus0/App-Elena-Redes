import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from app.services.apify_service import ApifyService
from app.services.ai_service import AIService
from app.schemas.competitor import CompetitorDiscoveryRequest, DiscoveredCompetitor

logger = logging.getLogger(__name__)

class DiscoveryService:
    def __init__(self):
        self.apify_service = ApifyService()
        self.ai_service = AIService()

    async def discover_competitors(self, request: CompetitorDiscoveryRequest) -> Tuple[List[DiscoveredCompetitor], List[str]]:
        """
        Main discovery method.
        Returns:
            Tuple containing:
            1. List of validated DiscoveredCompetitor objects
            2. List of raw handles returned by AI (or empty if fallback used)
        """
        # 1. Try Semantic Discovery
        raw_handles = []
        try:
            results, raw_handles = await self._discover_via_semantic_search(request)
            if results:
                logger.info(f"Semantic discovery returned {len(results)} valid competitors.")
                return results, raw_handles
            else:
                logger.warning("Semantic discovery returned 0 valid results. Triggering fallback.")
        except Exception as e:
            logger.error(f"Semantic discovery failed: {e}. Triggering fallback.")

        # 2. Fallback to Hashtag Search
        logger.info("Fallback: Executing hashtag-based discovery.")
        fallback_results = await self._discover_via_hashtags(request)
        return fallback_results, raw_handles

    async def _discover_via_semantic_search(self, request: CompetitorDiscoveryRequest) -> Tuple[List[DiscoveredCompetitor], List[str]]:
        """
        Use Grok to find handles, then verify them with Apify.
        Returns: (valid_competitors, raw_grok_handles)
        """
        # Prepare inputs for Grok
        topic = ", ".join(request.hashtags)
        location = ", ".join(request.location_keywords) if request.location_keywords else "Global"
        niche = ", ".join(request.niche_keywords) if request.niche_keywords else "General"

        logger.info(f"Starting Semantic Discovery: Topic='{topic}', Loc='{location}', Niche='{niche}'")

        # Call Grok
        handles = await self.ai_service.find_competitor_handles(topic, location, niche)
        if not handles:
            logger.warning("Grok returned no handles.")
            return [], []

        logger.info(f"Grok returned {len(handles)} handles: {handles}")

        # Verify handles with Apify (Concurrent batching)
        valid_competitors = []
        batch_size = 5

        for i in range(0, len(handles), batch_size):
            batch = handles[i:i + batch_size]
            tasks = [self.apify_service.get_profile_metadata(h, platform="instagram") for h in batch]

            # Execute batch
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for handle, result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.error(f"Error verifying handle {handle}: {result}")
                    continue

                if not result:
                    logger.debug(f"Handle {handle} not found or private (Apify returned None).")
                    continue

                # Filter Private accounts if needed (though get_profile_metadata might return them with limited info)
                # If 'is_private' is True, we might want to skip them if we can't analyze them later.
                # Usually we only want public accounts.
                if result.get("is_private"):
                    logger.debug(f"Handle {handle} is private. Skipping.")
                    continue

                # Check follower constraints
                followers = result.get("followers_count", 0)
                if followers < request.min_followers or followers > request.max_followers:
                    logger.debug(f"Handle {handle} followers ({followers}) out of range.")
                    continue

                # Create DiscoveredCompetitor
                # Note: Semantic search implies high relevance if Grok did its job.
                # We assign a high base relevance score.
                valid_competitors.append(DiscoveredCompetitor(
                    handle=handle,
                    full_name=result.get("full_name"),
                    platform="instagram",
                    followers=followers,
                    relevance_score=95, # High confidence from AI
                    activity_status="Unknown", # Lightweight endpoint doesn't give activity stats usually
                    last_post_date=None,
                    match_reasons=["🤖 AI Recommended", "✅ Verified Public"],
                    profile_pic_url=result.get("profile_pic_url")
                ))

        return valid_competitors, handles

    async def _discover_via_hashtags(self, request: CompetitorDiscoveryRequest) -> List[DiscoveredCompetitor]:
        logger.info(f"Starting hashtag discovery for: {request.hashtags}")
        discovered = {}
        now = datetime.utcnow()

        # Limit hashtags to avoid excessive API usage
        hashtags_to_search = request.hashtags[:5]

        for hashtag in hashtags_to_search:
            logger.debug(f"Searching hashtag: #{hashtag}")
            # Fetch raw posts (limit 50 per hashtag to keep it fast)
            posts = await self.apify_service.search_hashtag_posts_raw(hashtag, limit=50)
            logger.debug(f"Found {len(posts)} posts for #{hashtag}")

            for i, post in enumerate(posts):
                # --- 1. Robust Field Extraction ---
                username = self._extract_username(post)
                if not username:
                    logger.debug(f"Post {i}: Skipped - No username found. Keys: {list(post.keys())}")
                    continue

                username = username.lower().strip()
                if username in discovered:
                    continue

                followers = self._extract_followers(post)

                # --- 2. Size Filter ---
                if followers < request.min_followers or followers > request.max_followers:
                    logger.debug(f"@{username}: Filtered by size ({followers} not in {request.min_followers}-{request.max_followers})")
                    continue

                # --- 3. Activity Filter ---
                post_date = self._parse_date(post.get("timestamp") or post.get("takenAt") or post.get("date") or post.get("createTime"))
                days_since = (now - post_date).days if post_date else 999

                activity_status = "Unknown"
                if days_since <= 7: activity_status = "Very Active"
                elif days_since <= 30: activity_status = "Active"
                else: activity_status = "Inactive"

                if request.require_active:
                    if days_since > request.max_days_since_last_post:
                        logger.debug(f"@{username}: Filtered by activity (Last post: {days_since} days ago)")
                        continue # Skip inactive accounts

                # --- 4. Relevance Scoring (Plaintext) ---
                relevance_score = 0
                match_reasons = []

                caption = post.get("caption") or post.get("text") or ""
                # Ensure we are working with string
                if not isinstance(caption, str):
                    caption = str(caption) if caption else ""

                post_text = (caption + " " + username).lower()

                # Niche Match
                if request.niche_keywords:
                    matches = [k for k in request.niche_keywords if k.lower() in post_text]
                    if matches:
                        relevance_score += 25
                        match_reasons.append("🎯 Niche Match")

                # Location Match
                if request.location_keywords:
                    matches = [k for k in request.location_keywords if k.lower() in post_text]
                    if matches:
                        relevance_score += 30
                        match_reasons.append("📍 Location Match")

                if activity_status == "Very Active":
                    relevance_score += 15
                    match_reasons.append("🔥 Very Active")

                # Boost if followers are in sweet spot
                if 1000 <= followers <= 50000:
                    relevance_score += 10

                # Base score to ensure visibility if filters passed
                final_score = relevance_score + 10

                # --- Bypass Logic ---
                # If score is low but they passed size/activity filters, we KEEP them.
                # Just warn in reasons.
                if final_score < 30: # Arbitrary threshold for "Low"
                    match_reasons.append("⚠️ Low Keyword Match")

                # Profile pic extraction
                profile_pic = self._extract_profile_pic(post)

                # Full name extraction
                full_name = self._extract_full_name(post)

                # Detect platform
                platform = "tiktok" if "authorMeta" in post else "instagram"

                logger.debug(f"Perfil @{username} encontrado, Score: {final_score}, Seguidores: {followers}")

                discovered[username] = DiscoveredCompetitor(
                    handle=username,
                    full_name=full_name,
                    platform=platform,
                    followers=followers,
                    relevance_score=final_score,
                    activity_status=activity_status,
                    last_post_date=post_date.isoformat() if post_date else None,
                    match_reasons=match_reasons,
                    profile_pic_url=profile_pic
                )

        # Sort by relevance
        sorted_results = sorted(list(discovered.values()), key=lambda x: x.relevance_score, reverse=True)
        logger.info(f"Discovery complete. Found {len(sorted_results)} unique competitors.")
        return sorted_results[:50]

    def _extract_username(self, post: Dict[str, Any]) -> Optional[str]:
        return (
            post.get("ownerUsername") or
            post.get("username") or
            post.get("owner", {}).get("username") or
            post.get("authorMeta", {}).get("name") # TikTok
        )

    def _extract_full_name(self, post: Dict[str, Any]) -> Optional[str]:
        return (
            post.get("ownerFullName") or
            post.get("full_name") or
            post.get("owner", {}).get("full_name") or
            post.get("owner", {}).get("fullName") or
            post.get("authorMeta", {}).get("nickName") # TikTok
        )

    def _extract_followers(self, post: Dict[str, Any]) -> int:
        followers = post.get("ownerFollowerCount")
        if followers is not None: return int(followers)

        followers = post.get("followersCount")
        if followers is not None: return int(followers)

        owner = post.get("owner", {})
        followers = owner.get("followersCount")
        if followers is not None: return int(followers)

        # TikTok
        author = post.get("authorMeta", {})
        followers = author.get("fans")
        if followers is not None: return int(followers)

        return 0

    def _extract_profile_pic(self, post: Dict[str, Any]) -> Optional[str]:
        pic = post.get("displayUrl") # Thumbnail as fallback

        if "owner" in post:
            pic = post.get("owner", {}).get("profilePicUrl") or pic

        if "ownerProfilePicUrl" in post:
             pic = post.get("ownerProfilePicUrl") or pic

        if "authorMeta" in post: # TikTok
            pic = post.get("authorMeta", {}).get("avatar") or pic

        return pic

    def _parse_date(self, date_val) -> Optional[datetime]:
        if not date_val: return None
        try:
            if isinstance(date_val, (int, float)):
                 return datetime.fromtimestamp(date_val)
            # Try ISO formats
            formats = [
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d",
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(date_val, fmt)
                except ValueError:
                    continue
            return None
        except: return None
