import logging
from datetime import datetime
from typing import List, Dict, Optional
from app.services.apify_service import ApifyService
from app.schemas.competitor import CompetitorDiscoveryRequest, DiscoveredCompetitor

logger = logging.getLogger(__name__)

class DiscoveryService:
    def __init__(self):
        self.apify_service = ApifyService()

    async def discover_competitors(self, request: CompetitorDiscoveryRequest) -> List[DiscoveredCompetitor]:
        discovered = {}
        now = datetime.utcnow()

        # Limit hashtags to avoid excessive API usage
        hashtags_to_search = request.hashtags[:5]

        for hashtag in hashtags_to_search:
            # Fetch raw posts (limit 50 per hashtag to keep it fast)
            posts = await self.apify_service.search_hashtag_posts_raw(hashtag, limit=50)

            for post in posts:
                # Handle different field names depending on the scraper version
                username = (
                    post.get("ownerUsername") or
                    post.get("username") or
                    post.get("owner", {}).get("username")
                )

                if not username:
                    continue

                username = username.lower().strip()
                if username in discovered:
                    continue

                # --- 1. Size Filter ---
                followers = post.get("ownerFollowerCount", 0)
                # Some scrapers return 'followersCount' inside 'owner'
                if not followers and "owner" in post:
                    followers = post.get("owner", {}).get("followersCount", 0)

                # If we still don't have followers, we might be dealing with a different structure
                # For TikTok/others, we might need adjustments. Assuming Instagram for now.

                if followers < request.min_followers or followers > request.max_followers:
                    continue

                # --- 2. Activity Filter ---
                post_date = self._parse_date(post.get("timestamp") or post.get("takenAt") or post.get("date") or post.get("createTime"))
                days_since = (now - post_date).days if post_date else 999

                activity_status = "Unknown"
                if days_since <= 7: activity_status = "Very Active"
                elif days_since <= 30: activity_status = "Active"
                else: activity_status = "Inactive"

                if request.require_active:
                    if days_since > request.max_days_since_last_post:
                        continue # Skip inactive accounts

                # --- 3. Relevance Scoring ---
                relevance_score = 0
                match_reasons = []

                caption = post.get("caption") or post.get("text") or ""
                post_text = (caption + " " + username).lower()

                if request.location_keywords and any(k.lower() in post_text for k in request.location_keywords):
                    relevance_score += 30
                    match_reasons.append("📍 Location Match")

                if request.niche_keywords and any(k.lower() in post_text for k in request.niche_keywords):
                    relevance_score += 25
                    match_reasons.append("🎯 Niche Match")

                if activity_status == "Very Active":
                    relevance_score += 15
                    match_reasons.append("🔥 Very Active")

                # Boost if followers are in sweet spot (e.g. 1k-50k)
                if 1000 <= followers <= 50000:
                    relevance_score += 10

                # Profile pic
                profile_pic = post.get("displayUrl") # Sometimes post thumb is used if profile pic missing
                if "owner" in post:
                    profile_pic = post.get("owner", {}).get("profilePicUrl") or profile_pic
                if "authorMeta" in post: # TikTok
                    profile_pic = post.get("authorMeta", {}).get("avatar") or profile_pic

                discovered[username] = DiscoveredCompetitor(
                    handle=username,
                    platform="instagram" if "ownerUsername" in post or "owner" in post else "tiktok",
                    followers=followers,
                    relevance_score=relevance_score + 10, # Base score
                    activity_status=activity_status,
                    last_post_date=post_date.isoformat() if post_date else None,
                    match_reasons=match_reasons,
                    profile_pic_url=profile_pic
                )

        # Sort by relevance
        sorted_results = sorted(list(discovered.values()), key=lambda x: x.relevance_score, reverse=True)
        return sorted_results[:50]

    def _parse_date(self, date_val) -> Optional[datetime]:
        if not date_val: return None
        try:
            if isinstance(date_val, (int, float)):
                 return datetime.fromtimestamp(date_val)
            # Try ISO formats
            # Common formats from Instagram/Apify
            formats = [
                "%Y-%m-%dT%H:%M:%S.%fZ", # With micros and Z
                "%Y-%m-%dT%H:%M:%S.%f",  # With micros, no Z (Python isoformat default)
                "%Y-%m-%dT%H:%M:%SZ",    # No micros, with Z
                "%Y-%m-%dT%H:%M:%S",     # No micros, no Z
                "%Y-%m-%d",              # Date only
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(date_val, fmt)
                except ValueError:
                    continue
            return None
        except: return None
