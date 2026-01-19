import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from app.services.discovery_service import DiscoveryService
from app.schemas.competitor import CompetitorDiscoveryRequest

@pytest.mark.asyncio
async def test_discovery_service_filtering():
    # Mock ApifyService
    with patch("app.services.discovery_service.ApifyService") as MockApifyService:
        mock_apify = MockApifyService.return_value

        # Setup mock data
        now = datetime.utcnow()
        mock_posts = [
            # Active, good followers, niche match
            {
                "ownerUsername": "active_user",
                "ownerFollowerCount": 5000,
                "timestamp": (now - timedelta(days=2)).isoformat(),
                "caption": "Luxury real estate in Madrid #realestate",
                "displayUrl": "http://pic.url"
            },
            # Inactive (posted 40 days ago)
            {
                "ownerUsername": "inactive_user",
                "ownerFollowerCount": 5000,
                "timestamp": (now - timedelta(days=40)).isoformat(),
                "caption": "Old post",
            },
            # Too small
            {
                "ownerUsername": "small_user",
                "ownerFollowerCount": 50,
                "timestamp": (now - timedelta(days=1)).isoformat(),
                "caption": "New user",
            },
             # Too big
            {
                "ownerUsername": "huge_user",
                "ownerFollowerCount": 500000,
                "timestamp": (now - timedelta(days=1)).isoformat(),
                "caption": "Celebrity",
            }
        ]

        mock_apify.search_hashtag_posts_raw = AsyncMock(return_value=mock_posts)

        service = DiscoveryService()

        request = CompetitorDiscoveryRequest(
            hashtags=["realestate"],
            niche_keywords=["luxury"],
            min_followers=100,
            max_followers=100000,
            require_active=True,
            max_days_since_last_post=30
        )

        results = await service.discover_competitors(request)

        # Should only return 'active_user'
        assert len(results) == 1
        assert results[0].handle == "active_user"
        assert results[0].activity_status == "Very Active"

        # Check matching reasons logic
        # "luxury" is in niche_keywords and "Luxury" is in caption.
        # So we expect niche match.
        # But wait, in the implementation:
        # post_text = (caption + " " + username).lower() -> "luxury real estate in madrid #realestate active_user"
        # "luxury" in post_text is True.

        assert any("Niche Match" in r for r in results[0].match_reasons)
