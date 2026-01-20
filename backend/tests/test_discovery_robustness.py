import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from app.services.discovery_service import DiscoveryService
from app.schemas.competitor import CompetitorDiscoveryRequest

@pytest.mark.asyncio
async def test_robust_mapping_and_bypass():
    """
    Test ensuring DiscoveryService handles:
    1. Varied Apify data structures (nested keys).
    2. Bypass logic (low relevance but passes filters -> included).
    3. Plain text output.
    """
    with patch("app.services.discovery_service.ApifyService") as MockApifyService:
        mock_apify = MockApifyService.return_value

        now = datetime.utcnow()

        # Mock Data with various structural "issues" and scenarios
        mock_posts = [
            # Case 1: Standard structure (should work already)
            {
                "ownerUsername": "standard_user",
                "ownerFollowerCount": 5000,
                "timestamp": now.isoformat(),
                "caption": "Standard match #niche",
                "displayUrl": "http://img.com/1"
            },
            # Case 2: Nested structure (Apify variation) - previously might fail
            {
                "owner": {
                    "username": "nested_user",
                    "followersCount": 6000,
                    "full_name": "Nested User Name",
                    "profilePicUrl": "http://img.com/nested"
                },
                "takenAt": (now - timedelta(days=1)).timestamp(),
                "text": "Caption in text field #niche"
            },
            # Case 3: TikTok structure (authorMeta)
            {
                "authorMeta": {
                    "name": "tiktok_user",
                    "nickName": "TikTok Star",
                    "fans": 10000,
                    "avatar": "http://img.com/tiktok"
                },
                "createTime": int((now - timedelta(days=2)).timestamp()),
                "text": "TikTok caption" # No niche keyword, but valid size/activity -> Should pass via BYPASS
            },
            # Case 4: Missing username (should skip)
            {
                "caption": "Ghost post"
            },
            # Case 5: Valid user but ZERO relevance (no keywords) -> Should pass via BYPASS
            {
                "username": "random_active_user",
                "followersCount": 8000,
                "date": now.strftime("%Y-%m-%d"),
                "caption": "Just a random photo", # No keywords
            }
        ]

        mock_apify.search_hashtag_posts_raw = AsyncMock(return_value=mock_posts)

        service = DiscoveryService()

        request = CompetitorDiscoveryRequest(
            hashtags=["test"],
            niche_keywords=["niche"], # "Standard" and "Nested" should match this
            min_followers=1000,
            max_followers=50000,
            require_active=True,
            max_days_since_last_post=30
        )

        results = await service.discover_competitors(request)

        # Convert results to a dict for easy checking
        result_map = {r.handle: r for r in results}

        # --- Verification ---

        # 1. Standard User
        assert "standard_user" in result_map
        assert result_map["standard_user"].followers == 5000

        # 2. Nested User (Robust Mapping Check)
        assert "nested_user" in result_map, "Failed to map 'owner.username'"
        assert result_map["nested_user"].followers == 6000, "Failed to map 'owner.followersCount'"
        assert any("Niche Match" in r for r in result_map["nested_user"].match_reasons)

        # 3. TikTok User (Bypass Check 1)
        assert "tiktok_user" in result_map, "Failed to map TikTok 'authorMeta.name'"
        assert result_map["tiktok_user"].followers == 10000
        # Should have a base score even without niche match
        assert result_map["tiktok_user"].relevance_score >= 10

        # 4. Random Active User (Bypass Check 2)
        assert "random_active_user" in result_map
        assert result_map["random_active_user"].relevance_score >= 10
        # Should have specific reason requested by user
        assert any("Exploración general" in r for r in result_map["random_active_user"].match_reasons)

        # 5. Missing user
        assert len([r for r in results if not r.handle]) == 0

        print("\nTest Results Summary:")
        for r in results:
            print(f"@{r.handle} | Score: {r.relevance_score} | Reasons: {r.match_reasons}")
