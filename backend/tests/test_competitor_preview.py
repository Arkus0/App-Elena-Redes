import pytest
from unittest.mock import AsyncMock, patch
from app.schemas.competitor import CompetitorCreate
from app.services.apify_service import ApifyService

@pytest.mark.asyncio
async def test_apify_service_preview_mock():
    """Test that ApifyService falls back to mock data correctly when unavailable"""
    # Force is_available to False
    with patch("app.services.apify_service.ApifyService.is_available", return_value=False):
        service = ApifyService()

        # Test specific mock for cristiano
        result = await service.get_profile_metadata("cristiano", "instagram")
        assert result["handle"] == "cristiano"
        assert result["full_name"] == "Cristiano Ronaldo"
        assert result["followers_count"] == 627000000

        # Test generic mock
        result_generic = await service.get_profile_metadata("someuser", "instagram")
        assert result_generic["handle"] == "someuser"
        assert "someuser" in result_generic["full_name"].lower()
        assert result_generic["followers_count"] == 125400

@pytest.mark.asyncio
async def test_apify_service_preview_live_simulation():
    """Test the logic that would run if Apify was available (mocking the client call)"""
    with patch("app.services.apify_service.ApifyService.is_available", return_value=True):
        with patch("app.core.config.settings.APIFY_API_KEY", "fake_key"):
            # Mock the Apify client
            with patch("apify_client.ApifyClientAsync") as MockClient:
                mock_client_instance = MockClient.return_value
                mock_actor = mock_client_instance.actor.return_value

                # Fix: Make call awaitable
                mock_actor.call = AsyncMock(return_value={"defaultDatasetId": "dataset123"})

                # Mock dataset iterator
                mock_dataset = mock_client_instance.dataset.return_value

                # Setup async iterator for items
                async def async_iter(limit=None):
                    yield {
                        "ownerUsername": "realuser",
                        "ownerFullName": "Real User",
                        "biography": "Real bio",
                        "followersCount": 500,
                        "profilePicUrl": "http://pic.url",
                        "isPrivate": False,
                        "isVerified": False
                    }

                mock_dataset.iterate_items.side_effect = async_iter

                service = ApifyService()
                service.client = mock_client_instance

                result = await service.get_profile_metadata("realuser", "instagram")

                # Verify we got the "live" data, not mock fallback
                assert result["handle"] == "realuser"
                assert result["full_name"] == "Real User"
                assert result["followers_count"] == 500
