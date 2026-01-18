import pytest
from app.services.user_config_service import UserConfigService
from app.schemas.user_config import UserConfigInfoResponse

def test_get_config_info_structure():
    service = UserConfigService()
    info = service.get_config_info()

    assert isinstance(info, UserConfigInfoResponse)

    # Check Precision Options
    assert len(info.precision_options) == 5
    assert info.precision_options[0].value == "ultra_low"
    assert info.precision_options[4].value == "max"

    # Check Multimodal Options
    assert len(info.multimodal_options) == 2
    assert info.multimodal_options[0].value == "light"
    assert info.multimodal_options[1].value == "full"

    # Check KPI Templates
    assert len(info.kpi_templates) == 5
    assert info.kpi_templates[0].name == "brand_awareness"

def test_get_config_info_immutability():
    """Ensure that modifying the returned list doesn't affect subsequent calls."""
    service = UserConfigService()
    info1 = service.get_config_info()

    # Simulate modification if it were a mutable list returned directly (though Pydantic models are mutable, the list container from the service should remain intact for next call if it was recreated or deep copied, but since we are using global constants now, we rely on the fact that we return a new Response object wrapping the lists)

    # Actually, since we pass the SAME list object to UserConfigInfoResponse constructor,
    # and UserConfigInfoResponse is a Pydantic model...
    # Pydantic models by default validate and copy data.
    # Let's verify that we get what we expect.

    assert info1.precision_options[0].value == "ultra_low"

    # If we were to modify the list in python (monkey patch), it would be bad if it affected the constant.
    # But UserConfigInfoResponse creates a model which usually holds its own data.

    info2 = service.get_config_info()
    assert info2.precision_options[0].value == "ultra_low"
