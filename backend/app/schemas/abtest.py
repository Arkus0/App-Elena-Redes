from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ABTestVariantCreate(BaseModel):
    """Schema for creating a variant in an A/B test."""
    variant_name: str = Field(..., description="Label for the variant (e.g., 'A', 'B')")
    content_structure: Optional[Dict[str, Any]] = Field(None, description="Specific content data (e.g. hook, caption)")

class ABTestExperimentCreate(BaseModel):
    """Schema for creating a new A/B test experiment."""
    test_name: str = Field(..., description="Name of the experiment")
    original_content_id: Optional[int] = Field(None, description="ID of the original content being tested")
    variants: List[ABTestVariantCreate] = Field(..., min_items=2, description="List of variants to test")

class ABTestVariantResponse(BaseModel):
    """Response schema for a variant."""
    id: int
    variant_name: str
    alpha_param: int
    beta_param: int
    content_structure: Optional[Dict[str, Any]]

    class Config:
        from_attributes = True

class ABTestExperimentResponse(BaseModel):
    """Response schema for an experiment."""
    id: int
    business_id: int
    test_name: str
    original_content_id: Optional[int]
    is_active: bool
    variants: List[ABTestVariantResponse]

    class Config:
        from_attributes = True
