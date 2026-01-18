"""
Extension API - Public endpoints for Chrome extension sync
==========================================================

Provides endpoints that the Elena Bridge extension can call to sync configuration.
Uses a simple API key authentication (generated per business) instead of full OAuth.

This enables the extension to automatically fetch the own_username config from the
backend without requiring the user to manually copy values.
"""
import secrets
import hashlib
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Header, Depends, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, Column, String, Integer, DateTime, ForeignKey

from app.core.database import get_db, Base
from app.models.business import Business

router = APIRouter()


# =============================================================================
# Extension API Key Model
# =============================================================================

class ExtensionApiKey(Base):
    """
    Stores API keys for extension authentication.
    Each business can have one active extension API key.
    """
    __tablename__ = "extension_api_keys"

    id = Column(Integer, primary_key=True, index=True)
    business_id = Column(Integer, ForeignKey("businesses.id"), unique=True, nullable=False)
    # Stored as SHA256 hash for security
    key_hash = Column(String(64), nullable=False, index=True)
    # Last 4 chars for display (e.g., "****abcd")
    key_suffix = Column(String(4), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)


# =============================================================================
# Schemas
# =============================================================================

class ExtensionConfigResponse(BaseModel):
    """Configuration for the extension"""
    own_instagram_username: Optional[str] = None
    own_tiktok_username: Optional[str] = None
    business_name: str
    business_type: str
    feedback_loop_enabled: bool


class ApiKeyResponse(BaseModel):
    """Response when generating a new API key"""
    api_key: str  # Only returned once during generation
    key_suffix: str
    message: str


class ApiKeyStatusResponse(BaseModel):
    """Status of existing API key"""
    has_key: bool
    key_suffix: Optional[str] = None
    last_used_at: Optional[datetime] = None


# =============================================================================
# Helper Functions
# =============================================================================

def hash_api_key(api_key: str) -> str:
    """Hash API key using SHA256"""
    return hashlib.sha256(api_key.encode()).hexdigest()


def generate_api_key() -> str:
    """Generate a secure random API key"""
    # Format: elena_ext_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX (40 chars total)
    return f"elena_ext_{secrets.token_hex(16)}"


async def get_business_by_api_key(
    db: AsyncSession,
    api_key: str
) -> Optional[Business]:
    """Validate API key and return associated business"""
    key_hash = hash_api_key(api_key)

    result = await db.execute(
        select(ExtensionApiKey).where(ExtensionApiKey.key_hash == key_hash)
    )
    ext_key = result.scalar_one_or_none()

    if not ext_key:
        return None

    # Update last used timestamp
    ext_key.last_used_at = datetime.utcnow()
    await db.commit()

    # Get business
    result = await db.execute(
        select(Business).where(Business.id == ext_key.business_id)
    )
    return result.scalar_one_or_none()


# =============================================================================
# Public Endpoints (API Key Auth)
# =============================================================================

@router.get("/config", response_model=ExtensionConfigResponse)
async def get_extension_config(
    x_extension_api_key: str = Header(..., alias="X-Extension-API-Key"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get extension configuration using API key.

    This endpoint is called by the Elena Bridge extension to sync the
    own_username configuration without requiring user login.

    Headers:
        X-Extension-API-Key: elena_ext_XXXXXXXX...

    Returns:
        ExtensionConfigResponse with own_instagram_username, own_tiktok_username, etc.
    """
    business = await get_business_by_api_key(db, x_extension_api_key)

    if not business:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key. Generate a new key from the dashboard."
        )

    return ExtensionConfigResponse(
        own_instagram_username=business.own_instagram_username,
        own_tiktok_username=business.own_tiktok_username,
        business_name=business.name,
        business_type=business.business_type.value,
        feedback_loop_enabled=bool(
            business.own_instagram_username or business.own_tiktok_username
        )
    )


# =============================================================================
# Protected Endpoints (User Auth) - For generating/managing API keys
# =============================================================================

from app.api.deps import get_current_user
from app.models.user import User


@router.post("/api-key/{business_id}", response_model=ApiKeyResponse)
async def generate_extension_api_key(
    business_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Generate a new API key for the extension.

    This invalidates any existing key for the business.
    The key is only shown once - store it securely.

    Returns:
        ApiKeyResponse with the new API key (shown only once)
    """
    # Verify business ownership
    result = await db.execute(
        select(Business)
        .where(Business.id == business_id)
        .where(Business.user_id == current_user.id)
    )
    business = result.scalar_one_or_none()

    if not business:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Business not found"
        )

    # Generate new key
    api_key = generate_api_key()
    key_hash = hash_api_key(api_key)
    key_suffix = api_key[-4:]

    # Delete existing key if any
    result = await db.execute(
        select(ExtensionApiKey).where(ExtensionApiKey.business_id == business_id)
    )
    existing = result.scalar_one_or_none()

    if existing:
        await db.delete(existing)

    # Create new key
    new_key = ExtensionApiKey(
        business_id=business_id,
        key_hash=key_hash,
        key_suffix=key_suffix
    )
    db.add(new_key)
    await db.commit()

    return ApiKeyResponse(
        api_key=api_key,
        key_suffix=key_suffix,
        message="API key generated. Copy it now - it won't be shown again!"
    )


@router.get("/api-key/{business_id}/status", response_model=ApiKeyStatusResponse)
async def get_api_key_status(
    business_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Check if an API key exists for the business.
    """
    # Verify business ownership
    result = await db.execute(
        select(Business)
        .where(Business.id == business_id)
        .where(Business.user_id == current_user.id)
    )
    business = result.scalar_one_or_none()

    if not business:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Business not found"
        )

    # Check for existing key
    result = await db.execute(
        select(ExtensionApiKey).where(ExtensionApiKey.business_id == business_id)
    )
    ext_key = result.scalar_one_or_none()

    if ext_key:
        return ApiKeyStatusResponse(
            has_key=True,
            key_suffix=ext_key.key_suffix,
            last_used_at=ext_key.last_used_at
        )

    return ApiKeyStatusResponse(has_key=False)


@router.delete("/api-key/{business_id}")
async def revoke_extension_api_key(
    business_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Revoke the API key for the extension.
    """
    # Verify business ownership
    result = await db.execute(
        select(Business)
        .where(Business.id == business_id)
        .where(Business.user_id == current_user.id)
    )
    business = result.scalar_one_or_none()

    if not business:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Business not found"
        )

    # Delete key
    result = await db.execute(
        select(ExtensionApiKey).where(ExtensionApiKey.business_id == business_id)
    )
    ext_key = result.scalar_one_or_none()

    if ext_key:
        await db.delete(ext_key)
        await db.commit()

    return {"message": "API key revoked successfully"}
