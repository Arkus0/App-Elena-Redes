"""
UserConfig Schemas - Pydantic Models for User Configuration API

Schemas para el endpoint unificado /api/user/config que sincroniza
TODAS las configuraciones frontend-backend en tiempo real.

Author: BrandPulse AI
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


# =============================================================================
# Enums
# =============================================================================

class EmbeddingPrecision(str, Enum):
    """Niveles de precision de embeddings."""
    ULTRA_LOW = "ultra_low"  # 64 dims
    LOW = "low"              # 128 dims (DEFAULT)
    MEDIUM = "medium"        # 256 dims
    HIGH = "high"            # 384 dims
    MAX = "max"              # 384 dims raw


class MultimodalMode(str, Enum):
    """Modos de procesamiento multimodal."""
    LIGHT = "light"   # ~5s processing
    FULL = "full"     # ~20s processing


# Mapeo de precision a dimensiones
PRECISION_DIMS_MAP = {
    "ultra_low": 64,
    "low": 128,
    "medium": 256,
    "high": 384,
    "max": 384,
}


# =============================================================================
# Discovery Config Schemas
# =============================================================================

class DiscoveryConfig(BaseModel):
    """Configuracion de descubrimiento de contenido/competidores."""
    hashtags: List[str] = Field(default_factory=list, description="Hashtags de nicho a monitorear")
    location_keywords: List[str] = Field(default_factory=list, description="Keywords de ubicacion")
    niche_keywords: List[str] = Field(default_factory=list, description="Keywords de nicho")
    follower_min: int = Field(default=0, ge=0, description="Minimo de seguidores")
    follower_max: int = Field(default=1000000, ge=0, description="Maximo de seguidores")
    min_recent_posts: int = Field(default=5, ge=1, le=100, description="Minimo posts recientes")
    exclude_verified: bool = Field(default=False, description="Excluir cuentas verificadas")
    exclude_private: bool = Field(default=True, description="Excluir cuentas privadas")

    class Config:
        schema_extra = {
            "example": {
                "hashtags": ["inmobiliaria", "pisosalmeria", "ventacasa"],
                "location_keywords": ["almeria", "andalucia"],
                "niche_keywords": ["inmobiliaria", "venta pisos"],
                "follower_min": 500,
                "follower_max": 50000,
                "min_recent_posts": 10,
                "exclude_verified": True,
                "exclude_private": True,
            }
        }


# =============================================================================
# KPI Weights Schemas
# =============================================================================

class KPIWeights(BaseModel):
    """Pesos de KPI para calculo de RPI."""
    likes: float = Field(default=1.0, ge=0, le=20, description="Peso para likes")
    comments: float = Field(default=2.0, ge=0, le=20, description="Peso para comentarios")
    shares: float = Field(default=10.0, ge=0, le=20, description="Peso para compartidos")
    saves: float = Field(default=5.0, ge=0, le=20, description="Peso para guardados")
    views: float = Field(default=3.0, ge=0, le=20, description="Peso para views")

    def normalize(self) -> Dict[str, float]:
        """Retorna pesos normalizados (suma = 1.0)."""
        total = self.likes + self.comments + self.shares + self.saves + self.views
        if total == 0:
            return {"likes": 0.2, "comments": 0.2, "shares": 0.2, "saves": 0.2, "views": 0.2}
        return {
            "likes": self.likes / total,
            "comments": self.comments / total,
            "shares": self.shares / total,
            "saves": self.saves / total,
            "views": self.views / total,
        }

    def to_vector(self) -> List[float]:
        """Retorna como vector ordenado."""
        return [self.likes, self.comments, self.shares, self.saves, self.views]

    class Config:
        schema_extra = {
            "example": {
                "likes": 1.0,
                "comments": 2.0,
                "shares": 10.0,
                "saves": 5.0,
                "views": 3.0,
            }
        }


# =============================================================================
# Light Mode Config Schemas
# =============================================================================

class LightModeConfig(BaseModel):
    """Configuracion detallada de modo ligero multimodal."""
    whisper_model: str = Field(default="tiny", description="Modelo Whisper: 'tiny' o 'base'")
    max_duration_seconds: float = Field(default=3.0, ge=1.0, le=30.0, description="Duracion maxima audio")
    ocr_max_frames: int = Field(default=5, ge=1, le=30, description="Frames maximos para OCR")
    use_thumbnail: bool = Field(default=True, description="Usar thumbnail para OCR")
    cache_enabled: bool = Field(default=True, description="Habilitar cache de procesamiento")
    skip_non_video: bool = Field(default=True, description="Saltar contenido no video")

    @validator("whisper_model")
    def validate_whisper_model(cls, v):
        if v not in ("tiny", "base"):
            raise ValueError("whisper_model debe ser 'tiny' o 'base'")
        return v


# =============================================================================
# Main UserConfig Schemas
# =============================================================================

class UserConfigBase(BaseModel):
    """Base schema para UserConfig."""
    discovery: Optional[DiscoveryConfig] = Field(default_factory=DiscoveryConfig)
    kpi_weights: Optional[KPIWeights] = Field(default_factory=KPIWeights)
    kpi_template_name: Optional[str] = Field(None, max_length=50, description="Nombre de plantilla KPI")
    embedding_precision: EmbeddingPrecision = Field(
        default=EmbeddingPrecision.LOW,
        description="Precision de embeddings: ultra_low/low/medium/high/max"
    )
    multimodal_mode: MultimodalMode = Field(
        default=MultimodalMode.LIGHT,
        description="Modo multimodal: light (~5s) o full (~20s)"
    )
    light_mode_config: Optional[LightModeConfig] = Field(default_factory=LightModeConfig)
    own_instagram_username: Optional[str] = Field(None, max_length=100, description="Username Instagram propio (sin @)")
    own_tiktok_username: Optional[str] = Field(None, max_length=100, description="Username TikTok propio")
    description: Optional[str] = Field(None, max_length=500, description="Notas del usuario")

    @validator("own_instagram_username", "own_tiktok_username", pre=True)
    def strip_at_symbol(cls, v):
        """Elimina @ si el usuario lo incluye."""
        if v and isinstance(v, str):
            return v.lstrip("@").strip()
        return v


class UserConfigCreate(UserConfigBase):
    """Schema para crear UserConfig."""
    pass


class UserConfigUpdate(BaseModel):
    """Schema para actualizar UserConfig (todos los campos opcionales)."""
    discovery: Optional[DiscoveryConfig] = None
    kpi_weights: Optional[KPIWeights] = None
    kpi_template_name: Optional[str] = None
    embedding_precision: Optional[EmbeddingPrecision] = None
    multimodal_mode: Optional[MultimodalMode] = None
    light_mode_config: Optional[LightModeConfig] = None
    own_instagram_username: Optional[str] = None
    own_tiktok_username: Optional[str] = None
    description: Optional[str] = None

    @validator("own_instagram_username", "own_tiktok_username", pre=True)
    def strip_at_symbol(cls, v):
        if v and isinstance(v, str):
            return v.lstrip("@").strip()
        return v


class UserConfigResponse(UserConfigBase):
    """Schema de respuesta para UserConfig."""
    id: int
    user_id: int
    business_id: int
    embedding_dims: int = Field(..., description="Dimensiones reales de embeddings")
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

    @classmethod
    def from_orm_with_dims(cls, obj):
        """Crea respuesta desde ORM con dims calculados."""
        data = {
            "id": obj.id,
            "user_id": obj.user_id,
            "business_id": obj.business_id,
            "discovery": DiscoveryConfig(
                hashtags=obj.discovery_hashtags or [],
                location_keywords=obj.discovery_location_keywords or [],
                niche_keywords=obj.discovery_niche_keywords or [],
                follower_min=obj.discovery_follower_min,
                follower_max=obj.discovery_follower_max,
                min_recent_posts=obj.discovery_min_recent_posts,
                exclude_verified=obj.discovery_exclude_verified,
                exclude_private=obj.discovery_exclude_private,
            ),
            "kpi_weights": KPIWeights(
                likes=obj.kpi_likes_weight,
                comments=obj.kpi_comments_weight,
                shares=obj.kpi_shares_weight,
                saves=obj.kpi_saves_weight,
                views=obj.kpi_views_weight,
            ),
            "kpi_template_name": obj.kpi_template_name,
            "embedding_precision": obj.embedding_precision.value if hasattr(obj.embedding_precision, 'value') else obj.embedding_precision,
            "embedding_dims": PRECISION_DIMS_MAP.get(
                obj.embedding_precision.value if hasattr(obj.embedding_precision, 'value') else obj.embedding_precision,
                128
            ),
            "multimodal_mode": obj.multimodal_mode.value if hasattr(obj.multimodal_mode, 'value') else obj.multimodal_mode,
            "light_mode_config": LightModeConfig(**(obj.light_mode_config or {})),
            "own_instagram_username": obj.own_instagram_username,
            "own_tiktok_username": obj.own_tiktok_username,
            "description": obj.description,
            "is_active": obj.is_active,
            "created_at": obj.created_at,
            "updated_at": obj.updated_at,
        }
        return cls(**data)


# =============================================================================
# Config Info Response (para frontend)
# =============================================================================

class PrecisionOption(BaseModel):
    """Opcion de precision disponible."""
    value: str
    label: str
    dimensions: int
    description: str
    performance: str
    is_recommended: bool


class MultimodalOption(BaseModel):
    """Opcion de modo multimodal disponible."""
    value: str
    label: str
    description: str
    estimated_time_seconds: float
    is_recommended: bool


class KPITemplate(BaseModel):
    """Plantilla de KPI predefinida."""
    name: str
    display_name: str
    description: str
    icon: str
    weights: KPIWeights


class UserConfigInfoResponse(BaseModel):
    """Respuesta con opciones disponibles para configuracion."""
    precision_options: List[PrecisionOption]
    multimodal_options: List[MultimodalOption]
    kpi_templates: List[KPITemplate]
    default_config: UserConfigBase


# =============================================================================
# Config para Pipelines (uso interno)
# =============================================================================

class PipelineConfig(BaseModel):
    """
    Configuracion simplificada para uso en pipelines.

    Esta clase se usa internamente en ingest.py, ml_service.py,
    growth_prediction_engine.py para tener acceso rapido a la config.
    """
    user_id: int
    business_id: int

    # Embedding
    embedding_precision: str
    embedding_dims: int

    # Multimodal
    multimodal_mode: str
    light_mode_enabled: bool
    whisper_model: str
    ocr_max_frames: int

    # KPI Weights
    kpi_weights: Dict[str, float]
    kpi_weights_normalized: Dict[str, float]

    # Own profile
    own_instagram_username: Optional[str]
    own_tiktok_username: Optional[str]

    # Discovery
    discovery_hashtags: List[str]
    discovery_niche_keywords: List[str]

    def is_own_profile(self, username: str, platform: str = "instagram") -> bool:
        """Verifica si un username corresponde al perfil propio."""
        if not username:
            return False
        username_clean = username.lstrip("@").lower()

        if platform == "instagram":
            own = self.own_instagram_username
        elif platform == "tiktok":
            own = self.own_tiktok_username
        else:
            return False

        if not own:
            return False

        return own.lower() == username_clean

    def get_weighted_rpi(self, likes: int, comments: int, shares: int, saves: int, views: int) -> float:
        """Calcula RPI usando los pesos configurados."""
        w = self.kpi_weights
        weighted = (
            likes * w.get("likes", 1.0) +
            comments * w.get("comments", 2.0) +
            shares * w.get("shares", 10.0) +
            saves * w.get("saves", 5.0) +
            views * w.get("views", 3.0)
        )
        total_weight = sum(w.values())
        if total_weight > 0:
            return weighted / total_weight
        return weighted

    @classmethod
    def from_user_config(cls, config) -> "PipelineConfig":
        """Crea PipelineConfig desde UserConfig ORM."""
        precision = config.embedding_precision.value if hasattr(config.embedding_precision, 'value') else config.embedding_precision
        mode = config.multimodal_mode.value if hasattr(config.multimodal_mode, 'value') else config.multimodal_mode
        light_config = config.light_mode_config or {}

        kpi_weights = {
            "likes": config.kpi_likes_weight,
            "comments": config.kpi_comments_weight,
            "shares": config.kpi_shares_weight,
            "saves": config.kpi_saves_weight,
            "views": config.kpi_views_weight,
        }

        total = sum(kpi_weights.values())
        if total > 0:
            normalized = {k: v / total for k, v in kpi_weights.items()}
        else:
            normalized = {k: 0.2 for k in kpi_weights}

        return cls(
            user_id=config.user_id,
            business_id=config.business_id,
            embedding_precision=precision,
            embedding_dims=PRECISION_DIMS_MAP.get(precision, 128),
            multimodal_mode=mode,
            light_mode_enabled=(mode == "light"),
            whisper_model=light_config.get("whisper_model", "tiny"),
            ocr_max_frames=light_config.get("ocr_max_frames", 5),
            kpi_weights=kpi_weights,
            kpi_weights_normalized=normalized,
            own_instagram_username=config.own_instagram_username,
            own_tiktok_username=config.own_tiktok_username,
            discovery_hashtags=config.discovery_hashtags or [],
            discovery_niche_keywords=config.discovery_niche_keywords or [],
        )

    @classmethod
    def create_default(cls, user_id: int, business_id: int) -> "PipelineConfig":
        """Crea config por defecto seguro."""
        return cls(
            user_id=user_id,
            business_id=business_id,
            embedding_precision="low",
            embedding_dims=128,
            multimodal_mode="light",
            light_mode_enabled=True,
            whisper_model="tiny",
            ocr_max_frames=5,
            kpi_weights={"likes": 1.0, "comments": 2.0, "shares": 10.0, "saves": 5.0, "views": 3.0},
            kpi_weights_normalized={"likes": 0.048, "comments": 0.095, "shares": 0.476, "saves": 0.238, "views": 0.143},
            own_instagram_username=None,
            own_tiktok_username=None,
            discovery_hashtags=[],
            discovery_niche_keywords=[],
        )
