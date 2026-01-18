"""
UserConfig Model - Unified Configuration for All ML Pipelines

Este modelo unifica TODAS las configuraciones de usuario que afectan los pipelines:
- Discovery: hashtags, keywords de nicho, filtros de seguidores
- KPI Weights: pesos personalizados para calculo de RPI
- Embedding Precision: dimensiones de embeddings (ultra_low/low/medium/high/max)
- Multimodal Mode: procesamiento ligero vs completo (light/full)
- Own Username: identificador para feedback loop de perfil propio

IMPORTANTE: Este modelo reemplaza configuraciones dispersas y asegura que
frontend changes afecten 100% el backend (ingest/train/inference).

Author: BrandPulse AI
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean,
    ForeignKey, JSON, Text, Enum as SQLEnum
)
from sqlalchemy.orm import relationship
import enum
from app.core.database import Base


class EmbeddingPrecision(str, enum.Enum):
    """
    Niveles de precision de embeddings.

    - ULTRA_LOW (64 dims): Ultra rapido para PC modesto
    - LOW (128 dims): Recomendado sobremesa normal (DEFAULT)
    - MEDIUM (256 dims): Balance precision/velocidad
    - HIGH (384 dims): Full dims con TruncatedSVD
    - MAX (full raw): 384 dims sin reduccion
    """
    ULTRA_LOW = "ultra_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"


class MultimodalMode(str, enum.Enum):
    """
    Modos de procesamiento multimodal (Whisper/EasyOCR).

    - LIGHT: Whisper tiny, 3s audio, 5 frames OCR (~5s processing)
    - FULL: Whisper base, audio completo, todos frames (~20s processing)
    """
    LIGHT = "light"
    FULL = "full"


# Mapeo de precision a dimensiones
PRECISION_TO_DIMS = {
    "ultra_low": 64,
    "low": 128,
    "medium": 256,
    "high": 384,
    "max": 384,  # max = full raw 384
}


# KPI Weights por defecto (enfasis en shares)
DEFAULT_KPI_WEIGHTS = {
    "likes": 1.0,
    "comments": 2.0,
    "shares": 10.0,
    "saves": 5.0,
    "views": 3.0,
}


# Discovery config por defecto
DEFAULT_DISCOVERY_CONFIG = {
    "hashtags": [],
    "location_keywords": [],
    "niche_keywords": [],
    "follower_min": 0,
    "follower_max": 1000000,
    "min_recent_posts": 5,
    "exclude_verified": False,
    "exclude_private": True,
}


# Light mode config por defecto
DEFAULT_LIGHT_MODE_CONFIG = {
    "whisper_model": "tiny",
    "max_duration_seconds": 3.0,
    "ocr_max_frames": 5,
    "use_thumbnail": True,
    "cache_enabled": True,
    "skip_non_video": True,
}


class UserConfig(Base):
    """
    Configuracion unificada de usuario para todos los pipelines ML.

    Esta tabla centraliza TODAS las configuraciones que afectan:
    - Ingestion (ingest.py): own_username, light_mode, discovery
    - Training (train.py): embedding_precision, kpi_weights
    - Inference (ml_service.py, growth_engine.py): todos los campos

    SINCRONIZACION REAL-TIME:
    Los endpoints GET/POST /api/user/config cargan/actualizan esta tabla.
    Todos los pipelines DEBEN cargar config desde aqui, NO usar hardcodes.
    """
    __tablename__ = "user_configs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    business_id = Column(Integer, ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False, index=True)

    # =========================================================================
    # DISCOVERY CONFIG - Filtros para descubrimiento de contenido/competidores
    # =========================================================================
    # Hashtags de nicho para monitorear (JSON array)
    discovery_hashtags = Column(JSON, default=list, nullable=False)
    # Keywords de ubicacion (ej: ["almeria", "malaga", "andalucia"])
    discovery_location_keywords = Column(JSON, default=list, nullable=False)
    # Keywords de nicho (ej: ["inmobiliaria", "venta pisos"])
    discovery_niche_keywords = Column(JSON, default=list, nullable=False)
    # Rango de seguidores para filtrar cuentas
    discovery_follower_min = Column(Integer, default=0, nullable=False)
    discovery_follower_max = Column(Integer, default=1000000, nullable=False)
    # Minimo de posts recientes para considerar cuenta activa
    discovery_min_recent_posts = Column(Integer, default=5, nullable=False)
    # Excluir cuentas verificadas (competidores grandes)
    discovery_exclude_verified = Column(Boolean, default=False, nullable=False)
    # Excluir cuentas privadas
    discovery_exclude_private = Column(Boolean, default=True, nullable=False)

    # =========================================================================
    # KPI WEIGHTS - Pesos para calculo de RPI (Multi-Objetivo)
    # =========================================================================
    # Peso para likes (1.0 default - engagement basico)
    kpi_likes_weight = Column(Float, default=1.0, nullable=False)
    # Peso para comentarios (2.0 default - engagement profundo)
    kpi_comments_weight = Column(Float, default=2.0, nullable=False)
    # Peso para shares (10.0 default - potencial viral, mas valioso)
    kpi_shares_weight = Column(Float, default=10.0, nullable=False)
    # Peso para guardados (5.0 default - intencion de compra)
    kpi_saves_weight = Column(Float, default=5.0, nullable=False)
    # Peso para views (3.0 default - alcance Reels)
    kpi_views_weight = Column(Float, default=3.0, nullable=False)
    # Nombre de plantilla si se uso una predefinida
    kpi_template_name = Column(String(50), nullable=True)

    # =========================================================================
    # EMBEDDING PRECISION - Dimensiones para ML features
    # =========================================================================
    # Precision de embeddings: ultra_low/low/medium/high/max
    # DEFAULT: "low" (128 dims) - seguro para sobremesa normal
    embedding_precision = Column(
        SQLEnum(EmbeddingPrecision),
        default=EmbeddingPrecision.LOW,
        nullable=False
    )

    # =========================================================================
    # MULTIMODAL MODE - Procesamiento Whisper/EasyOCR
    # =========================================================================
    # Modo multimodal: light (rapido ~5s) o full (completo ~20s)
    multimodal_mode = Column(
        SQLEnum(MultimodalMode),
        default=MultimodalMode.LIGHT,
        nullable=False
    )
    # Configuracion detallada de light mode (JSON)
    light_mode_config = Column(JSON, default=DEFAULT_LIGHT_MODE_CONFIG, nullable=False)

    # =========================================================================
    # OWN PROFILE - Identificador para Human-in-the-Loop Feedback
    # =========================================================================
    # Username de Instagram propio (sin @) para feedback loop
    own_instagram_username = Column(String(100), nullable=True)
    # Username de TikTok propio para feedback loop
    own_tiktok_username = Column(String(100), nullable=True)

    # =========================================================================
    # METADATA
    # =========================================================================
    # Descripcion/notas del usuario
    description = Column(Text, nullable=True)
    # Esta config esta activa?
    is_active = Column(Boolean, default=True, nullable=False)
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def get_embedding_dims(self) -> int:
        """Obtiene el numero de dimensiones para la precision actual."""
        return PRECISION_TO_DIMS.get(self.embedding_precision.value, 128)

    def get_kpi_weights_dict(self) -> dict:
        """Retorna los pesos KPI como diccionario."""
        return {
            "likes": self.kpi_likes_weight,
            "comments": self.kpi_comments_weight,
            "shares": self.kpi_shares_weight,
            "saves": self.kpi_saves_weight,
            "views": self.kpi_views_weight,
        }

    def get_kpi_weights_vector(self) -> list:
        """Retorna los pesos KPI como vector ordenado [likes, comments, shares, saves, views]."""
        return [
            self.kpi_likes_weight,
            self.kpi_comments_weight,
            self.kpi_shares_weight,
            self.kpi_saves_weight,
            self.kpi_views_weight,
        ]

    def get_normalized_kpi_weights(self) -> dict:
        """Retorna pesos KPI normalizados (suma = 1.0)."""
        weights = self.get_kpi_weights_vector()
        total = sum(weights)
        if total == 0:
            return {k: 0.2 for k in ["likes", "comments", "shares", "saves", "views"]}
        return {
            "likes": self.kpi_likes_weight / total,
            "comments": self.kpi_comments_weight / total,
            "shares": self.kpi_shares_weight / total,
            "saves": self.kpi_saves_weight / total,
            "views": self.kpi_views_weight / total,
        }

    def get_discovery_config(self) -> dict:
        """Retorna configuracion de discovery como diccionario."""
        return {
            "hashtags": self.discovery_hashtags or [],
            "location_keywords": self.discovery_location_keywords or [],
            "niche_keywords": self.discovery_niche_keywords or [],
            "follower_min": self.discovery_follower_min,
            "follower_max": self.discovery_follower_max,
            "min_recent_posts": self.discovery_min_recent_posts,
            "exclude_verified": self.discovery_exclude_verified,
            "exclude_private": self.discovery_exclude_private,
        }

    def get_light_mode_config(self) -> dict:
        """Retorna configuracion de light mode."""
        base = DEFAULT_LIGHT_MODE_CONFIG.copy()
        if self.light_mode_config:
            base.update(self.light_mode_config)
        return base

    def is_light_mode_enabled(self) -> bool:
        """Verifica si light mode esta habilitado."""
        return self.multimodal_mode == MultimodalMode.LIGHT

    def to_dict(self) -> dict:
        """Convierte el modelo a diccionario completo."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "business_id": self.business_id,
            "discovery": self.get_discovery_config(),
            "kpi_weights": self.get_kpi_weights_dict(),
            "kpi_template_name": self.kpi_template_name,
            "embedding_precision": self.embedding_precision.value,
            "embedding_dims": self.get_embedding_dims(),
            "multimodal_mode": self.multimodal_mode.value,
            "light_mode_config": self.get_light_mode_config(),
            "own_instagram_username": self.own_instagram_username,
            "own_tiktok_username": self.own_tiktok_username,
            "description": self.description,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def create_default(cls, user_id: int, business_id: int) -> "UserConfig":
        """Crea una configuracion con valores por defecto seguros."""
        return cls(
            user_id=user_id,
            business_id=business_id,
            discovery_hashtags=[],
            discovery_location_keywords=[],
            discovery_niche_keywords=[],
            discovery_follower_min=0,
            discovery_follower_max=1000000,
            discovery_min_recent_posts=5,
            discovery_exclude_verified=False,
            discovery_exclude_private=True,
            kpi_likes_weight=1.0,
            kpi_comments_weight=2.0,
            kpi_shares_weight=10.0,
            kpi_saves_weight=5.0,
            kpi_views_weight=3.0,
            embedding_precision=EmbeddingPrecision.LOW,  # Safe default
            multimodal_mode=MultimodalMode.LIGHT,  # Fast default
            light_mode_config=DEFAULT_LIGHT_MODE_CONFIG,
            is_active=True,
        )

    def __repr__(self):
        return (
            f"<UserConfig(id={self.id}, user_id={self.user_id}, "
            f"business_id={self.business_id}, precision={self.embedding_precision.value})>"
        )
