"""
UserConfigService - Servicio Central de Configuracion de Usuario

Este servicio es el UNICO punto de acceso para cargar configuraciones de usuario
en todos los pipelines (ingest, train, inference). Elimina hardcodes y asegura
sincronizacion real-time entre frontend y backend.

Uso en pipelines:
    from app.services.user_config_service import get_pipeline_config, user_config_service

    # En cualquier pipeline:
    config = await get_pipeline_config(user_id=1, business_id=1, db=session)
    logger.info(f"User config loaded: precision={config.embedding_precision}, "
                f"multimodal={config.multimodal_mode}, own=@{config.own_instagram_username}")

    # Usar en embeddings:
    extractor = EmbeddingExtractor(precision=config.embedding_precision)

    # Usar en RPI:
    rpi = config.get_weighted_rpi(likes=100, comments=20, shares=5, saves=10, views=1000)

Author: BrandPulse AI
"""

import logging
from typing import Optional, Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from sqlalchemy.orm import selectinload
from functools import lru_cache
from datetime import datetime

from app.models.user_config import UserConfig, EmbeddingPrecision, MultimodalMode
from app.schemas.user_config import (
    UserConfigCreate, UserConfigUpdate, UserConfigResponse,
    PipelineConfig, DiscoveryConfig, KPIWeights, LightModeConfig,
    PrecisionOption, MultimodalOption, KPITemplate, UserConfigInfoResponse,
    PRECISION_DIMS_MAP, UserConfigBase,
    EmbeddingPrecision as EP, MultimodalMode as MM
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants (Static Config Info)
# =============================================================================

_PRECISION_OPTIONS = [
    PrecisionOption(
        value="ultra_low",
        label="Ultra Baja (64 dims)",
        dimensions=64,
        description="Ultra rapido para PC modesto",
        performance="Muy rapido, ~0.3GB RAM, train <15s",
        is_recommended=False,
    ),
    PrecisionOption(
        value="low",
        label="Baja (128 dims) - Recomendada",
        dimensions=128,
        description="Recomendado para sobremesa normal",
        performance="Rapido, ~0.5GB RAM, train <30s",
        is_recommended=True,
    ),
    PrecisionOption(
        value="medium",
        label="Media (256 dims)",
        dimensions=256,
        description="Balance precision/velocidad",
        performance="Moderado, ~1GB RAM, train <1min",
        is_recommended=False,
    ),
    PrecisionOption(
        value="high",
        label="Alta (384 dims)",
        dimensions=384,
        description="Full precision con TruncatedSVD",
        performance="Lento, ~1.5GB RAM, train <2min",
        is_recommended=False,
    ),
    PrecisionOption(
        value="max",
        label="Maxima (384 raw)",
        dimensions=384,
        description="Sin reduccion - mejor matices creativos",
        performance="Mas lento, ~2GB RAM",
        is_recommended=False,
    ),
]

_MULTIMODAL_OPTIONS = [
    MultimodalOption(
        value="light",
        label="Modo Ligero - Recomendado",
        description="Whisper tiny, 3s audio, 5 frames OCR. ~5s por video.",
        estimated_time_seconds=5.0,
        is_recommended=True,
    ),
    MultimodalOption(
        value="full",
        label="Modo Completo",
        description="Whisper base, audio completo, todos frames. ~20s por video.",
        estimated_time_seconds=20.0,
        is_recommended=False,
    ),
]

_KPI_TEMPLATES = [
    KPITemplate(
        name="brand_awareness",
        display_name="Brand Awareness",
        description="Maximiza alcance y visibilidad",
        icon="eye",
        weights=KPIWeights(likes=1.0, comments=1.0, shares=5.0, saves=2.0, views=10.0),
    ),
    KPITemplate(
        name="leads",
        display_name="Generacion de Leads",
        description="Enfoca en guardados e interaccion profunda",
        icon="target",
        weights=KPIWeights(likes=1.0, comments=5.0, shares=8.0, saves=10.0, views=2.0),
    ),
    KPITemplate(
        name="community",
        display_name="Comunidad",
        description="Prioriza comentarios y engagement profundo",
        icon="users",
        weights=KPIWeights(likes=3.0, comments=10.0, shares=5.0, saves=3.0, views=1.0),
    ),
    KPITemplate(
        name="viral",
        display_name="Potencial Viral",
        description="Maximiza shares y alcance organico",
        icon="share",
        weights=KPIWeights(likes=2.0, comments=3.0, shares=15.0, saves=5.0, views=8.0),
    ),
    KPITemplate(
        name="balanced",
        display_name="Balanceado",
        description="Equilibrio entre todas las metricas",
        icon="balance",
        weights=KPIWeights(likes=1.0, comments=2.0, shares=10.0, saves=5.0, views=3.0),
    ),
]


class UserConfigService:
    """
    Servicio para gestionar configuraciones de usuario.

    IMPORTANTE: Este servicio debe ser usado por TODOS los pipelines
    para cargar configuraciones. NO usar hardcodes en ninguna parte.
    """

    # Cache en memoria para configs frecuentes (TTL corto)
    _config_cache: Dict[str, tuple] = {}
    _cache_ttl_seconds = 60  # Refresca cada minuto

    # =========================================================================
    # Core CRUD Operations
    # =========================================================================

    async def get_config(
        self,
        db: AsyncSession,
        user_id: int,
        business_id: int
    ) -> Optional[UserConfig]:
        """
        Obtiene la configuracion activa para un usuario/business.

        Args:
            db: Session de base de datos
            user_id: ID del usuario
            business_id: ID del negocio

        Returns:
            UserConfig si existe, None si no
        """
        result = await db.execute(
            select(UserConfig).where(
                and_(
                    UserConfig.user_id == user_id,
                    UserConfig.business_id == business_id,
                    UserConfig.is_active == True
                )
            )
        )
        return result.scalar_one_or_none()

    async def get_or_create_config(
        self,
        db: AsyncSession,
        user_id: int,
        business_id: int
    ) -> UserConfig:
        """
        Obtiene config existente o crea una con defaults seguros.

        Esta es la funcion principal que deben usar los pipelines.
        Siempre retorna una config valida.
        """
        config = await self.get_config(db, user_id, business_id)

        if config is None:
            logger.info(
                f"Creating default config for user={user_id}, business={business_id}"
            )
            config = UserConfig.create_default(user_id, business_id)
            db.add(config)
            await db.commit()
            await db.refresh(config)

            logger.info(
                f"New user config created: precision={config.embedding_precision.value}, "
                f"multimodal={config.multimodal_mode.value}"
            )

        return config

    async def create_config(
        self,
        db: AsyncSession,
        user_id: int,
        business_id: int,
        config_data: UserConfigCreate
    ) -> UserConfig:
        """
        Crea una nueva configuracion de usuario.

        Desactiva configuraciones previas del mismo user/business.
        """
        # Desactivar configs anteriores
        await self._deactivate_existing(db, user_id, business_id)

        # Crear nueva
        discovery = config_data.discovery or DiscoveryConfig()
        kpi = config_data.kpi_weights or KPIWeights()
        light = config_data.light_mode_config or LightModeConfig()

        config = UserConfig(
            user_id=user_id,
            business_id=business_id,
            # Discovery
            discovery_hashtags=discovery.hashtags,
            discovery_location_keywords=discovery.location_keywords,
            discovery_niche_keywords=discovery.niche_keywords,
            discovery_follower_min=discovery.follower_min,
            discovery_follower_max=discovery.follower_max,
            discovery_min_recent_posts=discovery.min_recent_posts,
            discovery_exclude_verified=discovery.exclude_verified,
            discovery_exclude_private=discovery.exclude_private,
            # KPI Weights
            kpi_likes_weight=kpi.likes,
            kpi_comments_weight=kpi.comments,
            kpi_shares_weight=kpi.shares,
            kpi_saves_weight=kpi.saves,
            kpi_views_weight=kpi.views,
            kpi_template_name=config_data.kpi_template_name,
            # Embedding
            embedding_precision=EmbeddingPrecision(config_data.embedding_precision.value),
            # Multimodal
            multimodal_mode=MultimodalMode(config_data.multimodal_mode.value),
            light_mode_config=light.dict(),
            # Own profile
            own_instagram_username=config_data.own_instagram_username,
            own_tiktok_username=config_data.own_tiktok_username,
            description=config_data.description,
            is_active=True,
        )

        db.add(config)
        await db.commit()
        await db.refresh(config)

        # Invalidar cache
        self._invalidate_cache(user_id, business_id)

        logger.info(
            f"User config created: user={user_id}, business={business_id}, "
            f"precision={config.embedding_precision.value}, "
            f"multimodal={config.multimodal_mode.value}, "
            f"own=@{config.own_instagram_username or 'N/A'}"
        )

        return config

    async def update_config(
        self,
        db: AsyncSession,
        user_id: int,
        business_id: int,
        update_data: UserConfigUpdate
    ) -> UserConfig:
        """
        Actualiza la configuracion existente.

        Solo actualiza campos proporcionados (patch semantics).
        """
        config = await self.get_or_create_config(db, user_id, business_id)

        # Update discovery fields
        if update_data.discovery:
            d = update_data.discovery
            config.discovery_hashtags = d.hashtags
            config.discovery_location_keywords = d.location_keywords
            config.discovery_niche_keywords = d.niche_keywords
            config.discovery_follower_min = d.follower_min
            config.discovery_follower_max = d.follower_max
            config.discovery_min_recent_posts = d.min_recent_posts
            config.discovery_exclude_verified = d.exclude_verified
            config.discovery_exclude_private = d.exclude_private

        # Update KPI weights
        if update_data.kpi_weights:
            k = update_data.kpi_weights
            config.kpi_likes_weight = k.likes
            config.kpi_comments_weight = k.comments
            config.kpi_shares_weight = k.shares
            config.kpi_saves_weight = k.saves
            config.kpi_views_weight = k.views

        if update_data.kpi_template_name is not None:
            config.kpi_template_name = update_data.kpi_template_name

        # Update embedding precision
        if update_data.embedding_precision:
            config.embedding_precision = EmbeddingPrecision(update_data.embedding_precision.value)

        # Update multimodal mode
        if update_data.multimodal_mode:
            config.multimodal_mode = MultimodalMode(update_data.multimodal_mode.value)

        if update_data.light_mode_config:
            config.light_mode_config = update_data.light_mode_config.dict()

        # Update own profile
        if update_data.own_instagram_username is not None:
            config.own_instagram_username = update_data.own_instagram_username
        if update_data.own_tiktok_username is not None:
            config.own_tiktok_username = update_data.own_tiktok_username

        if update_data.description is not None:
            config.description = update_data.description

        config.updated_at = datetime.utcnow()

        await db.commit()
        await db.refresh(config)

        # Invalidar cache
        self._invalidate_cache(user_id, business_id)

        logger.info(
            f"User config updated: user={user_id}, business={business_id}, "
            f"precision={config.embedding_precision.value}, "
            f"multimodal={config.multimodal_mode.value}, "
            f"own=@{config.own_instagram_username or 'N/A'}"
        )

        return config

    async def _deactivate_existing(
        self,
        db: AsyncSession,
        user_id: int,
        business_id: int
    ):
        """Desactiva configs existentes para el user/business."""
        result = await db.execute(
            select(UserConfig).where(
                and_(
                    UserConfig.user_id == user_id,
                    UserConfig.business_id == business_id,
                    UserConfig.is_active == True
                )
            )
        )
        existing = result.scalars().all()
        for config in existing:
            config.is_active = False
        await db.flush()

    # =========================================================================
    # Pipeline Config - Formato simplificado para pipelines
    # =========================================================================

    async def get_pipeline_config(
        self,
        db: AsyncSession,
        user_id: int,
        business_id: int
    ) -> PipelineConfig:
        """
        Obtiene configuracion en formato optimizado para pipelines.

        Esta es la funcion que deben usar ingest.py, ml_service.py,
        growth_prediction_engine.py, etc.

        Incluye:
        - embedding_precision y dims
        - multimodal_mode y light_mode config
        - kpi_weights (raw y normalized)
        - own_username para feedback loop
        - discovery config

        Returns:
            PipelineConfig listo para usar en cualquier pipeline
        """
        # Check cache first
        cache_key = f"{user_id}:{business_id}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        # Load from DB
        config = await self.get_or_create_config(db, user_id, business_id)
        pipeline_config = PipelineConfig.from_user_config(config)

        # Cache it
        self._set_cache(cache_key, pipeline_config)

        # Log critical info
        logger.info(
            f"User config loaded: precision={pipeline_config.embedding_precision} ({pipeline_config.embedding_dims} dims), "
            f"multimodal={pipeline_config.multimodal_mode}, "
            f"own=@{pipeline_config.own_instagram_username or 'N/A'}"
        )

        return pipeline_config

    # =========================================================================
    # Available Options (para frontend)
    # =========================================================================

    def get_config_info(self) -> UserConfigInfoResponse:
        """Retorna opciones disponibles para el frontend."""
        default_config = UserConfigBase(
            embedding_precision=EP.LOW,
            multimodal_mode=MM.LIGHT,
        )

        return UserConfigInfoResponse(
            precision_options=_PRECISION_OPTIONS,
            multimodal_options=_MULTIMODAL_OPTIONS,
            kpi_templates=_KPI_TEMPLATES,
            default_config=default_config,
        )

    # =========================================================================
    # Cache Management
    # =========================================================================

    def _get_from_cache(self, key: str) -> Optional[PipelineConfig]:
        """Obtiene config de cache si no ha expirado."""
        if key in self._config_cache:
            config, timestamp = self._config_cache[key]
            if (datetime.utcnow() - timestamp).total_seconds() < self._cache_ttl_seconds:
                return config
            # Expired
            del self._config_cache[key]
        return None

    def _set_cache(self, key: str, config: PipelineConfig):
        """Guarda config en cache."""
        self._config_cache[key] = (config, datetime.utcnow())

    def _invalidate_cache(self, user_id: int, business_id: int):
        """Invalida cache para un user/business."""
        key = f"{user_id}:{business_id}"
        if key in self._config_cache:
            del self._config_cache[key]

    def clear_cache(self):
        """Limpia todo el cache."""
        self._config_cache.clear()


# =============================================================================
# Singleton instance
# =============================================================================

user_config_service = UserConfigService()


# =============================================================================
# Convenience Functions for Pipelines
# =============================================================================

async def get_pipeline_config(
    user_id: int,
    business_id: int,
    db: AsyncSession
) -> PipelineConfig:
    """
    Funcion de conveniencia para obtener config en pipelines.

    Uso:
        from app.services.user_config_service import get_pipeline_config

        config = await get_pipeline_config(user_id=1, business_id=1, db=session)
        extractor = EmbeddingExtractor(precision=config.embedding_precision)
    """
    return await user_config_service.get_pipeline_config(db, user_id, business_id)


def get_default_pipeline_config(user_id: int = 0, business_id: int = 0) -> PipelineConfig:
    """
    Retorna config por defecto (sin acceso a DB).

    Usar solo como fallback cuando no hay acceso a DB.
    """
    return PipelineConfig.create_default(user_id, business_id)
