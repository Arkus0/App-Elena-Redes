"""
Trend Velocity - Real-Time Trend Freshness Validation

Este módulo implementa un "Safety Check" para validar que los audios y hashtags
recomendados por el modelo XGBoost estén frescos y no sean tendencias caducadas.

PROBLEMA:
El modelo XGBoost es estático y puede recomendar tendencias pasadas de moda
basándose en likes históricos acumulados, cuando esas tendencias ya no funcionan.

SOLUCIÓN:
Verificar la "velocidad" de una tendencia consultando los últimos 50 videos
que usaron ese audio/hashtag y analizar sus fechas de publicación.

CLASIFICACIÓN:
- TRENDING: >= 50% de videos en las últimas 48h -> Tendencia fresca, USAR
- RISING: >= 30% de videos en última semana -> En crecimiento, USAR
- STABLE: Distribución normal -> Estable, USAR CON PRECAUCIÓN
- STALE: >= 80% de videos hace > 2 semanas -> CADUCADO, NO USAR

Autor: BrandPulse AI
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TrendStatus(str, Enum):
    """Estados de frescura de una tendencia."""
    TRENDING = "trending"      # >= 50% en últimas 48h - MUY FRESCO
    RISING = "rising"          # >= 30% en última semana - EN CRECIMIENTO
    STABLE = "stable"          # Distribución normal - ESTABLE
    STALE = "stale"            # >= 80% hace > 2 semanas - CADUCADO
    UNKNOWN = "unknown"        # No hay datos suficientes


class TrendType(str, Enum):
    """Tipos de elementos de tendencia."""
    AUDIO = "audio"
    HASHTAG = "hashtag"
    CHALLENGE = "challenge"
    EFFECT = "effect"


@dataclass
class TrendSample:
    """Muestra individual de un video/post que usa la tendencia."""
    post_id: str
    posted_at: datetime
    views_count: int = 0
    likes_count: int = 0
    platform: str = "instagram"

    @property
    def age_hours(self) -> float:
        """Edad del post en horas desde ahora."""
        now = datetime.now()
        delta = now - self.posted_at
        return delta.total_seconds() / 3600


@dataclass
class TrendVelocityResult:
    """Resultado del análisis de velocidad de una tendencia."""
    # Identificación
    trend_type: TrendType
    trend_identifier: str  # nombre del audio, hashtag, etc.

    # Estado calculado
    status: TrendStatus
    is_fresh: bool  # True si se puede usar (TRENDING, RISING, STABLE)
    is_stale: bool  # True si está caducado (STALE)

    # Métricas de velocidad
    samples_analyzed: int
    samples_last_48h: int
    samples_last_week: int
    samples_older_2weeks: int

    # Porcentajes calculados
    pct_last_48h: float
    pct_last_week: float
    pct_older_2weeks: float

    # Velocidad (derivada)
    velocity_score: float  # -1 (decayendo) a +1 (creciendo)
    acceleration: float    # Segunda derivada - cambio en velocidad

    # Metadata
    avg_engagement: float
    peak_date: Optional[datetime]
    analyzed_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Mensaje para el usuario
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            "trend_type": self.trend_type.value,
            "trend_identifier": self.trend_identifier,
            "status": self.status.value,
            "is_fresh": self.is_fresh,
            "is_stale": self.is_stale,
            "metrics": {
                "samples_analyzed": self.samples_analyzed,
                "samples_last_48h": self.samples_last_48h,
                "samples_last_week": self.samples_last_week,
                "samples_older_2weeks": self.samples_older_2weeks,
            },
            "percentages": {
                "pct_last_48h": round(self.pct_last_48h * 100, 1),
                "pct_last_week": round(self.pct_last_week * 100, 1),
                "pct_older_2weeks": round(self.pct_older_2weeks * 100, 1),
            },
            "velocity": {
                "score": round(self.velocity_score, 3),
                "acceleration": round(self.acceleration, 3),
            },
            "engagement": {
                "avg_engagement": round(self.avg_engagement, 2),
                "peak_date": self.peak_date.isoformat() if self.peak_date else None,
            },
            "recommendation": self.recommendation,
            "analyzed_at": self.analyzed_at,
        }


@dataclass
class BatchTrendResult:
    """Resultado del análisis de múltiples tendencias."""
    total_analyzed: int
    fresh_trends: List[TrendVelocityResult]
    stale_trends: List[TrendVelocityResult]
    filtered_count: int  # Cuántas se filtraron por ser STALE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_analyzed": self.total_analyzed,
            "fresh_count": len(self.fresh_trends),
            "stale_count": len(self.stale_trends),
            "filtered_count": self.filtered_count,
            "fresh_trends": [t.to_dict() for t in self.fresh_trends],
            "stale_trends": [t.to_dict() for t in self.stale_trends],
        }


class TrendVelocityChecker:
    """
    Motor de verificación de velocidad/frescura de tendencias.

    UMBRALES DE CLASIFICACIÓN:
    - TRENDING: >= 50% de muestras en últimas 48h
    - RISING: >= 30% de muestras en última semana (y < 50% en 48h)
    - STALE: >= 80% de muestras hace más de 2 semanas
    - STABLE: Todo lo demás

    FILTRADO:
    - STALE trends NUNCA se incluyen en recomendaciones
    - TRENDING y RISING tienen prioridad sobre STABLE

    Ejemplo de uso:
        checker = TrendVelocityChecker()

        # Verificar un hashtag
        result = await checker.check_trend_velocity(
            trend_type=TrendType.HASHTAG,
            trend_identifier="#smallbusiness",
            platform="instagram"
        )

        if result.is_stale:
            print("NO usar este hashtag - está caducado")

        # Filtrar lista de audios
        fresh_audios = await checker.filter_fresh_trends(
            trends=[...],
            trend_type=TrendType.AUDIO
        )
    """

    # =========================================================================
    # CONFIGURACIÓN DE UMBRALES
    # =========================================================================

    # Umbral para TRENDING: % de muestras en últimas 48h
    THRESHOLD_TRENDING = 0.50  # >= 50%

    # Umbral para RISING: % de muestras en última semana
    THRESHOLD_RISING = 0.30  # >= 30%

    # Umbral para STALE: % de muestras hace más de 2 semanas
    THRESHOLD_STALE = 0.80  # >= 80%

    # Número de muestras a recolectar
    DEFAULT_SAMPLE_SIZE = 50

    # Períodos de tiempo en horas
    PERIOD_48H = 48
    PERIOD_WEEK = 168  # 7 * 24
    PERIOD_2WEEKS = 336  # 14 * 24

    # Mensajes de recomendación
    RECOMMENDATIONS = {
        TrendStatus.TRENDING: (
            "Tendencia MUY FRESCA. Úsala ahora para máximo alcance. "
            "El algoritmo está priorizando este contenido activamente."
        ),
        TrendStatus.RISING: (
            "Tendencia en CRECIMIENTO. Buen momento para usarla. "
            "Todavía tiene momentum pero no está saturada."
        ),
        TrendStatus.STABLE: (
            "Tendencia ESTABLE. Puede usarse pero no garantiza boost algorítmico. "
            "Considera combinar con un elemento trending para más alcance."
        ),
        TrendStatus.STALE: (
            "TENDENCIA CADUCADA. NO USAR. El algoritmo ya no prioriza este contenido. "
            "Aunque tenga muchos likes históricos, los nuevos posts no rendirán igual."
        ),
        TrendStatus.UNKNOWN: (
            "No hay suficientes datos para evaluar esta tendencia. "
            "Procede con precaución o busca una alternativa más establecida."
        ),
    }

    def __init__(self, apify_service=None):
        """
        Inicializa el checker de velocidad de tendencias.

        Args:
            apify_service: Instancia de ApifyService para scraping (opcional).
                          Si no se proporciona, se usará mock data.
        """
        self.apify_service = apify_service
        logger.info("TrendVelocityChecker initialized")

    async def check_trend_velocity(
        self,
        trend_type: TrendType,
        trend_identifier: str,
        platform: str = "instagram",
        sample_size: int = DEFAULT_SAMPLE_SIZE
    ) -> TrendVelocityResult:
        """
        Verifica la velocidad/frescura de una tendencia específica.

        Args:
            trend_type: Tipo de tendencia (AUDIO, HASHTAG, etc.)
            trend_identifier: Identificador (nombre del audio, hashtag, etc.)
            platform: Plataforma a consultar
            sample_size: Número de muestras a recolectar

        Returns:
            TrendVelocityResult con el análisis completo.
        """
        logger.info(f"Checking velocity for {trend_type.value}: {trend_identifier}")

        # Obtener muestras de videos/posts que usan esta tendencia
        samples = await self._fetch_trend_samples(
            trend_type=trend_type,
            trend_identifier=trend_identifier,
            platform=platform,
            sample_size=sample_size
        )

        # Si no hay suficientes muestras, retornar UNKNOWN
        if len(samples) < 5:
            return self._create_unknown_result(trend_type, trend_identifier)

        # Analizar distribución temporal
        return self._analyze_samples(
            samples=samples,
            trend_type=trend_type,
            trend_identifier=trend_identifier
        )

    async def filter_fresh_trends(
        self,
        trends: List[Dict[str, Any]],
        trend_type: TrendType,
        platform: str = "instagram"
    ) -> BatchTrendResult:
        """
        Filtra una lista de tendencias, eliminando las STALE.

        Args:
            trends: Lista de tendencias a verificar.
                   Cada item debe tener 'identifier' o 'name'.
            trend_type: Tipo de tendencias en la lista.
            platform: Plataforma a consultar.

        Returns:
            BatchTrendResult con tendencias frescas y filtradas.
        """
        fresh = []
        stale = []

        # Procesar en paralelo para mayor velocidad
        tasks = []
        for trend in trends:
            identifier = trend.get("identifier") or trend.get("name") or str(trend)
            tasks.append(
                self.check_trend_velocity(
                    trend_type=trend_type,
                    trend_identifier=identifier,
                    platform=platform
                )
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error checking trend: {result}")
                continue

            if result.is_fresh:
                fresh.append(result)
            else:
                stale.append(result)

        # Ordenar frescos por velocity_score (más trending primero)
        fresh.sort(key=lambda x: x.velocity_score, reverse=True)

        return BatchTrendResult(
            total_analyzed=len(trends),
            fresh_trends=fresh,
            stale_trends=stale,
            filtered_count=len(stale)
        )

    async def validate_recipe_elements(
        self,
        audios: List[str],
        hashtags: List[str],
        platform: str = "instagram"
    ) -> Dict[str, Any]:
        """
        Valida los elementos de una receta viral antes de devolverla al usuario.

        Args:
            audios: Lista de audios recomendados.
            hashtags: Lista de hashtags recomendados.
            platform: Plataforma objetivo.

        Returns:
            Dict con elementos validados y filtrados.
        """
        logger.info(f"Validating recipe: {len(audios)} audios, {len(hashtags)} hashtags")

        # Verificar audios en paralelo
        audio_tasks = [
            self.check_trend_velocity(TrendType.AUDIO, audio, platform)
            for audio in audios
        ]

        # Verificar hashtags en paralelo
        hashtag_tasks = [
            self.check_trend_velocity(TrendType.HASHTAG, tag, platform)
            for tag in hashtags
        ]

        # Ejecutar todo en paralelo
        all_results = await asyncio.gather(
            *audio_tasks, *hashtag_tasks,
            return_exceptions=True
        )

        # Separar resultados
        audio_results = all_results[:len(audios)]
        hashtag_results = all_results[len(audios):]

        # Filtrar elementos frescos
        fresh_audios = []
        stale_audios = []
        for audio, result in zip(audios, audio_results):
            if isinstance(result, Exception):
                fresh_audios.append({"name": audio, "status": "unknown"})
            elif result.is_fresh:
                fresh_audios.append({
                    "name": audio,
                    "status": result.status.value,
                    "velocity_score": result.velocity_score
                })
            else:
                stale_audios.append({
                    "name": audio,
                    "status": result.status.value,
                    "reason": result.recommendation
                })

        fresh_hashtags = []
        stale_hashtags = []
        for tag, result in zip(hashtags, hashtag_results):
            if isinstance(result, Exception):
                fresh_hashtags.append({"name": tag, "status": "unknown"})
            elif result.is_fresh:
                fresh_hashtags.append({
                    "name": tag,
                    "status": result.status.value,
                    "velocity_score": result.velocity_score
                })
            else:
                stale_hashtags.append({
                    "name": tag,
                    "status": result.status.value,
                    "reason": result.recommendation
                })

        # Ordenar por velocity_score
        fresh_audios.sort(key=lambda x: x.get("velocity_score", 0), reverse=True)
        fresh_hashtags.sort(key=lambda x: x.get("velocity_score", 0), reverse=True)

        return {
            "validated_audios": fresh_audios,
            "filtered_audios": stale_audios,
            "validated_hashtags": fresh_hashtags,
            "filtered_hashtags": stale_hashtags,
            "summary": {
                "total_audios": len(audios),
                "fresh_audios": len(fresh_audios),
                "stale_audios": len(stale_audios),
                "total_hashtags": len(hashtags),
                "fresh_hashtags": len(fresh_hashtags),
                "stale_hashtags": len(stale_hashtags),
            },
            "has_stale_elements": len(stale_audios) > 0 or len(stale_hashtags) > 0,
        }

    async def _fetch_trend_samples(
        self,
        trend_type: TrendType,
        trend_identifier: str,
        platform: str,
        sample_size: int
    ) -> List[TrendSample]:
        """
        Obtiene muestras de videos/posts que usan una tendencia.

        En producción, esto usaría ApifyService para scraping real.
        """
        # Si tenemos servicio de Apify disponible, usarlo
        if self.apify_service and self.apify_service.is_available():
            return await self._fetch_from_apify(
                trend_type, trend_identifier, platform, sample_size
            )

        # Mock data para desarrollo/demo
        return self._generate_mock_samples(
            trend_type, trend_identifier, sample_size
        )

    async def _fetch_from_apify(
        self,
        trend_type: TrendType,
        trend_identifier: str,
        platform: str,
        sample_size: int
    ) -> List[TrendSample]:
        """Obtiene datos reales via Apify."""
        try:
            if trend_type == TrendType.HASHTAG:
                # Buscar por hashtag
                results = await self.apify_service.search_trending_content(
                    keyword=trend_identifier.lstrip("#"),
                    platform=platform,
                    max_results=sample_size
                )
            elif trend_type == TrendType.AUDIO:
                # Buscar por audio/sonido
                results = await self.apify_service.search_trending_content(
                    keyword=trend_identifier,
                    platform=platform,
                    max_results=sample_size
                )
            else:
                results = []

            # Convertir a TrendSample
            samples = []
            for item in results:
                posted_at = item.get("timestamp") or item.get("createTime")
                if posted_at:
                    if isinstance(posted_at, str):
                        try:
                            posted_at = datetime.fromisoformat(posted_at.replace("Z", "+00:00"))
                        except ValueError:
                            continue
                    elif isinstance(posted_at, (int, float)):
                        posted_at = datetime.fromtimestamp(posted_at)

                    samples.append(TrendSample(
                        post_id=str(item.get("id", "")),
                        posted_at=posted_at,
                        views_count=item.get("playCount") or item.get("viewCount") or 0,
                        likes_count=item.get("diggCount") or item.get("likesCount") or 0,
                        platform=platform
                    ))

            return samples

        except Exception as e:
            logger.error(f"Error fetching from Apify: {e}")
            return self._generate_mock_samples(trend_type, trend_identifier, sample_size)

    def _generate_mock_samples(
        self,
        trend_type: TrendType,
        trend_identifier: str,
        sample_size: int
    ) -> List[TrendSample]:
        """
        Genera datos mock para desarrollo.
        Simula diferentes escenarios de tendencias.
        """
        samples = []
        now = datetime.now()

        # Determinar el patrón de distribución basándose en el identificador
        # Esto permite testear diferentes escenarios
        identifier_lower = trend_identifier.lower()

        if "trending" in identifier_lower or "viral" in identifier_lower:
            # Simular tendencia MUY fresca (>50% en 48h)
            distribution = [0.55, 0.25, 0.20]  # 48h, semana, >2 semanas
        elif "rising" in identifier_lower or "new" in identifier_lower:
            # Simular tendencia en crecimiento
            distribution = [0.25, 0.45, 0.30]
        elif "old" in identifier_lower or "stale" in identifier_lower:
            # Simular tendencia caducada
            distribution = [0.05, 0.10, 0.85]
        else:
            # Distribución aleatoria basada en hash del identificador
            hash_val = hash(trend_identifier) % 100
            if hash_val < 25:
                distribution = [0.60, 0.25, 0.15]  # Trending
            elif hash_val < 50:
                distribution = [0.20, 0.50, 0.30]  # Rising
            elif hash_val < 75:
                distribution = [0.15, 0.35, 0.50]  # Stable
            else:
                distribution = [0.05, 0.12, 0.83]  # Stale

        # Generar muestras según distribución
        for i in range(sample_size):
            rand = np.random.random()

            if rand < distribution[0]:
                # Últimas 48h
                hours_ago = np.random.uniform(0, self.PERIOD_48H)
            elif rand < distribution[0] + distribution[1]:
                # Última semana (pero no 48h)
                hours_ago = np.random.uniform(self.PERIOD_48H, self.PERIOD_WEEK)
            else:
                # Más de 2 semanas
                hours_ago = np.random.uniform(self.PERIOD_2WEEKS, self.PERIOD_2WEEKS * 2)

            posted_at = now - timedelta(hours=hours_ago)

            samples.append(TrendSample(
                post_id=f"mock_{i}_{trend_identifier[:10]}",
                posted_at=posted_at,
                views_count=int(np.random.exponential(50000)),
                likes_count=int(np.random.exponential(5000)),
                platform="instagram"
            ))

        return samples

    def _analyze_samples(
        self,
        samples: List[TrendSample],
        trend_type: TrendType,
        trend_identifier: str
    ) -> TrendVelocityResult:
        """Analiza las muestras y calcula la velocidad de la tendencia."""
        now = datetime.now()

        # Categorizar por período
        last_48h = []
        last_week = []
        older_2weeks = []

        for sample in samples:
            age_hours = (now - sample.posted_at).total_seconds() / 3600

            if age_hours <= self.PERIOD_48H:
                last_48h.append(sample)
            elif age_hours <= self.PERIOD_WEEK:
                last_week.append(sample)
            elif age_hours > self.PERIOD_2WEEKS:
                older_2weeks.append(sample)

        total = len(samples)

        # Calcular porcentajes
        pct_48h = len(last_48h) / total if total > 0 else 0
        pct_week = len(last_week) / total if total > 0 else 0
        pct_2weeks = len(older_2weeks) / total if total > 0 else 0

        # Determinar estado
        status = self._determine_status(pct_48h, pct_week, pct_2weeks)

        # Calcular velocity score (-1 a +1)
        # Positivo = creciendo, Negativo = decayendo
        velocity_score = self._calculate_velocity(pct_48h, pct_week, pct_2weeks)

        # Calcular aceleración (segunda derivada)
        acceleration = self._calculate_acceleration(samples)

        # Calcular engagement promedio
        avg_engagement = np.mean([s.likes_count for s in samples]) if samples else 0

        # Encontrar fecha pico
        peak_date = self._find_peak_date(samples)

        return TrendVelocityResult(
            trend_type=trend_type,
            trend_identifier=trend_identifier,
            status=status,
            is_fresh=status in [TrendStatus.TRENDING, TrendStatus.RISING, TrendStatus.STABLE],
            is_stale=status == TrendStatus.STALE,
            samples_analyzed=total,
            samples_last_48h=len(last_48h),
            samples_last_week=len(last_week),
            samples_older_2weeks=len(older_2weeks),
            pct_last_48h=pct_48h,
            pct_last_week=pct_week,
            pct_older_2weeks=pct_2weeks,
            velocity_score=velocity_score,
            acceleration=acceleration,
            avg_engagement=avg_engagement,
            peak_date=peak_date,
            recommendation=self.RECOMMENDATIONS[status]
        )

    def _determine_status(
        self,
        pct_48h: float,
        pct_week: float,
        pct_2weeks: float
    ) -> TrendStatus:
        """Determina el estado de la tendencia basándose en los porcentajes."""
        # STALE: >= 80% hace más de 2 semanas
        if pct_2weeks >= self.THRESHOLD_STALE:
            return TrendStatus.STALE

        # TRENDING: >= 50% en últimas 48h
        if pct_48h >= self.THRESHOLD_TRENDING:
            return TrendStatus.TRENDING

        # RISING: >= 30% en última semana (combinado con 48h)
        if (pct_48h + pct_week) >= self.THRESHOLD_RISING:
            return TrendStatus.RISING

        # STABLE: todo lo demás
        return TrendStatus.STABLE

    def _calculate_velocity(
        self,
        pct_48h: float,
        pct_week: float,
        pct_2weeks: float
    ) -> float:
        """
        Calcula un score de velocidad de -1 a +1.

        Positivo = tendencia creciendo
        Negativo = tendencia decayendo
        """
        # Pesos para cada período (más reciente = más importante)
        weight_48h = 3.0
        weight_week = 1.0
        weight_2weeks = -2.0

        weighted_sum = (
            pct_48h * weight_48h +
            pct_week * weight_week +
            pct_2weeks * weight_2weeks
        )

        # Normalizar a rango -1 a +1
        max_positive = weight_48h  # Si todo está en 48h
        max_negative = weight_2weeks  # Si todo está hace >2 semanas

        if weighted_sum >= 0:
            return min(weighted_sum / max_positive, 1.0)
        else:
            return max(weighted_sum / abs(max_negative), -1.0)

    def _calculate_acceleration(self, samples: List[TrendSample]) -> float:
        """Calcula la aceleración (segunda derivada) de la tendencia."""
        if len(samples) < 10:
            return 0.0

        # Ordenar por fecha
        sorted_samples = sorted(samples, key=lambda x: x.posted_at)

        # Dividir en dos mitades
        mid = len(sorted_samples) // 2
        first_half = sorted_samples[:mid]
        second_half = sorted_samples[mid:]

        # Calcular densidad de posts por período
        if first_half and second_half:
            first_span = (first_half[-1].posted_at - first_half[0].posted_at).total_seconds() / 3600
            second_span = (second_half[-1].posted_at - second_half[0].posted_at).total_seconds() / 3600

            if first_span > 0 and second_span > 0:
                first_density = len(first_half) / first_span
                second_density = len(second_half) / second_span

                # Aceleración = cambio en densidad
                if first_density > 0:
                    return (second_density - first_density) / first_density

        return 0.0

    def _find_peak_date(self, samples: List[TrendSample]) -> Optional[datetime]:
        """Encuentra la fecha con más actividad."""
        if not samples:
            return None

        # Agrupar por día
        day_counts: Dict[str, int] = {}
        for sample in samples:
            day_key = sample.posted_at.strftime("%Y-%m-%d")
            day_counts[day_key] = day_counts.get(day_key, 0) + 1

        if not day_counts:
            return None

        # Encontrar día con más posts
        peak_day = max(day_counts, key=day_counts.get)
        return datetime.strptime(peak_day, "%Y-%m-%d")

    def _create_unknown_result(
        self,
        trend_type: TrendType,
        trend_identifier: str
    ) -> TrendVelocityResult:
        """Crea un resultado UNKNOWN cuando no hay datos suficientes."""
        return TrendVelocityResult(
            trend_type=trend_type,
            trend_identifier=trend_identifier,
            status=TrendStatus.UNKNOWN,
            is_fresh=True,  # Por defecto, asumir que se puede usar
            is_stale=False,
            samples_analyzed=0,
            samples_last_48h=0,
            samples_last_week=0,
            samples_older_2weeks=0,
            pct_last_48h=0.0,
            pct_last_week=0.0,
            pct_older_2weeks=0.0,
            velocity_score=0.0,
            acceleration=0.0,
            avg_engagement=0.0,
            peak_date=None,
            recommendation=self.RECOMMENDATIONS[TrendStatus.UNKNOWN]
        )


# =============================================================================
# SINGLETON Y FUNCIONES DE UTILIDAD
# =============================================================================

_trend_velocity_instance: Optional[TrendVelocityChecker] = None


def get_trend_velocity_checker(apify_service=None) -> TrendVelocityChecker:
    """
    Obtiene la instancia singleton del TrendVelocityChecker.

    Args:
        apify_service: Servicio de Apify para scraping real (opcional).

    Returns:
        Instancia del checker de velocidad.
    """
    global _trend_velocity_instance
    if _trend_velocity_instance is None:
        _trend_velocity_instance = TrendVelocityChecker(apify_service)
    return _trend_velocity_instance


def reset_trend_velocity_checker() -> None:
    """Reinicia la instancia singleton (útil para tests)."""
    global _trend_velocity_instance
    _trend_velocity_instance = None


async def quick_check_trend(
    identifier: str,
    trend_type: str = "hashtag",
    platform: str = "instagram"
) -> Dict[str, Any]:
    """
    Función de utilidad para verificación rápida de una tendencia.

    Args:
        identifier: Nombre del hashtag, audio, etc.
        trend_type: Tipo de tendencia ("hashtag", "audio", "challenge")
        platform: Plataforma a consultar

    Returns:
        Dict con estado y recomendación.
    """
    checker = get_trend_velocity_checker()

    type_map = {
        "hashtag": TrendType.HASHTAG,
        "audio": TrendType.AUDIO,
        "challenge": TrendType.CHALLENGE,
        "effect": TrendType.EFFECT,
    }

    result = await checker.check_trend_velocity(
        trend_type=type_map.get(trend_type, TrendType.HASHTAG),
        trend_identifier=identifier,
        platform=platform
    )

    return {
        "identifier": identifier,
        "status": result.status.value,
        "is_fresh": result.is_fresh,
        "is_stale": result.is_stale,
        "velocity_score": result.velocity_score,
        "recommendation": result.recommendation,
    }
