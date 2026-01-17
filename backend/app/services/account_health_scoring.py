"""
Account Health Scoring - Evaluación de salud de cuenta antes de predicciones

Este módulo analiza la salud de una cuenta de usuario basándose en sus últimos
posts para calibrar las predicciones del modelo XGBoost.

LÓGICA DE AUTORIDAD:
1. Analiza los últimos 10 posts de la cuenta del usuario (no competidores)
2. Calcula la media de views y desviación estándar
3. Clasifica la autoridad según la relación views/seguidores:
   - Normal: media views >= 10% de seguidores
   - Low_Authority: media views < 10% de seguidores
   - Possible_Shadowban: media views < 2% de seguidores

CALIBRACIÓN DE PREDICCIÓN:
- Low_Authority: Factor 0.3x en predicción de views absolutas
- Possible_Shadowban: Factor 0.1x + alerta crítica
- Normal: Sin modificación (factor 1.0x)

Autor: BrandPulse AI
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class AccountHealthStatus(str, Enum):
    """Estados de salud de la cuenta."""
    HEALTHY = "healthy"
    LOW_AUTHORITY = "low_authority"
    POSSIBLE_SHADOWBAN = "possible_shadowban"
    INSUFFICIENT_DATA = "insufficient_data"


class AuthorityLevel(str, Enum):
    """Niveles de autoridad basados en la relación views/seguidores."""
    HIGH = "high"          # >= 20% de seguidores en views
    NORMAL = "normal"      # >= 10% de seguidores en views
    LOW = "low"            # < 10% de seguidores en views (Low_Authority)
    CRITICAL = "critical"  # < 2% de seguidores en views (Possible_Shadowban)


@dataclass
class PostMetrics:
    """Métricas de un post individual."""
    post_id: str
    views_count: int
    likes_count: int = 0
    comments_count: int = 0
    shares_count: int = 0
    saves_count: int = 0
    posted_at: Optional[datetime] = None

    @property
    def engagement_rate(self) -> float:
        """Calcula el engagement rate del post."""
        if self.views_count == 0:
            return 0.0
        total_interactions = self.likes_count + self.comments_count + self.shares_count + self.saves_count
        return total_interactions / self.views_count


@dataclass
class AccountHealthResult:
    """Resultado completo del análisis de salud de la cuenta."""
    # Estado principal
    health_status: AccountHealthStatus
    authority_level: AuthorityLevel

    # Métricas calculadas
    avg_views: float
    std_views: float
    follower_count: int
    views_to_followers_ratio: float  # Porcentaje de seguidores que ven los posts

    # Factores de calibración
    prediction_penalty_factor: float  # 1.0 = sin penalización, 0.3 = Low Authority

    # Posts analizados
    posts_analyzed: int
    min_posts_required: int = 5

    # Mensajes para el usuario
    warning_message: Optional[str] = None
    recommendation: Optional[str] = None

    # Metadata
    analyzed_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convierte el resultado a diccionario."""
        return {
            "health_status": self.health_status.value,
            "authority_level": self.authority_level.value,
            "metrics": {
                "avg_views": round(self.avg_views, 2),
                "std_views": round(self.std_views, 2),
                "follower_count": self.follower_count,
                "views_to_followers_ratio": round(self.views_to_followers_ratio * 100, 2),
            },
            "calibration": {
                "prediction_penalty_factor": self.prediction_penalty_factor,
                "is_penalized": self.prediction_penalty_factor < 1.0,
            },
            "analysis": {
                "posts_analyzed": self.posts_analyzed,
                "min_posts_required": self.min_posts_required,
                "has_sufficient_data": self.posts_analyzed >= self.min_posts_required,
            },
            "user_feedback": {
                "warning_message": self.warning_message,
                "recommendation": self.recommendation,
            },
            "analyzed_at": self.analyzed_at,
        }


@dataclass
class CalibratedPrediction:
    """Predicción calibrada con información de salud de cuenta."""
    # Predicción original
    original_rpi_score: float
    original_rpi_raw: float

    # Predicción calibrada
    calibrated_rpi_score: float
    calibrated_rpi_raw: float

    # Factor aplicado
    penalty_factor: float

    # Información de salud
    account_health: AccountHealthResult

    # Mensaje combinado para el usuario
    combined_message: str

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            "original_prediction": {
                "rpi_score": round(self.original_rpi_score, 4),
                "rpi_raw": round(self.original_rpi_raw, 4),
            },
            "calibrated_prediction": {
                "rpi_score": round(self.calibrated_rpi_score, 4),
                "rpi_raw": round(self.calibrated_rpi_raw, 4),
            },
            "calibration_applied": {
                "penalty_factor": self.penalty_factor,
                "was_penalized": self.penalty_factor < 1.0,
            },
            "account_health": self.account_health.to_dict(),
            "combined_message": self.combined_message,
        }


class AccountHealthScoring:
    """
    Motor de evaluación de salud de cuenta para calibración de predicciones.

    Este sistema analiza la performance histórica de la cuenta del usuario
    para ajustar las predicciones del modelo XGBoost de manera realista.

    UMBRALES DE AUTORIDAD:
    - >= 20% views/followers: HIGH (factor 1.0x, bonus posible)
    - >= 10% views/followers: NORMAL (factor 1.0x)
    - < 10% views/followers: LOW_AUTHORITY (factor 0.3x)
    - < 2% views/followers: POSSIBLE_SHADOWBAN (factor 0.1x)

    Ejemplo de uso:
        scoring = AccountHealthScoring()

        # Evaluar salud con datos de posts
        health = scoring.evaluate_account_health(
            recent_posts=[...],  # Últimos 10 posts
            follower_count=10000
        )

        # Calibrar una predicción
        calibrated = scoring.calibrate_prediction(
            original_score=1.5,
            original_raw=3.48,
            account_health=health
        )
    """

    # =========================================================================
    # CONFIGURACIÓN DE UMBRALES
    # =========================================================================

    # Umbrales de ratio views/seguidores para determinar autoridad
    THRESHOLD_HIGH_AUTHORITY = 0.20      # >= 20% views de seguidores
    THRESHOLD_NORMAL_AUTHORITY = 0.10    # >= 10% views de seguidores
    THRESHOLD_LOW_AUTHORITY = 0.02       # < 2% indica posible shadowban

    # Factores de penalización por nivel de autoridad
    PENALTY_FACTOR_HIGH = 1.0            # Sin penalización
    PENALTY_FACTOR_NORMAL = 1.0          # Sin penalización
    PENALTY_FACTOR_LOW = 0.3             # Penalización severa para Low Authority
    PENALTY_FACTOR_CRITICAL = 0.1        # Penalización crítica para Shadowban

    # Mínimo de posts requeridos para análisis confiable
    MIN_POSTS_FOR_ANALYSIS = 5
    DEFAULT_POSTS_TO_ANALYZE = 10

    # Mensajes para el usuario
    MESSAGES = {
        AuthorityLevel.HIGH: {
            "warning": None,
            "recommendation": "Tu cuenta tiene excelente tracción. Las predicciones se aplican sin modificación."
        },
        AuthorityLevel.NORMAL: {
            "warning": None,
            "recommendation": "Tu cuenta tiene buena tracción. Las predicciones reflejan el potencial real del contenido."
        },
        AuthorityLevel.LOW: {
            "warning": (
                "Tu cuenta tiene baja tracción actualmente. "
                "Este video está optimizado, pero necesitarás subir 5-10 así de constantes "
                "para reactivar el algoritmo."
            ),
            "recommendation": (
                "Recomendamos: 1) Publicar consistentemente en horarios pico, "
                "2) Usar hooks más agresivos en los primeros 3 segundos, "
                "3) Responder a todos los comentarios en los primeros 30 minutos."
            )
        },
        AuthorityLevel.CRITICAL: {
            "warning": (
                "ALERTA: Tu cuenta muestra señales de posible shadowban o restricción algorítmica. "
                "Las views están muy por debajo de lo esperado para tu número de seguidores. "
                "Este video puede tener dificultades para alcanzar su potencial incluso si está bien optimizado."
            ),
            "recommendation": (
                "Acciones urgentes: 1) Revisa si tienes violaciones de comunidad pendientes, "
                "2) Evita hashtags prohibidos o shadowbaneados, "
                "3) Publica contenido original sin marcas de agua, "
                "4) Considera pausar 48-72h y reiniciar con contenido seguro, "
                "5) Interactúa genuinamente con tu comunidad antes de publicar nuevo contenido."
            )
        }
    }

    def __init__(self):
        """Inicializa el motor de scoring de salud de cuenta."""
        logger.info("AccountHealthScoring initialized")

    def evaluate_account_health(
        self,
        recent_posts: List[Dict[str, Any]],
        follower_count: int,
        posts_to_analyze: int = DEFAULT_POSTS_TO_ANALYZE
    ) -> AccountHealthResult:
        """
        Evalúa la salud de una cuenta basándose en sus posts recientes.

        Args:
            recent_posts: Lista de posts recientes con al menos 'views_count'.
                         Formato esperado: [{"views_count": int, "likes_count": int, ...}, ...]
            follower_count: Número de seguidores de la cuenta.
            posts_to_analyze: Número máximo de posts a analizar (default 10).

        Returns:
            AccountHealthResult con el análisis completo.
        """
        # Validar datos de entrada
        if follower_count <= 0:
            logger.warning("Follower count is 0 or negative, using 1 to avoid division by zero")
            follower_count = 1

        # Limitar posts a analizar
        posts = recent_posts[:posts_to_analyze]
        posts_analyzed = len(posts)

        # Verificar si hay suficientes datos
        if posts_analyzed < self.MIN_POSTS_FOR_ANALYSIS:
            return self._create_insufficient_data_result(
                posts_analyzed=posts_analyzed,
                follower_count=follower_count
            )

        # Extraer views de cada post
        views_list = self._extract_views(posts)

        if not views_list:
            return self._create_insufficient_data_result(
                posts_analyzed=0,
                follower_count=follower_count
            )

        # Calcular estadísticas
        avg_views = float(np.mean(views_list))
        std_views = float(np.std(views_list))

        # Calcular ratio views/seguidores
        views_to_followers_ratio = avg_views / follower_count

        # Determinar nivel de autoridad
        authority_level = self._determine_authority_level(views_to_followers_ratio)

        # Determinar estado de salud
        health_status = self._determine_health_status(authority_level)

        # Obtener factor de penalización
        penalty_factor = self._get_penalty_factor(authority_level)

        # Obtener mensajes para el usuario
        messages = self.MESSAGES.get(authority_level, self.MESSAGES[AuthorityLevel.NORMAL])

        return AccountHealthResult(
            health_status=health_status,
            authority_level=authority_level,
            avg_views=avg_views,
            std_views=std_views,
            follower_count=follower_count,
            views_to_followers_ratio=views_to_followers_ratio,
            prediction_penalty_factor=penalty_factor,
            posts_analyzed=posts_analyzed,
            min_posts_required=self.MIN_POSTS_FOR_ANALYSIS,
            warning_message=messages.get("warning"),
            recommendation=messages.get("recommendation")
        )

    def _extract_views(self, posts: List[Dict[str, Any]]) -> List[int]:
        """Extrae las views de una lista de posts."""
        views = []
        for post in posts:
            # Intentar diferentes nombres de campo para views
            view_count = (
                post.get("views_count") or
                post.get("view_count") or
                post.get("views") or
                post.get("play_count") or
                0
            )
            if view_count and view_count > 0:
                views.append(int(view_count))
        return views

    def _determine_authority_level(self, ratio: float) -> AuthorityLevel:
        """Determina el nivel de autoridad basándose en el ratio views/seguidores."""
        if ratio >= self.THRESHOLD_HIGH_AUTHORITY:
            return AuthorityLevel.HIGH
        elif ratio >= self.THRESHOLD_NORMAL_AUTHORITY:
            return AuthorityLevel.NORMAL
        elif ratio >= self.THRESHOLD_LOW_AUTHORITY:
            return AuthorityLevel.LOW
        else:
            return AuthorityLevel.CRITICAL

    def _determine_health_status(self, authority: AuthorityLevel) -> AccountHealthStatus:
        """Convierte el nivel de autoridad en estado de salud."""
        mapping = {
            AuthorityLevel.HIGH: AccountHealthStatus.HEALTHY,
            AuthorityLevel.NORMAL: AccountHealthStatus.HEALTHY,
            AuthorityLevel.LOW: AccountHealthStatus.LOW_AUTHORITY,
            AuthorityLevel.CRITICAL: AccountHealthStatus.POSSIBLE_SHADOWBAN,
        }
        return mapping.get(authority, AccountHealthStatus.HEALTHY)

    def _get_penalty_factor(self, authority: AuthorityLevel) -> float:
        """Obtiene el factor de penalización según el nivel de autoridad."""
        factors = {
            AuthorityLevel.HIGH: self.PENALTY_FACTOR_HIGH,
            AuthorityLevel.NORMAL: self.PENALTY_FACTOR_NORMAL,
            AuthorityLevel.LOW: self.PENALTY_FACTOR_LOW,
            AuthorityLevel.CRITICAL: self.PENALTY_FACTOR_CRITICAL,
        }
        return factors.get(authority, 1.0)

    def _create_insufficient_data_result(
        self,
        posts_analyzed: int,
        follower_count: int
    ) -> AccountHealthResult:
        """Crea un resultado para cuando no hay suficientes datos."""
        return AccountHealthResult(
            health_status=AccountHealthStatus.INSUFFICIENT_DATA,
            authority_level=AuthorityLevel.NORMAL,  # Asumimos normal por defecto
            avg_views=0.0,
            std_views=0.0,
            follower_count=follower_count,
            views_to_followers_ratio=0.0,
            prediction_penalty_factor=1.0,  # Sin penalización si no hay datos
            posts_analyzed=posts_analyzed,
            min_posts_required=self.MIN_POSTS_FOR_ANALYSIS,
            warning_message=(
                f"No hay suficientes datos para evaluar la salud de tu cuenta. "
                f"Se necesitan al menos {self.MIN_POSTS_FOR_ANALYSIS} posts recientes, "
                f"pero solo se encontraron {posts_analyzed}."
            ),
            recommendation=(
                "Las predicciones se aplicarán sin calibración de cuenta. "
                "Para obtener predicciones más precisas, publica más contenido "
                "y vuelve a analizar cuando tengas al menos 5-10 posts."
            )
        )

    def calibrate_prediction(
        self,
        original_score: float,
        original_raw: float,
        account_health: AccountHealthResult,
        prediction_explanation: str = ""
    ) -> CalibratedPrediction:
        """
        Calibra una predicción del modelo XGBoost según la salud de la cuenta.

        Args:
            original_score: RPI score original (log-transformed).
            original_raw: RPI raw original (views predichas).
            account_health: Resultado del análisis de salud.
            prediction_explanation: Explicación original del modelo (opcional).

        Returns:
            CalibratedPrediction con valores ajustados y mensaje combinado.
        """
        penalty_factor = account_health.prediction_penalty_factor

        # Aplicar penalización a las predicciones
        # El score log-transformed se ajusta logarítmicamente
        # El raw se ajusta directamente con el factor
        calibrated_raw = original_raw * penalty_factor

        # Para el score, aplicamos el log del factor de penalización
        # log(raw * factor) = log(raw) + log(factor)
        if penalty_factor > 0:
            calibrated_score = original_score + np.log(penalty_factor)
        else:
            calibrated_score = original_score - 2.0  # Penalización máxima

        # Construir mensaje combinado
        combined_message = self._build_combined_message(
            account_health=account_health,
            original_raw=original_raw,
            calibrated_raw=calibrated_raw,
            prediction_explanation=prediction_explanation
        )

        return CalibratedPrediction(
            original_rpi_score=original_score,
            original_rpi_raw=original_raw,
            calibrated_rpi_score=calibrated_score,
            calibrated_rpi_raw=calibrated_raw,
            penalty_factor=penalty_factor,
            account_health=account_health,
            combined_message=combined_message
        )

    def _build_combined_message(
        self,
        account_health: AccountHealthResult,
        original_raw: float,
        calibrated_raw: float,
        prediction_explanation: str
    ) -> str:
        """Construye un mensaje combinado para el usuario."""
        parts = []

        # Añadir explicación original si existe
        if prediction_explanation:
            parts.append(f"**Análisis del contenido:** {prediction_explanation}")

        # Añadir información de predicción
        if account_health.prediction_penalty_factor < 1.0:
            parts.append(
                f"**Predicción ajustada:** Este contenido tiene potencial para "
                f"~{int(original_raw):,} views en condiciones óptimas, pero dado el estado "
                f"actual de tu cuenta, estimamos ~{int(calibrated_raw):,} views reales."
            )
        else:
            parts.append(
                f"**Predicción:** Este contenido tiene potencial para "
                f"~{int(original_raw):,} views."
            )

        # Añadir advertencia si aplica
        if account_health.warning_message:
            parts.append(f"**Advertencia:** {account_health.warning_message}")

        # Añadir recomendación
        if account_health.recommendation:
            parts.append(f"**Recomendación:** {account_health.recommendation}")

        return "\n\n".join(parts)

    def evaluate_from_scraped_posts(
        self,
        business_id: str,
        follower_count: int,
        db_session = None
    ) -> AccountHealthResult:
        """
        Evalúa la salud de una cuenta usando posts scrapeados de la base de datos.

        Esta es una función de conveniencia que obtiene los posts directamente
        de la base de datos si se proporciona una sesión.

        Args:
            business_id: ID del negocio a evaluar.
            follower_count: Número de seguidores.
            db_session: Sesión de base de datos SQLAlchemy (opcional).

        Returns:
            AccountHealthResult con el análisis.
        """
        if db_session is None:
            logger.warning("No database session provided, returning insufficient data result")
            return self._create_insufficient_data_result(
                posts_analyzed=0,
                follower_count=follower_count
            )

        # Esta función debería integrarse con el modelo ScrapedPost
        # Por ahora, retornamos datos insuficientes
        # TODO: Implementar query real cuando se tenga acceso a los posts del usuario
        logger.info(f"Evaluating account health for business {business_id}")
        return self._create_insufficient_data_result(
            posts_analyzed=0,
            follower_count=follower_count
        )

    def get_health_summary(self, health: AccountHealthResult) -> str:
        """
        Genera un resumen legible del estado de salud.

        Args:
            health: Resultado del análisis de salud.

        Returns:
            String con resumen formateado.
        """
        status_emoji = {
            AccountHealthStatus.HEALTHY: "✅",
            AccountHealthStatus.LOW_AUTHORITY: "⚠️",
            AccountHealthStatus.POSSIBLE_SHADOWBAN: "🚨",
            AccountHealthStatus.INSUFFICIENT_DATA: "❓",
        }

        emoji = status_emoji.get(health.health_status, "❓")
        ratio_pct = health.views_to_followers_ratio * 100

        summary = f"""
{emoji} Estado de cuenta: {health.health_status.value.upper()}

📊 Métricas:
   • Promedio de views: {health.avg_views:,.0f}
   • Desviación estándar: {health.std_views:,.0f}
   • Seguidores: {health.follower_count:,}
   • Ratio views/seguidores: {ratio_pct:.1f}%

🎯 Nivel de autoridad: {health.authority_level.value.upper()}
📉 Factor de calibración: {health.prediction_penalty_factor}x

📝 Posts analizados: {health.posts_analyzed}/{health.min_posts_required} (mínimo requerido)
"""

        if health.warning_message:
            summary += f"\n⚠️ Advertencia:\n{health.warning_message}\n"

        if health.recommendation:
            summary += f"\n💡 Recomendación:\n{health.recommendation}\n"

        return summary.strip()


# =============================================================================
# SINGLETON Y FUNCIONES DE UTILIDAD
# =============================================================================

_account_health_instance: Optional[AccountHealthScoring] = None


def get_account_health_scoring() -> AccountHealthScoring:
    """
    Obtiene la instancia singleton del AccountHealthScoring.

    Returns:
        Instancia del motor de scoring de salud.
    """
    global _account_health_instance
    if _account_health_instance is None:
        _account_health_instance = AccountHealthScoring()
    return _account_health_instance


def reset_account_health_scoring() -> None:
    """Reinicia la instancia singleton (útil para tests)."""
    global _account_health_instance
    _account_health_instance = None
