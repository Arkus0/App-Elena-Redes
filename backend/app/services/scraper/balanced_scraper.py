"""
Balanced Scraper - Fix Survivor Bias by collecting both Top and Flop posts

PROBLEMA ORIGINAL:
El scraper solo recolectaba Top Posts (ordenados por engagement descendente),
causando Survivor Bias: el modelo nunca aprende qué características causan fracasos.

SOLUCION:
Implementar "Balanced Sampling" que recolecta:
- Top N posts (high performers / viral)
- Bottom N posts (low performers / flops)

Esto permite al modelo XGBoost discriminar entre features de exito y fracaso.
"""

import logging
from enum import Enum
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class SamplingStrategy(str, Enum):
    """Estrategias de muestreo para la recoleccion de datos"""
    TOP_ONLY = "top_only"           # Solo top posts (comportamiento antiguo - CAUSA BIAS)
    BOTTOM_ONLY = "bottom_only"     # Solo flops (para debug)
    BALANCED = "balanced"           # Top + Bottom (RECOMENDADO)
    STRATIFIED = "stratified"       # Muestreo estratificado por percentiles


@dataclass
class SamplingConfig:
    """Configuracion de muestreo balanceado"""
    strategy: SamplingStrategy = SamplingStrategy.BALANCED
    top_n: int = 25              # Numero de top posts a recolectar
    bottom_n: int = 5            # Numero de flop posts a recolectar (minimo 5)
    min_posts_required: int = 20  # Minimo de posts para aplicar estrategia
    exclude_middle_percent: float = 0.6  # Excluir el 60% del medio para maximizar contraste

    def __post_init__(self):
        """Validar configuracion"""
        if self.bottom_n < 5:
            logger.warning(f"bottom_n={self.bottom_n} es muy bajo. Usando minimo de 5.")
            self.bottom_n = 5


class BalancedScraper:
    """
    Scraper con estrategia de muestreo balanceado para eliminar Survivor Bias.

    USO:
    ```python
    scraper = BalancedScraper(strategy=SamplingStrategy.BALANCED)
    posts = scraper.apply_balanced_sampling(raw_posts, follower_count=10000)
    ```
    """

    def __init__(
        self,
        strategy: SamplingStrategy = SamplingStrategy.BALANCED,
        config: Optional[SamplingConfig] = None
    ):
        self.strategy = strategy
        self.config = config or SamplingConfig(strategy=strategy)
        logger.info(f"BalancedScraper initialized with strategy: {strategy.value}")

    def apply_balanced_sampling(
        self,
        posts: List[Dict[str, Any]],
        follower_count: int = 1,
        sort_key: str = "engagement_score"
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Aplica muestreo balanceado a una lista de posts.

        Args:
            posts: Lista de posts scrapeados (sin ordenar)
            follower_count: Numero de seguidores para calcular engagement rate
            sort_key: Campo por el cual ordenar (engagement_score, views, likes, etc.)

        Returns:
            Tuple[List[Dict], Dict]: (posts_balanceados, metadata_del_sampling)
        """
        if not posts:
            return [], {"error": "No posts provided", "strategy": self.strategy.value}

        n_total = len(posts)
        metadata = {
            "strategy": self.strategy.value,
            "total_scraped": n_total,
            "follower_count": follower_count,
            "sort_key": sort_key,
        }

        # Calcular engagement score si no existe
        posts = self._ensure_engagement_scores(posts, follower_count)

        # Ordenar por engagement (descendente)
        sorted_posts = sorted(posts, key=lambda x: x.get(sort_key, 0), reverse=True)

        # Aplicar estrategia de muestreo
        if self.strategy == SamplingStrategy.TOP_ONLY:
            result = self._sample_top_only(sorted_posts)
        elif self.strategy == SamplingStrategy.BOTTOM_ONLY:
            result = self._sample_bottom_only(sorted_posts)
        elif self.strategy == SamplingStrategy.BALANCED:
            result = self._sample_balanced(sorted_posts)
        elif self.strategy == SamplingStrategy.STRATIFIED:
            result = self._sample_stratified(sorted_posts)
        else:
            result = sorted_posts[:self.config.top_n]

        # Etiquetar posts segun su posicion en el ranking original
        result = self._label_post_performance(result, sorted_posts)

        # Actualizar metadata
        metadata.update({
            "top_posts_count": sum(1 for p in result if p.get("_performance_tier") == "top"),
            "bottom_posts_count": sum(1 for p in result if p.get("_performance_tier") == "bottom"),
            "middle_posts_count": sum(1 for p in result if p.get("_performance_tier") == "middle"),
            "final_count": len(result),
            "sampling_ratio": len(result) / n_total if n_total > 0 else 0,
        })

        logger.info(
            f"Balanced sampling applied: {metadata['top_posts_count']} top, "
            f"{metadata['bottom_posts_count']} bottom, {metadata['final_count']} total"
        )

        return result, metadata

    def _ensure_engagement_scores(
        self,
        posts: List[Dict[str, Any]],
        follower_count: int
    ) -> List[Dict[str, Any]]:
        """Asegura que todos los posts tengan engagement_score calculado"""
        for post in posts:
            if "engagement_score" not in post or post["engagement_score"] is None:
                post["engagement_score"] = self._calculate_engagement_score(post, follower_count)
        return posts

    def _calculate_engagement_score(self, post: Dict[str, Any], follower_count: int) -> float:
        """
        Calcula engagement score ponderado.
        Pesos basados en intencionalidad del usuario:
        - Saves (5x): Mayor intencion de volver
        - Shares (4x): Mayor alcance organico
        - Comments (3x): Engagement activo
        - Likes (1x): Engagement pasivo
        """
        likes = post.get("likes", post.get("likes_count", 0)) or 0
        comments = post.get("comments", post.get("comments_count", 0)) or 0
        saves = post.get("saves", post.get("saves_count", 0)) or 0
        shares = post.get("shares", post.get("shares_count", 0)) or 0
        views = post.get("video_views", post.get("views_count", post.get("plays", 0))) or 0

        weighted_engagement = (
            likes +
            (comments * 3) +
            (saves * 5) +
            (shares * 4)
        )

        # Normalizar por views si disponible, sino por followers
        if views > 0:
            engagement_rate = (weighted_engagement / views) * 100
            return min(engagement_rate * 10, 100)
        elif follower_count > 0:
            engagement_rate = (weighted_engagement / follower_count) * 100
            return min(engagement_rate * 10, 100)

        # Fallback: log scaling
        import math
        if weighted_engagement > 0:
            return min(math.log10(weighted_engagement + 1) * 20, 100)
        return 0.0

    def _sample_top_only(self, sorted_posts: List[Dict]) -> List[Dict]:
        """
        Muestreo solo de top posts (comportamiento antiguo).
        ADVERTENCIA: Esto causa Survivor Bias!
        """
        logger.warning("Using TOP_ONLY strategy - this causes Survivor Bias!")
        return sorted_posts[:self.config.top_n]

    def _sample_bottom_only(self, sorted_posts: List[Dict]) -> List[Dict]:
        """Muestreo solo de flop posts (para debug/testing)"""
        return sorted_posts[-self.config.bottom_n:]

    def _sample_balanced(self, sorted_posts: List[Dict]) -> List[Dict]:
        """
        Muestreo balanceado: Top N + Bottom N
        ESTRATEGIA RECOMENDADA para eliminar Survivor Bias
        """
        n_total = len(sorted_posts)

        if n_total < self.config.min_posts_required:
            logger.warning(
                f"Not enough posts ({n_total}) for balanced sampling. "
                f"Returning all posts."
            )
            return sorted_posts

        # Top posts (exitos)
        top_posts = sorted_posts[:self.config.top_n]

        # Bottom posts (fracasos) - asegurar que no se solapan con top
        # Tomar los ultimos N posts que NO estan en top
        available_for_bottom = n_total - self.config.top_n
        actual_bottom_n = min(self.config.bottom_n, available_for_bottom)

        if actual_bottom_n < self.config.bottom_n:
            logger.warning(
                f"Only {actual_bottom_n} posts available for bottom sampling "
                f"(requested {self.config.bottom_n})"
            )

        bottom_posts = sorted_posts[-actual_bottom_n:] if actual_bottom_n > 0 else []

        # Combinar y mezclar para evitar bias de orden
        combined = top_posts + bottom_posts

        logger.info(
            f"Balanced sampling: {len(top_posts)} top + {len(bottom_posts)} bottom = "
            f"{len(combined)} total"
        )

        return combined

    def _sample_stratified(self, sorted_posts: List[Dict]) -> List[Dict]:
        """
        Muestreo estratificado por percentiles.
        Toma muestras de diferentes niveles de performance.
        """
        n_total = len(sorted_posts)

        if n_total < 10:
            return sorted_posts

        # Dividir en 5 estratos (quintiles)
        strata_size = n_total // 5
        samples_per_stratum = max(self.config.top_n // 5, 2)

        result = []
        for i in range(5):
            start = i * strata_size
            end = start + strata_size if i < 4 else n_total
            stratum = sorted_posts[start:end]

            # Mas muestras de extremos (Q1 y Q5), menos del medio
            if i == 0:  # Top quintile
                n_samples = samples_per_stratum * 2
            elif i == 4:  # Bottom quintile
                n_samples = samples_per_stratum * 2
            else:  # Middle quintiles
                n_samples = samples_per_stratum // 2

            result.extend(stratum[:n_samples])

        return result

    def _label_post_performance(
        self,
        sampled_posts: List[Dict],
        all_sorted_posts: List[Dict]
    ) -> List[Dict]:
        """
        Etiqueta cada post con su tier de performance.
        Esto es metadata auxiliar para debugging/analisis.
        """
        n_total = len(all_sorted_posts)
        if n_total == 0:
            return sampled_posts

        # Calcular umbrales de percentiles
        top_threshold_idx = int(n_total * 0.2)  # Top 20%
        bottom_threshold_idx = int(n_total * 0.8)  # Bottom 20%

        # Crear set de IDs para busqueda rapida
        all_ids = {
            p.get("platform_id", p.get("id", i)): i
            for i, p in enumerate(all_sorted_posts)
        }

        for post in sampled_posts:
            post_id = post.get("platform_id", post.get("id"))
            rank = all_ids.get(post_id, n_total // 2)

            if rank < top_threshold_idx:
                post["_performance_tier"] = "top"
                post["_performance_percentile"] = round((1 - rank / n_total) * 100, 1)
            elif rank >= bottom_threshold_idx:
                post["_performance_tier"] = "bottom"
                post["_performance_percentile"] = round((1 - rank / n_total) * 100, 1)
            else:
                post["_performance_tier"] = "middle"
                post["_performance_percentile"] = round((1 - rank / n_total) * 100, 1)

        return sampled_posts

    def get_flop_posts(
        self,
        posts: List[Dict[str, Any]],
        n: int = 5,
        follower_count: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Metodo de conveniencia para obtener solo los N peores posts.

        Args:
            posts: Lista de posts
            n: Numero de flops a retornar
            follower_count: Seguidores para calcular engagement

        Returns:
            Lista de los N posts con peor performance
        """
        posts = self._ensure_engagement_scores(posts, follower_count)
        sorted_posts = sorted(posts, key=lambda x: x.get("engagement_score", 0))
        flops = sorted_posts[:n]

        # Etiquetar como flops
        for post in flops:
            post["_performance_tier"] = "bottom"
            post["_is_flop"] = True

        logger.info(f"Extracted {len(flops)} flop posts (lowest engagement)")
        return flops

    def get_top_posts(
        self,
        posts: List[Dict[str, Any]],
        n: int = 25,
        follower_count: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Metodo de conveniencia para obtener solo los N mejores posts.

        Args:
            posts: Lista de posts
            n: Numero de top posts a retornar
            follower_count: Seguidores para calcular engagement

        Returns:
            Lista de los N posts con mejor performance
        """
        posts = self._ensure_engagement_scores(posts, follower_count)
        sorted_posts = sorted(posts, key=lambda x: x.get("engagement_score", 0), reverse=True)
        tops = sorted_posts[:n]

        # Etiquetar como tops
        for post in tops:
            post["_performance_tier"] = "top"
            post["_is_viral"] = True

        logger.info(f"Extracted {len(tops)} top posts (highest engagement)")
        return tops


# Factory function para crear scraper con configuracion por defecto
def create_balanced_scraper(
    top_n: int = 25,
    bottom_n: int = 5,
    strategy: SamplingStrategy = SamplingStrategy.BALANCED
) -> BalancedScraper:
    """
    Factory function para crear un BalancedScraper con configuracion comun.

    Args:
        top_n: Numero de top posts a recolectar (default: 25)
        bottom_n: Numero de flop posts a recolectar (default: 5, minimo recomendado)
        strategy: Estrategia de muestreo (default: BALANCED)

    Returns:
        BalancedScraper configurado

    Ejemplo:
        scraper = create_balanced_scraper(top_n=20, bottom_n=10)
        posts, meta = scraper.apply_balanced_sampling(raw_posts, follower_count=5000)
    """
    config = SamplingConfig(
        strategy=strategy,
        top_n=top_n,
        bottom_n=bottom_n
    )
    return BalancedScraper(strategy=strategy, config=config)
