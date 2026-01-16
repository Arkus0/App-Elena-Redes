"""
Metric Normalization Service - Relative Performance Index (RPI)

Este módulo implementa la normalización de métricas de engagement para eliminar
el sesgo de tamaño de cuenta. En lugar de predecir métricas absolutas (likes, views),
el modelo predice el "Relative Performance Index" (RPI).

PROBLEMA RESUELTO:
- Las métricas absolutas introducen sesgo masivo: cuentas grandes parecen
  tener mejor contenido solo por tener más seguidores.
- Solución: Normalizar cada post contra el baseline histórico de ESA cuenta.

LÓGICA MATEMÁTICA:
1. Rolling Baseline: Mediana de engagement de últimos N posts del autor
2. RPI = Engagement_actual / Baseline_engagement
3. Log-Transform: np.log1p(RPI) para reducir outliers virales
4. Cold Start: Si cuenta tiene < min_posts, usar mediana del nicho

EJEMPLO:
- Cuenta con mediana de 1000 likes
- Post actual tiene 1500 likes
- RPI = 1500/1000 = 1.5
- RPI_log = log1p(1.5) ≈ 0.916
- Interpretación: Este post tuvo 50% más engagement que lo habitual

Autor: BrandPulse AI
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class NormalizationStrategy(str, Enum):
    """Estrategias de normalización disponibles"""
    MEDIAN = "median"           # Mediana (robusta a outliers)
    MEAN = "mean"               # Media (sensible a outliers)
    TRIMMED_MEAN = "trimmed_mean"  # Media recortada (20% extremos)
    WEIGHTED_MEDIAN = "weighted_median"  # Mediana ponderada por recencia


@dataclass
class NormalizationConfig:
    """
    Configuración para el cálculo del Relative Performance Index (RPI).

    Attributes:
        window_size: Número de posts históricos para calcular baseline (default: 20)
        min_posts_for_baseline: Mínimo de posts para usar baseline personal (default: 5)
        strategy: Estrategia de agregación para el baseline
        apply_log_transform: Si aplicar np.log1p al resultado
        clip_rpi_max: Valor máximo de RPI antes de log (evita explosión)
        clip_rpi_min: Valor mínimo de RPI (evita división por 0)
        engagement_weights: Pesos para calcular engagement ponderado
    """
    window_size: int = 20
    min_posts_for_baseline: int = 5
    strategy: NormalizationStrategy = NormalizationStrategy.MEDIAN
    apply_log_transform: bool = True
    clip_rpi_max: float = 100.0     # Max 100x engagement normal
    clip_rpi_min: float = 0.01      # Min 1% engagement normal

    # Pesos por defecto para engagement ponderado
    engagement_weights: Dict[str, float] = field(default_factory=lambda: {
        "likes": 1.0,
        "comments": 3.0,
        "saves": 5.0,
        "shares": 4.0,
        "views": 0.1,  # Views tienen peso bajo (fáciles de obtener)
    })


@dataclass
class NicheBaseline:
    """Baseline agregado por nicho/categoría para cold start"""
    niche: str
    median_engagement: float
    mean_engagement: float
    std_engagement: float
    sample_size: int
    percentile_25: float
    percentile_75: float


class MetricNormalizer:
    """
    Normalizador de métricas de engagement usando Relative Performance Index (RPI).

    El RPI transforma métricas absolutas en rendimiento relativo al histórico
    del autor, eliminando el sesgo de tamaño de cuenta.

    Uso típico:
        normalizer = MetricNormalizer()

        # Normalizar un DataFrame completo
        df_normalized = normalizer.normalize_dataframe(df, author_col="author_id")

        # El modelo XGBoost usa df_normalized["rpi_score"] como target

    Ejemplo numérico:
        - Autor con mediana histórica de 1000 likes
        - Post actual: 1500 likes
        - RPI = 1500 / 1000 = 1.5
        - RPI_log = log1p(1.5) ≈ 0.916
    """

    def __init__(self, config: Optional[NormalizationConfig] = None):
        """
        Inicializa el normalizador.

        Args:
            config: Configuración de normalización. Usa defaults si None.
        """
        self.config = config or NormalizationConfig()
        self._niche_baselines: Dict[str, NicheBaseline] = {}
        self._global_baseline: Optional[float] = None

        logger.info(
            f"MetricNormalizer initialized: window={self.config.window_size}, "
            f"min_posts={self.config.min_posts_for_baseline}, "
            f"strategy={self.config.strategy.value}"
        )

    def calculate_weighted_engagement(
        self,
        post: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calcula el engagement ponderado de un post.

        Args:
            post: Diccionario con métricas del post
            weights: Pesos personalizados (usa config si None)

        Returns:
            Engagement ponderado como float

        Ejemplo:
            >>> post = {"likes": 500, "comments": 50, "saves": 100}
            >>> engagement = normalizer.calculate_weighted_engagement(post)
            >>> # 500*1 + 50*3 + 100*5 = 500 + 150 + 500 = 1150
        """
        weights = weights or self.config.engagement_weights

        # Mapeo de campos (soportar múltiples formatos)
        field_mappings = {
            "likes": ["likes_count", "likes", "likeCount"],
            "comments": ["comments_count", "comments", "commentCount"],
            "saves": ["saves_count", "saves", "saveCount"],
            "shares": ["shares_count", "shares", "shareCount"],
            "views": ["views_count", "views", "viewCount", "play_count"],
        }

        total = 0.0
        for metric, fields in field_mappings.items():
            value = 0
            for field_name in fields:
                if field_name in post and post[field_name] is not None:
                    value = float(post[field_name])
                    break
            total += value * weights.get(metric, 0.0)

        return total

    def _calculate_baseline_for_author(
        self,
        df: pd.DataFrame,
        current_idx: int,
        author_col: str,
        engagement_col: str
    ) -> Optional[float]:
        """
        Calcula el baseline rolling para un autor específico.

        IMPORTANTE: Excluye el post actual del cálculo para evitar data leakage.

        Args:
            df: DataFrame ordenado por fecha
            current_idx: Índice del post actual
            author_col: Columna con ID del autor
            engagement_col: Columna con engagement calculado

        Returns:
            Baseline (mediana) o None si datos insuficientes
        """
        current_author = df.loc[current_idx, author_col]

        # Filtrar posts anteriores del mismo autor
        mask = (
            (df[author_col] == current_author) &
            (df.index < current_idx)
        )
        author_history = df.loc[mask, engagement_col].tail(self.config.window_size)

        if len(author_history) < self.config.min_posts_for_baseline:
            return None

        # Calcular baseline según estrategia
        if self.config.strategy == NormalizationStrategy.MEDIAN:
            return author_history.median()
        elif self.config.strategy == NormalizationStrategy.MEAN:
            return author_history.mean()
        elif self.config.strategy == NormalizationStrategy.TRIMMED_MEAN:
            # Recortar 20% extremos
            q_low = author_history.quantile(0.1)
            q_high = author_history.quantile(0.9)
            trimmed = author_history[(author_history >= q_low) & (author_history <= q_high)]
            return trimmed.mean() if len(trimmed) > 0 else author_history.median()
        elif self.config.strategy == NormalizationStrategy.WEIGHTED_MEDIAN:
            # Posts más recientes tienen más peso
            n = len(author_history)
            weights = np.linspace(0.5, 1.0, n)  # Peso crece linealmente
            sorted_vals = author_history.sort_values()
            cumsum = np.cumsum(weights[np.argsort(author_history.values)])
            median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
            return sorted_vals.iloc[min(median_idx, len(sorted_vals) - 1)]

        return author_history.median()

    def fit_niche_baselines(
        self,
        df: pd.DataFrame,
        niche_col: str = "niche",
        engagement_col: str = "weighted_engagement"
    ) -> Dict[str, NicheBaseline]:
        """
        Calcula baselines agregados por nicho/categoría para cold start.

        Args:
            df: DataFrame con datos de todos los posts
            niche_col: Columna con el nicho/categoría
            engagement_col: Columna con engagement

        Returns:
            Diccionario de NicheBaseline por nicho
        """
        logger.info(f"Fitting niche baselines from {len(df)} posts")

        for niche, group in df.groupby(niche_col):
            engagement = group[engagement_col].dropna()

            if len(engagement) < 10:
                logger.warning(f"Nicho '{niche}' tiene pocos datos ({len(engagement)} posts)")
                continue

            self._niche_baselines[niche] = NicheBaseline(
                niche=niche,
                median_engagement=engagement.median(),
                mean_engagement=engagement.mean(),
                std_engagement=engagement.std(),
                sample_size=len(engagement),
                percentile_25=engagement.quantile(0.25),
                percentile_75=engagement.quantile(0.75),
            )

            logger.debug(
                f"Nicho '{niche}': median={self._niche_baselines[niche].median_engagement:.2f}, "
                f"n={len(engagement)}"
            )

        # Calcular baseline global como fallback final
        all_engagement = df[engagement_col].dropna()
        self._global_baseline = all_engagement.median()

        logger.info(
            f"Fitted {len(self._niche_baselines)} niche baselines. "
            f"Global baseline: {self._global_baseline:.2f}"
        )

        return self._niche_baselines

    def get_fallback_baseline(
        self,
        niche: Optional[str] = None
    ) -> float:
        """
        Obtiene baseline de fallback para cold start.

        Jerarquía:
        1. Baseline del nicho específico
        2. Baseline global
        3. Valor default (100.0)

        Args:
            niche: Nicho/categoría del autor

        Returns:
            Baseline para usar como fallback
        """
        if niche and niche in self._niche_baselines:
            return self._niche_baselines[niche].median_engagement

        if self._global_baseline is not None:
            return self._global_baseline

        logger.warning("No baseline available, using default=100.0")
        return 100.0

    def calculate_rpi(
        self,
        current_engagement: float,
        baseline_engagement: float
    ) -> float:
        """
        Calcula el Relative Performance Index.

        RPI = current_engagement / baseline_engagement

        Args:
            current_engagement: Engagement del post actual
            baseline_engagement: Baseline histórico

        Returns:
            RPI (opcionalmente log-transformado)

        Ejemplo:
            >>> rpi = normalizer.calculate_rpi(1500, 1000)
            >>> # RPI = 1.5 (50% mejor que baseline)
            >>> # Con log: log1p(1.5) ≈ 0.916
        """
        # Evitar división por cero
        if baseline_engagement <= 0:
            baseline_engagement = 1.0

        rpi = current_engagement / baseline_engagement

        # Clipear valores extremos
        rpi = np.clip(rpi, self.config.clip_rpi_min, self.config.clip_rpi_max)

        # Aplicar log transform si configurado
        if self.config.apply_log_transform:
            rpi = np.log1p(rpi)

        return float(rpi)

    def normalize_dataframe(
        self,
        df: pd.DataFrame,
        author_col: str = "author_id",
        niche_col: Optional[str] = "niche",
        date_col: str = "posted_at",
        engagement_cols: Optional[List[str]] = None,
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        Normaliza un DataFrame completo calculando RPI para cada post.

        Este es el método principal que debe usarse antes de entrenar XGBoost.

        Args:
            df: DataFrame con posts
            author_col: Columna con ID del autor
            niche_col: Columna con nicho (para cold start)
            date_col: Columna con fecha (para ordenar cronológicamente)
            engagement_cols: Columnas de engagement a agregar
            inplace: Si modificar df original

        Returns:
            DataFrame con columnas adicionales:
            - weighted_engagement: Engagement ponderado
            - baseline_engagement: Baseline usado
            - rpi_raw: RPI sin transformar
            - rpi_score: RPI final (target para XGBoost)
            - baseline_source: "personal", "niche", o "global"
        """
        if not inplace:
            df = df.copy()

        logger.info(f"Normalizing {len(df)} posts with RPI calculation")

        # 1. Ordenar por fecha (cronológico)
        if date_col in df.columns:
            df = df.sort_values(date_col).reset_index(drop=True)

        # 2. Calcular engagement ponderado si no existe
        if "weighted_engagement" not in df.columns:
            df["weighted_engagement"] = df.apply(
                lambda row: self.calculate_weighted_engagement(row.to_dict()),
                axis=1
            )

        # 3. Fit niche baselines si no existen
        if not self._niche_baselines and niche_col and niche_col in df.columns:
            self.fit_niche_baselines(df, niche_col, "weighted_engagement")

        # 4. Calcular RPI para cada post
        baselines = []
        baseline_sources = []
        rpi_raw_values = []
        rpi_scores = []

        for idx in df.index:
            current_engagement = df.loc[idx, "weighted_engagement"]

            # Intentar calcular baseline personal
            baseline = self._calculate_baseline_for_author(
                df, idx, author_col, "weighted_engagement"
            )

            if baseline is not None:
                source = "personal"
            else:
                # Cold start: usar baseline de nicho o global
                niche = df.loc[idx, niche_col] if niche_col and niche_col in df.columns else None
                baseline = self.get_fallback_baseline(niche)
                source = "niche" if niche and niche in self._niche_baselines else "global"

            # Calcular RPI
            rpi_raw = current_engagement / max(baseline, 1.0)
            rpi_raw = np.clip(rpi_raw, self.config.clip_rpi_min, self.config.clip_rpi_max)

            if self.config.apply_log_transform:
                rpi_score = np.log1p(rpi_raw)
            else:
                rpi_score = rpi_raw

            baselines.append(baseline)
            baseline_sources.append(source)
            rpi_raw_values.append(rpi_raw)
            rpi_scores.append(rpi_score)

        # 5. Agregar columnas al DataFrame
        df["baseline_engagement"] = baselines
        df["baseline_source"] = baseline_sources
        df["rpi_raw"] = rpi_raw_values
        df["rpi_score"] = rpi_scores

        # 6. Log estadísticas
        source_counts = df["baseline_source"].value_counts()
        logger.info(
            f"RPI calculation complete. "
            f"Baseline sources: {source_counts.to_dict()}"
        )
        logger.info(
            f"RPI stats: mean={df['rpi_score'].mean():.3f}, "
            f"std={df['rpi_score'].std():.3f}, "
            f"min={df['rpi_score'].min():.3f}, "
            f"max={df['rpi_score'].max():.3f}"
        )

        return df

    def normalize_single_post(
        self,
        post: Dict[str, Any],
        author_history: List[Dict[str, Any]],
        niche: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Normaliza un único post (útil para predicción en tiempo real).

        Args:
            post: Diccionario con datos del post
            author_history: Lista de posts históricos del autor
            niche: Nicho del autor para fallback

        Returns:
            Diccionario con RPI y metadatos
        """
        current_engagement = self.calculate_weighted_engagement(post)

        # Calcular baseline del histórico
        if len(author_history) >= self.config.min_posts_for_baseline:
            history_engagements = [
                self.calculate_weighted_engagement(p)
                for p in author_history[-self.config.window_size:]
            ]
            baseline = float(np.median(history_engagements))
            source = "personal"
        else:
            baseline = self.get_fallback_baseline(niche)
            source = "niche" if niche and niche in self._niche_baselines else "global"

        rpi_raw = current_engagement / max(baseline, 1.0)
        rpi_raw = np.clip(rpi_raw, self.config.clip_rpi_min, self.config.clip_rpi_max)

        if self.config.apply_log_transform:
            rpi_score = np.log1p(rpi_raw)
        else:
            rpi_score = rpi_raw

        return {
            "weighted_engagement": current_engagement,
            "baseline_engagement": baseline,
            "baseline_source": source,
            "rpi_raw": rpi_raw,
            "rpi_score": rpi_score,
        }

    def inverse_transform(
        self,
        rpi_score: float,
        baseline_engagement: float
    ) -> float:
        """
        Transforma RPI de vuelta a engagement absoluto (para interpretación).

        Args:
            rpi_score: RPI score (potencialmente log-transformado)
            baseline_engagement: Baseline del autor

        Returns:
            Engagement absoluto estimado
        """
        if self.config.apply_log_transform:
            rpi_raw = np.expm1(rpi_score)
        else:
            rpi_raw = rpi_score

        return rpi_raw * baseline_engagement

    def get_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Obtiene estadísticas del proceso de normalización.

        Args:
            df: DataFrame normalizado

        Returns:
            Diccionario con estadísticas
        """
        return {
            "total_posts": len(df),
            "baseline_sources": df["baseline_source"].value_counts().to_dict(),
            "rpi_score": {
                "mean": df["rpi_score"].mean(),
                "std": df["rpi_score"].std(),
                "min": df["rpi_score"].min(),
                "max": df["rpi_score"].max(),
                "median": df["rpi_score"].median(),
                "q25": df["rpi_score"].quantile(0.25),
                "q75": df["rpi_score"].quantile(0.75),
            },
            "cold_start_rate": (
                df["baseline_source"].isin(["niche", "global"]).sum() / len(df)
            ),
            "niche_baselines": {
                k: v.median_engagement
                for k, v in self._niche_baselines.items()
            },
        }


# ============================================================================
# Funciones de utilidad para integración rápida
# ============================================================================

def normalize_for_training(
    df: pd.DataFrame,
    author_col: str = "author_id",
    niche_col: str = "niche",
    date_col: str = "posted_at",
    config: Optional[NormalizationConfig] = None
) -> Tuple[pd.DataFrame, MetricNormalizer]:
    """
    Función de conveniencia para preparar datos de entrenamiento.

    Uso:
        df_train, normalizer = normalize_for_training(df_raw)

        # Entrenar XGBoost con rpi_score como target
        X = df_train[feature_cols]
        y = df_train["rpi_score"]  # <-- Target normalizado
        model.fit(X, y)

    Args:
        df: DataFrame con posts crudos
        author_col: Columna de autor
        niche_col: Columna de nicho
        date_col: Columna de fecha
        config: Configuración personalizada

    Returns:
        Tuple de (DataFrame normalizado, Normalizer instance)
    """
    normalizer = MetricNormalizer(config)
    df_normalized = normalizer.normalize_dataframe(
        df,
        author_col=author_col,
        niche_col=niche_col,
        date_col=date_col
    )

    logger.info(
        f"Data ready for training. "
        f"Use 'rpi_score' column as target variable. "
        f"Cold start rate: {normalizer.get_stats(df_normalized)['cold_start_rate']:.1%}"
    )

    return df_normalized, normalizer


def create_rpi_from_raw_metrics(
    likes: int,
    comments: int,
    saves: int,
    shares: int,
    author_baseline: float,
    apply_log: bool = True
) -> float:
    """
    Calcula RPI directamente desde métricas crudas.

    Función de utilidad para cálculos rápidos sin instanciar clase.

    Args:
        likes, comments, saves, shares: Métricas del post
        author_baseline: Baseline histórico del autor
        apply_log: Si aplicar log transform

    Returns:
        RPI score

    Ejemplo:
        >>> rpi = create_rpi_from_raw_metrics(
        ...     likes=1500, comments=50, saves=100, shares=20,
        ...     author_baseline=1000
        ... )
        >>> print(f"RPI: {rpi:.3f}")  # ~0.916 si apply_log=True
    """
    # Pesos por defecto
    weighted = likes * 1.0 + comments * 3.0 + saves * 5.0 + shares * 4.0

    rpi_raw = weighted / max(author_baseline, 1.0)
    rpi_raw = np.clip(rpi_raw, 0.01, 100.0)

    if apply_log:
        return float(np.log1p(rpi_raw))
    return float(rpi_raw)
