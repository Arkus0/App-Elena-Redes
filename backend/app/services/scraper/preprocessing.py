"""
Data Preprocessing Module - Labeling and Dataset Balancing

Este modulo implementa:
1. Calculo de engagement_ratio (Engagement / Followers)
2. Etiquetado binario is_viral (Top 20% = 1, Bottom 20% = 0)
3. Balanceo del dataset para entrenamiento (minimo 30% casos negativos)

OBJETIVO:
Preparar datos para el modelo XGBoost que pueda discriminar entre
caracteristicas de exito y fracaso.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LabelingConfig:
    """Configuracion para etiquetado de posts"""
    top_percentile: float = 0.20       # Top 20% son "virales" (is_viral=1)
    bottom_percentile: float = 0.20    # Bottom 20% son "flops" (is_viral=0)
    min_engagement_threshold: float = 0.0  # Engagement minimo para considerar
    exclude_zero_engagement: bool = True    # Excluir posts sin engagement


@dataclass
class BalancingConfig:
    """Configuracion para balanceo de dataset"""
    min_negative_ratio: float = 0.30   # Minimo 30% de casos negativos
    max_negative_ratio: float = 0.50   # Maximo 50% para balance perfecto
    strategy: str = "hybrid"           # undersampling, oversampling, hybrid, smote
    random_state: int = 42


def calculate_engagement_ratio(
    post: Dict[str, Any],
    follower_count: int,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Calcula el ratio de engagement normalizado por followers.

    Formula: (weighted_engagement / followers) * 100

    Args:
        post: Diccionario con metricas del post
        follower_count: Numero de seguidores de la cuenta
        weights: Pesos personalizados para cada metrica

    Returns:
        Engagement ratio como porcentaje (0-100+)

    Ejemplo:
        >>> post = {"likes": 500, "comments": 50, "saves": 100}
        >>> ratio = calculate_engagement_ratio(post, follower_count=10000)
        >>> print(f"Engagement ratio: {ratio:.2f}%")
    """
    # Pesos por defecto basados en intencionalidad
    default_weights = {
        "likes": 1.0,
        "comments": 3.0,    # Engagement activo
        "saves": 5.0,       # Mayor intencion
        "shares": 4.0,      # Alcance organico
    }
    weights = weights or default_weights

    # Extraer metricas (soportar multiples nombres de campos)
    likes = post.get("likes_count", post.get("likes", 0)) or 0
    comments = post.get("comments_count", post.get("comments", 0)) or 0
    saves = post.get("saves_count", post.get("saves", 0)) or 0
    shares = post.get("shares_count", post.get("shares", 0)) or 0

    # Calcular engagement ponderado
    weighted_engagement = (
        likes * weights.get("likes", 1.0) +
        comments * weights.get("comments", 3.0) +
        saves * weights.get("saves", 5.0) +
        shares * weights.get("shares", 4.0)
    )

    # Normalizar por followers
    if follower_count > 0:
        engagement_ratio = (weighted_engagement / follower_count) * 100
    else:
        engagement_ratio = 0.0

    return round(engagement_ratio, 4)


def calculate_engagement_ratio_batch(
    posts: List[Dict[str, Any]],
    follower_count: int,
    weights: Optional[Dict[str, float]] = None
) -> List[Dict[str, Any]]:
    """
    Calcula engagement_ratio para una lista de posts.

    Args:
        posts: Lista de posts
        follower_count: Numero de seguidores
        weights: Pesos personalizados

    Returns:
        Lista de posts con engagement_ratio agregado
    """
    for post in posts:
        post["engagement_ratio"] = calculate_engagement_ratio(post, follower_count, weights)

    logger.info(f"Calculated engagement_ratio for {len(posts)} posts")
    return posts


def label_viral_status(
    posts: List[Dict[str, Any]],
    config: Optional[LabelingConfig] = None,
    engagement_key: str = "engagement_ratio"
) -> List[Dict[str, Any]]:
    """
    Etiqueta posts con columna binaria is_viral.

    Logica:
    - is_viral = 1: Top 20% del dataset (Exitos)
    - is_viral = 0: Bottom 20% del dataset (Fracasos)
    - is_viral = None: Middle 60% (excluidos del entrenamiento binario)

    Args:
        posts: Lista de posts con engagement calculado
        config: Configuracion de etiquetado
        engagement_key: Campo a usar para ordenar

    Returns:
        Lista de posts con is_viral agregado

    Ejemplo:
        >>> labeled_posts = label_viral_status(posts)
        >>> viral_count = sum(1 for p in labeled_posts if p.get("is_viral") == 1)
        >>> flop_count = sum(1 for p in labeled_posts if p.get("is_viral") == 0)
    """
    config = config or LabelingConfig()

    if not posts:
        logger.warning("No posts to label")
        return posts

    # Filtrar posts sin engagement si esta configurado
    if config.exclude_zero_engagement:
        valid_posts = [p for p in posts if p.get(engagement_key, 0) > config.min_engagement_threshold]
        excluded_count = len(posts) - len(valid_posts)
        if excluded_count > 0:
            logger.info(f"Excluded {excluded_count} posts with zero/low engagement")
    else:
        valid_posts = posts

    if not valid_posts:
        logger.warning("No valid posts after filtering")
        return posts

    # Ordenar por engagement
    sorted_posts = sorted(valid_posts, key=lambda x: x.get(engagement_key, 0), reverse=True)
    n_total = len(sorted_posts)

    # Calcular indices de corte
    top_cutoff = int(n_total * config.top_percentile)
    bottom_cutoff = int(n_total * (1 - config.bottom_percentile))

    # Obtener valores de engagement en los puntos de corte
    if top_cutoff > 0 and top_cutoff < n_total:
        top_threshold = sorted_posts[top_cutoff - 1].get(engagement_key, 0)
    else:
        top_threshold = float('inf')

    if bottom_cutoff < n_total and bottom_cutoff > 0:
        bottom_threshold = sorted_posts[bottom_cutoff].get(engagement_key, 0)
    else:
        bottom_threshold = 0

    logger.info(
        f"Labeling thresholds - Top {config.top_percentile*100}%: {top_threshold:.4f}, "
        f"Bottom {config.bottom_percentile*100}%: {bottom_threshold:.4f}"
    )

    # Etiquetar posts
    viral_count = 0
    flop_count = 0
    middle_count = 0

    for post in posts:
        engagement = post.get(engagement_key, 0)

        if engagement >= top_threshold and top_threshold > 0:
            post["is_viral"] = 1
            post["viral_label"] = "viral"
            viral_count += 1
        elif engagement <= bottom_threshold:
            post["is_viral"] = 0
            post["viral_label"] = "flop"
            flop_count += 1
        else:
            post["is_viral"] = None  # Excluir del entrenamiento binario
            post["viral_label"] = "middle"
            middle_count += 1

    logger.info(
        f"Labeling complete: {viral_count} viral (is_viral=1), "
        f"{flop_count} flops (is_viral=0), {middle_count} middle (excluded)"
    )

    return posts


def label_viral_status_df(
    df: pd.DataFrame,
    engagement_col: str = "engagement_ratio",
    config: Optional[LabelingConfig] = None
) -> pd.DataFrame:
    """
    Version para DataFrames de label_viral_status.

    Args:
        df: DataFrame con datos de posts
        engagement_col: Columna con engagement
        config: Configuracion de etiquetado

    Returns:
        DataFrame con columnas is_viral y viral_label agregadas
    """
    config = config or LabelingConfig()

    if df.empty:
        df["is_viral"] = pd.Series(dtype=float)
        df["viral_label"] = pd.Series(dtype=str)
        return df

    # Calcular percentiles
    top_threshold = df[engagement_col].quantile(1 - config.top_percentile)
    bottom_threshold = df[engagement_col].quantile(config.bottom_percentile)

    # Etiquetar
    conditions = [
        df[engagement_col] >= top_threshold,
        df[engagement_col] <= bottom_threshold
    ]
    choices = [1, 0]

    df["is_viral"] = np.select(conditions, choices, default=np.nan)
    df["viral_label"] = np.select(
        conditions,
        ["viral", "flop"],
        default="middle"
    )

    # Log stats
    viral_count = (df["is_viral"] == 1).sum()
    flop_count = (df["is_viral"] == 0).sum()
    middle_count = df["is_viral"].isna().sum()

    logger.info(
        f"DataFrame labeling: {viral_count} viral, {flop_count} flops, {middle_count} middle"
    )

    return df


def balance_dataset(
    posts: List[Dict[str, Any]],
    config: Optional[BalancingConfig] = None,
    label_key: str = "is_viral"
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Balancea el dataset para entrenamiento del modelo.

    Asegura minimo 30% de casos negativos (flops) usando:
    - Undersampling de clase mayoritaria
    - Oversampling de clase minoritaria (con ruido)
    - SMOTE si imbalanced-learn esta disponible

    Args:
        posts: Lista de posts etiquetados
        config: Configuracion de balanceo
        label_key: Campo con la etiqueta

    Returns:
        Tuple[List[Dict], Dict]: (posts_balanceados, metadata)

    Ejemplo:
        >>> balanced, meta = balance_dataset(labeled_posts)
        >>> print(f"Ratio negativo: {meta['negative_ratio']:.1%}")
    """
    config = config or BalancingConfig()

    # Filtrar posts con etiqueta valida (excluir middle)
    labeled_posts = [p for p in posts if p.get(label_key) is not None]

    if not labeled_posts:
        return posts, {"error": "No labeled posts", "strategy": "none"}

    # Separar por clase
    positive_posts = [p for p in labeled_posts if p.get(label_key) == 1]
    negative_posts = [p for p in labeled_posts if p.get(label_key) == 0]

    n_positive = len(positive_posts)
    n_negative = len(negative_posts)
    n_total = n_positive + n_negative

    current_negative_ratio = n_negative / n_total if n_total > 0 else 0

    metadata = {
        "original_positive": n_positive,
        "original_negative": n_negative,
        "original_ratio": current_negative_ratio,
        "strategy": config.strategy,
    }

    logger.info(
        f"Original distribution: {n_positive} positive, {n_negative} negative "
        f"({current_negative_ratio:.1%} negative)"
    )

    # Si ya cumple el minimo, retornar sin cambios
    if current_negative_ratio >= config.min_negative_ratio:
        logger.info("Dataset already balanced, no changes needed")
        metadata["action"] = "none"
        metadata["final_positive"] = n_positive
        metadata["final_negative"] = n_negative
        metadata["final_ratio"] = current_negative_ratio
        return labeled_posts, metadata

    # Calcular cuantos negativos necesitamos
    # Para ratio minimo: n_neg / (n_pos + n_neg) >= min_ratio
    # n_neg >= min_ratio * n_pos / (1 - min_ratio)
    target_negative = int(config.min_negative_ratio * n_positive / (1 - config.min_negative_ratio))
    needed_negative = target_negative - n_negative

    logger.info(f"Need {needed_negative} more negative samples to reach {config.min_negative_ratio:.0%}")

    # Aplicar estrategia de balanceo
    if config.strategy == "undersampling":
        result = _undersample_majority(positive_posts, negative_posts, config)
    elif config.strategy == "oversampling":
        result = _oversample_minority(positive_posts, negative_posts, needed_negative, config)
    elif config.strategy == "smote":
        result = _apply_smote(positive_posts, negative_posts, config)
    else:  # hybrid
        result = _hybrid_balance(positive_posts, negative_posts, needed_negative, config)

    # Actualizar metadata
    final_positive = sum(1 for p in result if p.get(label_key) == 1)
    final_negative = sum(1 for p in result if p.get(label_key) == 0)
    final_total = final_positive + final_negative

    metadata["action"] = config.strategy
    metadata["final_positive"] = final_positive
    metadata["final_negative"] = final_negative
    metadata["final_ratio"] = final_negative / final_total if final_total > 0 else 0
    metadata["samples_added"] = len(result) - n_total

    logger.info(
        f"Balanced distribution: {final_positive} positive, {final_negative} negative "
        f"({metadata['final_ratio']:.1%} negative)"
    )

    return result, metadata


def _undersample_majority(
    positive: List[Dict],
    negative: List[Dict],
    config: BalancingConfig
) -> List[Dict]:
    """Reduce la clase mayoritaria (positivos/virales)"""
    np.random.seed(config.random_state)

    n_negative = len(negative)
    # Calcular cuantos positivos mantener para lograr el ratio deseado
    # ratio = n_neg / (n_pos + n_neg) => n_pos = n_neg * (1-ratio) / ratio
    target_positive = int(n_negative * (1 - config.min_negative_ratio) / config.min_negative_ratio)
    target_positive = max(target_positive, 1)  # Al menos 1

    if target_positive < len(positive):
        # Random sampling sin reemplazo
        indices = np.random.choice(len(positive), target_positive, replace=False)
        sampled_positive = [positive[i] for i in indices]
        logger.info(f"Undersampled positive class: {len(positive)} -> {len(sampled_positive)}")
    else:
        sampled_positive = positive

    return sampled_positive + negative


def _oversample_minority(
    positive: List[Dict],
    negative: List[Dict],
    needed: int,
    config: BalancingConfig
) -> List[Dict]:
    """Aumenta la clase minoritaria (negativos/flops) con duplicados + ruido"""
    np.random.seed(config.random_state)

    if needed <= 0 or not negative:
        return positive + negative

    # Duplicar samples con pequeno ruido en engagement
    augmented_negative = negative.copy()

    for i in range(needed):
        # Seleccionar un sample aleatorio para duplicar
        base_sample = negative[i % len(negative)].copy()

        # Agregar pequeno ruido a metricas para evitar duplicados exactos
        noise_factor = np.random.uniform(0.9, 1.1)
        for key in ["likes_count", "likes", "comments_count", "comments",
                    "engagement_ratio", "engagement_score"]:
            if key in base_sample and base_sample[key] is not None:
                base_sample[key] = base_sample[key] * noise_factor

        # Marcar como augmentado
        base_sample["_is_augmented"] = True
        base_sample["_augment_source_id"] = negative[i % len(negative)].get("id", i)

        augmented_negative.append(base_sample)

    logger.info(f"Oversampled negative class: {len(negative)} -> {len(augmented_negative)}")

    return positive + augmented_negative


def _apply_smote(
    positive: List[Dict],
    negative: List[Dict],
    config: BalancingConfig
) -> List[Dict]:
    """Aplica SMOTE usando imbalanced-learn si esta disponible"""
    try:
        from imblearn.over_sampling import SMOTE
        from app.services.ml_service import FeatureExtractor

        # Convertir a features numericas
        all_posts = positive + negative
        df = FeatureExtractor.extract_batch(all_posts)

        # Preparar X e y
        y = np.array([1] * len(positive) + [0] * len(negative))

        # Seleccionar solo columnas numericas
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        X = df[numeric_cols].fillna(0).values

        # Aplicar SMOTE
        smote = SMOTE(
            sampling_strategy=config.min_negative_ratio / (1 - config.min_negative_ratio),
            random_state=config.random_state
        )
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Reconstruir posts (aproximado - los nuevos son sinteticos)
        result = []
        for i, (x, label) in enumerate(zip(X_resampled, y_resampled)):
            if i < len(all_posts):
                # Post original
                result.append(all_posts[i])
            else:
                # Post sintetico - crear estructura basica
                synthetic = {
                    "is_viral": int(label),
                    "viral_label": "viral" if label == 1 else "flop",
                    "_is_synthetic": True,
                    "_smote_features": x.tolist(),
                }
                # Mapear features de vuelta
                for j, col in enumerate(numeric_cols):
                    synthetic[col] = float(x[j])
                result.append(synthetic)

        logger.info(f"SMOTE applied: {len(all_posts)} -> {len(result)} samples")
        return result

    except ImportError:
        logger.warning("imbalanced-learn not installed. Falling back to hybrid strategy.")
        return _hybrid_balance(positive, negative, 0, config)


def _hybrid_balance(
    positive: List[Dict],
    negative: List[Dict],
    needed: int,
    config: BalancingConfig
) -> List[Dict]:
    """
    Estrategia hibrida: combina undersampling ligero + oversampling.
    Mejor para datasets pequenos.
    """
    np.random.seed(config.random_state)

    n_positive = len(positive)
    n_negative = len(negative)

    # Calcular target sizes
    # Queremos: n_neg_final / (n_pos_final + n_neg_final) = target_ratio
    target_ratio = (config.min_negative_ratio + config.max_negative_ratio) / 2

    # Opcion 1: Undersample positivos
    target_positive_under = int(n_negative * (1 - target_ratio) / target_ratio)

    # Opcion 2: Oversample negativos
    target_negative_over = int(n_positive * target_ratio / (1 - target_ratio))

    # Elegir la opcion que minimize la perdida de datos
    if target_positive_under >= n_positive * 0.7:  # No perder mas del 30% de positivos
        # Undersample positivos
        indices = np.random.choice(n_positive, target_positive_under, replace=False)
        final_positive = [positive[i] for i in indices]
        final_negative = negative
        logger.info(f"Hybrid: undersampled positive {n_positive} -> {len(final_positive)}")
    else:
        # Oversample negativos
        needed = target_negative_over - n_negative
        final_positive = positive
        final_negative = negative.copy()

        for i in range(max(0, needed)):
            base = negative[i % len(negative)].copy()
            noise = np.random.uniform(0.85, 1.15)
            for key in ["likes_count", "comments_count", "engagement_ratio"]:
                if key in base and base[key]:
                    base[key] = base[key] * noise
            base["_is_augmented"] = True
            final_negative.append(base)

        logger.info(f"Hybrid: oversampled negative {n_negative} -> {len(final_negative)}")

    return final_positive + final_negative


class DataPreprocessor:
    """
    Clase principal para preprocesamiento de datos.
    Combina todas las funciones de preprocesamiento en un pipeline.
    """

    def __init__(
        self,
        labeling_config: Optional[LabelingConfig] = None,
        balancing_config: Optional[BalancingConfig] = None
    ):
        self.labeling_config = labeling_config or LabelingConfig()
        self.balancing_config = balancing_config or BalancingConfig()
        self._processing_stats = {}

    def preprocess_for_training(
        self,
        posts: List[Dict[str, Any]],
        follower_count: int
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Pipeline completo de preprocesamiento para entrenamiento.

        Pasos:
        1. Calcular engagement_ratio
        2. Etiquetar is_viral
        3. Balancear dataset

        Args:
            posts: Lista de posts crudos
            follower_count: Numero de seguidores

        Returns:
            Tuple[List[Dict], Dict]: (posts_procesados, estadisticas)
        """
        stats = {
            "input_count": len(posts),
            "follower_count": follower_count,
        }

        # Paso 1: Calcular engagement ratio
        posts = calculate_engagement_ratio_batch(posts, follower_count)
        stats["after_engagement_calc"] = len(posts)

        # Paso 2: Etiquetar viral status
        posts = label_viral_status(posts, self.labeling_config)
        stats["viral_count"] = sum(1 for p in posts if p.get("is_viral") == 1)
        stats["flop_count"] = sum(1 for p in posts if p.get("is_viral") == 0)
        stats["middle_count"] = sum(1 for p in posts if p.get("is_viral") is None)

        # Paso 3: Balancear dataset
        balanced_posts, balance_meta = balance_dataset(posts, self.balancing_config)
        stats["balancing"] = balance_meta
        stats["output_count"] = len(balanced_posts)

        self._processing_stats = stats
        logger.info(f"Preprocessing complete: {stats['input_count']} -> {stats['output_count']} posts")

        return balanced_posts, stats

    def prepare_dataframe(
        self,
        posts: List[Dict[str, Any]],
        include_middle: bool = False
    ) -> pd.DataFrame:
        """
        Convierte posts procesados a DataFrame para ML.

        Args:
            posts: Lista de posts preprocesados
            include_middle: Si incluir posts del medio (is_viral=None)

        Returns:
            DataFrame listo para entrenamiento
        """
        df = pd.DataFrame(posts)

        # Filtrar si no incluir middle
        if not include_middle:
            df = df[df["is_viral"].notna()].copy()

        # Convertir is_viral a int
        df["is_viral"] = df["is_viral"].astype(int)

        return df

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadisticas del ultimo preprocesamiento"""
        return self._processing_stats


# Convenience functions para uso directo
def preprocess_posts(
    posts: List[Dict[str, Any]],
    follower_count: int,
    top_percentile: float = 0.20,
    bottom_percentile: float = 0.20,
    min_negative_ratio: float = 0.30
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Funcion de conveniencia para preprocesar posts de una sola vez.

    Args:
        posts: Lista de posts crudos
        follower_count: Numero de seguidores
        top_percentile: Percentil para etiquetar como viral
        bottom_percentile: Percentil para etiquetar como flop
        min_negative_ratio: Ratio minimo de negativos despues del balanceo

    Returns:
        Tuple[List[Dict], Dict]: (posts_procesados, estadisticas)

    Ejemplo:
        >>> from app.services.scraper import preprocess_posts
        >>> processed, stats = preprocess_posts(raw_posts, follower_count=10000)
        >>> print(f"Dataset balanceado: {stats['output_count']} posts")
    """
    labeling_config = LabelingConfig(
        top_percentile=top_percentile,
        bottom_percentile=bottom_percentile
    )
    balancing_config = BalancingConfig(
        min_negative_ratio=min_negative_ratio
    )

    preprocessor = DataPreprocessor(labeling_config, balancing_config)
    return preprocessor.preprocess_for_training(posts, follower_count)
