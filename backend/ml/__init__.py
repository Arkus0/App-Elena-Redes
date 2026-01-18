"""
Backend ML Module - Multimodal Feature Engineering & Online Learning
====================================================================

This module provides multimodal late fusion capabilities and
online/incremental learning for engagement prediction.

Components:
-----------
- multimodal_fusion: Late fusion for combining text, audio, and visual features
- online_update: River-based incremental learning for live model updates

Usage:
------
    # Multimodal fusion
    from backend.ml.multimodal_fusion import (
        fuse_multimodal_features,
        get_multimodal_extractor,
        MULTIMODAL_FEATURE_COLUMNS
    )
    df_fused = fuse_multimodal_features(df)

    # Online learning
    from backend.ml.online_update import (
        online_update,
        get_online_predictor,
    )
    result = online_update("niche", features_df, targets_df)
"""

from .multimodal_fusion import (
    # Main fusion function
    fuse_multimodal_features,
    add_multimodal_features_conditional,
    is_video_content,

    # Extractor class and singleton
    MultimodalEmbeddingExtractor,
    get_multimodal_extractor,

    # Feature column names
    TRANSCRIPT_FEATURE_COLUMNS,
    OCR_FEATURE_COLUMNS,
    INTERACTION_FEATURE_COLUMNS,
    MULTIMODAL_FEATURE_COLUMNS,
    get_transcript_feature_names,
    get_ocr_feature_names,
    get_interaction_feature_names,
    get_multimodal_feature_names,

    # Configuration
    CAPTION_PCA_DIM,
    TRANSCRIPT_PCA_DIM,
    OCR_PCA_DIM,
)

# Online learning imports (optional - requires river)
try:
    from .online_update import (
        # Main functions
        online_update,
        online_update_single,
        get_online_predictor,
        reset_online_predictor,
        detect_drift,
        compare_online_vs_batch,

        # Classes
        OnlineEngagementPredictor,
        OnlineUpdateResult,
        OnlineModelMetrics,

        # Configuration
        RIVER_AVAILABLE,
        TARGET_NAMES as ONLINE_TARGET_NAMES,
        EVALUATION_INTERVAL,
        IMPROVEMENT_THRESHOLD,
    )
    ONLINE_LEARNING_AVAILABLE = True
except ImportError:
    ONLINE_LEARNING_AVAILABLE = False
    RIVER_AVAILABLE = False


__all__ = [
    # Main functions
    'fuse_multimodal_features',
    'add_multimodal_features_conditional',
    'is_video_content',

    # Classes
    'MultimodalEmbeddingExtractor',
    'get_multimodal_extractor',

    # Feature names
    'TRANSCRIPT_FEATURE_COLUMNS',
    'OCR_FEATURE_COLUMNS',
    'INTERACTION_FEATURE_COLUMNS',
    'MULTIMODAL_FEATURE_COLUMNS',
    'get_transcript_feature_names',
    'get_ocr_feature_names',
    'get_interaction_feature_names',
    'get_multimodal_feature_names',

    # Config
    'CAPTION_PCA_DIM',
    'TRANSCRIPT_PCA_DIM',
    'OCR_PCA_DIM',

    # Online Learning (if available)
    'ONLINE_LEARNING_AVAILABLE',
    'RIVER_AVAILABLE',
]

# Conditionally add online learning exports
if ONLINE_LEARNING_AVAILABLE:
    __all__.extend([
        'online_update',
        'online_update_single',
        'get_online_predictor',
        'reset_online_predictor',
        'detect_drift',
        'compare_online_vs_batch',
        'OnlineEngagementPredictor',
        'OnlineUpdateResult',
        'OnlineModelMetrics',
        'ONLINE_TARGET_NAMES',
        'EVALUATION_INTERVAL',
        'IMPROVEMENT_THRESHOLD',
    ])
