"""
Backend ML Module - Multimodal Feature Engineering
===================================================

This module provides multimodal late fusion capabilities for
engagement prediction in Reels/TikTok content.

Components:
-----------
- multimodal_fusion: Late fusion for combining text, audio, and visual features

Usage:
------
    from backend.ml.multimodal_fusion import (
        fuse_multimodal_features,
        get_multimodal_extractor,
        MULTIMODAL_FEATURE_COLUMNS
    )

    # Apply multimodal fusion to your DataFrame
    df_fused = fuse_multimodal_features(df)
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
]
