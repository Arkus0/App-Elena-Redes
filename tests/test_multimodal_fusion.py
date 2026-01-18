#!/usr/bin/env python3
"""
Tests for Multimodal Late Fusion Module
=======================================

Tests the multimodal fusion pipeline for engagement prediction.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest


class TestMultimodalFusion:
    """Test suite for multimodal fusion functionality."""

    def test_import_multimodal_fusion(self):
        """Test that multimodal fusion module can be imported."""
        from backend.ml.multimodal_fusion import (
            fuse_multimodal_features,
            get_multimodal_extractor,
            MULTIMODAL_FEATURE_COLUMNS,
            TRANSCRIPT_FEATURE_COLUMNS,
            OCR_FEATURE_COLUMNS,
            INTERACTION_FEATURE_COLUMNS,
        )
        assert len(TRANSCRIPT_FEATURE_COLUMNS) == 20
        assert len(OCR_FEATURE_COLUMNS) == 20
        assert len(INTERACTION_FEATURE_COLUMNS) == 8
        assert len(MULTIMODAL_FEATURE_COLUMNS) == 48

    def test_feature_column_names(self):
        """Test that feature column names are correctly formatted."""
        from backend.ml.multimodal_fusion import (
            get_transcript_feature_names,
            get_ocr_feature_names,
            get_interaction_feature_names,
        )

        transcript_cols = get_transcript_feature_names()
        assert transcript_cols[0] == "transcript_emb_1"
        assert transcript_cols[-1] == "transcript_emb_20"

        ocr_cols = get_ocr_feature_names()
        assert ocr_cols[0] == "ocr_emb_1"
        assert ocr_cols[-1] == "ocr_emb_20"

        interaction_cols = get_interaction_feature_names()
        assert "interaction_hook_x_sentiment" in interaction_cols
        assert "interaction_multimodal_text_density" in interaction_cols

    def test_fuse_multimodal_features_basic(self):
        """Test basic multimodal fusion with sample data."""
        from backend.ml.multimodal_fusion import (
            fuse_multimodal_features,
            MULTIMODAL_FEATURE_COLUMNS,
        )

        # Create test DataFrame
        df = pd.DataFrame({
            'caption': [
                'Nuevo Reel increíble! Mira este truco',
                'POV: cuando tu café está perfecto',
            ],
            'whisper_transcript': [
                'Hola a todos, hoy les voy a mostrar un truco increíble',
                'El secreto de un buen café está en la temperatura del agua',
            ],
            'easyocr_text': [
                'TRUCO #1',
                'CAFÉ PERFECTO',
            ],
            'hook_score': [0.85, 0.72],
            'sentiment_compound': [0.6, 0.4],
            'is_reel': [1, 1],
            'cta_count': [2, 1],
            'video_optimal_length': [1, 1],
            'caption_length': [40, 35],
            'media_type': ['reel', 'reel'],
        })

        original_cols = len(df.columns)

        # Apply fusion (without saving PCA)
        df_fused = fuse_multimodal_features(df, fit_pca_if_needed=True, save_pca=False)

        # Check new columns were added
        assert len(df_fused.columns) > original_cols
        for col in MULTIMODAL_FEATURE_COLUMNS:
            assert col in df_fused.columns, f"Missing column: {col}"

        # Check no NaN values
        assert not df_fused[MULTIMODAL_FEATURE_COLUMNS].isna().any().any()

    def test_fuse_without_multimodal_columns(self):
        """Test fusion when multimodal columns are missing."""
        from backend.ml.multimodal_fusion import (
            fuse_multimodal_features,
            MULTIMODAL_FEATURE_COLUMNS,
        )

        # DataFrame without whisper_transcript or easyocr_text
        df = pd.DataFrame({
            'caption': ['Test caption'],
            'hook_score': [0.5],
            'sentiment_compound': [0.3],
            'is_reel': [1],
            'cta_count': [1],
        })

        df_fused = fuse_multimodal_features(df, fit_pca_if_needed=True, save_pca=False)

        # Should add zero columns
        for col in MULTIMODAL_FEATURE_COLUMNS:
            assert col in df_fused.columns
            assert df_fused[col].iloc[0] == 0.0

    def test_interaction_features_calculation(self):
        """Test that interaction features are calculated correctly."""
        from backend.ml.multimodal_fusion import fuse_multimodal_features

        df = pd.DataFrame({
            'caption': ['Test'],
            'whisper_transcript': ['Test transcript'],
            'easyocr_text': ['TEST'],
            'hook_score': [0.8],
            'sentiment_compound': [0.5],
            'is_reel': [1],
            'cta_count': [3],
            'video_optimal_length': [1],
            'caption_length': [10],
            'media_type': ['reel'],
        })

        df_fused = fuse_multimodal_features(df, fit_pca_if_needed=True, save_pca=False)

        # Check interaction calculations
        assert abs(df_fused['interaction_hook_x_sentiment'].iloc[0] - 0.8 * 0.5) < 0.01
        assert abs(df_fused['interaction_hook_x_is_reel'].iloc[0] - 0.8 * 1) < 0.01
        assert abs(df_fused['interaction_hook_x_cta_count'].iloc[0] - 0.8 * 3) < 0.01

    def test_is_video_content_detection(self):
        """Test video content detection function."""
        from backend.ml.multimodal_fusion import is_video_content

        # Test with media_type
        row_reel = pd.Series({'media_type': 'reel'})
        assert is_video_content(row_reel) == True

        row_video = pd.Series({'media_type': 'video'})
        assert is_video_content(row_video) == True

        row_image = pd.Series({'media_type': 'image'})
        assert is_video_content(row_image) == False

        # Test with is_reel
        row_is_reel = pd.Series({'is_reel': 1})
        assert is_video_content(row_is_reel) == True

        row_not_reel = pd.Series({'is_reel': 0})
        assert is_video_content(row_not_reel) == False

        # Test with video_duration
        row_duration = pd.Series({'video_duration': 30})
        assert is_video_content(row_duration) == True

    def test_multimodal_extractor_singleton(self):
        """Test that multimodal extractor uses singleton pattern."""
        from backend.ml.multimodal_fusion import get_multimodal_extractor

        extractor1 = get_multimodal_extractor()
        extractor2 = get_multimodal_extractor()

        # Should be same instance
        assert extractor1 is extractor2

    def test_empty_text_handling(self):
        """Test handling of empty/null text fields."""
        from backend.ml.multimodal_fusion import (
            fuse_multimodal_features,
            TRANSCRIPT_FEATURE_COLUMNS,
            OCR_FEATURE_COLUMNS,
        )

        df = pd.DataFrame({
            'caption': ['Test'],
            'whisper_transcript': [''],  # Empty
            'easyocr_text': [None],  # None
            'hook_score': [0.5],
            'sentiment_compound': [0.3],
            'is_reel': [1],
            'cta_count': [1],
            'video_optimal_length': [1],
            'caption_length': [4],
            'media_type': ['reel'],
        })

        df_fused = fuse_multimodal_features(df, fit_pca_if_needed=True, save_pca=False)

        # Empty/null should result in zeros
        for col in TRANSCRIPT_FEATURE_COLUMNS:
            assert df_fused[col].iloc[0] == 0.0
        for col in OCR_FEATURE_COLUMNS:
            assert df_fused[col].iloc[0] == 0.0

    def test_add_multimodal_features_conditional(self):
        """Test conditional multimodal feature addition."""
        from backend.ml.multimodal_fusion import (
            add_multimodal_features_conditional,
            MULTIMODAL_FEATURE_COLUMNS,
        )

        # Video content
        df_video = pd.DataFrame({
            'caption': ['Video post'],
            'media_type': ['reel'],
        })

        df_result = add_multimodal_features_conditional(df_video)
        assert all(col in df_result.columns for col in MULTIMODAL_FEATURE_COLUMNS)

        # Image content (no multimodal data)
        df_image = pd.DataFrame({
            'caption': ['Image post'],
            'media_type': ['image'],
        })

        df_result = add_multimodal_features_conditional(df_image)
        assert all(col in df_result.columns for col in MULTIMODAL_FEATURE_COLUMNS)
        # Should be all zeros
        for col in MULTIMODAL_FEATURE_COLUMNS:
            assert df_result[col].iloc[0] == 0.0


class TestMultimodalWithMLService:
    """Integration tests with ML service."""

    def test_feature_extractor_includes_multimodal(self):
        """Test that FeatureExtractor includes multimodal features."""
        from backend.app.services.ml_service import FeatureExtractor

        content = {
            'caption': 'Nuevo Reel con tips increíbles!',
            'whisper_transcript': 'Hola, les voy a dar tres tips importantes',
            'easyocr_text': '3 TIPS',
            'content_format': 'reel',
            'media_type': 'reel',
        }

        features = FeatureExtractor.extract_features(content)

        # Check multimodal features are present
        assert 'interaction_hook_x_sentiment' in features
        assert 'interaction_multimodal_text_density' in features
        assert 'transcript_emb_1' in features
        assert 'ocr_emb_1' in features

    def test_ml_predictor_feature_columns(self):
        """Test that MLPredictor has multimodal features in FEATURE_COLUMNS."""
        from backend.app.services.ml_service import MLPredictor

        # Check that multimodal columns are included
        assert any('transcript_emb' in col for col in MLPredictor.FEATURE_COLUMNS)
        assert any('ocr_emb' in col for col in MLPredictor.FEATURE_COLUMNS)
        assert any('interaction_' in col for col in MLPredictor.FEATURE_COLUMNS)


class TestMultimodalWithTraining:
    """Integration tests with training pipeline."""

    def test_training_feature_columns_include_multimodal(self):
        """Test that training FEATURE_COLUMNS include multimodal."""
        from ml.train import FEATURE_COLUMNS, MULTIMODAL_FEATURE_COLUMNS

        for col in MULTIMODAL_FEATURE_COLUMNS:
            assert col in FEATURE_COLUMNS, f"Missing {col} in FEATURE_COLUMNS"

    def test_prepare_features_adds_multimodal(self):
        """Test that prepare_features adds multimodal columns."""
        from ml.train import prepare_features, MULTIMODAL_FEATURE_COLUMNS

        df = pd.DataFrame({
            'caption': ['Test caption'],
            'engagement_rate': [50.0],
            'business_type': ['restaurante'],
            'is_reel': [1],
            'media_type': ['reel'],
        })

        X, y = prepare_features(df)

        # Multimodal columns should be added (as zeros since no transcript/ocr)
        multimodal_in_X = [col for col in MULTIMODAL_FEATURE_COLUMNS if col in X.columns]
        assert len(multimodal_in_X) > 0


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
