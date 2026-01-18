#!/usr/bin/env python3
"""
Tests for Embedding Precision Configuration
===========================================

Validates that the embedding precision system works correctly:
- ultra_low (64 dims): Ultra rapido para PC modesto
- low (128 dims): Recomendado sobremesa normal Almeria
- medium (256 dims): Balance precision/velocidad
- high (384 dims): Full dims con TruncatedSVD
- max (full raw 384): Sin reduccion

Author: BrandPulse AI
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Test imports
import sys
sys.path.insert(0, '/home/user/App-Elena-Redes')


class TestPrecisionTodimsMapping:
    """Test PRECISION_TO_DIMS configuration."""

    def test_import_precision_config(self):
        """Test that precision config can be imported."""
        from ml.features_embeddings import (
            PRECISION_TO_DIMS,
            EmbeddingPrecision,
        )
        assert PRECISION_TO_DIMS is not None
        assert EmbeddingPrecision is not None

    def test_ultra_low_is_64_dims(self):
        """Test ultra_low precision is 64 dims."""
        from ml.features_embeddings import PRECISION_TO_DIMS
        assert PRECISION_TO_DIMS["ultra_low"] == 64

    def test_low_is_128_dims(self):
        """Test low precision is 128 dims."""
        from ml.features_embeddings import PRECISION_TO_DIMS
        assert PRECISION_TO_DIMS["low"] == 128

    def test_medium_is_256_dims(self):
        """Test medium precision is 256 dims."""
        from ml.features_embeddings import PRECISION_TO_DIMS
        assert PRECISION_TO_DIMS["medium"] == 256

    def test_high_is_384_dims(self):
        """Test high precision is 384 dims."""
        from ml.features_embeddings import PRECISION_TO_DIMS
        assert PRECISION_TO_DIMS["high"] == 384

    def test_max_is_none_full_raw(self):
        """Test max precision is None (full raw embeddings)."""
        from ml.features_embeddings import PRECISION_TO_DIMS
        assert PRECISION_TO_DIMS["max"] is None


class TestEmbeddingPrecisionEnum:
    """Test EmbeddingPrecision enum."""

    def test_enum_values(self):
        """Test all enum values exist."""
        from ml.features_embeddings import EmbeddingPrecision
        assert EmbeddingPrecision.ULTRA_LOW.value == "ultra_low"
        assert EmbeddingPrecision.LOW.value == "low"
        assert EmbeddingPrecision.MEDIUM.value == "medium"
        assert EmbeddingPrecision.HIGH.value == "high"
        assert EmbeddingPrecision.MAX.value == "max"

    def test_dimensions_property_ultra_low(self):
        """Test ultra_low dimensions property."""
        from ml.features_embeddings import EmbeddingPrecision
        assert EmbeddingPrecision.ULTRA_LOW.dimensions == 64

    def test_dimensions_property_low(self):
        """Test low dimensions property."""
        from ml.features_embeddings import EmbeddingPrecision
        assert EmbeddingPrecision.LOW.dimensions == 128

    def test_dimensions_property_max(self):
        """Test max dimensions property returns full 384."""
        from ml.features_embeddings import EmbeddingPrecision, EMBEDDING_DIM
        # max returns None in PRECISION_TO_DIMS, but dimensions property returns EMBEDDING_DIM
        assert EmbeddingPrecision.MAX.dimensions == EMBEDDING_DIM
        assert EmbeddingPrecision.MAX.dimensions == 384

    def test_uses_reduction_property(self):
        """Test uses_reduction property."""
        from ml.features_embeddings import EmbeddingPrecision
        # ultra_low, low, medium, high use reduction
        assert EmbeddingPrecision.ULTRA_LOW.uses_reduction is True
        assert EmbeddingPrecision.LOW.uses_reduction is True
        assert EmbeddingPrecision.MEDIUM.uses_reduction is True
        assert EmbeddingPrecision.HIGH.uses_reduction is True
        # max does NOT use reduction (full raw)
        assert EmbeddingPrecision.MAX.uses_reduction is False


class TestPrecisionRequired:
    """Test that precision is required (no default)."""

    def test_extractor_requires_precision(self):
        """Test EmbeddingExtractor raises error without precision."""
        from ml.features_embeddings import EmbeddingExtractor
        with pytest.raises(ValueError, match="precision es REQUERIDO"):
            EmbeddingExtractor(precision=None)

    def test_get_embedding_extractor_requires_precision(self):
        """Test get_embedding_extractor raises error without precision."""
        from ml.features_embeddings import get_embedding_extractor
        with pytest.raises(ValueError, match="precision es REQUERIDO"):
            get_embedding_extractor(precision=None)

    def test_invalid_precision_raises_error(self):
        """Test invalid precision raises error."""
        from ml.features_embeddings import EmbeddingExtractor
        with pytest.raises(ValueError, match="no valida"):
            EmbeddingExtractor(precision="invalid_precision")


class TestEmbeddingExtractorDimensions:
    """Test EmbeddingExtractor output dimensions."""

    @pytest.fixture
    def mock_sentence_transformer(self):
        """Mock SentenceTransformer for testing."""
        with patch('ml.features_embeddings.SentenceTransformer') as mock:
            # Mock model that returns 384-dim embeddings
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.randn(384)
            mock.return_value = mock_model
            yield mock

    def test_ultra_low_returns_64_dims(self, mock_sentence_transformer):
        """Test ultra_low precision returns 64 dimensions."""
        from ml.features_embeddings import EmbeddingExtractor
        extractor = EmbeddingExtractor(precision="ultra_low")
        assert extractor.target_dims == 64

    def test_low_returns_128_dims(self, mock_sentence_transformer):
        """Test low precision returns 128 dimensions."""
        from ml.features_embeddings import EmbeddingExtractor
        extractor = EmbeddingExtractor(precision="low")
        assert extractor.target_dims == 128

    def test_medium_returns_256_dims(self, mock_sentence_transformer):
        """Test medium precision returns 256 dimensions."""
        from ml.features_embeddings import EmbeddingExtractor
        extractor = EmbeddingExtractor(precision="medium")
        assert extractor.target_dims == 256

    def test_high_returns_384_dims(self, mock_sentence_transformer):
        """Test high precision returns 384 dimensions."""
        from ml.features_embeddings import EmbeddingExtractor
        extractor = EmbeddingExtractor(precision="high")
        assert extractor.target_dims == 384

    def test_max_returns_384_dims_raw(self, mock_sentence_transformer):
        """Test max precision returns full 384 dimensions (raw)."""
        from ml.features_embeddings import EmbeddingExtractor
        extractor = EmbeddingExtractor(precision="max")
        assert extractor.target_dims == 384
        assert extractor.uses_reduction is False  # max = no reduction


class TestBusinessModelPrecision:
    """Test Business model embedding precision."""

    def test_business_model_has_ultra_low(self):
        """Test Business model has ULTRA_LOW enum."""
        from backend.app.models.business import EmbeddingPrecision
        assert hasattr(EmbeddingPrecision, 'ULTRA_LOW')
        assert EmbeddingPrecision.ULTRA_LOW.value == "ultra_low"

    def test_business_model_default_is_low(self):
        """Test Business model default is LOW (not MAX)."""
        from backend.app.models.business import Business, EmbeddingPrecision
        # Check the column default
        col = Business.__table__.columns['embedding_precision']
        assert col.default.arg == EmbeddingPrecision.LOW


class TestAPIEndpointPrecisionInfo:
    """Test API endpoint precision info."""

    def test_precision_info_has_ultra_low(self):
        """Test PRECISION_INFO has ultra_low."""
        from backend.app.api.embeddings import PRECISION_INFO
        assert "ultra_low" in PRECISION_INFO
        assert PRECISION_INFO["ultra_low"]["dimensions"] == 64

    def test_precision_info_recommended_is_low(self):
        """Test recommended precision is low."""
        from backend.app.api.embeddings import PRECISION_INFO
        # low should be marked as recommended in the label
        assert "Recomendada" in PRECISION_INFO["low"]["label"]


class TestTextIntelligencePrecision:
    """Test text_intelligence.py precision parameter."""

    def test_encode_and_reduce_accepts_precision(self):
        """Test encode_and_reduce accepts precision parameter."""
        from backend.app.services.text_intelligence import SemanticEncoder
        import inspect
        sig = inspect.signature(SemanticEncoder.encode_and_reduce)
        assert "precision" in sig.parameters

    def test_encode_new_format_accepts_precision(self):
        """Test _encode_new_format accepts precision parameter."""
        from backend.app.services.text_intelligence import SemanticEncoder
        import inspect
        sig = inspect.signature(SemanticEncoder._encode_new_format)
        assert "precision" in sig.parameters

    def test_extract_all_features_accepts_precision(self):
        """Test extract_all_features accepts precision parameter."""
        from backend.app.services.text_intelligence import TextIntelligenceEngine
        import inspect
        sig = inspect.signature(TextIntelligenceEngine.extract_all_features)
        assert "precision" in sig.parameters


class TestFeatureCountByPrecision:
    """Test expected feature counts by precision level."""

    @pytest.mark.parametrize("precision,expected_dims", [
        ("ultra_low", 64),
        ("low", 128),
        ("medium", 256),
        ("high", 384),
        ("max", 384),
    ])
    def test_feature_count_per_precision(self, precision, expected_dims):
        """Test feature count for each precision level."""
        from ml.features_embeddings import PRECISION_TO_DIMS, EMBEDDING_DIM

        dims = PRECISION_TO_DIMS.get(precision)
        if dims is None:  # max = full raw
            dims = EMBEDDING_DIM

        assert dims == expected_dims


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
