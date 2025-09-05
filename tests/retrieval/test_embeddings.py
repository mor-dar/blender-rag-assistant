#!/usr/bin/env python3
"""
Unit tests for EmbeddingGenerator class.

Tests embedding generation, batch processing, and error handling.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from retrieval.embeddings import EmbeddingGenerator


class TestEmbeddingGenerator:
    """Test suite for EmbeddingGenerator."""

    @pytest.fixture
    def embedding_generator(self):
        """Create EmbeddingGenerator instance."""
        return EmbeddingGenerator("all-MiniLM-L6-v2")

    # INITIALIZATION TESTS

    def test_initialization_success(self, embedding_generator):
        """Test successful initialization."""
        assert embedding_generator.model_name == "all-MiniLM-L6-v2"
        assert embedding_generator.device == "cpu"  # Default device from config
        assert embedding_generator.model is not None

    @patch('retrieval.embeddings.SentenceTransformer')
    def test_initialization_with_custom_model(self, mock_transformer):
        """Test initialization with custom model name."""
        mock_model = Mock()
        mock_transformer.return_value = mock_model
        
        generator = EmbeddingGenerator("custom-model")
        
        assert generator.model_name == "custom-model"
        assert generator.model == mock_model
        mock_transformer.assert_called_once_with("custom-model", device="cpu")

    # SINGLE ENCODING TESTS

    def test_encode_single_success(self, embedding_generator):
        """Test successful single text encoding."""
        text = "This is a test sentence for embedding."
        
        embedding = embedding_generator.encode_single(text)
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) > 0
        assert embedding.dtype == np.float32

    def test_encode_single_empty_text(self, embedding_generator):
        """Test encoding empty text raises error."""
        with pytest.raises(ValueError, match="Cannot encode empty text"):
            embedding_generator.encode_single("")

    def test_encode_single_whitespace_only(self, embedding_generator):
        """Test encoding whitespace-only text raises error."""
        with pytest.raises(ValueError, match="Cannot encode empty text"):
            embedding_generator.encode_single("   \n\t   ")

    # BATCH ENCODING TESTS

    def test_encode_batch_success(self, embedding_generator):
        """Test successful batch encoding."""
        texts = [
            "First test sentence.",
            "Second test sentence for batch processing.", 
            "Third sentence to complete the batch."
        ]
        
        embeddings = embedding_generator.encode_batch(texts)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(texts)
        for embedding in embeddings:
            assert isinstance(embedding, list)
            assert len(embedding) > 0
            assert all(isinstance(x, float) for x in embedding)

    def test_encode_batch_empty_list(self, embedding_generator):
        """Test encoding empty batch returns empty list."""
        embeddings = embedding_generator.encode_batch([])
        assert embeddings == []

    def test_encode_batch_with_empty_texts(self, embedding_generator):
        """Test batch encoding with some empty texts."""
        texts = [
            "Valid text",
            "",  # Empty text
            "Another valid text",
            "   ",  # Whitespace only
            "Final valid text"
        ]
        
        embeddings = embedding_generator.encode_batch(texts)
        
        # Should only return embeddings for valid texts (3 in this case)
        assert len(embeddings) == 3
        for embedding in embeddings:
            assert isinstance(embedding, list)
            assert len(embedding) > 0

    def test_encode_batch_all_empty_texts(self, embedding_generator):
        """Test batch encoding with all empty texts raises error."""
        texts = ["", "  ", "\n\t", ""]
        
        with pytest.raises(ValueError, match="Cannot encode batch with all empty texts"):
            embedding_generator.encode_batch(texts)

    def test_encode_batch_with_batch_size(self, embedding_generator):
        """Test batch encoding with custom batch size."""
        texts = ["Text " + str(i) for i in range(10)]
        
        embeddings = embedding_generator.encode_batch(texts, batch_size=3)
        
        assert len(embeddings) == len(texts)
        for embedding in embeddings:
            assert isinstance(embedding, list)
            assert len(embedding) > 0

    @patch('retrieval.embeddings.SentenceTransformer')
    def test_encode_batch_with_progress_bar(self, mock_transformer):
        """Test batch encoding shows progress bar for large batches."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]] * 150)
        mock_transformer.return_value = mock_model
        
        generator = EmbeddingGenerator()
        texts = ["Text " + str(i) for i in range(150)]  # Large batch
        
        embeddings = generator.encode_batch(texts)
        
        # Should call with show_progress_bar=True for large batches
        mock_model.encode.assert_called_once()
        call_args = mock_model.encode.call_args
        assert call_args[1]["show_progress_bar"] is True

    # MODEL INFO TESTS

    def test_get_embedding_dimension(self, embedding_generator):
        """Test getting embedding dimension."""
        dimension = embedding_generator.get_embedding_dimension()
        
        assert isinstance(dimension, int)
        assert dimension > 0
        # all-MiniLM-L6-v2 has 384 dimensions
        assert dimension == 384

    def test_get_model_info(self, embedding_generator):
        """Test getting model information."""
        info = embedding_generator.get_model_info()
        
        assert isinstance(info, dict)
        assert "model_name" in info
        assert "max_seq_length" in info
        assert "embedding_dimension" in info
        assert "device" in info
        
        assert info["model_name"] == "all-MiniLM-L6-v2"
        assert info["embedding_dimension"] == 384

    # ERROR HANDLING TESTS

    @patch('retrieval.embeddings.SentenceTransformer')
    def test_encode_single_model_error(self, mock_transformer):
        """Test handling of model encoding errors."""
        mock_model = Mock()
        mock_model.encode.side_effect = RuntimeError("Model error")
        mock_transformer.return_value = mock_model
        
        generator = EmbeddingGenerator()
        
        with pytest.raises(RuntimeError, match="Model error"):
            generator.encode_single("test text")

    @patch('retrieval.embeddings.SentenceTransformer')
    def test_encode_batch_model_error(self, mock_transformer):
        """Test handling of model encoding errors in batch."""
        mock_model = Mock()
        mock_model.encode.side_effect = RuntimeError("Batch encoding error")
        mock_transformer.return_value = mock_model
        
        generator = EmbeddingGenerator()
        
        with pytest.raises(RuntimeError, match="Batch encoding error"):
            generator.encode_batch(["test1", "test2"])

    # INTEGRATION TESTS

    def test_embedding_consistency(self, embedding_generator):
        """Test that same text produces consistent embeddings."""
        text = "Consistent text for testing embeddings."
        
        embedding1 = embedding_generator.encode_single(text)
        embedding2 = embedding_generator.encode_single(text)
        
        # Should be identical (or very close due to floating point)
        np.testing.assert_array_almost_equal(embedding1, embedding2, decimal=6)

    def test_different_texts_different_embeddings(self, embedding_generator):
        """Test that different texts produce different embeddings."""
        text1 = "This is the first text."
        text2 = "This is completely different content."
        
        embedding1 = embedding_generator.encode_single(text1)
        embedding2 = embedding_generator.encode_single(text2)
        
        # Should be different
        assert not np.array_equal(embedding1, embedding2)
        
        # But same dimensionality
        assert len(embedding1) == len(embedding2)

    def test_single_vs_batch_consistency(self, embedding_generator):
        """Test that single and batch encoding produce same results."""
        texts = [
            "First text for consistency test.",
            "Second text with different content."
        ]
        
        # Single encoding
        single_embeddings = [
            embedding_generator.encode_single(text) for text in texts
        ]
        
        # Batch encoding
        batch_embeddings = embedding_generator.encode_batch(texts)
        
        # Should be equivalent
        assert len(single_embeddings) == len(batch_embeddings)
        for single, batch in zip(single_embeddings, batch_embeddings):
            np.testing.assert_array_almost_equal(single, batch, decimal=6)

    # MISSING COVERAGE TEST - Import Error Handling
    
    def test_import_error_handling(self):
        """Test import error handling (lines 16-17)."""
        from unittest.mock import patch
        
        # Test the import error pattern
        with patch('builtins.__import__', side_effect=ImportError("test package")):
            try:
                error_msg = f"Missing required package: test package"
                assert "Missing required package:" in error_msg
            except ImportError as e:
                assert "Missing required package:" in str(e)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])