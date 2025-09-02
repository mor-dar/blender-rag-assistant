#!/usr/bin/env python3
"""
Unit tests for SemanticRetriever class.

Tests semantic search functionality, query processing, and result formatting.
"""

import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from retrieval.retriever import SemanticRetriever, RetrievalResult


class TestRetrievalResult:
    """Test suite for RetrievalResult dataclass."""

    def test_retrieval_result_creation(self):
        """Test RetrievalResult creation."""
        text = "Sample document text"
        metadata = {"title": "Sample Doc", "section": "test"}
        score = 0.85
        
        result = RetrievalResult(text=text, metadata=metadata, score=score)
        
        assert result.text == text
        assert result.metadata == metadata
        assert result.score == score

    def test_retrieval_result_equality(self):
        """Test RetrievalResult equality comparison."""
        result1 = RetrievalResult("text", {"key": "value"}, 0.9)
        result2 = RetrievalResult("text", {"key": "value"}, 0.9)
        result3 = RetrievalResult("different", {"key": "value"}, 0.9)
        
        assert result1 == result2
        assert result1 != result3


class TestSemanticRetriever:
    """Test suite for SemanticRetriever."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_components(self):
        """Mock embedding generator and vector store."""
        with patch('retrieval.retriever.EmbeddingGenerator') as mock_embedding, \
             patch('retrieval.retriever.VectorStore') as mock_vector_store:
            
            # Setup mock embedding generator
            mock_embed_instance = Mock()
            mock_embed_instance.encode_single.return_value = np.array([0.1, 0.2, 0.3])
            mock_embed_instance.get_embedding_dimension.return_value = 384
            mock_embed_instance.get_model_info.return_value = {
                "model_name": "test-model",
                "embedding_dimension": 384
            }
            mock_embedding.return_value = mock_embed_instance
            
            # Setup mock vector store
            mock_store_instance = Mock()
            mock_store_instance.list_collections.return_value = ["blender_docs_demo"]
            mock_store_instance.get_database_info.return_value = {
                "total_collections": 1,
                "collections": {"blender_docs_demo": {"count": 100}}
            }
            mock_vector_store.return_value = mock_store_instance
            
            yield mock_embed_instance, mock_store_instance

    @pytest.fixture
    def retriever(self, temp_db_path, mock_components):
        """Create SemanticRetriever instance with mocked components."""
        mock_embedding, mock_vector_store = mock_components
        return SemanticRetriever(temp_db_path, "test-model", "blender_docs_demo")

    # INITIALIZATION TESTS

    def test_initialization_success(self, temp_db_path, mock_components):
        """Test successful SemanticRetriever initialization."""
        retriever = SemanticRetriever(temp_db_path, "test-model", "test_collection")
        
        assert retriever.db_path == temp_db_path
        assert retriever.collection_name == "test_collection"
        assert retriever.embedding_generator is not None
        assert retriever.vector_store is not None

    def test_initialization_with_defaults(self, temp_db_path, mock_components):
        """Test initialization with default parameters."""
        retriever = SemanticRetriever(temp_db_path)
        
        assert retriever.collection_name == "blender_docs"  # Uses config default

    def test_validation_collection_exists(self, retriever, mock_components):
        """Test collection validation when collection exists."""
        mock_embedding, mock_vector_store = mock_components
        mock_vector_store.list_collections.return_value = ["blender_docs_demo"]
        
        assert retriever._validate_collection() is True

    def test_validation_collection_missing(self, retriever, mock_components):
        """Test collection validation when collection is missing."""
        mock_embedding, mock_vector_store = mock_components
        mock_vector_store.list_collections.return_value = ["other_collection"]
        
        assert retriever._validate_collection() is False

    # RETRIEVAL TESTS

    def test_retrieve_success(self, retriever, mock_components):
        """Test successful document retrieval."""
        mock_embedding, mock_vector_store = mock_components
        
        # Setup mock query results
        mock_results = {
            "documents": [["Document 1 content", "Document 2 content"]],
            "metadatas": [[
                {"title": "Doc 1", "section": "modeling"},
                {"title": "Doc 2", "section": "animation"}
            ]],
            "distances": [[0.2, 0.4]]
        }
        mock_vector_store.query_collection.return_value = mock_results
        
        results = retriever.retrieve("test query", k=2)
        
        assert len(results) == 2
        assert isinstance(results[0], RetrievalResult)
        assert results[0].text == "Document 1 content"
        assert results[0].metadata["title"] == "Doc 1"
        assert results[0].score == 0.8  # 1 - 0.2 distance
        assert results[1].score == 0.6  # 1 - 0.4 distance

    def test_retrieve_empty_query(self, retriever):
        """Test retrieval with empty query returns empty results."""
        results = retriever.retrieve("")
        assert results == []

    def test_retrieve_whitespace_query(self, retriever):
        """Test retrieval with whitespace-only query returns empty results."""
        results = retriever.retrieve("   \n\t   ")
        assert results == []

    def test_retrieve_with_min_score_filter(self, retriever, mock_components):
        """Test retrieval with minimum score filtering."""
        mock_embedding, mock_vector_store = mock_components
        
        # Setup results with varying distances
        mock_results = {
            "documents": [["Doc 1", "Doc 2", "Doc 3"]],
            "metadatas": [[{"title": f"Doc {i}"} for i in range(1, 4)]],
            "distances": [[0.1, 0.5, 0.8]]  # Scores: 0.9, 0.5, 0.2
        }
        mock_vector_store.query_collection.return_value = mock_results
        
        results = retriever.retrieve("test", k=3, min_score=0.6)
        
        # Should only return documents with score >= 0.6 (first doc with score 0.9)
        assert len(results) == 1
        assert results[0].score == 0.9

    def test_retrieve_with_custom_collection(self, retriever, mock_components):
        """Test retrieval with custom collection name."""
        mock_embedding, mock_vector_store = mock_components
        
        mock_results = {
            "documents": [["Test content"]],
            "metadatas": [[{"title": "Test"}]],
            "distances": [[0.3]]
        }
        mock_vector_store.query_collection.return_value = mock_results
        
        results = retriever.retrieve("query", collection_name="custom_collection")
        
        # Verify custom collection was used
        mock_vector_store.query_collection.assert_called_once()
        call_args = mock_vector_store.query_collection.call_args
        assert call_args[1]["collection_name"] == "custom_collection"

    def test_retrieve_with_metadata_filter(self, retriever, mock_components):
        """Test retrieval with metadata filtering."""
        mock_embedding, mock_vector_store = mock_components
        
        mock_results = {
            "documents": [["Filtered content"]],
            "metadatas": [[{"title": "Filtered", "section": "modeling"}]],
            "distances": [[0.2]]
        }
        mock_vector_store.query_collection.return_value = mock_results
        
        metadata_filter = {"section": "modeling"}
        results = retriever.retrieve("query", metadata_filter=metadata_filter)
        
        # Verify filter was passed
        mock_vector_store.query_collection.assert_called_once()
        call_args = mock_vector_store.query_collection.call_args
        assert call_args[1]["where"] == metadata_filter

    def test_retrieve_no_results(self, retriever, mock_components):
        """Test retrieval when no results are found."""
        mock_embedding, mock_vector_store = mock_components
        mock_vector_store.query_collection.return_value = None
        
        results = retriever.retrieve("no results query")
        assert results == []

    def test_retrieve_empty_results(self, retriever, mock_components):
        """Test retrieval when empty results are returned."""
        mock_embedding, mock_vector_store = mock_components
        mock_vector_store.query_collection.return_value = {"documents": []}
        
        results = retriever.retrieve("empty results query")
        assert results == []

    def test_retrieve_error_handling(self, retriever, mock_components):
        """Test retrieval error handling."""
        mock_embedding, mock_vector_store = mock_components
        mock_embedding.encode_single.side_effect = Exception("Encoding error")
        
        results = retriever.retrieve("error query")
        assert results == []

    # CONTEXT RETRIEVAL TESTS

    def test_retrieve_with_context_basic(self, retriever, mock_components):
        """Test context retrieval (currently returns primary results)."""
        mock_embedding, mock_vector_store = mock_components
        
        mock_results = {
            "documents": [["Context document"]],
            "metadatas": [[{"title": "Context Doc"}]],
            "distances": [[0.3]]
        }
        mock_vector_store.query_collection.return_value = mock_results
        
        results = retriever.retrieve_with_context("query", k=1, context_window=2)
        
        # Currently should return same as regular retrieve
        assert len(results) == 1
        assert results[0].text == "Context document"

    def test_retrieve_with_context_zero_window(self, retriever, mock_components):
        """Test context retrieval with zero context window."""
        mock_embedding, mock_vector_store = mock_components
        
        mock_results = {
            "documents": [["Primary document"]],
            "metadatas": [[{"title": "Primary"}]],
            "distances": [[0.2]]
        }
        mock_vector_store.query_collection.return_value = mock_results
        
        results = retriever.retrieve_with_context("query", context_window=0)
        
        # Should return primary results
        assert len(results) == 1

    # METADATA SEARCH TESTS

    def test_search_by_metadata_success(self, retriever, mock_components):
        """Test successful metadata-only search."""
        mock_embedding, mock_vector_store = mock_components
        
        mock_results = {
            "documents": [["Metadata filtered doc"]],
            "metadatas": [[{"title": "Filtered", "section": "target_section"}]]
        }
        mock_vector_store.query_collection.return_value = mock_results
        
        metadata_filter = {"section": "target_section"}
        results = retriever.search_by_metadata(metadata_filter, k=5)
        
        assert len(results) == 1
        assert results[0].text == "Metadata filtered doc"
        assert results[0].score == 0.0  # No semantic scoring
        assert results[0].metadata["section"] == "target_section"

    def test_search_by_metadata_no_results(self, retriever, mock_components):
        """Test metadata search with no results."""
        mock_embedding, mock_vector_store = mock_components
        mock_vector_store.query_collection.return_value = None
        
        results = retriever.search_by_metadata({"nonexistent": "value"})
        assert results == []

    def test_search_by_metadata_error(self, retriever, mock_components):
        """Test metadata search error handling."""
        mock_embedding, mock_vector_store = mock_components
        mock_vector_store.query_collection.side_effect = Exception("Search error")
        
        results = retriever.search_by_metadata({"section": "test"})
        assert results == []

    # SIMILARITY SEARCH TESTS

    def test_get_similar_documents_success(self, retriever, mock_components):
        """Test finding similar documents."""
        mock_embedding, mock_vector_store = mock_components
        
        mock_results = {
            "documents": [["Original document", "Similar document 1", "Similar document 2"]],
            "metadatas": [[
                {"title": "Original"},
                {"title": "Similar 1"},
                {"title": "Similar 2"}
            ]],
            "distances": [[0.0, 0.2, 0.3]]
        }
        mock_vector_store.query_collection.return_value = mock_results
        
        results = retriever.get_similar_documents("Original document", k=2)
        
        # Should exclude exact match and return 2 similar docs
        assert len(results) == 2
        assert results[0].text == "Similar document 1"
        assert results[1].text == "Similar document 2"

    def test_get_similar_documents_include_exact(self, retriever, mock_components):
        """Test finding similar documents including exact matches."""
        mock_embedding, mock_vector_store = mock_components
        
        mock_results = {
            "documents": [["Original", "Similar 1", "Similar 2"]],
            "metadatas": [[{"title": f"Doc {i}"} for i in range(3)]],
            "distances": [[0.0, 0.2, 0.3]]
        }
        mock_vector_store.query_collection.return_value = mock_results
        
        results = retriever.get_similar_documents("Original", k=2, exclude_exact_match=False)
        
        # Should include exact match
        assert len(results) == 2
        assert results[0].text == "Original"

    # CONFIGURATION AND INFO TESTS

    def test_get_retriever_info(self, retriever, mock_components):
        """Test getting retriever information."""
        mock_embedding, mock_vector_store = mock_components
        
        info = retriever.get_retriever_info()
        
        assert "default_collection" in info
        assert "database_path" in info
        assert "embedding_model" in info
        assert "database_info" in info
        assert "available_collections" in info
        
        assert info["default_collection"] == "blender_docs_demo"
        assert str(info["database_path"]) == str(retriever.db_path)

    def test_set_default_collection_success(self, retriever, mock_components):
        """Test setting default collection successfully."""
        mock_embedding, mock_vector_store = mock_components
        mock_vector_store.list_collections.return_value = ["new_collection"]
        
        success = retriever.set_default_collection("new_collection")
        
        assert success is True
        assert retriever.collection_name == "new_collection"

    def test_set_default_collection_nonexistent(self, retriever, mock_components):
        """Test setting default collection that doesn't exist."""
        mock_embedding, mock_vector_store = mock_components
        mock_vector_store.list_collections.return_value = ["other_collection"]
        
        success = retriever.set_default_collection("nonexistent_collection")
        
        assert success is False
        assert retriever.collection_name == "blender_docs_demo"  # Unchanged

    # HEALTH CHECK TESTS

    def test_health_check_healthy(self, retriever, mock_components):
        """Test health check when system is healthy."""
        mock_embedding, mock_vector_store = mock_components
        mock_vector_store.list_collections.return_value = ["blender_docs_demo"]
        mock_embedding.encode_single.return_value = np.array([0.1, 0.2, 0.3])
        
        health = retriever.health_check()
        
        assert health["status"] == "healthy"
        assert health["issues"] == []

    def test_health_check_missing_collection(self, retriever, mock_components):
        """Test health check when default collection is missing."""
        mock_embedding, mock_vector_store = mock_components
        mock_vector_store.list_collections.return_value = ["other_collection"]
        mock_embedding.encode_single.return_value = np.array([0.1, 0.2, 0.3])
        
        health = retriever.health_check()
        
        assert health["status"] == "unhealthy"
        assert len(health["issues"]) > 0
        assert any("collection" in issue.lower() for issue in health["issues"])

    def test_health_check_no_collections(self, retriever, mock_components):
        """Test health check when no collections exist."""
        mock_embedding, mock_vector_store = mock_components
        mock_vector_store.list_collections.return_value = []
        mock_embedding.encode_single.return_value = np.array([0.1, 0.2, 0.3])
        
        health = retriever.health_check()
        
        assert health["status"] == "unhealthy"
        assert any("No collections" in issue for issue in health["issues"])

    def test_health_check_database_error(self, retriever, mock_components):
        """Test health check when database access fails."""
        mock_embedding, mock_vector_store = mock_components
        mock_vector_store.list_collections.side_effect = Exception("DB error")
        
        health = retriever.health_check()
        
        assert health["status"] == "unhealthy"
        assert any("Database access error" in issue for issue in health["issues"])

    def test_health_check_embedding_error(self, retriever, mock_components):
        """Test health check when embedding model fails."""
        mock_embedding, mock_vector_store = mock_components
        mock_vector_store.list_collections.return_value = ["blender_docs_demo"]
        mock_embedding.encode_single.side_effect = Exception("Embedding error")
        
        health = retriever.health_check()
        
        assert health["status"] == "unhealthy"
        assert any("Embedding model error" in issue for issue in health["issues"])

    def test_health_check_empty_embeddings(self, retriever, mock_components):
        """Test health check when embedding model returns empty results."""
        mock_embedding, mock_vector_store = mock_components
        mock_vector_store.list_collections.return_value = ["blender_docs_demo"]
        mock_embedding.encode_single.return_value = np.array([])
        
        health = retriever.health_check()
        
        assert health["status"] == "unhealthy"
        assert any("not working properly" in issue for issue in health["issues"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])