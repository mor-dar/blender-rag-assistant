#!/usr/bin/env python3
"""
Unit tests for VectorStore class.

Tests ChromaDB operations, collection management, and persistence.
"""

import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from retrieval.vector_store import VectorStore


class TestVectorStore:
    """Test suite for VectorStore."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture 
    def vector_store(self, temp_db_path):
        """Create VectorStore instance."""
        return VectorStore(temp_db_path)

    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return {
            "documents": ["Document 1 content", "Document 2 content", "Document 3 content"],
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            "metadatas": [
                {"title": "Doc 1", "section": "modeling"},
                {"title": "Doc 2", "section": "animation"}, 
                {"title": "Doc 3", "section": "rendering"}
            ],
            "ids": ["doc1", "doc2", "doc3"]
        }

    # INITIALIZATION TESTS

    def test_initialization_success(self, temp_db_path):
        """Test successful VectorStore initialization."""
        vector_store = VectorStore(temp_db_path)
        
        assert vector_store.db_path == temp_db_path
        assert vector_store.client is not None
        assert vector_store.logger is not None
        assert temp_db_path.exists()

    def test_initialization_creates_directory(self):
        """Test initialization creates database directory."""
        temp_dir = tempfile.mkdtemp()
        try:
            db_path = Path(temp_dir) / "nested" / "db" / "path"
            vector_store = VectorStore(db_path)
            
            assert db_path.exists()
            assert vector_store.db_path == db_path
        finally:
            shutil.rmtree(temp_dir)

    # COLLECTION MANAGEMENT TESTS

    def test_create_collection_success(self, vector_store):
        """Test successful collection creation."""
        collection_name = "test_collection"
        metadata = {"tier": "demo", "description": "Test collection"}
        
        collection = vector_store.create_collection(collection_name, metadata)
        
        assert collection is not None
        
        # Verify collection was created
        collections = vector_store.list_collections()
        assert collection_name in collections

    def test_create_collection_replace_existing(self, vector_store):
        """Test creating collection replaces existing one."""
        collection_name = "replace_test"
        
        # Create first collection
        collection1 = vector_store.create_collection(collection_name, {"version": "1"})
        assert collection1 is not None
        
        # Create second collection with same name (should replace)
        collection2 = vector_store.create_collection(collection_name, {"version": "2"}, replace=True)
        assert collection2 is not None
        
        # Should still only have one collection with this name
        collections = vector_store.list_collections()
        assert collections.count(collection_name) == 1

    def test_create_collection_no_replace(self, vector_store):
        """Test creating collection without replacing existing one."""
        collection_name = "no_replace_test"
        
        # Create first collection
        vector_store.create_collection(collection_name, {"version": "1"})
        
        # ChromaDB will raise an error if we try to create with same name without replace
        # So our VectorStore should handle this gracefully
        with pytest.raises(Exception):
            vector_store.create_collection(collection_name, {"version": "2"}, replace=False)

    def test_get_collection_success(self, vector_store):
        """Test successfully getting existing collection."""
        collection_name = "get_test"
        vector_store.create_collection(collection_name)
        
        collection = vector_store.get_collection(collection_name)
        assert collection is not None

    def test_get_collection_nonexistent(self, vector_store):
        """Test getting non-existent collection returns None."""
        collection = vector_store.get_collection("nonexistent_collection")
        assert collection is None

    def test_delete_collection_success(self, vector_store):
        """Test successful collection deletion."""
        collection_name = "delete_test"
        vector_store.create_collection(collection_name)
        
        # Verify collection exists
        assert collection_name in vector_store.list_collections()
        
        # Delete collection
        success = vector_store.delete_collection(collection_name)
        assert success is True
        
        # Verify collection is gone
        assert collection_name not in vector_store.list_collections()

    def test_delete_collection_nonexistent(self, vector_store):
        """Test deleting non-existent collection."""
        success = vector_store.delete_collection("nonexistent_collection")
        assert success is False

    def test_list_collections_empty(self, vector_store):
        """Test listing collections when none exist."""
        collections = vector_store.list_collections()
        assert collections == []

    def test_list_collections_multiple(self, vector_store):
        """Test listing multiple collections."""
        collection_names = ["collection_1", "collection_2", "collection_3"]
        
        for name in collection_names:
            vector_store.create_collection(name)
        
        collections = vector_store.list_collections()
        for name in collection_names:
            assert name in collections

    def test_get_collection_info_success(self, vector_store, sample_documents):
        """Test getting collection information."""
        collection_name = "info_test"
        metadata = {"tier": "demo"}
        
        vector_store.create_collection(collection_name, metadata)
        
        # Add some documents
        vector_store.add_documents(
            collection_name,
            sample_documents["documents"],
            sample_documents["embeddings"], 
            sample_documents["metadatas"],
            sample_documents["ids"]
        )
        
        info = vector_store.get_collection_info(collection_name)
        
        assert "name" in info
        assert "count" in info
        assert "metadata" in info
        assert info["name"] == collection_name
        assert info["count"] == len(sample_documents["documents"])

    def test_get_collection_info_nonexistent(self, vector_store):
        """Test getting info for non-existent collection."""
        info = vector_store.get_collection_info("nonexistent_collection")
        assert "error" in info

    # DOCUMENT OPERATIONS TESTS

    def test_add_documents_success(self, vector_store, sample_documents):
        """Test successful document addition."""
        collection_name = "add_docs_test"
        vector_store.create_collection(collection_name)
        
        success = vector_store.add_documents(
            collection_name,
            sample_documents["documents"],
            sample_documents["embeddings"],
            sample_documents["metadatas"], 
            sample_documents["ids"]
        )
        
        assert success is True
        
        # Verify documents were added
        info = vector_store.get_collection_info(collection_name)
        assert info["count"] == len(sample_documents["documents"])

    def test_add_documents_nonexistent_collection(self, vector_store, sample_documents):
        """Test adding documents to non-existent collection fails."""
        success = vector_store.add_documents(
            "nonexistent_collection",
            sample_documents["documents"],
            sample_documents["embeddings"],
            sample_documents["metadatas"],
            sample_documents["ids"]
        )
        
        assert success is False

    def test_query_collection_success(self, vector_store, sample_documents):
        """Test successful collection querying."""
        collection_name = "query_test"
        vector_store.create_collection(collection_name)
        
        # Add documents
        vector_store.add_documents(
            collection_name,
            sample_documents["documents"],
            sample_documents["embeddings"],
            sample_documents["metadatas"],
            sample_documents["ids"]
        )
        
        # Query collection
        query_embedding = [[0.15, 0.25, 0.35]]  # Similar to first document
        results = vector_store.query_collection(
            collection_name,
            query_embedding,
            n_results=2
        )
        
        assert results is not None
        assert "documents" in results
        assert "metadatas" in results
        assert "distances" in results
        assert len(results["documents"][0]) <= 2  # Requested max 2 results

    def test_query_collection_with_filter(self, vector_store, sample_documents):
        """Test querying collection with metadata filter."""
        collection_name = "filter_test"
        vector_store.create_collection(collection_name)
        
        # Add documents
        vector_store.add_documents(
            collection_name,
            sample_documents["documents"],
            sample_documents["embeddings"],
            sample_documents["metadatas"],
            sample_documents["ids"]
        )
        
        # Query with filter
        query_embedding = [[0.5, 0.5, 0.5]]
        results = vector_store.query_collection(
            collection_name,
            query_embedding,
            n_results=5,
            where={"section": "modeling"}
        )
        
        assert results is not None
        # Should only return documents from modeling section
        if results.get("metadatas") and results["metadatas"][0]:
            for metadata in results["metadatas"][0]:
                assert metadata["section"] == "modeling"

    def test_query_collection_nonexistent(self, vector_store):
        """Test querying non-existent collection."""
        query_embedding = [[0.1, 0.2, 0.3]]
        results = vector_store.query_collection(
            "nonexistent_collection",
            query_embedding
        )
        
        assert results is None

    def test_update_documents_success(self, vector_store, sample_documents):
        """Test successful document updates."""
        collection_name = "update_test"
        vector_store.create_collection(collection_name)
        
        # Add initial documents
        vector_store.add_documents(
            collection_name,
            sample_documents["documents"],
            sample_documents["embeddings"],
            sample_documents["metadatas"],
            sample_documents["ids"]
        )
        
        # Update documents with new embeddings (matching dimension)
        new_documents = ["Updated document 1", "Updated document 2"]
        new_embeddings = [[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]]  # Same dimension as sample
        new_metadatas = [
            {"title": "Updated Doc 1", "section": "updated"},
            {"title": "Updated Doc 2", "section": "updated"}
        ]
        ids_to_update = ["doc1", "doc2"]
        
        success = vector_store.update_documents(
            collection_name,
            ids_to_update,
            documents=new_documents,
            embeddings=new_embeddings,
            metadatas=new_metadatas
        )
        
        assert success is True

    def test_update_documents_nonexistent_collection(self, vector_store):
        """Test updating documents in non-existent collection."""
        success = vector_store.update_documents(
            "nonexistent_collection",
            ["doc1"],
            documents=["Updated doc"]
        )
        
        assert success is False

    def test_delete_documents_success(self, vector_store, sample_documents):
        """Test successful document deletion."""
        collection_name = "delete_docs_test"
        vector_store.create_collection(collection_name)
        
        # Add documents
        vector_store.add_documents(
            collection_name,
            sample_documents["documents"],
            sample_documents["embeddings"],
            sample_documents["metadatas"],
            sample_documents["ids"]
        )
        
        # Delete some documents
        ids_to_delete = ["doc1", "doc2"]
        success = vector_store.delete_documents(collection_name, ids_to_delete)
        
        assert success is True
        
        # Verify documents were deleted
        info = vector_store.get_collection_info(collection_name)
        expected_remaining = len(sample_documents["documents"]) - len(ids_to_delete)
        assert info["count"] == expected_remaining

    def test_delete_documents_nonexistent_collection(self, vector_store):
        """Test deleting documents from non-existent collection."""
        success = vector_store.delete_documents("nonexistent_collection", ["doc1"])
        assert success is False

    # METADATA PERSISTENCE TESTS

    def test_save_build_metadata_success(self, vector_store):
        """Test successful metadata saving."""
        collection_name = "metadata_save_test"
        metadata = {
            "collection_name": collection_name,
            "chunks_added": 100,
            "built_at": "2024-01-01T00:00:00"
        }
        
        success = vector_store.save_build_metadata(collection_name, metadata)
        assert success is True
        
        # Verify file was created
        metadata_file = vector_store.db_path / f"build_metadata_{collection_name}.json"
        assert metadata_file.exists()
        
        # Verify content
        with open(metadata_file) as f:
            saved_metadata = json.load(f)
        assert saved_metadata == metadata

    def test_load_build_metadata_success(self, vector_store):
        """Test successful metadata loading."""
        collection_name = "metadata_load_test"
        metadata = {
            "collection_name": collection_name,
            "chunks_added": 50,
            "built_at": "2024-01-01T12:00:00"
        }
        
        # Save metadata first
        vector_store.save_build_metadata(collection_name, metadata)
        
        # Load metadata
        loaded_metadata = vector_store.load_build_metadata(collection_name)
        
        assert loaded_metadata == metadata

    def test_load_build_metadata_nonexistent(self, vector_store):
        """Test loading metadata for non-existent file."""
        metadata = vector_store.load_build_metadata("nonexistent_collection")
        assert metadata is None

    # DATABASE INFO TESTS

    def test_get_database_info_empty(self, vector_store):
        """Test getting database info when empty."""
        info = vector_store.get_database_info()
        
        assert "database_path" in info
        assert "total_collections" in info
        assert "collections" in info
        assert info["total_collections"] == 0
        assert info["collections"] == {}

    def test_get_database_info_with_collections(self, vector_store, sample_documents):
        """Test getting database info with collections."""
        # Create collections
        collections = ["collection_1", "collection_2"]
        for collection_name in collections:
            vector_store.create_collection(collection_name, {"tier": "test"})
            vector_store.add_documents(
                collection_name,
                sample_documents["documents"][:1],  # Just one document
                sample_documents["embeddings"][:1],
                sample_documents["metadatas"][:1],
                sample_documents["ids"][:1]
            )
        
        info = vector_store.get_database_info()
        
        assert info["total_collections"] == len(collections)
        assert len(info["collections"]) == len(collections)
        
        for collection_name in collections:
            assert collection_name in info["collections"]
            assert info["collections"][collection_name]["count"] == 1

    # ERROR HANDLING TESTS

    @patch('retrieval.vector_store.chromadb.PersistentClient')
    def test_initialization_chromadb_error(self, mock_client_class, temp_db_path):
        """Test handling ChromaDB initialization errors."""
        mock_client_class.side_effect = Exception("ChromaDB error")
        
        with pytest.raises(Exception, match="ChromaDB error"):
            VectorStore(temp_db_path)

    @patch('retrieval.vector_store.json.dump')
    def test_save_build_metadata_write_error(self, mock_json_dump, vector_store):
        """Test handling file write errors in save_build_metadata."""
        mock_json_dump.side_effect = IOError("Write error")
        
        success = vector_store.save_build_metadata("test", {"data": "test"})
        assert success is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])