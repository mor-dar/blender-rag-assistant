#!/usr/bin/env python3
"""
Unit tests for VectorDBBuilder class.

Tests include smoke tests, edge cases, and error handling.
"""

import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

from data.processing.vector_builder import VectorDBBuilder


class TestVectorDBBuilder:
    """Test suite for VectorDBBuilder."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def temp_raw_dir(self):
        """Create temporary raw documents directory with sample HTML."""
        temp_dir = tempfile.mkdtemp()
        raw_dir = Path(temp_dir) / "raw"
        raw_dir.mkdir()
        
        # Create sample HTML files
        sample_html = """
        <html>
        <head><title>Test Blender Doc</title></head>
        <body>
            <main>
                <h1>Modeling Basics</h1>
                <p>This is how you model in Blender.</p>
                <p>Use the Tab key to enter Edit mode.</p>
            </main>
        </body>
        </html>
        """
        
        (raw_dir / "modeling.html").write_text(sample_html)
        (raw_dir / "animation.html").write_text(sample_html.replace("Modeling", "Animation"))
        
        yield raw_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def test_config(self):
        """Standard test configuration."""
        return {
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size": 100,  # Small for testing
            "chunk_overlap": 20,
            "batch_size": 5
        }

    # SMOKE TESTS
    
    def test_initialization_success(self, test_config, temp_db_path):
        """Test successful VectorDBBuilder initialization."""
        builder = VectorDBBuilder(test_config, temp_db_path)
        
        assert builder.config == test_config
        assert builder.db_path == temp_db_path
        assert builder.vector_store is not None
        assert builder.embedding_generator is not None
        assert builder.processor is not None

    def test_build_collection_success(self, test_config, temp_db_path, temp_raw_dir):
        """Test successful collection building."""
        builder = VectorDBBuilder(test_config, temp_db_path)
        
        result = builder.build_collection(
            collection_name="test_collection",
            raw_dir=temp_raw_dir,
            collection_metadata={"test": True}
        )
        
        assert result["collection_name"] == "test_collection"
        assert result["chunks_added"] > 0
        assert "built_at" in result
        assert "config" in result

    def test_list_collections_success(self, test_config, temp_db_path, temp_raw_dir):
        """Test listing collections."""
        builder = VectorDBBuilder(test_config, temp_db_path)
        
        # Build a test collection
        builder.build_collection("test_collection", temp_raw_dir)
        
        # List collections
        collections = builder.list_collections()
        assert "test_collection" in collections

    def test_get_collection_info_success(self, test_config, temp_db_path, temp_raw_dir):
        """Test getting collection information."""
        builder = VectorDBBuilder(test_config, temp_db_path)
        
        # Build a test collection
        builder.build_collection("test_collection", temp_raw_dir)
        
        # Get collection info
        info = builder.get_collection_info("test_collection")
        assert info["name"] == "test_collection"
        assert info["count"] > 0
        assert "metadata" in info

    # EDGE CASES

    def test_build_collection_empty_directory(self, test_config, temp_db_path):
        """Test building collection from empty directory."""
        empty_dir = Path(tempfile.mkdtemp())
        try:
            builder = VectorDBBuilder(test_config, temp_db_path)
            
            result = builder.build_collection("empty_collection", empty_dir)
            
            assert result["chunks_added"] == 0
            assert result["collection_name"] == "empty_collection"
        finally:
            shutil.rmtree(empty_dir)

    def test_build_collection_nonexistent_directory(self, test_config, temp_db_path):
        """Test building collection from non-existent directory."""
        nonexistent_dir = Path("/definitely/does/not/exist")
        
        builder = VectorDBBuilder(test_config, temp_db_path)
        
        result = builder.build_collection("test_collection", nonexistent_dir)
        assert result["chunks_added"] == 0

    def test_build_collection_replaces_existing(self, test_config, temp_db_path, temp_raw_dir):
        """Test that building overwrites existing collection."""
        builder = VectorDBBuilder(test_config, temp_db_path)
        
        # Build first collection
        result1 = builder.build_collection("duplicate_collection", temp_raw_dir)
        chunks_first = result1["chunks_added"]
        
        # Build again with same name
        result2 = builder.build_collection("duplicate_collection", temp_raw_dir)
        chunks_second = result2["chunks_added"]
        
        # Should have same number of chunks (replaced, not added to)
        assert chunks_first == chunks_second

    def test_build_collection_invalid_html(self, test_config, temp_db_path):
        """Test handling of malformed HTML files."""
        temp_dir = tempfile.mkdtemp()
        try:
            raw_dir = Path(temp_dir) / "raw"
            raw_dir.mkdir()
            
            # Create invalid HTML
            (raw_dir / "broken.html").write_text("<html><title>Broken</title><body><p>Unclosed paragraph")
            
            builder = VectorDBBuilder(test_config, temp_db_path)
            result = builder.build_collection("broken_collection", raw_dir)
            
            # Should handle gracefully
            assert isinstance(result, dict)
            assert "chunks_added" in result
            
        finally:
            shutil.rmtree(temp_dir)

    def test_get_collection_info_nonexistent(self, test_config, temp_db_path):
        """Test getting info for non-existent collection."""
        builder = VectorDBBuilder(test_config, temp_db_path)
        
        info = builder.get_collection_info("nonexistent_collection")
        assert "error" in info

    def test_very_small_chunks(self, temp_db_path, temp_raw_dir):
        """Test with very small chunk size."""
        config = {
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size": 10,  # Very small
            "chunk_overlap": 2,
            "batch_size": 1
        }
        
        builder = VectorDBBuilder(config, temp_db_path)
        result = builder.build_collection("tiny_chunks", temp_raw_dir)
        
        # Should create many small chunks
        assert result["chunks_added"] > 5

    def test_zero_overlap_chunks(self, temp_db_path, temp_raw_dir):
        """Test chunking with zero overlap."""
        config = {
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size": 50,
            "chunk_overlap": 0,  # No overlap
            "batch_size": 5
        }
        
        builder = VectorDBBuilder(config, temp_db_path)
        result = builder.build_collection("no_overlap", temp_raw_dir)
        
        assert result["chunks_added"] > 0

    # ERROR HANDLING TESTS

    @patch('retrieval.vector_store.chromadb.PersistentClient')
    def test_chromadb_initialization_failure(self, mock_client_class, test_config, temp_db_path):
        """Test handling of ChromaDB initialization failure."""
        mock_client_class.side_effect = Exception("Database connection failed")
        
        with pytest.raises(Exception, match="Database connection failed"):
            VectorDBBuilder(test_config, temp_db_path)

    def test_invalid_config_missing_fields(self, temp_db_path):
        """Test handling of invalid configuration."""
        invalid_config = {"embedding_model": "all-MiniLM-L6-v2"}  # Missing required fields
        
        builder = VectorDBBuilder(invalid_config, temp_db_path)
        
        # Should handle gracefully when accessing missing config
        temp_dir = tempfile.mkdtemp()
        try:
            raw_dir = Path(temp_dir)
            # This should not crash but may have reduced functionality
            result = builder.build_collection("test", raw_dir)
            assert isinstance(result, dict)
        finally:
            shutil.rmtree(temp_dir)

    @patch('data.processing.vector_builder.DocumentProcessor')
    def test_document_processing_failure(self, mock_processor_class, test_config, temp_db_path, temp_raw_dir):
        """Test handling of document processing failure."""
        # Mock processor to raise exception
        mock_processor = Mock()
        mock_processor.process_documents.side_effect = Exception("Processing failed")
        mock_processor_class.return_value = mock_processor
        
        builder = VectorDBBuilder(test_config, temp_db_path)
        builder.processor = mock_processor
        
        with pytest.raises(Exception, match="Processing failed"):
            builder.build_collection("test_collection", temp_raw_dir)

    def test_metadata_persistence(self, test_config, temp_db_path, temp_raw_dir):
        """Test that build metadata is properly saved."""
        builder = VectorDBBuilder(test_config, temp_db_path)
        
        collection_name = "metadata_test"
        collection_metadata = {"tier": "demo", "description": "Test collection"}
        
        result = builder.build_collection(collection_name, temp_raw_dir, collection_metadata)
        
        # Check metadata file was created
        metadata_file = temp_db_path / f"build_metadata_{collection_name}.json"
        assert metadata_file.exists()
        
        # Check metadata contents
        with open(metadata_file) as f:
            saved_metadata = json.load(f)
        
        assert saved_metadata["collection_name"] == collection_name
        assert saved_metadata["chunks_added"] == result["chunks_added"]
        assert saved_metadata["collection_metadata"]["tier"] == "demo"

    # INTEGRATION TESTS

    def test_full_workflow_demo_tier(self, test_config, temp_db_path, temp_raw_dir):
        """Test complete workflow for demo tier."""
        builder = VectorDBBuilder(test_config, temp_db_path)
        
        # Build collection
        result = builder.build_collection(
            "blender_docs_demo", 
            temp_raw_dir,
            {"tier": "demo", "description": "Demo dataset"}
        )
        
        # Verify results
        assert result["chunks_added"] > 0
        assert result["collection_name"] == "blender_docs_demo"
        
        # Verify collection exists and is queryable
        info = builder.get_collection_info("blender_docs_demo")
        assert info["count"] > 0
        assert info["metadata"]["tier"] == "demo"

    def test_multiple_collections(self, test_config, temp_db_path, temp_raw_dir):
        """Test creating multiple collections in same database."""
        builder = VectorDBBuilder(test_config, temp_db_path)
        
        # Build first collection
        result1 = builder.build_collection("collection_1", temp_raw_dir)
        
        # Build second collection
        result2 = builder.build_collection("collection_2", temp_raw_dir)
        
        # Both should exist
        collections = builder.list_collections()
        assert "collection_1" in collections
        assert "collection_2" in collections
        
        # Both should have data
        assert builder.get_collection_info("collection_1")["count"] > 0
        assert builder.get_collection_info("collection_2")["count"] > 0


# PERFORMANCE TESTS

class TestVectorDBBuilderPerformance:
    """Performance-focused tests for VectorDBBuilder."""

    def test_large_batch_processing(self, temp_db):
        """Test handling of large batch sizes."""
        config = {
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size": 100,
            "chunk_overlap": 10,
            "batch_size": 500  # Large batch
        }
        
        # Create directory with many small files
        temp_dir = tempfile.mkdtemp()
        try:
            raw_dir = Path(temp_dir) / "raw"
            raw_dir.mkdir()
            
            # Create 20 small HTML files
            for i in range(20):
                html = f"<html><body><p>Document {i} content here.</p></body></html>"
                (raw_dir / f"doc_{i}.html").write_text(html)
            
            builder = VectorDBBuilder(config, temp_db)
            result = builder.build_collection("large_batch_test", raw_dir)
            
            assert result["chunks_added"] > 0
            
        finally:
            shutil.rmtree(temp_dir)

    def test_memory_usage_with_large_files(self, standard_config, temp_db):
        """Test memory handling with large document."""
        temp_dir = tempfile.mkdtemp()
        try:
            raw_dir = Path(temp_dir) / "raw"  
            raw_dir.mkdir()
            
            # Create large HTML file
            large_content = "<html><body>" + "<p>Large document content. " * 1000 + "</p></body></html>"
            (raw_dir / "large_doc.html").write_text(large_content)
            
            builder = VectorDBBuilder(standard_config, temp_db)
            result = builder.build_collection("large_file_test", raw_dir)
            
            # Should handle large file without crashing
            assert result["chunks_added"] > 10  # Should create multiple chunks
            
        finally:
            shutil.rmtree(temp_dir)


# FIXTURES AND HELPERS

@pytest.fixture
def sample_html_content():
    """Sample HTML content for testing."""
    return """
    <html>
    <head>
        <title>Blender Modeling Guide</title>
    </head>
    <body>
        <nav>Navigation menu</nav>
        <main>
            <h1>3D Modeling in Blender</h1>
            <section>
                <h2>Basic Operations</h2>
                <p>To start modeling in Blender, first enter Edit mode by pressing Tab.</p>
                <p>Use the Extrude tool (E key) to add geometry to your mesh.</p>
                <ul>
                    <li>Select faces to extrude</li>
                    <li>Press E to activate extrude</li>
                    <li>Move mouse to extrude in normal direction</li>
                    <li>Click to confirm</li>
                </ul>
            </section>
        </main>
        <footer>Footer content</footer>
        <script>Some JavaScript</script>
    </body>
    </html>
    """

    # MISSING COVERAGE TEST - Batch Addition Failure
    
    def test_build_collection_batch_failure_logging(self, vector_builder, sample_docs):
        """Test logging when batch addition fails (line 110)."""
        from unittest.mock import patch
        
        # Mock the vector store to fail batch additions
        with patch.object(vector_builder.vector_store, 'add_documents', return_value=False):
            # This should log errors for failed batches
            collection_name = vector_builder.build_collection(sample_docs, batch_size=10)
            
            # Should still return a collection name even if some batches fail
            assert collection_name is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])