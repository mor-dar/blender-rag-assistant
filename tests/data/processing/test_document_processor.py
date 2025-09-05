#!/usr/bin/env python3
"""
Unit tests for DocumentProcessor class.

Tests text extraction, chunking, and HTML processing.
"""

import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, Mock
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

from data.processing.document_processor import DocumentProcessor


class TestDocumentProcessor:
    """Test suite for DocumentProcessor."""

    @pytest.fixture
    def test_config(self):
        """Standard test configuration."""
        return {
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size": 100,
            "chunk_overlap": 20
        }

    @pytest.fixture
    def processor(self, test_config):
        """Create DocumentProcessor instance."""
        return DocumentProcessor(test_config)

    # TEXT EXTRACTION TESTS

    def test_extract_text_from_html_basic(self, processor, sample_html):
        """Test basic HTML text extraction."""
        file_path = Path("/fake/path/test.html")
        
        text, metadata = processor.extract_text_from_html(sample_html, file_path)
        
        # Should extract main content
        assert "Test Content" in text
        assert "test document for unit testing" in text
        assert "Subsection" in text
        
        # Should exclude navigation and scripts
        assert "Skip navigation" not in text
        assert "JavaScript to remove" not in text
        
        # Check metadata
        assert metadata["title"] == "Test Document"
        assert metadata["section"] == "unknown"  # No 'raw' in path
        assert "extracted_at" in metadata

    def test_extract_text_from_html_no_title(self, processor):
        """Test HTML without title tag."""
        html_no_title = "<html><body><p>Content without title</p></body></html>"
        file_path = Path("/test/document.html")
        
        text, metadata = processor.extract_text_from_html(html_no_title, file_path)
        
        assert text == "Content without title"
        assert metadata["title"] == "document"  # Uses filename

    def test_extract_text_from_html_malformed(self, processor):
        """Test malformed HTML handling."""
        malformed_html = "<html><body><p>Unclosed paragraph<div>Mixed content</html>"
        file_path = Path("/test/broken.html")
        
        text, metadata = processor.extract_text_from_html(malformed_html, file_path)
        
        # BeautifulSoup should handle gracefully
        assert "Unclosed paragraph" in text
        assert "Mixed content" in text

    def test_extract_text_from_html_empty(self, processor):
        """Test empty HTML handling."""
        empty_html = "<html></html>"
        file_path = Path("/test/empty.html")
        
        text, metadata = processor.extract_text_from_html(empty_html, file_path)
        
        assert text == ""
        assert metadata["title"] == "empty"

    # SECTION EXTRACTION TESTS

    def test_extract_section_from_path_with_raw(self, processor):
        """Test section extraction from path containing 'raw'."""
        path = Path("/project/data/raw/modeling/basics.html")
        
        section = processor._extract_section_from_path(path)
        
        assert section == "modeling"

    def test_extract_section_from_path_no_raw(self, processor):
        """Test section extraction from path without 'raw'."""
        path = Path("/project/documents/modeling.html")
        
        section = processor._extract_section_from_path(path)
        
        assert section == "unknown"

    # URL CONVERSION TESTS

    def test_path_to_url_conversion(self, processor):
        """Test file path to URL conversion."""
        path = Path("/data/raw/modeling/basics.html")
        
        url = processor._path_to_url(path)
        
        assert url == "https://docs.blender.org/manual/en/4.5/modeling/basics.html"

    # CHUNKING TESTS

    def test_chunk_text_basic(self, processor):
        """Test basic text chunking."""
        text = "This is a test document. " * 50  # Long enough to require multiple chunks
        metadata = {"title": "Test", "source_file": "/test.html"}
        
        chunks = processor.chunk_text(text, metadata)
        
        assert len(chunks) > 1  # Should create multiple chunks
        for chunk in chunks:
            assert "text" in chunk
            assert "metadata" in chunk
            assert chunk["metadata"]["title"] == "Test"
            assert "chunk_id" in chunk["metadata"]
            assert "token_count" in chunk["metadata"]

    def test_chunk_text_empty(self, processor):
        """Test chunking empty text."""
        text = ""
        metadata = {"title": "Empty"}
        
        chunks = processor.chunk_text(text, metadata)
        
        assert chunks == []

    def test_chunk_text_small(self, processor):
        """Test chunking text smaller than chunk size."""
        text = "Short text."
        metadata = {"title": "Short", "source_file": "/short.html"}
        
        # Test with legacy chunker for consistent behavior
        chunks = processor.chunk_text(text, metadata, use_semantic=False)
        
        assert len(chunks) == 1
        assert chunks[0]["text"] == text
        assert chunks[0]["metadata"]["chunk_index"] == 0

    def test_chunk_text_overlap(self, processor):
        """Test that chunking creates proper overlap."""
        # Create config with specific overlap
        config = {
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size": 50,
            "chunk_overlap": 10
        }
        processor_with_overlap = DocumentProcessor(config)
        
        # Long text that will definitely need multiple chunks
        text = "Word " * 100  # 500 characters, much larger than chunk size
        metadata = {"title": "Overlap Test", "source_file": "/test.html"}
        
        # Test with legacy chunker for consistent behavior
        chunks = processor_with_overlap.chunk_text(text, metadata, use_semantic=False)
        
        assert len(chunks) > 1
        # Check that chunks have proper metadata
        for i, chunk in enumerate(chunks):
            assert chunk["metadata"]["chunk_index"] == i
            assert chunk["metadata"]["start_token"] >= 0
            assert chunk["metadata"]["end_token"] > chunk["metadata"]["start_token"]

    # DOCUMENT PROCESSING TESTS

    def test_process_documents_success(self, processor, test_data_dir):
        """Test successful document processing."""
        raw_dir = test_data_dir / "raw"
        
        chunks = processor.process_documents(raw_dir)
        
        assert len(chunks) > 0
        # Should have chunks from all three test files
        titles = [chunk["metadata"]["title"] for chunk in chunks]
        assert any("Modeling" in title for title in titles)
        assert any("Animation" in title for title in titles)
        assert any("Rendering" in title for title in titles)

    def test_process_documents_empty_directory(self, processor):
        """Test processing empty directory."""
        temp_dir = tempfile.mkdtemp()
        try:
            empty_dir = Path(temp_dir) / "empty"
            empty_dir.mkdir()
            
            chunks = processor.process_documents(empty_dir)
            
            assert chunks == []
        finally:
            shutil.rmtree(temp_dir)

    def test_process_documents_with_errors(self, processor):
        """Test processing directory with some problematic files."""
        temp_dir = tempfile.mkdtemp()
        try:
            raw_dir = Path(temp_dir) / "raw"
            raw_dir.mkdir()
            
            # Create good file
            (raw_dir / "good.html").write_text("<html><body>Good content</body></html>")
            
            # Create file that will cause read error (not actually possible in this test, 
            # but we can test the error handling by mocking)
            with patch('builtins.open', side_effect=Exception("Read error")):
                chunks = processor.process_documents(raw_dir)
                # Should handle errors gracefully
                assert isinstance(chunks, list)
        finally:
            shutil.rmtree(temp_dir)

    # EDGE CASES AND ERROR HANDLING

    def test_chunk_text_very_small_chunks(self, processor):
        """Test with very small chunk size."""
        # Override config for this test
        processor.config["chunk_size"] = 5  # Very small
        processor.config["chunk_overlap"] = 1
        
        text = "This is a longer text that should be split into many small chunks."
        metadata = {"title": "Small Chunks", "source_file": "/test.html"}
        
        # Test with legacy chunker for consistent behavior
        chunks = processor.chunk_text(text, metadata, use_semantic=False)
        
        assert len(chunks) > 3  # Should create multiple small chunks
        for chunk in chunks:
            assert len(chunk["text"]) > 0

    def test_chunk_text_zero_overlap(self, processor):
        """Test chunking with zero overlap."""
        processor.config["chunk_overlap"] = 0
        
        text = "This is some test text. " * 20
        metadata = {"title": "No Overlap", "source_file": "/test.html"}
        
        # Test with legacy chunker for consistent behavior
        chunks = processor.chunk_text(text, metadata, use_semantic=False)
        
        assert len(chunks) > 1
        # Verify no overlap in text content
        for i in range(len(chunks) - 1):
            current_end = chunks[i]["metadata"]["end_token"]
            next_start = chunks[i + 1]["metadata"]["start_token"]
            assert current_end <= next_start  # No overlap

    def test_initialization_without_embedding_model(self):
        """Test initialization without embedding model (refactored)."""
        config = {"chunk_size": 100, "chunk_overlap": 20}
        processor = DocumentProcessor(config)
        
        assert processor.config == config
        assert processor.tokenizer is not None
        # No embedding model in refactored version

    def test_html_cleaning_removes_unwanted_elements(self, processor):
        """Test that HTML cleaning removes unwanted elements."""
        html_with_unwanted = """
        <html>
        <head><title>Test</title></head>
        <body>
            <nav>Navigation to remove</nav>
            <header>Header to remove</header>
            <main>
                <p>Keep this content</p>
                <script>Remove this script</script>
                <style>Remove this style</style>
            </main>
            <footer>Footer to remove</footer>
        </body>
        </html>
        """
        
        file_path = Path("/test/clean.html")
        text, metadata = processor.extract_text_from_html(html_with_unwanted, file_path)
        
        # Should keep main content
        assert "Keep this content" in text
        
        # Should remove unwanted elements
        assert "Navigation to remove" not in text
        assert "Header to remove" not in text
        assert "Footer to remove" not in text
        assert "Remove this script" not in text
        assert "Remove this style" not in text

    # ENHANCED METADATA TESTS

    def test_subsection_metadata_extraction(self, processor):
        """Test extraction of subsection metadata from HTML hierarchy."""
        html_with_hierarchy = """
        <html>
        <head><title>Blender Animation Guide</title></head>
        <body>
            <main>
                <h1>Animation in Blender</h1>
                <h2>Keyframes</h2>
                <p>Learn about keyframe animation.</p>
                <h3>Insert Keyframes</h3>
                <p>Press I to insert keyframes.</p>
                <h2>Timeline</h2>
                <p>The timeline controls playback.</p>
            </main>
        </body>
        </html>
        """
        
        file_path = Path("/data/raw/animation/keyframes.html")
        text, metadata = processor.extract_text_from_html(html_with_hierarchy, file_path)
        
        # Check enhanced metadata fields
        assert "subsection" in metadata
        assert "heading_hierarchy_json" in metadata
        assert "content_type" in metadata
        
        # Check subsection extraction (should be first h2)
        assert metadata["subsection"] == "Keyframes"
        
        # Check heading hierarchy (now serialized as JSON)
        import json
        headings = json.loads(metadata["heading_hierarchy_json"])
        assert len(headings) == 4  # h1, 2*h2, h3
        assert headings[0]["level"] == 1
        assert headings[0]["text"] == "Animation in Blender"
        assert headings[1]["level"] == 2
        assert headings[1]["text"] == "Keyframes"

    def test_content_type_classification(self, processor):
        """Test automatic content type classification."""
        test_cases = [
            # Procedural content
            ("<h2>How to Model in Blender</h2><p>Step by step tutorial.</p>", "procedural"),
            # Reference content  
            ("<h2>Tool Properties</h2><p>Settings and options.</p>", "reference"),
            # Conceptual content
            ("<h2>Overview of Rendering</h2><p>Introduction to concepts.</p>", "conceptual"),
            # Code content
            ("<pre><code>bpy.ops.mesh.primitive_cube_add()</code></pre>", "code"),
            # Structured content (multiple lists)
            ("<ul><li>Item 1</li></ul><ol><li>Step 1</li></ol><ul><li>Item 2</li></ul>", "structured")
        ]
        
        for html_content, expected_type in test_cases:
            full_html = f"<html><body><main>{html_content}</main></body></html>"
            file_path = Path("/test/content_type.html")
            
            text, metadata = processor.extract_text_from_html(full_html, file_path)
            assert metadata["content_type"] == expected_type, f"Expected {expected_type}, got {metadata['content_type']}"

    def test_heading_hierarchy_with_ids_and_classes(self, processor):
        """Test that heading IDs and classes are captured."""
        html_with_attributes = """
        <html>
        <body>
            <main>
                <h1 id="main-title" class="title primary">Main Title</h1>
                <h2 id="section-1" class="section-header">Section 1</h2>
                <h3>Subsection without attributes</h3>
            </main>
        </body>
        </html>
        """
        
        file_path = Path("/test/attributes.html")
        text, metadata = processor.extract_text_from_html(html_with_attributes, file_path)
        
        import json
        headings = json.loads(metadata["heading_hierarchy_json"])
        
        # Check first heading has ID and classes
        assert headings[0]["id"] == "main-title"
        assert "title" in headings[0]["classes"]
        assert "primary" in headings[0]["classes"]
        
        # Check second heading has ID
        assert headings[1]["id"] == "section-1"
        assert "section-header" in headings[1]["classes"]
        
        # Check third heading has empty attributes
        assert headings[2]["id"] == ""
        assert headings[2]["classes"] == []

    def test_empty_headings_ignored(self, processor):
        """Test that empty headings are ignored."""
        html_with_empty_headings = """
        <html>
        <body>
            <main>
                <h1>Valid Title</h1>
                <h2></h2>
                <h3>   </h3>
                <h2>Valid Subtitle</h2>
            </main>
        </body>
        </html>
        """
        
        file_path = Path("/test/empty_headings.html")
        text, metadata = processor.extract_text_from_html(html_with_empty_headings, file_path)
        
        import json
        headings = json.loads(metadata["heading_hierarchy_json"])
        
        # Should only have 2 valid headings
        assert len(headings) == 2
        assert headings[0]["text"] == "Valid Title"
        assert headings[1]["text"] == "Valid Subtitle"
        
        # Subsection should be the first h2 with content
        assert metadata["subsection"] == "Valid Subtitle"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])