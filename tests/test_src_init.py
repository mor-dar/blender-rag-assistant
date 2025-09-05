"""
Tests for the main src package initialization.

This module tests the package-level constants and metadata.
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import src


class TestPackageMetadata:
    """Test package-level metadata and constants."""
    
    def test_version_exists(self):
        """Test that version constant exists and is a string."""
        assert hasattr(src, '__version__')
        assert isinstance(src.__version__, str)
        assert len(src.__version__) > 0
    
    def test_version_format(self):
        """Test that version follows semantic versioning pattern."""
        version = src.__version__
        
        # Basic format check: should have at least X.Y.Z
        parts = version.split('.')
        assert len(parts) >= 3
        
        # First three parts should be numeric
        for i in range(3):
            assert parts[i].isdigit(), f"Version part {i} '{parts[i]}' is not numeric"
    
    def test_author_exists(self):
        """Test that author constant exists and is a string."""
        assert hasattr(src, '__author__')
        assert isinstance(src.__author__, str)
        assert len(src.__author__) > 0
        assert src.__author__ == "Ready Tensor"
    
    def test_description_exists(self):
        """Test that description constant exists and is a string."""
        assert hasattr(src, '__description__')
        assert isinstance(src.__description__, str)
        assert len(src.__description__) > 0
    
    def test_description_content(self):
        """Test that description contains expected keywords."""
        description = src.__description__
        
        # Should mention key concepts
        assert "Retrieval-Augmented Generation" in description or "RAG" in description
        assert "Blender" in description
        assert "documentation" in description
    
    def test_package_docstring(self):
        """Test that package has a proper docstring."""
        assert src.__doc__ is not None
        assert len(src.__doc__.strip()) > 0
        assert "Blender Bot" in src.__doc__
        assert "documentation assistant" in src.__doc__
    
    def test_all_metadata_types(self):
        """Test that all metadata fields are strings."""
        metadata_fields = ['__version__', '__author__', '__description__']
        
        for field in metadata_fields:
            if hasattr(src, field):
                value = getattr(src, field)
                assert isinstance(value, str), f"{field} should be a string, got {type(value)}"
                assert len(value) > 0, f"{field} should not be empty"
    
    def test_version_immutability(self):
        """Test that version can be accessed multiple times consistently."""
        version1 = src.__version__
        version2 = src.__version__
        assert version1 == version2
    
    def test_package_importable(self):
        """Test that the package can be imported successfully."""
        # This test is somewhat redundant since we already imported it,
        # but it confirms the import works without errors
        import src as reimported_src
        assert reimported_src.__version__ == src.__version__
        assert reimported_src.__author__ == src.__author__
        assert reimported_src.__description__ == src.__description__


class TestPackageStructure:
    """Test the overall package structure and accessibility."""
    
    def test_src_is_module(self):
        """Test that src behaves as a proper Python module."""
        assert hasattr(src, '__name__')
        assert hasattr(src, '__file__')
        assert hasattr(src, '__package__')
    
    def test_no_unexpected_exports(self):
        """Test that package doesn't export unexpected attributes."""
        # Get all public attributes (not starting with _)
        public_attrs = [attr for attr in dir(src) if not attr.startswith('_')]
        
        # Should only have our expected public constants
        expected_attrs = []  # We don't expect any public attributes beyond metadata
        
        for attr in public_attrs:
            # All public attributes should be either expected or metadata
            assert attr in expected_attrs or attr.startswith('__'), f"Unexpected public attribute: {attr}"


class TestVersionSpecifics:
    """Test specific version-related functionality."""
    
    def test_version_is_development(self):
        """Test that current version indicates development status."""
        version = src.__version__
        
        # Should be 0.x.x for development
        major_version = int(version.split('.')[0])
        assert major_version == 0, "Should be in development (0.x.x) version"
    
    def test_version_components_accessible(self):
        """Test that version components can be extracted."""
        version = src.__version__
        parts = version.split('.')
        
        major = int(parts[0])
        minor = int(parts[1]) 
        patch = int(parts[2])
        
        assert major >= 0
        assert minor >= 0
        assert patch >= 0
        
        # Reconstruct version
        reconstructed = f"{major}.{minor}.{patch}"
        assert reconstructed in version  # Should be contained (might have additional parts)


class TestIntegrationWithProject:
    """Test integration aspects with the broader project."""
    
    def test_consistent_with_project_description(self):
        """Test that package metadata is consistent with project purpose."""
        description = src.__description__
        
        # Should align with project being a RAG system
        rag_keywords = ["RAG", "Retrieval", "Generation", "retrieval", "augmented"]
        assert any(keyword in description for keyword in rag_keywords)
        
        # Should mention Blender
        assert "Blender" in description
    
    def test_author_matches_project(self):
        """Test that author matches project attribution."""
        assert src.__author__ == "Ready Tensor"
    
    def test_package_serves_as_namespace(self):
        """Test that package can serve as a namespace for submodules."""
        # This is more of a structural test
        # The package should be importable and allow submodule access
        
        # Test that we can access package attributes
        assert hasattr(src, '__version__')
        assert hasattr(src, '__author__')
        assert hasattr(src, '__description__')
        
        # Package should have proper module structure
        assert hasattr(src, '__path__') or hasattr(src, '__file__')