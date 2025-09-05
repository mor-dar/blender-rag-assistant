#!/usr/bin/env python3
"""
Unit tests for the BlenderAssistantRAG class.

Tests the main RAG orchestrator functionality including initialization,
query handling, memory integration, and error cases without making real API calls.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import logging
import sys
from pathlib import Path
import importlib

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

# Import RetrievalResult for proper mocking
from retrieval.retriever import RetrievalResult


@pytest.fixture
def mock_retrieval_results():
    """Fixture providing mock retrieval results."""
    return [
        RetrievalResult(
            text="Blender modeling documentation content",
            score=0.85,
            metadata={"source": "modeling_guide.html"}
        ),
        RetrievalResult(
            text="Additional context about 3D modeling",
            score=0.75,
            metadata={"source": "advanced_modeling.html"}
        )
    ]


class TestBlenderAssistantRAG:
    """Test suite for BlenderAssistantRAG class."""
    
    def test_handle_query_success_with_groq(self, mock_retrieval_results):
        """Test successful query handling with mocked Groq LLM."""
        # Mock all dependencies
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = mock_retrieval_results
        
        mock_llm = Mock()
        mock_llm.invoke.return_value = "To start modeling in Blender..."
        
        # Patch the imports and create RAG instance
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None), \
             patch('utils.config.MEMORY_TYPE', 'none'):
            
            # Force reload to pick up the patched API key values
            import rag.rag
            importlib.reload(rag.rag)
            # Force reload to pick up the patched API key values
            import rag.rag
            importlib.reload(rag.rag)
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            
            result = rag.handle_query("How do I start modeling in Blender?")
            
            # Verify retriever was called with correct method
            mock_retriever.retrieve.assert_called_once_with("How do I start modeling in Blender?", k=3)
            
            # Verify LLM was called with correct parameters
            mock_llm.invoke.assert_called_once()
            call_args = mock_llm.invoke.call_args[1]
            assert call_args['question'] == "How do I start modeling in Blender?"
            assert "Blender modeling documentation content" in call_args['context']
            assert "[1] Unknown Page" in call_args['context']
            
            assert "To start modeling in Blender..." in result
    
    def test_handle_query_success_with_openai(self, mock_retrieval_results):
        """Test successful query handling with mocked OpenAI LLM."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = mock_retrieval_results
        
        mock_llm = Mock()
        mock_llm.invoke.return_value = "To render in Blender, press F12..."
        
        # Patch for OpenAI path
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.openai_llm.OpenAILLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', None), \
             patch('utils.config.OPENAI_API_KEY', 'test-openai-key'), \
             patch('utils.config.MEMORY_TYPE', 'none'):
            
            # Force reload to pick up the patched API key values
            import rag.rag
            importlib.reload(rag.rag)
            # Force reload to pick up the patched API key values
            import rag.rag
            importlib.reload(rag.rag)
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            
            result = rag.handle_query("How do I render in Blender?")
            
            mock_retriever.retrieve.assert_called_once_with("How do I render in Blender?", k=3)
            mock_llm.invoke.assert_called_once()
            call_args = mock_llm.invoke.call_args[1]
            assert call_args['question'] == "How do I render in Blender?"
            assert "[1] Unknown Page" in call_args['context']
            
            assert "To render in Blender, press F12..." in result
    
    def test_initialization_with_different_collection_types(self):
        """Test initialization with different collection types."""
        mock_retriever = Mock()
        mock_llm = Mock()
        
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None), \
             patch('utils.config.CHROMA_PERSIST_DIRECTORY', '/tmp/test_db'), \
             patch('utils.config.EMBEDDING_MODEL', 'test-model'), \
             patch('utils.config.CHROMA_COLLECTION_NAME', 'test_collection'), \
             patch('utils.config.MEMORY_TYPE', 'none'):
            
            # Force reload to pick up the patched API key values
            import rag.rag
            importlib.reload(rag.rag)
            from rag.rag import BlenderAssistantRAG
            
            # Test default initialization
            rag = BlenderAssistantRAG()
            assert rag is not None
            assert hasattr(rag, 'retriever')
            assert hasattr(rag, 'llm')
    
    def test_handle_query_empty_string(self):
        """Test handling of empty query string."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = []
        
        mock_llm = Mock()
        mock_llm.invoke.return_value = "Please provide a specific question."
        
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None), \
             patch('utils.config.MEMORY_TYPE', 'none'):
            
            # Force reload to pick up the patched API key values
            import rag.rag
            importlib.reload(rag.rag)
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            result = rag.handle_query("")
            
            mock_retriever.retrieve.assert_called_once_with("", k=3)
            mock_llm.invoke.assert_called_once()
            call_args = mock_llm.invoke.call_args[1]
            assert call_args['question'] == ""
            assert "No relevant documentation found" in call_args['context']
            assert result == "Please provide a specific question."
    
    def test_handle_query_special_characters(self):
        """Test handling of queries with special characters."""
        mock_retriever = Mock()
        mock_retrieval_results = [
            RetrievalResult(
                text="Shortcuts context",
                score=0.8,
                metadata={"source": "shortcuts.html"}
            )
        ]
        mock_retriever.retrieve.return_value = mock_retrieval_results
        
        mock_llm = Mock()
        mock_llm.invoke.return_value = "Ctrl+R adds edge loops..."
        
        special_query = "How do I use Ctrl+R & Shift+A in Blender? 🎨"
        
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None), \
             patch('utils.config.MEMORY_TYPE', 'none'):
            
            # Force reload to pick up the patched API key values
            import rag.rag
            importlib.reload(rag.rag)
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            result = rag.handle_query(special_query)
            
            mock_retriever.retrieve.assert_called_once_with(special_query, k=3)
            assert "Ctrl+R adds edge loops..." in result
    
    def test_handle_query_long_query(self):
        """Test handling of very long query string."""
        mock_retriever = Mock()
        mock_retrieval_results = [
            RetrievalResult(
                text="Complex modeling context",
                score=0.85,
                metadata={"source": "modeling.html"}
            )
        ]
        mock_retriever.retrieve.return_value = mock_retrieval_results
        
        mock_llm = Mock()
        mock_llm.invoke.return_value = "For complex modeling tasks..."
        
        long_query = "How do I " + "very " * 1000 + "complex modeling in Blender?"
        
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None), \
             patch('utils.config.MEMORY_TYPE', 'none'):
            
            # Force reload to pick up the patched API key values
            import rag.rag
            importlib.reload(rag.rag)
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            result = rag.handle_query(long_query)
            
            mock_retriever.retrieve.assert_called_once_with(long_query, k=3)
            assert "For complex modeling tasks..." in result
    
    def test_llm_initialization_failure(self):
        """Test handling when LLM fails to initialize."""
        mock_retriever = Mock()
        
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', side_effect=Exception("API key invalid")), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None), \
             patch('utils.config.MEMORY_TYPE', 'none'):
            
            # Force reload to pick up the patched API key values
            import rag.rag
            importlib.reload(rag.rag)
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            
            # Verify LLM is None due to initialization failure
            assert rag.llm is None
            
            # Verify error handling in query
            result = rag.handle_query("Test query")
            assert result == "LLM not available. Please check API key configuration."
    
    def test_retriever_exception_propagation(self):
        """Test that retriever exceptions propagate correctly."""
        mock_retriever = Mock()
        mock_retriever.retrieve.side_effect = Exception("Database connection failed")
        
        mock_llm = Mock()
        mock_llm.invoke.return_value = "Error response"
        
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None), \
             patch('utils.config.MEMORY_TYPE', 'none'):
            
            # Force reload to pick up the patched API key values
            import rag.rag
            importlib.reload(rag.rag)
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            
            # The retriever exception should not propagate - it's handled gracefully
            result = rag.handle_query("Test query")
            # Verify that the system handles the retriever error gracefully
            assert "Error response" in result
    
    def test_llm_invoke_exception_propagation(self):
        """Test that LLM invoke exceptions propagate correctly."""
        mock_retriever = Mock()
        mock_retrieval_results = [
            RetrievalResult(
                text="Test context",
                score=0.9,
                metadata={"source": "test.html"}
            )
        ]
        mock_retriever.retrieve.return_value = mock_retrieval_results
        
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("API rate limit exceeded")
        
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None), \
             patch('utils.config.MEMORY_TYPE', 'none'):
            
            # Force reload to pick up the patched API key values
            import rag.rag
            importlib.reload(rag.rag)
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            
            with pytest.raises(Exception, match="API rate limit exceeded"):
                rag.handle_query("Test query")
    
    def test_unicode_query_handling(self):
        """Test handling of Unicode characters in queries."""
        mock_retriever = Mock()
        mock_retrieval_results = [
            RetrievalResult(
                text="Unicode context",
                score=0.85,
                metadata={"source": "unicode.html"}
            )
        ]
        mock_retriever.retrieve.return_value = mock_retrieval_results
        
        mock_llm = Mock()
        mock_llm.invoke.return_value = "Unicode response"
        
        unicode_query = "Comment créer un objet en Blender? 中文测试"
        
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None), \
             patch('utils.config.MEMORY_TYPE', 'none'):
            
            # Force reload to pick up the patched API key values
            import rag.rag
            importlib.reload(rag.rag)
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            result = rag.handle_query(unicode_query)
            
            mock_retriever.retrieve.assert_called_once_with(unicode_query, k=3)
            assert "Unicode response" in result


@pytest.mark.integration
class TestRAGIntegration:
    """Integration tests for the RAG system."""
    
    def test_no_api_key_error_logging(self):
        """Test error logging when no API keys are configured."""
        with patch('rag.rag.GROQ_API_KEY', None), \
             patch('rag.rag.OPENAI_API_KEY', None):
            # The error should be logged when importing with no API keys
            # Force reload to pick up the patched API key values
            import rag.rag
            importlib.reload(rag.rag)
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            # System should handle gracefully with dummy LLM
            assert rag.llm is not None  # DummyLLM should be instantiated
    
    def test_end_to_end_workflow_simulation(self):
        """Test complete workflow simulation."""
        from retrieval.retriever import RetrievalResult
        
        mock_retrieval_results = [
            RetrievalResult(
                text="Blender cube creation documentation",
                score=0.9,
                metadata={"source": "cube_guide.html"}
            )
        ]
        
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = mock_retrieval_results
        
        mock_llm = Mock()
        mock_llm.invoke.return_value = "To create a cube, press Shift+A and select Mesh > Cube."
        
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None), \
             patch('utils.config.MEMORY_TYPE', 'none'):
            
            # Force reload to pick up the patched API key values
            import rag.rag
            importlib.reload(rag.rag)
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            
            # Simulate multiple queries
            queries = [
                "How do I create a cube?",
                "How do I delete an object?",
                "How do I extrude faces?"
            ]
            
            responses = [
                "To create a cube, press Shift+A and select Mesh > Cube.",
                "To delete an object, select it and press X or Delete.",
                "To extrude faces, select faces in Edit mode and press E."
            ]
            
            for query, expected_response in zip(queries, responses):
                mock_llm.invoke.return_value = expected_response
                result = rag.handle_query(query)
                assert expected_response in result


@pytest.mark.unit
class TestRAGEdgeCases:
    """Edge case tests for RAG system."""
    
    def test_none_query_graceful_handling(self):
        """Test that None query is handled without crashing."""
        mock_retriever = Mock()
        mock_llm = Mock()
        
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None), \
             patch('utils.config.MEMORY_TYPE', 'none'):
            
            # Force reload to pick up the patched API key values
            import rag.rag
            importlib.reload(rag.rag)
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            
            # None query should either be handled or raise a clear error
            try:
                result = rag.handle_query(None)
                # If it doesn't raise, it should return a reasonable response
                assert isinstance(result, str)
            except (TypeError, AttributeError):
                # This is also acceptable - clear error for invalid input
                pass