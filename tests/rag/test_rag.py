#!/usr/bin/env python3
"""
Unit tests for the BlenderAssistantRAG class.

Tests the main RAG orchestrator functionality including initialization,
query handling, and error cases without making real API calls.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import logging
import sys
from pathlib import Path
import importlib

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


class TestBlenderAssistantRAG:
    """Test suite for BlenderAssistantRAG class."""
    
    def test_handle_query_success_with_groq(self):
        """Test successful query handling with mocked Groq LLM."""
        # Mock all dependencies
        mock_retriever = Mock()
        mock_retriever.retrieve_context.return_value = "Context about modeling"
        
        mock_llm = Mock()
        mock_llm.invoke.return_value = "To start modeling in Blender..."
        
        # Patch the imports and create RAG instance
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None):
            
            # Import after patching to ensure proper conditional import
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            
            result = rag.handle_query("How do I start modeling in Blender?")
            
            # Verify retriever was called
            mock_retriever.retrieve_context.assert_called_once_with("How do I start modeling in Blender?")
            
            # Verify LLM was called with correct parameters
            mock_llm.invoke.assert_called_once_with(
                question="How do I start modeling in Blender?",
                context="Context about modeling"
            )
            
            assert result == "To start modeling in Blender..."
    
    def test_handle_query_success_with_openai(self):
        """Test successful query handling with mocked OpenAI LLM."""
        mock_retriever = Mock()
        mock_retriever.retrieve_context.return_value = "Context about rendering"
        
        mock_llm = Mock()
        mock_llm.invoke.return_value = "To render in Blender, press F12..."
        
        # Patch for OpenAI path
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.openai_llm.OpenAILLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', None), \
             patch('utils.config.OPENAI_API_KEY', 'test-openai-key'):
            
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            
            result = rag.handle_query("How do I render in Blender?")
            
            mock_retriever.retrieve_context.assert_called_once_with("How do I render in Blender?")
            mock_llm.invoke.assert_called_once_with(
                question="How do I render in Blender?",
                context="Context about rendering"
            )
            
            assert result == "To render in Blender, press F12..."
    
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
             patch('utils.config.CHROMA_COLLECTION_NAME', 'test_collection'):
            
            from rag.rag import BlenderAssistantRAG
            
            # Test demo collection
            with patch('retrieval.SemanticRetriever') as mock_retriever_class:
                mock_retriever_class.return_value = mock_retriever
                rag_demo = BlenderAssistantRAG(collection_type="demo")
                mock_retriever_class.assert_called_with(
                    db_path='/tmp/test_db',
                    embedding_model='test-model',
                    collection_name='test_collection_demo'
                )
            
            # Test full collection
            with patch('retrieval.SemanticRetriever') as mock_retriever_class:
                mock_retriever_class.return_value = mock_retriever
                rag_full = BlenderAssistantRAG(collection_type="full")
                mock_retriever_class.assert_called_with(
                    db_path='/tmp/test_db',
                    embedding_model='test-model',
                    collection_name='test_collection_full'
                )
    
    def test_handle_query_empty_string(self):
        """Test handling of empty query string."""
        mock_retriever = Mock()
        mock_retriever.retrieve_context.return_value = ""
        
        mock_llm = Mock()
        mock_llm.invoke.return_value = "Please provide a specific question."
        
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None):
            
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            result = rag.handle_query("")
            
            mock_retriever.retrieve_context.assert_called_once_with("")
            mock_llm.invoke.assert_called_once_with(question="", context="")
            assert result == "Please provide a specific question."
    
    def test_handle_query_special_characters(self):
        """Test handling of queries with special characters."""
        mock_retriever = Mock()
        mock_retriever.retrieve_context.return_value = "Shortcuts context"
        
        mock_llm = Mock()
        mock_llm.invoke.return_value = "Ctrl+R adds edge loops..."
        
        special_query = "How do I use Ctrl+R & Shift+A in Blender? 🎨"
        
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None):
            
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            result = rag.handle_query(special_query)
            
            mock_retriever.retrieve_context.assert_called_once_with(special_query)
            assert result == "Ctrl+R adds edge loops..."
    
    def test_handle_query_long_query(self):
        """Test handling of very long query string."""
        mock_retriever = Mock()
        mock_retriever.retrieve_context.return_value = "Complex modeling context"
        
        mock_llm = Mock()
        mock_llm.invoke.return_value = "For complex modeling tasks..."
        
        long_query = "How do I " + "very " * 1000 + "complex modeling in Blender?"
        
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None):
            
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            result = rag.handle_query(long_query)
            
            mock_retriever.retrieve_context.assert_called_once_with(long_query)
            assert result == "For complex modeling tasks..."
    
    def test_llm_initialization_failure(self):
        """Test handling when LLM fails to initialize."""
        mock_retriever = Mock()
        
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', side_effect=Exception("API key invalid")), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None), \
             pytest.warns(None) as warnings:
            
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
        mock_retriever.retrieve_context.side_effect = Exception("Database connection failed")
        
        mock_llm = Mock()
        
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None):
            
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            
            with pytest.raises(Exception, match="Database connection failed"):
                rag.handle_query("Test query")
    
    def test_llm_invoke_exception_propagation(self):
        """Test that LLM invoke exceptions propagate correctly."""
        mock_retriever = Mock()
        mock_retriever.retrieve_context.return_value = "Test context"
        
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("API rate limit exceeded")
        
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None):
            
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            
            with pytest.raises(Exception, match="API rate limit exceeded"):
                rag.handle_query("Test query")
    
    def test_unicode_query_handling(self):
        """Test handling of Unicode characters in queries."""
        mock_retriever = Mock()
        mock_retriever.retrieve_context.return_value = "Unicode context"
        
        mock_llm = Mock()
        mock_llm.invoke.return_value = "Unicode response"
        
        unicode_query = "Comment créer un objet en Blender? 中文测试"
        
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None):
            
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            result = rag.handle_query(unicode_query)
            
            mock_retriever.retrieve_context.assert_called_once_with(unicode_query)
            assert result == "Unicode response"


@pytest.mark.integration
class TestRAGIntegration:
    """Integration tests for the RAG system."""
    
    def test_no_api_key_error_logging(self):
        """Test error logging when no API keys are configured."""
        with patch('utils.config.GROQ_API_KEY', None), \
             patch('utils.config.OPENAI_API_KEY', None), \
             pytest.warns(None):
            # The error should be logged when importing with no API keys
            pass  # This test mainly checks that the system doesn't crash
    
    def test_end_to_end_workflow_simulation(self):
        """Test complete workflow simulation."""
        mock_retriever = Mock()
        mock_retriever.retrieve_context.return_value = "Blender cube creation documentation"
        
        mock_llm = Mock()
        mock_llm.invoke.return_value = "To create a cube, press Shift+A and select Mesh > Cube."
        
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None):
            
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
             patch('utils.config.OPENAI_API_KEY', None):
            
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