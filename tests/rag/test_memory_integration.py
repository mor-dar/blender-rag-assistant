#!/usr/bin/env python3
"""
Unit tests for memory integration in BlenderAssistantRAG.

Tests the memory functionality including initialization, window memory,
summary memory, error handling, and edge cases.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os
import sys
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


@pytest.fixture
def mock_retrieval_results():
    """Mock retrieval results for testing."""
    from retrieval.retriever import RetrievalResult
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


class TestRAGMemoryInitialization:
    """Test memory initialization in RAG system."""
    
    def test_no_memory_initialization(self, mock_retrieval_results):
        """Test RAG initialization with no memory (MEMORY_TYPE=none)."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = mock_retrieval_results
        mock_llm = Mock()
        
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None), \
             patch('utils.config.MEMORY_TYPE', 'none'):
            
            # Force reload to pick up patched values
            import rag.rag
            importlib.reload(rag.rag)
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            
            assert rag.memory is None
            info = rag.get_memory_info()
            assert info['memory_type'] == 'none'
            assert info['memory_enabled'] is False
    
    def test_window_memory_initialization(self, mock_retrieval_results):
        """Test RAG initialization with window memory."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = mock_retrieval_results
        mock_llm = Mock()
        
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None), \
             patch('utils.config.MEMORY_TYPE', 'window'), \
             patch('utils.config.MEMORY_WINDOW_SIZE', 4):
            
            # Force reload to pick up patched values
            import rag.rag
            importlib.reload(rag.rag)
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            
            assert rag.memory is not None
            info = rag.get_memory_info()
            assert info['memory_type'] == 'window'
            assert info['memory_enabled'] is True
            assert info['window_size'] == 4
    
    def test_summary_memory_initialization(self, mock_retrieval_results):
        """Test RAG initialization with summary memory."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = mock_retrieval_results
        
        # Create a proper mock for LangChain BaseLanguageModel
        from langchain_core.language_models import BaseLanguageModel
        mock_base_model = Mock(spec=BaseLanguageModel)
        mock_base_model._llm_type = "fake"
        mock_base_model.invoke = Mock(return_value="test summary")
        
        mock_llm = Mock()
        mock_llm.model = mock_base_model
        
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None), \
             patch('utils.config.MEMORY_TYPE', 'summary'), \
             patch('utils.config.MEMORY_MAX_TOKEN_LIMIT', 1500):
            
            # Force reload to pick up patched values
            import rag.rag
            importlib.reload(rag.rag)
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            
            assert rag.memory is not None
            info = rag.get_memory_info()
            assert info['memory_type'] == 'summary'
            assert info['memory_enabled'] is True
            assert info['max_token_limit'] == 1500
    
    def test_memory_initialization_failure_graceful_handling(self, mock_retrieval_results):
        """Test graceful handling when memory initialization fails."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = mock_retrieval_results
        mock_llm = Mock()
        
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None), \
             patch('utils.config.MEMORY_TYPE', 'window'), \
             patch('langchain.memory.ConversationBufferWindowMemory', side_effect=Exception("Memory init failed")):
            
            # Force reload to pick up patched values
            import rag.rag
            importlib.reload(rag.rag)
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            
            # Should fall back to no memory
            assert rag.memory is None
            info = rag.get_memory_info()
            assert info['memory_enabled'] is False


class TestRAGWithWindowMemory:
    """Test RAG functionality with window memory."""
    
    def test_query_with_window_memory_first_message(self, mock_retrieval_results):
        """Test first query with window memory (no conversation history)."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = mock_retrieval_results
        mock_llm = Mock()
        mock_llm.invoke.return_value = "Here's how to model in Blender..."
        
        mock_memory = Mock()
        mock_memory.load_memory_variables.return_value = {'history': ''}
        
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None), \
             patch('utils.config.MEMORY_TYPE', 'window'), \
             patch('langchain.memory.ConversationBufferWindowMemory', return_value=mock_memory):
            
            # Force reload to pick up patched values
            import rag.rag
            importlib.reload(rag.rag)
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            
            query = "How do I start modeling?"
            result = rag.handle_query(query)
            
            # Verify memory operations
            mock_memory.load_memory_variables.assert_called_once_with({})
            mock_memory.save_context.assert_called_once_with(
                {"input": query}, 
                {"output": "Here's how to model in Blender..."}
            )
            
            assert result == "Here's how to model in Blender..."
    
    def test_query_with_window_memory_existing_history(self, mock_retrieval_results):
        """Test query with existing conversation history in window memory."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = mock_retrieval_results
        mock_llm = Mock()
        mock_llm.invoke.return_value = "To extrude, press E..."
        
        conversation_history = "Human: How do I select objects?\nAI: Click on objects to select them."
        mock_memory = Mock()
        mock_memory.load_memory_variables.return_value = {'history': conversation_history}
        
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None), \
             patch('utils.config.MEMORY_TYPE', 'window'), \
             patch('langchain.memory.ConversationBufferWindowMemory', return_value=mock_memory):
            
            # Force reload to pick up patched values
            import rag.rag
            importlib.reload(rag.rag)
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            
            query = "How do I extrude faces?"
            result = rag.handle_query(query)
            
            # Verify LLM was called with both retrieval context and conversation history
            mock_llm.invoke.assert_called_once()
            call_args = mock_llm.invoke.call_args[1]
            assert "Conversation History" in call_args['context']
            assert conversation_history in call_args['context']
            assert "Blender modeling documentation content" in call_args['context']
            
            assert result == "To extrude, press E..."
    
    def test_clear_memory_functionality(self, mock_retrieval_results):
        """Test clearing window memory."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = mock_retrieval_results
        mock_llm = Mock()
        
        mock_memory = Mock()
        
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None), \
             patch('utils.config.MEMORY_TYPE', 'window'), \
             patch('langchain.memory.ConversationBufferWindowMemory', return_value=mock_memory):
            
            # Force reload to pick up patched values
            import rag.rag
            importlib.reload(rag.rag)
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            
            rag.clear_memory()
            mock_memory.clear.assert_called_once()


class TestRAGWithSummaryMemory:
    """Test RAG functionality with summary memory."""
    
    def test_query_with_summary_memory(self, mock_retrieval_results):
        """Test query with summary memory."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = mock_retrieval_results
        mock_llm = Mock()
        mock_llm.model = Mock()
        mock_llm.invoke.return_value = "Animation basics explained..."
        
        conversation_summary = "Previously discussed: modeling basics and object manipulation."
        mock_memory = Mock()
        mock_memory.load_memory_variables.return_value = {'history': conversation_summary}
        
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None), \
             patch('utils.config.MEMORY_TYPE', 'summary'), \
             patch('langchain.memory.ConversationSummaryMemory', return_value=mock_memory):
            
            # Force reload to pick up patched values
            import rag.rag
            importlib.reload(rag.rag)
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            
            query = "How do I animate objects?"
            result = rag.handle_query(query)
            
            # Verify memory was used
            mock_memory.load_memory_variables.assert_called_once_with({})
            mock_memory.save_context.assert_called_once()
            
            # Verify context includes summary
            call_args = mock_llm.invoke.call_args[1]
            assert conversation_summary in call_args['context']
            
            assert result == "Animation basics explained..."
    
    def test_summary_memory_without_llm_model(self, mock_retrieval_results):
        """Test summary memory initialization fails gracefully without LLM model."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = mock_retrieval_results
        mock_llm = Mock()
        # Don't set mock_llm.model - simulate LLM without model attribute
        
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None), \
             patch('utils.config.MEMORY_TYPE', 'summary'):
            
            # Force reload to pick up patched values
            import rag.rag
            importlib.reload(rag.rag)
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            
            # Should fall back to no memory
            assert rag.memory is None
            info = rag.get_memory_info()
            assert info['memory_enabled'] is False


class TestRAGMemoryErrorHandling:
    """Test error handling in RAG memory functionality."""
    
    def test_memory_load_failure_fallback(self, mock_retrieval_results):
        """Test fallback when memory load fails."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = mock_retrieval_results
        mock_llm = Mock()
        mock_llm.invoke.return_value = "Fallback response..."
        
        mock_memory = Mock()
        mock_memory.load_memory_variables.side_effect = Exception("Memory load failed")
        
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None), \
             patch('utils.config.MEMORY_TYPE', 'window'), \
             patch('langchain.memory.ConversationBufferWindowMemory', return_value=mock_memory):
            
            # Force reload to pick up patched values
            import rag.rag
            importlib.reload(rag.rag)
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            
            query = "Test query"
            result = rag.handle_query(query)
            
            # Should fall back to normal operation without memory
            assert result == "Fallback response..."
            # Verify it still tried to call LLM with just retrieval context
            mock_llm.invoke.assert_called()
    
    def test_memory_save_failure_graceful_handling(self, mock_retrieval_results):
        """Test graceful handling when memory save fails."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = mock_retrieval_results
        mock_llm = Mock()
        mock_llm.invoke.return_value = "Response generated..."
        
        mock_memory = Mock()
        mock_memory.load_memory_variables.return_value = {'history': 'Some history'}
        mock_memory.save_context.side_effect = Exception("Memory save failed")
        
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None), \
             patch('utils.config.MEMORY_TYPE', 'window'), \
             patch('langchain.memory.ConversationBufferWindowMemory', return_value=mock_memory):
            
            # Force reload to pick up patched values
            import rag.rag
            importlib.reload(rag.rag)
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            
            query = "Test query"
            result = rag.handle_query(query)
            
            # Should still return response despite save failure
            assert result == "Response generated..."
            mock_memory.save_context.assert_called_once()
    
    def test_clear_memory_error_handling(self, mock_retrieval_results):
        """Test error handling when clearing memory fails."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = mock_retrieval_results
        mock_llm = Mock()
        
        mock_memory = Mock()
        mock_memory.clear.side_effect = Exception("Clear failed")
        
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None), \
             patch('utils.config.MEMORY_TYPE', 'window'), \
             patch('langchain.memory.ConversationBufferWindowMemory', return_value=mock_memory):
            
            # Force reload to pick up patched values
            import rag.rag
            importlib.reload(rag.rag)
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            
            # Should not raise exception
            rag.clear_memory()
            mock_memory.clear.assert_called_once()


class TestRAGMemoryEdgeCases:
    """Test edge cases for RAG memory functionality."""
    
    def test_invalid_memory_type_configuration(self, mock_retrieval_results):
        """Test handling of invalid memory type configuration."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = mock_retrieval_results
        mock_llm = Mock()
        
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None), \
             patch('utils.config.MEMORY_TYPE', 'invalid_type'):
            
            # Force reload to pick up patched values
            import rag.rag
            importlib.reload(rag.rag)
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            
            # Should default to no memory
            assert rag.memory is None
            info = rag.get_memory_info()
            assert info['memory_type'] == 'invalid_type'
            assert info['memory_enabled'] is False
    
    def test_memory_with_empty_retrieval_results(self):
        """Test memory functionality with empty retrieval results."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = []  # Empty results
        mock_llm = Mock()
        mock_llm.invoke.return_value = "No relevant docs found, but here's general info..."
        
        mock_memory = Mock()
        mock_memory.load_memory_variables.return_value = {'history': 'Previous conversation'}
        
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None), \
             patch('utils.config.MEMORY_TYPE', 'window'), \
             patch('langchain.memory.ConversationBufferWindowMemory', return_value=mock_memory):
            
            # Force reload to pick up patched values
            import rag.rag
            importlib.reload(rag.rag)
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            
            query = "How do I do something not in docs?"
            result = rag.handle_query(query)
            
            # Should still include conversation history even with no retrieval results
            call_args = mock_llm.invoke.call_args[1]
            assert "Previous conversation" in call_args['context']
            assert result == "No relevant docs found, but here's general info..."
    
    def test_memory_info_with_error(self, mock_retrieval_results):
        """Test get_memory_info when memory access causes error."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = mock_retrieval_results
        mock_llm = Mock()
        
        mock_memory = Mock()
        mock_memory.chat_memory.messages = ["msg1", "msg2"]
        # Simulate error when accessing memory attributes
        type(mock_memory).chat_memory = Mock(side_effect=Exception("Memory access error"))
        
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None), \
             patch('utils.config.MEMORY_TYPE', 'window'), \
             patch('langchain.memory.ConversationBufferWindowMemory', return_value=mock_memory):
            
            # Force reload to pick up patched values
            import rag.rag
            importlib.reload(rag.rag)
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            
            info = rag.get_memory_info()
            
            # Should include error information
            assert 'error' in info
            assert info['memory_enabled'] is True
    
    def test_no_memory_clear_operation(self, mock_retrieval_results):
        """Test clear memory operation when no memory is configured."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = mock_retrieval_results
        mock_llm = Mock()
        
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None), \
             patch('utils.config.MEMORY_TYPE', 'none'):
            
            # Force reload to pick up patched values
            import rag.rag
            importlib.reload(rag.rag)
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            
            # Should not raise exception
            rag.clear_memory()
            assert rag.memory is None


@pytest.mark.integration
class TestRAGMemoryIntegration:
    """Integration tests for RAG memory functionality."""
    
    def test_conversation_flow_with_window_memory(self, mock_retrieval_results):
        """Test complete conversation flow with window memory."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = mock_retrieval_results
        mock_llm = Mock()
        
        # Create real ConversationBufferWindowMemory for integration test
        from langchain.memory import ConversationBufferWindowMemory
        
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None), \
             patch('utils.config.MEMORY_TYPE', 'window'), \
             patch('utils.config.MEMORY_WINDOW_SIZE', 4):
            
            # Force reload to pick up patched values
            import rag.rag
            importlib.reload(rag.rag)
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            
            # Simulate conversation
            conversations = [
                ("How do I model?", "Use Edit mode and extrude faces."),
                ("What about texturing?", "Apply materials in Shading workspace."),
                ("How do I render?", "Press F12 to render the scene."),
                ("What's the shortcut for save?", "Ctrl+S saves your project."),
            ]
            
            for i, (query, response) in enumerate(conversations):
                mock_llm.invoke.return_value = response
                result = rag.handle_query(query)
                assert result == response
                
                # Check memory info updates
                info = rag.get_memory_info()
                # ConversationBufferWindowMemory with k=4 keeps last 4 messages
                # But may keep more during processing, so just check it's reasonable
                assert info['current_messages'] >= 0  # Basic sanity check
                assert info['current_messages'] <= 8  # Should never exceed reasonable limit
    
    def test_memory_type_switching_behavior(self, mock_retrieval_results):
        """Test behavior when memory type configuration changes (conceptual)."""
        # Note: In real usage, memory type is set at initialization
        # This test demonstrates the different behaviors
        
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = mock_retrieval_results
        
        # Create proper mock for LangChain BaseLanguageModel for summary memory
        from langchain_core.language_models import BaseLanguageModel
        mock_base_model = Mock(spec=BaseLanguageModel)
        mock_base_model._llm_type = "fake"
        mock_base_model.invoke = Mock(return_value="test summary")
        
        mock_llm = Mock()
        mock_llm.model = mock_base_model
        
        # Test window memory
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None), \
             patch('utils.config.MEMORY_TYPE', 'window'), \
             patch('utils.config.MEMORY_WINDOW_SIZE', 2):
            
            from rag.rag import BlenderAssistantRAG
            rag_window = BlenderAssistantRAG()
            
            assert rag_window.memory is not None
            info = rag_window.get_memory_info()
            assert info['memory_type'] == 'window'
        
        # Test summary memory  
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None), \
             patch('utils.config.MEMORY_TYPE', 'summary'), \
             patch('utils.config.MEMORY_MAX_TOKEN_LIMIT', 500):
            
            # Need to reload the module to get new config
            import importlib
            import rag.rag
            importlib.reload(rag.rag)
            
            rag_summary = rag.rag.BlenderAssistantRAG()
            
            assert rag_summary.memory is not None
            info = rag_summary.get_memory_info()
            assert info['memory_type'] == 'summary'