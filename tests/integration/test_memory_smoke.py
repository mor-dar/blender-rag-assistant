#!/usr/bin/env python3
"""
Smoke tests for memory functionality in the RAG system.

These are quick integration tests to verify basic memory functionality works
without deep testing of edge cases or complex scenarios.
"""

import pytest
from unittest.mock import Mock, patch
import os
import sys
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


@pytest.mark.smoke
class TestMemorySmokeTests:
    """Smoke tests for memory functionality."""
    
    def test_rag_initialization_with_each_memory_type(self):
        """Smoke test: RAG initializes successfully with each memory type."""
        mock_retriever = Mock()
        
        # Create proper mock for LangChain BaseLanguageModel
        from langchain_core.language_models import BaseLanguageModel
        mock_base_model = Mock(spec=BaseLanguageModel)
        mock_base_model._llm_type = "fake"
        mock_base_model.invoke = Mock(return_value="test summary")
        
        mock_llm = Mock()
        mock_llm.model = mock_base_model
        
        memory_types = ['none', 'window', 'summary']
        
        for memory_type in memory_types:
            with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
                 patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
                 patch('utils.config.GROQ_API_KEY', 'test-key'), \
                 patch('utils.config.OPENAI_API_KEY', None), \
                 patch('utils.config.MEMORY_TYPE', memory_type):
                
                # Force reload to pick up patched values
                import rag.rag
                importlib.reload(rag.rag)
                from rag.rag import BlenderAssistantRAG
                rag = BlenderAssistantRAG()
                
                # Should initialize without errors
                assert rag is not None
                assert hasattr(rag, 'memory')
                
                # Memory should be present for window/summary, None for none
                if memory_type == 'none':
                    assert rag.memory is None
                else:
                    assert rag.memory is not None
    
    def test_basic_query_with_each_memory_type(self):
        """Smoke test: Basic query works with each memory type."""
        from retrieval.retriever import RetrievalResult
        
        mock_retrieval_results = [
            RetrievalResult(
                text="Basic Blender documentation",
                score=0.8,
                metadata={"source": "test.html"}
            )
        ]
        
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = mock_retrieval_results
        mock_llm = Mock()
        # Create proper mock for LangChain BaseLanguageModel
        from langchain_core.language_models import BaseLanguageModel
        mock_base_model = Mock(spec=BaseLanguageModel)
        mock_base_model._llm_type = "fake"
        mock_base_model.invoke = Mock(return_value="test summary")
        mock_llm.model = mock_base_model
        mock_llm.invoke.return_value = "Basic response"
        
        memory_types = ['none', 'window', 'summary']
        
        for memory_type in memory_types:
            with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
                 patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
                 patch('utils.config.GROQ_API_KEY', 'test-key'), \
                 patch('utils.config.OPENAI_API_KEY', None), \
                 patch('utils.config.MEMORY_TYPE', memory_type):
                
                # Force reload to pick up patched values
                import rag.rag
                importlib.reload(rag.rag)
                from rag.rag import BlenderAssistantRAG
                rag = BlenderAssistantRAG()
                
                result = rag.handle_query("How do I use Blender?")
                
                # Should return a response that contains the basic response and citations
                assert "Basic response" in result
                assert "**Sources:**" in result
                assert "[1] Unknown Page" in result
                # LLM should have been called
                assert mock_llm.invoke.called
    
    def test_memory_info_returns_valid_data(self):
        """Smoke test: Memory info returns valid data for each type."""
        mock_retriever = Mock()
        mock_llm = Mock()
        # Create proper mock for LangChain BaseLanguageModel
        from langchain_core.language_models import BaseLanguageModel
        mock_base_model = Mock(spec=BaseLanguageModel)
        mock_base_model._llm_type = "fake"
        mock_base_model.invoke = Mock(return_value="test summary")
        mock_llm.model = mock_base_model
        
        memory_types = ['none', 'window', 'summary']
        
        for memory_type in memory_types:
            with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
                 patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
                 patch('utils.config.GROQ_API_KEY', 'test-key'), \
                 patch('utils.config.OPENAI_API_KEY', None), \
                 patch('utils.config.MEMORY_TYPE', memory_type):
                
                # Force reload to pick up patched values
                import rag.rag
                importlib.reload(rag.rag)
                from rag.rag import BlenderAssistantRAG
                rag = BlenderAssistantRAG()
                
                info = rag.get_memory_info()
                
                # Should return a dictionary with expected keys
                assert isinstance(info, dict)
                assert 'memory_type' in info
                assert 'memory_enabled' in info
                assert info['memory_type'] == memory_type
                assert info['memory_enabled'] == (memory_type != 'none')
    
    def test_clear_memory_doesnt_crash(self):
        """Smoke test: Clear memory doesn't crash for any memory type."""
        mock_retriever = Mock()
        mock_llm = Mock()
        # Create proper mock for LangChain BaseLanguageModel
        from langchain_core.language_models import BaseLanguageModel
        mock_base_model = Mock(spec=BaseLanguageModel)
        mock_base_model._llm_type = "fake"
        mock_base_model.invoke = Mock(return_value="test summary")
        mock_llm.model = mock_base_model
        
        memory_types = ['none', 'window', 'summary']
        
        for memory_type in memory_types:
            with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
                 patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
                 patch('utils.config.GROQ_API_KEY', 'test-key'), \
                 patch('utils.config.OPENAI_API_KEY', None), \
                 patch('utils.config.MEMORY_TYPE', memory_type):
                
                # Force reload to pick up patched values
                import rag.rag
                importlib.reload(rag.rag)
                from rag.rag import BlenderAssistantRAG
                rag = BlenderAssistantRAG()
                
                # Should not raise an exception
                rag.clear_memory()
    
    def test_memory_configuration_variables_accessible(self):
        """Smoke test: Memory configuration variables are accessible."""
        # Just verify we can import and access memory config variables
        from utils.config import MEMORY_TYPE, MEMORY_WINDOW_SIZE, MEMORY_MAX_TOKEN_LIMIT
        
        assert isinstance(MEMORY_TYPE, str)
        assert isinstance(MEMORY_WINDOW_SIZE, int)
        assert isinstance(MEMORY_MAX_TOKEN_LIMIT, int)
        assert MEMORY_WINDOW_SIZE > 0
        assert MEMORY_MAX_TOKEN_LIMIT > 0
    
    def test_langchain_memory_imports_work(self):
        """Smoke test: LangChain memory imports work correctly."""
        # Should be able to import LangChain memory classes
        try:
            from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory
        except ImportError as e:
            pytest.fail(f"Failed to import LangChain memory classes: {e}")
        
        # Should be able to create instances (with minimal config)
        try:
            window_memory = ConversationBufferWindowMemory(k=2, return_messages=False)
            assert window_memory is not None
        except Exception as e:
            pytest.fail(f"Failed to create ConversationBufferWindowMemory: {e}")


@pytest.mark.smoke
@pytest.mark.integration  
class TestMemoryEnvironmentIntegration:
    """Smoke tests for memory environment variable integration."""
    
    def test_env_var_changes_affect_rag_behavior(self):
        """Smoke test: Environment variable changes affect RAG behavior."""
        mock_retriever = Mock()
        mock_llm = Mock()
        # Create proper mock for LangChain BaseLanguageModel
        from langchain_core.language_models import BaseLanguageModel
        mock_base_model = Mock(spec=BaseLanguageModel)
        mock_base_model._llm_type = "fake"
        mock_base_model.invoke = Mock(return_value="test summary")
        mock_llm.model = mock_base_model
        
        # Test with window memory
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
            
            info = rag.get_memory_info()
            assert info['memory_type'] == 'window'
            assert info['window_size'] == 4
    
    def test_invalid_env_vars_dont_break_system(self):
        """Smoke test: Invalid environment variables don't break the system."""
        mock_retriever = Mock()
        mock_llm = Mock()
        
        # Test with invalid values
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None), \
             patch.dict(os.environ, {
                 'MEMORY_TYPE': 'invalid_type',
                 'MEMORY_WINDOW_SIZE': 'not_a_number',
                 'MEMORY_MAX_TOKEN_LIMIT': 'also_not_a_number'
             }):
            
            # Force reload config
            import importlib
            import utils.config
            importlib.reload(utils.config)
            
            # Should still be able to create RAG instance
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            
            # Should fall back to safe defaults/no memory
            info = rag.get_memory_info()
            assert info is not None
            assert 'memory_type' in info


@pytest.mark.smoke
@pytest.mark.performance
class TestMemoryPerformanceSmokeTests:
    """Basic performance smoke tests for memory functionality."""
    
    def test_memory_initialization_performance(self):
        """Smoke test: Memory initialization is reasonably fast."""
        import time
        
        mock_retriever = Mock()
        mock_llm = Mock()
        # Create proper mock for LangChain BaseLanguageModel
        from langchain_core.language_models import BaseLanguageModel
        mock_base_model = Mock(spec=BaseLanguageModel)
        mock_base_model._llm_type = "fake"
        mock_base_model.invoke = Mock(return_value="test summary")
        mock_llm.model = mock_base_model
        
        with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
             patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
             patch('utils.config.GROQ_API_KEY', 'test-key'), \
             patch('utils.config.OPENAI_API_KEY', None), \
             patch('utils.config.MEMORY_TYPE', 'window'):
            
            start_time = time.time()
            
            from rag.rag import BlenderAssistantRAG
            rag = BlenderAssistantRAG()
            
            end_time = time.time()
            init_time = end_time - start_time
            
            # Should initialize in reasonable time (less than 5 seconds)
            assert init_time < 5.0, f"RAG initialization with memory took too long: {init_time}s"
    
    def test_memory_operations_dont_significantly_slow_queries(self):
        """Smoke test: Memory operations don't significantly slow down queries."""
        from retrieval.retriever import RetrievalResult
        
        mock_retrieval_results = [RetrievalResult(text="Test", score=0.8, metadata={})]
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = mock_retrieval_results
        mock_llm = Mock()
        # Create proper mock for LangChain BaseLanguageModel
        from langchain_core.language_models import BaseLanguageModel
        mock_base_model = Mock(spec=BaseLanguageModel)
        mock_base_model._llm_type = "fake"
        mock_base_model.invoke = Mock(return_value="test summary")
        mock_llm.model = mock_base_model
        mock_llm.invoke.return_value = "Response"
        
        # Test both with and without memory
        times = {}
        
        for memory_type in ['none', 'window']:
            with patch('retrieval.SemanticRetriever', return_value=mock_retriever), \
                 patch('rag.llms.groq_llm.GroqLLM', return_value=mock_llm), \
                 patch('utils.config.GROQ_API_KEY', 'test-key'), \
                 patch('utils.config.OPENAI_API_KEY', None), \
                 patch('utils.config.MEMORY_TYPE', memory_type):
                
                # Force reload to pick up patched values
                import rag.rag
                importlib.reload(rag.rag)
                from rag.rag import BlenderAssistantRAG
                rag = BlenderAssistantRAG()
                
                import time
                start_time = time.time()
                
                # Run a few queries
                for i in range(3):
                    result = rag.handle_query(f"Test query {i}")
                    assert "Response" in result  # Response should be in the result (along with citations)
                
                end_time = time.time()
                times[memory_type] = end_time - start_time
        
        # Memory version shouldn't be dramatically slower
        # Allow up to 3x slower (very generous threshold for smoke test)
        if times['none'] > 0:  # Avoid division by zero
            slowdown_factor = times['window'] / times['none']
            assert slowdown_factor < 3.0, f"Memory operations slowed queries by {slowdown_factor}x"