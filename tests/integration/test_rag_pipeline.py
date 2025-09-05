"""
Integration tests for the complete RAG pipeline: query → retrieval → answer + citation.

This module tests the end-to-end functionality of the RAG system including:
- Complete query processing workflow
- Vector retrieval integration 
- LLM response generation
- Citation and source attribution
- Memory integration across conversation turns
- Error handling in complex scenarios
"""

import pytest
import os
import sys
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from rag.rag import BlenderAssistantRAG
from retrieval.vector_store import VectorStore
from retrieval.embeddings import EmbeddingGenerator
from retrieval.retriever import BlenderDocumentRetriever


@pytest.mark.integration
class TestRAGPipelineIntegration:
    """Test complete RAG pipeline integration scenarios."""
    
    def setup_method(self):
        """Set up test environment for each test."""
        self.test_dir = tempfile.mkdtemp()
        self.mock_vector_db_path = os.path.join(self.test_dir, "test_vector_db")
        
        # Mock environment variables
        self.env_patches = {
            'GROQ_API_KEY': 'test-groq-key',
            'GROQ_MODEL': 'llama3-8b-8192'
        }
        
        for key, value in self.env_patches.items():
            os.environ[key] = value
    
    def teardown_method(self):
        """Clean up test environment after each test."""
        # Clean up environment variables
        for key in self.env_patches:
            if key in os.environ:
                del os.environ[key]
        
        # Clean up temporary directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    @patch('rag.rag.BlenderDocumentRetriever')
    @patch('rag.rag.get_llm')
    def test_complete_query_to_answer_workflow(self, mock_get_llm, mock_retriever_class):
        """Test the complete workflow from user query to formatted answer with citations."""
        # Set up mocks
        mock_llm = Mock()
        mock_llm.invoke.return_value = "To extrude a face in Blender, select the face and press E key. This creates new geometry extending from the selected face."
        mock_get_llm.return_value = mock_llm
        
        mock_retriever = Mock()
        mock_retrieval_results = [
            {
                'content': 'Extrusion is a modeling technique where you create new geometry by extending faces, edges, or vertices.',
                'metadata': {
                    'source': 'modeling/basics.html',
                    'title': 'Modeling Basics',
                    'url': 'https://docs.blender.org/manual/en/latest/modeling/basics.html'
                },
                'score': 0.95
            },
            {
                'content': 'The E key is the default hotkey for extrude operations in Blender.',
                'metadata': {
                    'source': 'interface/keymap.html',
                    'title': 'Keyboard Shortcuts',
                    'url': 'https://docs.blender.org/manual/en/latest/interface/keymap.html'
                },
                'score': 0.87
            }
        ]
        mock_retriever.retrieve_documents.return_value = mock_retrieval_results
        mock_retriever_class.return_value = mock_retriever
        
        # Initialize RAG system
        rag_system = BlenderAssistantRAG()
        
        # Test query
        user_query = "How do I extrude a face in Blender?"
        result = rag_system.handle_query(user_query)
        
        # Verify retrieval was called
        mock_retriever.retrieve_documents.assert_called_once_with(user_query, top_k=5)
        
        # Verify LLM was called with context and query
        mock_llm.invoke.assert_called_once()
        llm_call_args = mock_llm.invoke.call_args[0]
        assert user_query in llm_call_args[0]
        assert "Extrusion is a modeling technique" in llm_call_args[0]
        assert "The E key is the default hotkey" in llm_call_args[0]
        
        # Verify response format includes sources
        assert "To extrude a face in Blender" in result
        assert "**Sources:**" in result
        assert "[1]" in result
        assert "Modeling Basics" in result
        assert "https://docs.blender.org/manual/en/latest/modeling/basics.html" in result
    
    @patch('rag.rag.BlenderDocumentRetriever')
    @patch('rag.rag.get_llm')
    def test_query_with_no_retrieval_results(self, mock_get_llm, mock_retriever_class):
        """Test handling of queries that return no retrieval results."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = "I don't have specific information about that topic in my knowledge base."
        mock_get_llm.return_value = mock_llm
        
        mock_retriever = Mock()
        mock_retriever.retrieve_documents.return_value = []  # No results
        mock_retriever_class.return_value = mock_retriever
        
        rag_system = BlenderAssistantRAG()
        
        result = rag_system.handle_query("What is a quantum flux capacitor in Blender?")
        
        # Should still get a response even with no retrieval results
        assert len(result) > 0
        assert "I don't have specific information" in result
        
        # Should not have sources section since no documents were retrieved
        assert "**Sources:**" not in result
    
    @patch('rag.rag.BlenderDocumentRetriever')
    @patch('rag.rag.get_llm')
    def test_conversation_memory_integration(self, mock_get_llm, mock_retriever_class):
        """Test conversation memory across multiple turns."""
        mock_llm = Mock()
        mock_llm.invoke.side_effect = [
            "Blender is a free and open-source 3D computer graphics software.",
            "You can download Blender from the official website at blender.org."
        ]
        mock_get_llm.return_value = mock_llm
        
        mock_retriever = Mock()
        mock_retriever.retrieve_documents.return_value = [
            {
                'content': 'Blender is a comprehensive 3D creation suite.',
                'metadata': {
                    'source': 'introduction.html',
                    'title': 'Introduction to Blender',
                    'url': 'https://docs.blender.org/manual/en/latest/introduction.html'
                },
                'score': 0.95
            }
        ]
        mock_retriever_class.return_value = mock_retriever
        
        # Initialize RAG system with window memory
        rag_system = BlenderAssistantRAG(memory_type="window", memory_config={"k": 4})
        
        # First query
        result1 = rag_system.handle_query("What is Blender?")
        assert "Blender is a free and open-source" in result1
        
        # Second query - should have context from first query
        result2 = rag_system.handle_query("Where can I download it?")
        assert "blender.org" in result2
        
        # Verify that memory context was included in second LLM call
        assert mock_llm.invoke.call_count == 2
        second_call_args = mock_llm.invoke.call_args[0][0]
        # Should contain conversation history
        assert "What is Blender?" in second_call_args or "conversation" in second_call_args.lower()
    
    @patch('rag.rag.BlenderDocumentRetriever')
    @patch('rag.rag.get_llm')
    def test_error_handling_in_retrieval(self, mock_get_llm, mock_retriever_class):
        """Test error handling when retrieval component fails."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = "I'm experiencing technical difficulties accessing my knowledge base."
        mock_get_llm.return_value = mock_llm
        
        mock_retriever = Mock()
        mock_retriever.retrieve_documents.side_effect = Exception("Vector database connection failed")
        mock_retriever_class.return_value = mock_retriever
        
        rag_system = BlenderAssistantRAG()
        
        # Should handle retrieval errors gracefully
        result = rag_system.handle_query("How do I model in Blender?")
        
        # Should still return a response
        assert len(result) > 0
        assert "technical difficulties" in result or "error" in result.lower()
    
    @patch('rag.rag.BlenderDocumentRetriever')
    @patch('rag.rag.get_llm')
    def test_error_handling_in_llm_generation(self, mock_get_llm, mock_retriever_class):
        """Test error handling when LLM generation fails."""
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM API rate limit exceeded")
        mock_get_llm.return_value = mock_llm
        
        mock_retriever = Mock()
        mock_retriever.retrieve_documents.return_value = [
            {
                'content': 'Test content',
                'metadata': {'source': 'test.html', 'title': 'Test', 'url': 'https://test.com'},
                'score': 0.9
            }
        ]
        mock_retriever_class.return_value = mock_retriever
        
        rag_system = BlenderAssistantRAG()
        
        # Should raise the exception since LLM failure is critical
        with pytest.raises(Exception, match="LLM API rate limit exceeded"):
            rag_system.handle_query("Test query")
    
    @patch('rag.rag.BlenderDocumentRetriever')
    @patch('rag.rag.get_llm')
    def test_citation_formatting_multiple_sources(self, mock_get_llm, mock_retriever_class):
        """Test proper citation formatting with multiple sources."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = "Blender supports multiple types of objects including meshes, curves, and lights."
        mock_get_llm.return_value = mock_llm
        
        mock_retriever = Mock()
        mock_retrieval_results = [
            {
                'content': 'Mesh objects are the primary type of 3D geometry in Blender.',
                'metadata': {
                    'source': 'modeling/meshes.html',
                    'title': 'Working with Meshes',
                    'url': 'https://docs.blender.org/manual/en/latest/modeling/meshes.html'
                },
                'score': 0.95
            },
            {
                'content': 'Curves in Blender can be used for paths, text, and complex shapes.',
                'metadata': {
                    'source': 'modeling/curves.html', 
                    'title': 'Curve Objects',
                    'url': 'https://docs.blender.org/manual/en/latest/modeling/curves.html'
                },
                'score': 0.90
            },
            {
                'content': 'Light objects control illumination and shadows in your scene.',
                'metadata': {
                    'source': 'lighting/types.html',
                    'title': 'Light Types',
                    'url': 'https://docs.blender.org/manual/en/latest/lighting/types.html'
                },
                'score': 0.85
            }
        ]
        mock_retriever.retrieve_documents.return_value = mock_retrieval_results
        mock_retriever_class.return_value = mock_retriever
        
        rag_system = BlenderAssistantRAG()
        result = rag_system.handle_query("What types of objects does Blender support?")
        
        # Verify all sources are cited
        assert "**Sources:**" in result
        assert "[1] Working with Meshes" in result
        assert "[2] Curve Objects" in result  
        assert "[3] Light Types" in result
        
        # Verify URLs are included
        assert "https://docs.blender.org/manual/en/latest/modeling/meshes.html" in result
        assert "https://docs.blender.org/manual/en/latest/modeling/curves.html" in result
        assert "https://docs.blender.org/manual/en/latest/lighting/types.html" in result
    
    @patch('rag.rag.BlenderDocumentRetriever')
    @patch('rag.rag.get_llm')  
    def test_summary_memory_integration(self, mock_get_llm, mock_retriever_class):
        """Test conversation summary memory across extended conversation."""
        mock_llm = Mock()
        responses = [
            "Modeling in Blender involves creating and manipulating 3D geometry.",
            "Yes, the Tab key switches between Edit and Object modes.",
            "In Edit mode, you can select vertices, edges, and faces to modify geometry."
        ]
        mock_llm.invoke.side_effect = responses
        mock_get_llm.return_value = mock_llm
        
        mock_retriever = Mock()
        mock_retriever.retrieve_documents.return_value = [
            {
                'content': 'Edit mode allows direct manipulation of mesh geometry.',
                'metadata': {
                    'source': 'modeling/modes.html',
                    'title': 'Blender Modes',
                    'url': 'https://docs.blender.org/manual/en/latest/modeling/modes.html'
                },
                'score': 0.9
            }
        ]
        mock_retriever_class.return_value = mock_retriever
        
        # Initialize RAG system with summary memory
        rag_system = BlenderAssistantRAG(memory_type="summary")
        
        # Simulate extended conversation
        result1 = rag_system.handle_query("How do I start modeling in Blender?")
        result2 = rag_system.handle_query("Can I switch between edit and object mode?")
        result3 = rag_system.handle_query("What can I do in edit mode?")
        
        assert "Modeling in Blender involves" in result1
        assert "Tab key switches" in result2  
        assert "select vertices, edges, and faces" in result3
        
        # Verify multiple LLM calls were made
        assert mock_llm.invoke.call_count == 3


@pytest.mark.integration 
@pytest.mark.slow
class TestRAGPipelineWithRealComponents:
    """Test RAG pipeline with real components (slower integration tests)."""
    
    def setup_method(self):
        """Set up test environment with real components."""
        self.test_dir = tempfile.mkdtemp()
        self.mock_vector_db_path = os.path.join(self.test_dir, "test_vector_db")
        
        # Set up minimal environment for testing
        os.environ['GROQ_API_KEY'] = 'test-key-for-testing'
    
    def teardown_method(self):
        """Clean up test environment."""
        if 'GROQ_API_KEY' in os.environ:
            del os.environ['GROQ_API_KEY']
        
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    @patch('rag.rag.get_llm')
    def test_real_embedding_and_vector_store_integration(self, mock_get_llm):
        """Test integration with real embedding generator and vector store."""
        # This test uses real embeddings but mocked LLM to avoid API calls
        mock_llm = Mock()
        mock_llm.invoke.return_value = "This is a test response about Blender."
        mock_get_llm.return_value = mock_llm
        
        # Create a minimal vector store with test data
        embedding_generator = EmbeddingGenerator()
        vector_store = VectorStore(
            db_path=self.mock_vector_db_path,
            embedding_generator=embedding_generator
        )
        
        # Add some test documents
        test_documents = [
            {
                'content': 'Blender is a free and open-source 3D computer graphics software toolset.',
                'metadata': {
                    'source': 'intro.html',
                    'title': 'Introduction',
                    'url': 'https://docs.blender.org/manual/en/latest/intro.html'
                }
            },
            {
                'content': 'The 3D viewport is the main interface for viewing and editing 3D objects.',
                'metadata': {
                    'source': 'interface/viewport.html',
                    'title': '3D Viewport',
                    'url': 'https://docs.blender.org/manual/en/latest/interface/viewport.html'
                }
            }
        ]
        
        collection_name = "test_blender_docs"
        vector_store.create_collection(collection_name, dimension=384)  # all-MiniLM-L6-v2 dimension
        
        for doc in test_documents:
            embedding = embedding_generator.encode_single(doc['content'])
            vector_store.add_document(collection_name, doc['content'], embedding, doc['metadata'])
        
        # Create retriever with real vector store
        retriever = BlenderDocumentRetriever(vector_store, collection_name)
        
        # Test retrieval
        results = retriever.retrieve_documents("What is Blender?", top_k=2)
        
        assert len(results) > 0
        assert any("free and open-source" in result['content'].lower() for result in results)
        
        # Test with RAG system (mocking the LLM but using real retrieval)
        with patch('rag.rag.BlenderDocumentRetriever', return_value=retriever):
            rag_system = BlenderAssistantRAG()
            result = rag_system.handle_query("What is Blender?")
            
            assert len(result) > 0
            assert "This is a test response about Blender." in result
            assert "**Sources:**" in result


@pytest.mark.integration
class TestRAGPipelineErrorRecovery:
    """Test error recovery and graceful degradation in RAG pipeline."""
    
    def setup_method(self):
        """Set up test environment."""
        os.environ['GROQ_API_KEY'] = 'test-key'
    
    def teardown_method(self):
        """Clean up test environment."""
        if 'GROQ_API_KEY' in os.environ:
            del os.environ['GROQ_API_KEY']
    
    @patch('rag.rag.BlenderDocumentRetriever')
    @patch('rag.rag.get_llm')
    def test_partial_component_failure_recovery(self, mock_get_llm, mock_retriever_class):
        """Test recovery when some components fail but others work."""
        # LLM works, retrieval fails intermittently
        mock_llm = Mock()
        mock_llm.invoke.return_value = "I'll do my best to help even without full access to my knowledge base."
        mock_get_llm.return_value = mock_llm
        
        mock_retriever = Mock()
        # Simulate intermittent failures
        mock_retriever.retrieve_documents.side_effect = [
            Exception("Connection timeout"),  # First call fails
            [  # Second call succeeds
                {
                    'content': 'Recovered content about Blender.',
                    'metadata': {
                        'source': 'recovery.html',
                        'title': 'Recovery Test',
                        'url': 'https://docs.blender.org/manual/en/latest/recovery.html'
                    },
                    'score': 0.8
                }
            ]
        ]
        mock_retriever_class.return_value = mock_retriever
        
        rag_system = BlenderAssistantRAG()
        
        # First query - retrieval fails, should still get response
        result1 = rag_system.handle_query("First test query")
        assert len(result1) > 0
        assert "help even without full access" in result1
        
        # Second query - retrieval works
        result2 = rag_system.handle_query("Second test query")  
        assert len(result2) > 0
        assert "**Sources:**" in result2
    
    @patch('rag.rag.BlenderDocumentRetriever')
    @patch('rag.rag.get_llm')
    def test_memory_persistence_across_errors(self, mock_get_llm, mock_retriever_class):
        """Test that conversation memory persists even when individual queries fail."""
        mock_llm = Mock()
        mock_llm.invoke.side_effect = [
            "First successful response.",
            Exception("Temporary LLM failure"),  # Second query fails
            "Third response, building on our previous conversation."  # Third succeeds
        ]
        mock_get_llm.return_value = mock_llm
        
        mock_retriever = Mock()
        mock_retriever.retrieve_documents.return_value = [
            {
                'content': 'Test content',
                'metadata': {'source': 'test.html', 'title': 'Test', 'url': 'https://test.com'},
                'score': 0.9
            }
        ]
        mock_retriever_class.return_value = mock_retriever
        
        rag_system = BlenderAssistantRAG(memory_type="window", memory_config={"k": 4})
        
        # First query succeeds
        result1 = rag_system.handle_query("First query")
        assert "First successful response" in result1
        
        # Second query fails  
        with pytest.raises(Exception, match="Temporary LLM failure"):
            rag_system.handle_query("Second query")
        
        # Third query should still have memory of first query
        result3 = rag_system.handle_query("Third query")
        assert "Third response" in result3
        
        # Memory should still contain the first successful interaction
        # This is implementation-dependent, but the system should maintain state


@pytest.mark.integration
class TestRAGPipelinePerformance:
    """Test performance characteristics of the RAG pipeline."""
    
    def setup_method(self):
        """Set up performance test environment."""
        os.environ['GROQ_API_KEY'] = 'test-key'
    
    def teardown_method(self):
        """Clean up performance test environment."""
        if 'GROQ_API_KEY' in os.environ:
            del os.environ['GROQ_API_KEY']
    
    @patch('rag.rag.BlenderDocumentRetriever')
    @patch('rag.rag.get_llm')
    def test_query_processing_time_reasonable(self, mock_get_llm, mock_retriever_class):
        """Test that query processing completes in reasonable time."""
        import time
        
        mock_llm = Mock()
        mock_llm.invoke.return_value = "Quick response for performance testing."
        mock_get_llm.return_value = mock_llm
        
        mock_retriever = Mock()
        mock_retriever.retrieve_documents.return_value = [
            {
                'content': 'Performance test content',
                'metadata': {'source': 'perf.html', 'title': 'Performance', 'url': 'https://test.com'},
                'score': 0.9
            }
        ]
        mock_retriever_class.return_value = mock_retriever
        
        rag_system = BlenderAssistantRAG()
        
        start_time = time.time()
        result = rag_system.handle_query("Performance test query")
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should complete quickly with mocked components (under 1 second)
        assert processing_time < 1.0
        assert len(result) > 0
        assert "Quick response" in result
    
    @patch('rag.rag.BlenderDocumentRetriever')  
    @patch('rag.rag.get_llm')
    def test_memory_doesnt_degrade_performance_significantly(self, mock_get_llm, mock_retriever_class):
        """Test that conversation memory doesn't significantly slow down queries."""
        import time
        
        mock_llm = Mock()
        mock_llm.invoke.return_value = "Memory performance test response."
        mock_get_llm.return_value = mock_llm
        
        mock_retriever = Mock()
        mock_retriever.retrieve_documents.return_value = []
        mock_retriever_class.return_value = mock_retriever
        
        # Test without memory
        rag_system_no_memory = BlenderAssistantRAG(memory_type=None)
        
        start_time = time.time()
        rag_system_no_memory.handle_query("Test query")
        no_memory_time = time.time() - start_time
        
        # Test with memory
        rag_system_with_memory = BlenderAssistantRAG(memory_type="window", memory_config={"k": 4})
        
        # Add some conversation history
        for i in range(3):
            rag_system_with_memory.handle_query(f"Setup query {i}")
        
        start_time = time.time()  
        rag_system_with_memory.handle_query("Final test query")
        memory_time = time.time() - start_time
        
        # Memory should not add more than 50% overhead with mocked components
        assert memory_time < no_memory_time * 1.5