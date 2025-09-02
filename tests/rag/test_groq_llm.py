#!/usr/bin/env python3
"""
Unit tests for the GroqLLM class.

Tests the Groq language model wrapper functionality including initialization,
query processing, and error handling without making real API calls.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from rag.llms.groq_llm import GroqLLM


class TestGroqLLM:
    """Test suite for GroqLLM class."""
    
    @pytest.fixture
    def mock_chat_groq(self):
        """Mock ChatGroq instance."""
        mock = Mock()
        mock_response = Mock()
        mock_response.content = "Mocked response from Groq"
        mock.invoke.return_value = mock_response
        return mock
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration values."""
        return {
            'GROQ_MODEL': 'llama3-8b-8192',
            'GROQ_API_KEY': 'test-groq-api-key',
            'GROQ_TEMPERATURE': 0.1
        }
    
    @patch('rag.llms.groq_llm.ChatGroq')
    @patch('rag.llms.groq_llm.GROQ_MODEL', 'llama3-8b-8192')
    @patch('rag.llms.groq_llm.GROQ_API_KEY', 'test-api-key')
    @patch('rag.llms.groq_llm.GROQ_TEMPERATURE', 0.1)
    def test_initialization_success(self, mock_chat_groq_class, mock_chat_groq, caplog):
        """Test successful initialization of GroqLLM."""
        mock_chat_groq_class.return_value = mock_chat_groq
        
        with caplog.at_level(logging.INFO):
            groq_llm = GroqLLM()
        
        # Verify ChatGroq was initialized with correct parameters
        mock_chat_groq_class.assert_called_once_with(
            model='llama3-8b-8192',
            temperature=0.1,
            api_key='test-api-key'
        )
        
        # Verify model attribute is set
        assert groq_llm.model == mock_chat_groq
        
        # Verify initialization log message
        assert "Initialized GroqLLM with model: llama3-8b-8192" in caplog.text
    
    @patch('rag.llms.groq_llm.ChatGroq')
    @patch('rag.llms.groq_llm.GROQ_MODEL', 'test-model')
    @patch('rag.llms.groq_llm.GROQ_API_KEY', 'different-key')
    @patch('rag.llms.groq_llm.GROQ_TEMPERATURE', 0.5)
    def test_initialization_different_config(self, mock_chat_groq_class, caplog):
        """Test initialization with different configuration values."""
        with caplog.at_level(logging.INFO):
            GroqLLM()
        
        # Verify different configuration is used
        mock_chat_groq_class.assert_called_once_with(
            model='test-model',
            temperature=0.5,
            api_key='different-key'
        )
        
        assert "Initialized GroqLLM with model: test-model" in caplog.text
    
    @patch('rag.llms.groq_llm.ChatGroq')
    def test_invoke_success(self, mock_chat_groq_class, mock_chat_groq):
        """Test successful invoke operation."""
        mock_chat_groq_class.return_value = mock_chat_groq
        
        # Setup mock response
        mock_response = Mock()
        mock_response.content = "To extrude in Blender, select faces and press E."
        mock_chat_groq.invoke.return_value = mock_response
        
        groq_llm = GroqLLM()
        result = groq_llm.invoke(
            question="How do I extrude in Blender?",
            context="Blender modeling documentation about extrusion tools."
        )
        
        # Verify ChatGroq invoke was called
        assert mock_chat_groq.invoke.call_count == 1
        call_args = mock_chat_groq.invoke.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0].content is not None
        
        # Verify formatted prompt contains question and context
        formatted_prompt = call_args[0].content
        assert "How do I extrude in Blender?" in formatted_prompt
        assert "Blender modeling documentation about extrusion tools." in formatted_prompt
        
        # Verify response
        assert result == "To extrude in Blender, select faces and press E."
    
    @patch('rag.llms.groq_llm.ChatGroq')
    @patch('rag.llms.groq_llm.blender_bot_template')
    def test_invoke_prompt_formatting(self, mock_template, mock_chat_groq_class, mock_chat_groq):
        """Test that prompt template is formatted correctly."""
        mock_chat_groq_class.return_value = mock_chat_groq
        mock_template.format.return_value = "Formatted prompt with context and question"
        
        mock_response = Mock()
        mock_response.content = "Formatted response"
        mock_chat_groq.invoke.return_value = mock_response
        
        groq_llm = GroqLLM()
        groq_llm.invoke(
            question="Test question",
            context="Test context"
        )
        
        # Verify template.format was called with correct arguments
        mock_template.format.assert_called_once_with(
            context="Test context",
            question="Test question"
        )
    
    @patch('rag.llms.groq_llm.ChatGroq')
    def test_invoke_empty_inputs(self, mock_chat_groq_class, mock_chat_groq):
        """Test invoke with empty question and context."""
        mock_chat_groq_class.return_value = mock_chat_groq
        
        mock_response = Mock()
        mock_response.content = "Please provide a specific question about Blender."
        mock_chat_groq.invoke.return_value = mock_response
        
        groq_llm = GroqLLM()
        result = groq_llm.invoke(question="", context="")
        
        # Verify empty inputs are handled
        assert result == "Please provide a specific question about Blender."
        mock_chat_groq.invoke.assert_called_once()
    
    @patch('rag.llms.groq_llm.ChatGroq')
    def test_invoke_long_inputs(self, mock_chat_groq_class, mock_chat_groq):
        """Test invoke with very long question and context."""
        mock_chat_groq_class.return_value = mock_chat_groq
        
        long_question = "How do I " + "really " * 500 + "complex modeling?"
        long_context = "This is " + "very " * 1000 + "detailed documentation."
        
        mock_response = Mock()
        mock_response.content = "For complex modeling tasks..."
        mock_chat_groq.invoke.return_value = mock_response
        
        groq_llm = GroqLLM()
        result = groq_llm.invoke(question=long_question, context=long_context)
        
        # Verify long inputs are processed
        assert result == "For complex modeling tasks..."
        mock_chat_groq.invoke.assert_called_once()
    
    @patch('rag.llms.groq_llm.ChatGroq')
    def test_invoke_special_characters(self, mock_chat_groq_class, mock_chat_groq):
        """Test invoke with special characters and Unicode."""
        mock_chat_groq_class.return_value = mock_chat_groq
        
        special_question = "How do I use Ctrl+R & Shift+A? 🎨 中文"
        special_context = "Context with symbols: @#$%^&*() and unicode: café"
        
        mock_response = Mock()
        mock_response.content = "Special characters are handled correctly."
        mock_chat_groq.invoke.return_value = mock_response
        
        groq_llm = GroqLLM()
        result = groq_llm.invoke(question=special_question, context=special_context)
        
        # Verify special characters are handled
        assert result == "Special characters are handled correctly."
        call_args = mock_chat_groq.invoke.call_args[0][0][0]
        assert special_question in call_args.content
        assert special_context in call_args.content
    
    @patch('rag.llms.groq_llm.ChatGroq')
    def test_invoke_api_exception(self, mock_chat_groq_class, mock_chat_groq, caplog):
        """Test invoke error handling when API call fails."""
        mock_chat_groq_class.return_value = mock_chat_groq
        mock_chat_groq.invoke.side_effect = Exception("API rate limit exceeded")
        
        groq_llm = GroqLLM()
        
        with caplog.at_level(logging.ERROR):
            result = groq_llm.invoke(
                question="Test question",
                context="Test context"
            )
        
        # Verify error handling
        assert result == "Error occurred while processing your request."
        assert "Error occurred while invoking GroqLLM: API rate limit exceeded" in caplog.text
    
    @patch('rag.llms.groq_llm.ChatGroq')
    def test_invoke_connection_error(self, mock_chat_groq_class, mock_chat_groq, caplog):
        """Test invoke error handling for connection errors."""
        mock_chat_groq_class.return_value = mock_chat_groq
        mock_chat_groq.invoke.side_effect = ConnectionError("Network connection failed")
        
        groq_llm = GroqLLM()
        
        with caplog.at_level(logging.ERROR):
            result = groq_llm.invoke(
                question="Test question",
                context="Test context"
            )
        
        assert result == "Error occurred while processing your request."
        assert "Error occurred while invoking GroqLLM: Network connection failed" in caplog.text
    
    @patch('rag.llms.groq_llm.ChatGroq')
    def test_invoke_authentication_error(self, mock_chat_groq_class, mock_chat_groq, caplog):
        """Test invoke error handling for authentication errors."""
        mock_chat_groq_class.return_value = mock_chat_groq
        mock_chat_groq.invoke.side_effect = Exception("Invalid API key")
        
        groq_llm = GroqLLM()
        
        with caplog.at_level(logging.ERROR):
            result = groq_llm.invoke(
                question="Test question",
                context="Test context"
            )
        
        assert result == "Error occurred while processing your request."
        assert "Error occurred while invoking GroqLLM: Invalid API key" in caplog.text
    
    @patch('rag.llms.groq_llm.ChatGroq')
    def test_invoke_debug_logging(self, mock_chat_groq_class, mock_chat_groq, caplog):
        """Test debug logging during invoke operation."""
        mock_chat_groq_class.return_value = mock_chat_groq
        
        mock_response = Mock()
        mock_response.content = "Debug test response"
        mock_chat_groq.invoke.return_value = mock_response
        
        groq_llm = GroqLLM()
        
        with caplog.at_level(logging.DEBUG):
            groq_llm.invoke(
                question="Debug test question",
                context="Debug test context"
            )
        
        # Verify debug logs contain request and response
        debug_messages = [record.message for record in caplog.records if record.levelno == logging.DEBUG]
        
        # Should have request and response debug messages
        request_logged = any("GroqLLM request:" in msg for msg in debug_messages)
        response_logged = any("GroqLLM response: Debug test response" in msg for msg in debug_messages)
        
        assert request_logged
        assert response_logged


@pytest.mark.unit
class TestGroqLLMEdgeCases:
    """Edge case tests for GroqLLM."""
    
    @patch('rag.llms.groq_llm.ChatGroq')
    def test_invoke_none_inputs(self, mock_chat_groq_class):
        """Test invoke with None inputs."""
        mock_chat_groq = Mock()
        mock_response = Mock()
        mock_response.content = "Response with None inputs"
        mock_chat_groq.invoke.return_value = mock_response
        mock_chat_groq_class.return_value = mock_chat_groq
        
        groq_llm = GroqLLM()
        
        # LangChain template apparently handles None gracefully
        result = groq_llm.invoke(question=None, context=None)
        assert result == "Response with None inputs"
    
    @patch('rag.llms.groq_llm.ChatGroq')
    def test_invoke_non_string_inputs(self, mock_chat_groq_class):
        """Test invoke with non-string inputs."""
        mock_chat_groq = Mock()
        mock_chat_groq_class.return_value = mock_chat_groq
        
        mock_response = Mock()
        mock_response.content = "Handled numeric inputs"
        mock_chat_groq.invoke.return_value = mock_response
        
        groq_llm = GroqLLM()
        
        # Should handle type conversion gracefully or raise appropriate error
        try:
            result = groq_llm.invoke(question=123, context=456)
            # If it succeeds, verify the result
            assert isinstance(result, str)
        except (TypeError, AttributeError):
            # This is also acceptable behavior
            pass
    
    @patch('rag.llms.groq_llm.ChatGroq')
    def test_response_content_none(self, mock_chat_groq_class):
        """Test handling when response.content is None."""
        mock_chat_groq = Mock()
        mock_chat_groq_class.return_value = mock_chat_groq
        
        mock_response = Mock()
        mock_response.content = None
        mock_chat_groq.invoke.return_value = mock_response
        
        groq_llm = GroqLLM()
        result = groq_llm.invoke(
            question="Test question",
            context="Test context"
        )
        
        # Should handle None content gracefully
        assert result is None
    
    @patch('rag.llms.groq_llm.ChatGroq')
    def test_response_missing_content_attribute(self, mock_chat_groq_class):
        """Test handling when response object lacks content attribute."""
        mock_chat_groq = Mock()
        mock_chat_groq_class.return_value = mock_chat_groq
        
        # Create a response object without content attribute
        class ResponseWithoutContent:
            pass
        
        mock_response = ResponseWithoutContent()
        mock_chat_groq.invoke.return_value = mock_response
        
        groq_llm = GroqLLM()
        
        # The LLM class catches the AttributeError and returns error message
        result = groq_llm.invoke(
            question="Test question",
            context="Test context"
        )
        
        assert result == "Error occurred while processing your request."


@pytest.mark.integration
class TestGroqLLMIntegration:
    """Integration-style tests for GroqLLM (with mocked external calls)."""
    
    @patch('rag.llms.groq_llm.ChatGroq')
    def test_end_to_end_workflow(self, mock_chat_groq_class):
        """Test complete workflow from initialization to response."""
        mock_chat_groq = Mock()
        mock_chat_groq_class.return_value = mock_chat_groq
        
        mock_response = Mock()
        mock_response.content = "To create a cube in Blender, press Shift+A and select Mesh > Cube."
        mock_chat_groq.invoke.return_value = mock_response
        
        # Initialize and use GroqLLM
        groq_llm = GroqLLM()
        result = groq_llm.invoke(
            question="How do I create a cube in Blender?",
            context="Blender documentation about mesh creation and basic objects."
        )
        
        # Verify complete workflow
        assert "cube in Blender" in result
        mock_chat_groq.invoke.assert_called_once()
    
    @patch('rag.llms.groq_llm.ChatGroq')
    @patch('rag.llms.groq_llm.GROQ_API_KEY', None)
    def test_missing_api_key_initialization(self, mock_chat_groq_class):
        """Test initialization behavior when API key is None."""
        # ChatGroq should still be called with None API key
        GroqLLM()
        mock_chat_groq_class.assert_called_once()
        call_kwargs = mock_chat_groq_class.call_args.kwargs
        assert call_kwargs['api_key'] is None