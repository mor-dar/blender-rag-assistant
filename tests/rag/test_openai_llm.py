#!/usr/bin/env python3
"""
Unit tests for the OpenAILLM class.

Tests the OpenAI language model wrapper functionality including initialization,
query processing, and error handling without making real API calls.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from rag.llms.openai_llm import OpenAILLM


class TestOpenAILLM:
    """Test suite for OpenAILLM class."""
    
    @pytest.fixture
    def mock_chat_openai(self):
        """Mock ChatOpenAI instance."""
        mock = Mock()
        mock_response = Mock()
        mock_response.content = "Mocked response from OpenAI"
        mock.invoke.return_value = mock_response
        return mock
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration values."""
        return {
            'OPENAI_MODEL': 'gpt-3.5-turbo',
            'OPENAI_API_KEY': 'test-openai-api-key',
            'OPENAI_TEMPERATURE': 0.1,
            'OPENAI_MAX_TOKENS': 1000
        }
    
    @patch('rag.llms.openai_llm.ChatOpenAI')
    @patch('rag.llms.openai_llm.OPENAI_MODEL', 'gpt-3.5-turbo')
    @patch('rag.llms.openai_llm.OPENAI_API_KEY', 'test-api-key')
    @patch('rag.llms.openai_llm.OPENAI_TEMPERATURE', 0.1)
    @patch('rag.llms.openai_llm.OPENAI_MAX_TOKENS', 1000)
    def test_initialization_success(self, mock_chat_openai_class, mock_chat_openai, caplog):
        """Test successful initialization of OpenAILLM."""
        mock_chat_openai_class.return_value = mock_chat_openai
        
        with caplog.at_level(logging.INFO):
            openai_llm = OpenAILLM()
        
        # Verify ChatOpenAI was initialized with correct parameters
        mock_chat_openai_class.assert_called_once_with(
            model='gpt-3.5-turbo',
            temperature=0.1,
            api_key='test-api-key',
            max_tokens=1000
        )
        
        # Verify model attribute is set
        assert openai_llm.model == mock_chat_openai
        
        # Verify initialization log message
        assert "Initialized OpenAI LLM with model: gpt-3.5-turbo" in caplog.text
    
    @patch('rag.llms.openai_llm.ChatOpenAI')
    @patch('rag.llms.openai_llm.OPENAI_MODEL', 'gpt-4')
    @patch('rag.llms.openai_llm.OPENAI_API_KEY', 'different-key')
    @patch('rag.llms.openai_llm.OPENAI_TEMPERATURE', 0.7)
    @patch('rag.llms.openai_llm.OPENAI_MAX_TOKENS', 2000)
    def test_initialization_different_config(self, mock_chat_openai_class, caplog):
        """Test initialization with different configuration values."""
        with caplog.at_level(logging.INFO):
            OpenAILLM()
        
        # Verify different configuration is used
        mock_chat_openai_class.assert_called_once_with(
            model='gpt-4',
            temperature=0.7,
            api_key='different-key',
            max_tokens=2000
        )
        
        assert "Initialized OpenAI LLM with model: gpt-4" in caplog.text
    
    @patch('rag.llms.openai_llm.ChatOpenAI')
    def test_invoke_success(self, mock_chat_openai_class, mock_chat_openai):
        """Test successful invoke operation."""
        mock_chat_openai_class.return_value = mock_chat_openai
        
        # Setup mock response
        mock_response = Mock()
        mock_response.content = "To subdivide in Blender, select faces and press Ctrl+2."
        mock_chat_openai.invoke.return_value = mock_response
        
        openai_llm = OpenAILLM()
        result = openai_llm.invoke(
            question="How do I subdivide in Blender?",
            context="Blender modeling documentation about subdivision surface modifier."
        )
        
        # Verify ChatOpenAI invoke was called
        assert mock_chat_openai.invoke.call_count == 1
        call_args = mock_chat_openai.invoke.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0].content is not None
        
        # Verify formatted prompt contains question and context
        formatted_prompt = call_args[0].content
        assert "How do I subdivide in Blender?" in formatted_prompt
        assert "Blender modeling documentation about subdivision surface modifier." in formatted_prompt
        
        # Verify response
        assert result == "To subdivide in Blender, select faces and press Ctrl+2."
    
    @patch('rag.llms.openai_llm.ChatOpenAI')
    @patch('rag.llms.openai_llm.blender_bot_template')
    def test_invoke_prompt_formatting(self, mock_template, mock_chat_openai_class, mock_chat_openai):
        """Test that prompt template is formatted correctly."""
        mock_chat_openai_class.return_value = mock_chat_openai
        mock_template.format.return_value = "Formatted prompt with context and question"
        
        mock_response = Mock()
        mock_response.content = "Formatted response"
        mock_chat_openai.invoke.return_value = mock_response
        
        openai_llm = OpenAILLM()
        openai_llm.invoke(
            question="Test question",
            context="Test context"
        )
        
        # Verify template.format was called with correct arguments
        mock_template.format.assert_called_once_with(
            context="Test context",
            question="Test question"
        )
    
    @patch('rag.llms.openai_llm.ChatOpenAI')
    def test_invoke_empty_inputs(self, mock_chat_openai_class, mock_chat_openai):
        """Test invoke with empty question and context."""
        mock_chat_openai_class.return_value = mock_chat_openai
        
        mock_response = Mock()
        mock_response.content = "Please provide a specific question about Blender."
        mock_chat_openai.invoke.return_value = mock_response
        
        openai_llm = OpenAILLM()
        result = openai_llm.invoke(question="", context="")
        
        # Verify empty inputs are handled
        assert result == "Please provide a specific question about Blender."
        mock_chat_openai.invoke.assert_called_once()
    
    @patch('rag.llms.openai_llm.ChatOpenAI')
    def test_invoke_long_inputs(self, mock_chat_openai_class, mock_chat_openai):
        """Test invoke with very long question and context."""
        mock_chat_openai_class.return_value = mock_chat_openai
        
        long_question = "How do I " + "efficiently " * 500 + "render animations?"
        long_context = "This is " + "detailed " * 1000 + "rendering documentation."
        
        mock_response = Mock()
        mock_response.content = "For efficient rendering..."
        mock_chat_openai.invoke.return_value = mock_response
        
        openai_llm = OpenAILLM()
        result = openai_llm.invoke(question=long_question, context=long_context)
        
        # Verify long inputs are processed
        assert result == "For efficient rendering..."
        mock_chat_openai.invoke.assert_called_once()
    
    @patch('rag.llms.openai_llm.ChatOpenAI')
    def test_invoke_special_characters(self, mock_chat_openai_class, mock_chat_openai):
        """Test invoke with special characters and Unicode."""
        mock_chat_openai_class.return_value = mock_chat_openai
        
        special_question = "How do I use F12 & Ctrl+Z? 🎬 日本語"
        special_context = "Context with symbols: !@#$%^&*() and unicode: naïve"
        
        mock_response = Mock()
        mock_response.content = "Special characters are processed correctly."
        mock_chat_openai.invoke.return_value = mock_response
        
        openai_llm = OpenAILLM()
        result = openai_llm.invoke(question=special_question, context=special_context)
        
        # Verify special characters are handled
        assert result == "Special characters are processed correctly."
        call_args = mock_chat_openai.invoke.call_args[0][0][0]
        assert special_question in call_args.content
        assert special_context in call_args.content
    
    @patch('rag.llms.openai_llm.ChatOpenAI')
    def test_invoke_api_exception(self, mock_chat_openai_class, mock_chat_openai, caplog):
        """Test invoke error handling when API call fails."""
        mock_chat_openai_class.return_value = mock_chat_openai
        mock_chat_openai.invoke.side_effect = Exception("OpenAI API quota exceeded")
        
        openai_llm = OpenAILLM()
        
        with caplog.at_level(logging.ERROR):
            result = openai_llm.invoke(
                question="Test question",
                context="Test context"
            )
        
        # Verify error handling
        assert result == "Error occurred while processing your request."
        assert "Error occurred while invoking OpenAI LLM: OpenAI API quota exceeded" in caplog.text
    
    @patch('rag.llms.openai_llm.ChatOpenAI')
    def test_invoke_connection_error(self, mock_chat_openai_class, mock_chat_openai, caplog):
        """Test invoke error handling for connection errors."""
        mock_chat_openai_class.return_value = mock_chat_openai
        mock_chat_openai.invoke.side_effect = ConnectionError("Unable to reach OpenAI servers")
        
        openai_llm = OpenAILLM()
        
        with caplog.at_level(logging.ERROR):
            result = openai_llm.invoke(
                question="Test question",
                context="Test context"
            )
        
        assert result == "Error occurred while processing your request."
        assert "Error occurred while invoking OpenAI LLM: Unable to reach OpenAI servers" in caplog.text
    
    @patch('rag.llms.openai_llm.ChatOpenAI')
    def test_invoke_authentication_error(self, mock_chat_openai_class, mock_chat_openai, caplog):
        """Test invoke error handling for authentication errors."""
        mock_chat_openai_class.return_value = mock_chat_openai
        mock_chat_openai.invoke.side_effect = Exception("Invalid OpenAI API key")
        
        openai_llm = OpenAILLM()
        
        with caplog.at_level(logging.ERROR):
            result = openai_llm.invoke(
                question="Test question",
                context="Test context"
            )
        
        assert result == "Error occurred while processing your request."
        assert "Error occurred while invoking OpenAI LLM: Invalid OpenAI API key" in caplog.text
    
    @patch('rag.llms.openai_llm.ChatOpenAI')
    def test_invoke_rate_limit_error(self, mock_chat_openai_class, mock_chat_openai, caplog):
        """Test invoke error handling for rate limit errors."""
        mock_chat_openai_class.return_value = mock_chat_openai
        mock_chat_openai.invoke.side_effect = Exception("Rate limit exceeded")
        
        openai_llm = OpenAILLM()
        
        with caplog.at_level(logging.ERROR):
            result = openai_llm.invoke(
                question="Test question",
                context="Test context"
            )
        
        assert result == "Error occurred while processing your request."
        assert "Error occurred while invoking OpenAI LLM: Rate limit exceeded" in caplog.text
    
    @patch('rag.llms.openai_llm.ChatOpenAI')
    def test_invoke_debug_logging(self, mock_chat_openai_class, mock_chat_openai, caplog):
        """Test debug logging during invoke operation."""
        mock_chat_openai_class.return_value = mock_chat_openai
        
        mock_response = Mock()
        mock_response.content = "Debug test response"
        mock_chat_openai.invoke.return_value = mock_response
        
        openai_llm = OpenAILLM()
        
        with caplog.at_level(logging.DEBUG):
            openai_llm.invoke(
                question="Debug test question",
                context="Debug test context"
            )
        
        # Verify debug logs contain request and response
        debug_messages = [record.message for record in caplog.records if record.levelno == logging.DEBUG]
        
        # Should have request and response debug messages
        request_logged = any("OpenAI LLM request:" in msg for msg in debug_messages)
        response_logged = any("OpenAI LLM response: Debug test response" in msg for msg in debug_messages)
        
        assert request_logged
        assert response_logged
    
    @patch('rag.llms.openai_llm.ChatOpenAI')
    def test_invoke_token_limit_handling(self, mock_chat_openai_class, mock_chat_openai, caplog):
        """Test handling when response exceeds token limits."""
        mock_chat_openai_class.return_value = mock_chat_openai
        mock_chat_openai.invoke.side_effect = Exception("Maximum context length exceeded")
        
        openai_llm = OpenAILLM()
        
        with caplog.at_level(logging.ERROR):
            result = openai_llm.invoke(
                question="Very long question...",
                context="Very long context..."
            )
        
        assert result == "Error occurred while processing your request."
        assert "Maximum context length exceeded" in caplog.text


@pytest.mark.unit
class TestOpenAILLMEdgeCases:
    """Edge case tests for OpenAILLM."""
    
    @patch('rag.llms.openai_llm.ChatOpenAI')
    def test_invoke_none_inputs(self, mock_chat_openai_class):
        """Test invoke with None inputs."""
        mock_chat_openai = Mock()
        mock_response = Mock()
        mock_response.content = "Response with None inputs"
        mock_chat_openai.invoke.return_value = mock_response
        mock_chat_openai_class.return_value = mock_chat_openai
        
        openai_llm = OpenAILLM()
        
        # LangChain template apparently handles None gracefully
        result = openai_llm.invoke(question=None, context=None)
        assert result == "Response with None inputs"
    
    @patch('rag.llms.openai_llm.ChatOpenAI')
    def test_invoke_non_string_inputs(self, mock_chat_openai_class):
        """Test invoke with non-string inputs."""
        mock_chat_openai = Mock()
        mock_chat_openai_class.return_value = mock_chat_openai
        
        mock_response = Mock()
        mock_response.content = "Handled numeric inputs"
        mock_chat_openai.invoke.return_value = mock_response
        
        openai_llm = OpenAILLM()
        
        # Should handle type conversion gracefully or raise appropriate error
        try:
            result = openai_llm.invoke(question=789, context=101112)
            # If it succeeds, verify the result
            assert isinstance(result, str)
        except (TypeError, AttributeError):
            # This is also acceptable behavior
            pass
    
    @patch('rag.llms.openai_llm.ChatOpenAI')
    def test_response_content_none(self, mock_chat_openai_class):
        """Test handling when response.content is None."""
        mock_chat_openai = Mock()
        mock_chat_openai_class.return_value = mock_chat_openai
        
        mock_response = Mock()
        mock_response.content = None
        mock_chat_openai.invoke.return_value = mock_response
        
        openai_llm = OpenAILLM()
        result = openai_llm.invoke(
            question="Test question",
            context="Test context"
        )
        
        # Should handle None content gracefully
        assert result is None
    
    @patch('rag.llms.openai_llm.ChatOpenAI')
    def test_response_missing_content_attribute(self, mock_chat_openai_class):
        """Test handling when response object lacks content attribute."""
        mock_chat_openai = Mock()
        mock_chat_openai_class.return_value = mock_chat_openai
        
        # Create a response object without content attribute
        class ResponseWithoutContent:
            pass
        
        mock_response = ResponseWithoutContent()
        mock_chat_openai.invoke.return_value = mock_response
        
        openai_llm = OpenAILLM()
        
        # The LLM class catches the AttributeError and returns error message
        result = openai_llm.invoke(
            question="Test question",
            context="Test context"
        )
        
        assert result == "Error occurred while processing your request."
    
    @patch('rag.llms.openai_llm.ChatOpenAI')
    def test_empty_response_content(self, mock_chat_openai_class):
        """Test handling when response.content is empty string."""
        mock_chat_openai = Mock()
        mock_chat_openai_class.return_value = mock_chat_openai
        
        mock_response = Mock()
        mock_response.content = ""
        mock_chat_openai.invoke.return_value = mock_response
        
        openai_llm = OpenAILLM()
        result = openai_llm.invoke(
            question="Test question",
            context="Test context"
        )
        
        # Should return empty string
        assert result == ""


@pytest.mark.integration
class TestOpenAILLMIntegration:
    """Integration-style tests for OpenAILLM (with mocked external calls)."""
    
    @patch('rag.llms.openai_llm.ChatOpenAI')
    def test_end_to_end_workflow(self, mock_chat_openai_class):
        """Test complete workflow from initialization to response."""
        mock_chat_openai = Mock()
        mock_chat_openai_class.return_value = mock_chat_openai
        
        mock_response = Mock()
        mock_response.content = "To create materials in Blender, go to the Shading workspace and add a new material."
        mock_chat_openai.invoke.return_value = mock_response
        
        # Initialize and use OpenAILLM
        openai_llm = OpenAILLM()
        result = openai_llm.invoke(
            question="How do I create materials in Blender?",
            context="Blender documentation about material creation and the Shading workspace."
        )
        
        # Verify complete workflow
        assert "materials in Blender" in result
        mock_chat_openai.invoke.assert_called_once()
    
    @patch('rag.llms.openai_llm.ChatOpenAI')
    @patch('rag.llms.openai_llm.OPENAI_API_KEY', None)
    def test_missing_api_key_initialization(self, mock_chat_openai_class):
        """Test initialization behavior when API key is None."""
        # ChatOpenAI should still be called with None API key
        OpenAILLM()
        mock_chat_openai_class.assert_called_once()
        call_kwargs = mock_chat_openai_class.call_args.kwargs
        assert call_kwargs['api_key'] is None
    
    @patch('rag.llms.openai_llm.ChatOpenAI')
    def test_complex_blender_workflow_simulation(self, mock_chat_openai_class):
        """Test simulation of complex Blender workflow questions."""
        mock_chat_openai = Mock()
        mock_chat_openai_class.return_value = mock_chat_openai
        
        # Simulate multiple related questions in a workflow
        questions_and_responses = [
            ("How do I start a new project?", "To start a new project in Blender..."),
            ("How do I add lighting?", "To add lighting, press Shift+A and select Light..."),
            ("How do I render the scene?", "To render, press F12 or go to Render menu...")
        ]
        
        openai_llm = OpenAILLM()
        
        for question, expected_response in questions_and_responses:
            mock_response = Mock()
            mock_response.content = expected_response
            mock_chat_openai.invoke.return_value = mock_response
            
            result = openai_llm.invoke(
                question=question,
                context="Relevant Blender documentation context"
            )
            
            assert result == expected_response
        
        # Verify all questions were processed
        assert mock_chat_openai.invoke.call_count == 3


@pytest.mark.performance
class TestOpenAILLMPerformance:
    """Performance-related tests for OpenAILLM."""
    
    @patch('rag.llms.openai_llm.ChatOpenAI')
    def test_large_context_handling(self, mock_chat_openai_class):
        """Test handling of large context without performance degradation."""
        mock_chat_openai = Mock()
        mock_chat_openai_class.return_value = mock_chat_openai
        
        # Create very large context (simulating extensive documentation)
        large_context = "Blender documentation section: " + "content " * 10000
        
        mock_response = Mock()
        mock_response.content = "Processed large context successfully"
        mock_chat_openai.invoke.return_value = mock_response
        
        openai_llm = OpenAILLM()
        result = openai_llm.invoke(
            question="How do I work with large scenes?",
            context=large_context
        )
        
        # Should handle large context without issues
        assert result == "Processed large context successfully"
        mock_chat_openai.invoke.assert_called_once()
        
        # Verify the large context was passed through
        call_args = mock_chat_openai.invoke.call_args[0][0][0]
        assert "content " * 10000 in call_args.content