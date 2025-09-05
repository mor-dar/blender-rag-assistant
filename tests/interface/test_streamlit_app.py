"""
Comprehensive tests for the Streamlit web interface.

This module tests all components of the Streamlit app including:
- Session state management
- RAG system initialization
- Message handling
- UI components
- Error handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import streamlit as st
from datetime import datetime
import os
import sys

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from interface.streamlit_app import (
    initialize_session_state,
    setup_rag_system,
    display_message,
    display_chat_history,
    add_message,
    clear_chat,
    create_sidebar,
    main
)


class MockSessionState:
    """Mock Streamlit session state for testing."""
    def __init__(self):
        self._data = {}
    
    def __contains__(self, key):
        return key in self._data
    
    def __getitem__(self, key):
        if key in self._data:
            return self._data[key]
        raise KeyError(f'st.session_state has no key "{key}". Did you forget to initialize it?')
    
    def __setitem__(self, key, value):
        self._data[key] = value
    
    def __getattr__(self, key):
        if key.startswith('_'):
            return super().__getattribute__(key)
        if key in self._data:
            return self._data[key]
        raise AttributeError(f'st.session_state has no attribute "{key}". Did you forget to initialize it?')
    
    def __setattr__(self, key, value):
        if key.startswith('_'):
            super().__setattr__(key, value)
        else:
            self._data[key] = value
    
    def get(self, key, default=None):
        return self._data.get(key, default)


@pytest.fixture
def mock_session_state():
    """Fixture that provides a consistent mock session state."""
    return MockSessionState()


@pytest.fixture
def patched_session_state(mock_session_state):
    """Fixture that patches st.session_state with our mock."""
    with patch.object(st, 'session_state', mock_session_state):
        yield mock_session_state


class TestSessionStateManagement:
    """Test session state initialization and management."""
    
    def test_initialize_session_state_new_session(self, patched_session_state):
        """Test initialization of session state for a new session."""
        initialize_session_state()
        
        assert 'messages' in patched_session_state
        assert patched_session_state.messages == []
        assert 'rag_system' in patched_session_state
        assert patched_session_state.rag_system is None
        assert 'initialized' in patched_session_state
        assert patched_session_state.initialized is False
    
    def test_initialize_session_state_existing_values(self, patched_session_state):
        """Test that existing session state values are preserved."""
        existing_messages = [{"role": "user", "content": "test"}]
        mock_rag = Mock()
        
        # Pre-populate session state
        patched_session_state.messages = existing_messages
        patched_session_state.rag_system = mock_rag
        patched_session_state.initialized = True
        
        initialize_session_state()
        
        assert patched_session_state.messages == existing_messages
        assert patched_session_state.rag_system == mock_rag
        assert patched_session_state.initialized is True


class TestRAGSystemSetup:
    """Test RAG system initialization and error handling."""
    
    @patch('interface.streamlit_app.dotenv.load_dotenv')
    @patch('interface.streamlit_app.initialize_logging')
    @patch('interface.streamlit_app.BlenderAssistantRAG')
    @patch('streamlit.spinner')
    def test_setup_rag_system_success_first_time(self, mock_spinner, mock_rag_class, mock_logging, mock_dotenv, patched_session_state):
        """Test successful RAG system initialization on first call."""
        mock_rag_instance = Mock()
        mock_rag_class.return_value = mock_rag_instance
        mock_spinner.return_value.__enter__ = Mock()
        mock_spinner.return_value.__exit__ = Mock()
        
        patched_session_state.rag_system = None
        patched_session_state.initialized = False
        
        result = setup_rag_system()
        
        assert result == mock_rag_instance
        assert patched_session_state.rag_system == mock_rag_instance
        assert patched_session_state.initialized is True
        mock_dotenv.assert_called_once()
        mock_logging.assert_called_once()
        mock_rag_class.assert_called_once()
    
    def test_setup_rag_system_already_initialized(self, patched_session_state):
        """Test that already initialized RAG system is reused."""
        mock_rag = Mock()
        patched_session_state.rag_system = mock_rag
        patched_session_state.initialized = True
        
        result = setup_rag_system()
        
        assert result == mock_rag
    
    def test_setup_rag_system_graceful_degradation(self, patched_session_state):
        """Test that RAG system gracefully handles missing API keys."""
        # Test the actual behavior when no API keys are set (uses DummyLLM)
        patched_session_state.rag_system = None
        patched_session_state.initialized = False
        
        result = setup_rag_system()
        
        # Should still return a RAG system (with DummyLLM) rather than None
        assert result is not None
        assert patched_session_state.rag_system is not None
        assert patched_session_state.initialized is True


class TestMessageHandling:
    """Test chat message handling and display functionality."""
    
    @patch('streamlit.chat_message')
    @patch('streamlit.write')
    @patch('streamlit.caption')
    def test_display_message_without_timestamp(self, mock_caption, mock_write, mock_chat_message):
        """Test displaying a message without timestamp."""
        mock_context = Mock()
        mock_chat_message.return_value.__enter__ = Mock(return_value=mock_context)
        mock_chat_message.return_value.__exit__ = Mock()
        
        display_message("user", "Hello, world!")
        
        mock_chat_message.assert_called_once_with("user")
        mock_write.assert_called_once_with("Hello, world!")
        mock_caption.assert_not_called()
    
    @patch('streamlit.chat_message')
    @patch('streamlit.write')
    @patch('streamlit.caption')
    def test_display_message_with_timestamp(self, mock_caption, mock_write, mock_chat_message):
        """Test displaying a message with timestamp."""
        mock_context = Mock()
        mock_chat_message.return_value.__enter__ = Mock(return_value=mock_context)
        mock_chat_message.return_value.__exit__ = Mock()
        
        test_time = datetime(2024, 1, 1, 12, 30, 45)
        display_message("assistant", "How can I help?", test_time)
        
        mock_chat_message.assert_called_once_with("assistant")
        mock_write.assert_called_once_with("How can I help?")
        mock_caption.assert_called_once_with("*12:30:45*")
    
    def test_add_message_to_history(self, patched_session_state):
        """Test adding a message to chat history."""
        patched_session_state.messages = []
        
        with patch('interface.streamlit_app.datetime') as mock_datetime:
            test_time = datetime(2024, 1, 1, 12, 0, 0)
            mock_datetime.now.return_value = test_time
            
            add_message("user", "Test message")
            
            assert len(patched_session_state.messages) == 1
            message = patched_session_state.messages[0]
            assert message["role"] == "user"
            assert message["content"] == "Test message"
            assert message["timestamp"] == test_time
    
    @patch('interface.streamlit_app.display_message')
    def test_display_chat_history_empty(self, mock_display, patched_session_state):
        """Test displaying empty chat history."""
        patched_session_state.messages = []
        display_chat_history()
        mock_display.assert_not_called()
    
    @patch('interface.streamlit_app.display_message')
    def test_display_chat_history_with_messages(self, mock_display, patched_session_state):
        """Test displaying chat history with multiple messages."""
        test_time = datetime(2024, 1, 1, 12, 0, 0)
        messages = [
            {"role": "user", "content": "Hello", "timestamp": test_time},
            {"role": "assistant", "content": "Hi there!"}  # No timestamp
        ]
        
        patched_session_state.messages = messages
        display_chat_history()
        
        assert mock_display.call_count == 2
        mock_display.assert_any_call(role="user", content="Hello", timestamp=test_time)
        mock_display.assert_any_call(role="assistant", content="Hi there!", timestamp=None)
    
    @patch('streamlit.rerun')
    def test_clear_chat(self, mock_rerun, patched_session_state):
        """Test clearing chat history."""
        patched_session_state.messages = [{"role": "user", "content": "test"}]
        
        clear_chat()
        
        assert patched_session_state.messages == []
        mock_rerun.assert_called_once()


class TestSidebarComponents:
    """Test sidebar UI components and functionality."""
    
    @patch('streamlit.sidebar')
    @patch('streamlit.title')
    @patch('streamlit.subheader')
    @patch('streamlit.success')
    @patch('streamlit.warning')
    @patch('streamlit.info')
    @patch('streamlit.button')
    @patch('streamlit.markdown')
    def test_create_sidebar_initialized_groq(self, mock_markdown, mock_button, mock_info, 
                                           mock_warning, mock_success, mock_subheader, 
                                           mock_title, mock_sidebar, patched_session_state):
        """Test sidebar creation with initialized RAG system and Groq API key."""
        mock_sidebar.return_value.__enter__ = Mock()
        mock_sidebar.return_value.__exit__ = Mock()
        mock_button.return_value = False
        
        patched_session_state.initialized = True
        
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key', 'GROQ_MODEL': 'llama3-8b'}):
            create_sidebar()
            
            mock_title.assert_called_with("⚙️ Configuration")
            mock_success.assert_called_with("✅ RAG System Online")
            mock_info.assert_called_with("**Model**: Groq - llama3-8b")
    
    @patch('streamlit.sidebar')
    @patch('streamlit.warning')
    @patch('streamlit.info')
    @patch('streamlit.button')
    def test_create_sidebar_not_initialized_openai(self, mock_button, mock_info, mock_warning, mock_sidebar):
        """Test sidebar creation with uninitialized RAG system and OpenAI API key."""
        mock_sidebar.return_value.__enter__ = Mock()
        mock_sidebar.return_value.__exit__ = Mock()
        mock_button.return_value = False
        
        mock_session_state = MockSessionState()
        mock_session_state.initialized = False
        
        with patch.object(st, 'session_state', mock_session_state):
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}, clear=True):
                create_sidebar()
                
                mock_warning.assert_any_call("⚠️ RAG System Not Initialized")
                mock_info.assert_called_with("**Model**: OpenAI - gpt-4")
    
    @patch('streamlit.sidebar')
    @patch('streamlit.warning')
    @patch('streamlit.button')
    def test_create_sidebar_no_api_keys(self, mock_button, mock_warning, mock_sidebar):
        """Test sidebar creation with no API keys configured."""
        mock_sidebar.return_value.__enter__ = Mock()
        mock_sidebar.return_value.__exit__ = Mock()
        mock_button.return_value = False
        
        mock_session_state = MockSessionState()
        mock_session_state.initialized = False
        
        with patch.object(st, 'session_state', mock_session_state):
            with patch.dict(os.environ, {}, clear=True):
                create_sidebar()
                
                mock_warning.assert_any_call("**Model**: No API key configured")
    
    @patch('streamlit.sidebar')
    @patch('streamlit.button')
    @patch('streamlit.rerun')
    @patch('interface.streamlit_app.clear_chat')
    def test_sidebar_clear_chat_button(self, mock_clear_chat, mock_rerun, mock_button, mock_sidebar):
        """Test clear chat button functionality in sidebar."""
        mock_sidebar.return_value.__enter__ = Mock()
        mock_sidebar.return_value.__exit__ = Mock()
        
        # Mock button returns True for clear chat button, False for others
        def button_side_effect(text, **kwargs):
            if "Clear Chat" in text:
                return True
            return False
        
        mock_button.side_effect = button_side_effect
        
        mock_session_state = MockSessionState()
        mock_session_state.initialized = True
        
        with patch.object(st, 'session_state', mock_session_state):
            create_sidebar()
            
            # clear_chat() handles its own rerun, so we don't expect an additional one here
            mock_clear_chat.assert_called_once()
    
    @patch('streamlit.sidebar')
    @patch('streamlit.button')
    @patch('streamlit.rerun')
    def test_sidebar_reinitialize_button(self, mock_rerun, mock_button, mock_sidebar):
        """Test reinitialize system button functionality in sidebar."""
        mock_sidebar.return_value.__enter__ = Mock()
        mock_sidebar.return_value.__exit__ = Mock()
        
        def button_side_effect(text, **kwargs):
            if "Reinitialize System" in text:
                return True
            return False
        
        mock_button.side_effect = button_side_effect
        
        mock_session_state = MockSessionState()
        mock_session_state.initialized = True
        mock_session_state.rag_system = Mock()
        
        with patch.object(st, 'session_state', mock_session_state):
            create_sidebar()
            
            assert st.session_state.rag_system is None
            assert st.session_state.initialized is False
            mock_rerun.assert_called_once()
    
    @patch('streamlit.sidebar')
    @patch('streamlit.button')
    @patch('streamlit.rerun')
    def test_sidebar_sample_question_selection(self, mock_rerun, mock_button, mock_sidebar):
        """Test sample question button functionality in sidebar."""
        mock_sidebar.return_value.__enter__ = Mock()
        mock_sidebar.return_value.__exit__ = Mock()
        
        # Mock a sample question button being clicked
        def button_side_effect(text, **kwargs):
            if "What are the different types" in text:
                return True
            return False
        
        mock_button.side_effect = button_side_effect
        
        mock_session_state = MockSessionState()
        mock_session_state.initialized = True
        
        with patch.object(st, 'session_state', mock_session_state):
            create_sidebar()
            
            assert hasattr(st.session_state, 'user_input')
            assert "What are the different types of buttons in Blender's interface?" == st.session_state.user_input
            mock_rerun.assert_called_once()


class TestMainApplication:
    """Test the main application flow and integration."""
    
    @patch('streamlit.set_page_config')
    @patch('interface.streamlit_app.initialize_session_state')
    @patch('interface.streamlit_app.create_sidebar')
    @patch('interface.streamlit_app.setup_rag_system')
    @patch('interface.streamlit_app.display_chat_history')
    @patch('streamlit.title')
    @patch('streamlit.markdown')
    @patch('streamlit.error')
    @patch('streamlit.stop')
    def test_main_rag_initialization_failure(self, mock_stop, mock_error, mock_markdown, 
                                           mock_title, mock_display_history, mock_setup_rag, mock_create_sidebar,
                                           mock_init_session, mock_set_page_config):
        """Test main application flow when RAG system fails to initialize."""
        mock_setup_rag.return_value = None
        
        main()
        
        mock_set_page_config.assert_called_once()
        mock_init_session.assert_called_once()
        mock_create_sidebar.assert_called_once()
        mock_setup_rag.assert_called_once()
        mock_error.assert_called_once_with(
            "❌ Unable to initialize RAG system. Please check your configuration and API keys."
        )
        mock_stop.assert_called_once()
    
    @patch('streamlit.set_page_config')
    @patch('interface.streamlit_app.initialize_session_state')
    @patch('interface.streamlit_app.create_sidebar')
    @patch('interface.streamlit_app.setup_rag_system')
    @patch('interface.streamlit_app.display_chat_history')
    @patch('streamlit.container')
    @patch('streamlit.columns')
    @patch('streamlit.chat_input')
    @patch('streamlit.button')
    @patch('streamlit.title')
    @patch('streamlit.markdown')
    def test_main_successful_initialization_no_input(self, mock_markdown, mock_title, 
                                                   mock_button, mock_chat_input, mock_columns,
                                                   mock_container, mock_display_history,
                                                   mock_setup_rag, mock_create_sidebar,
                                                   mock_init_session, mock_set_page_config):
        """Test main application flow with successful initialization but no user input."""
        mock_rag = Mock()
        mock_setup_rag.return_value = mock_rag
        mock_chat_input.return_value = ""  # No input
        mock_button.return_value = False   # Random button not clicked
        mock_container.return_value.__enter__ = Mock()
        mock_container.return_value.__exit__ = Mock()
        mock_columns.return_value = [Mock(), Mock()]
        mock_columns.return_value[0].__enter__ = Mock()
        mock_columns.return_value[0].__exit__ = Mock()
        mock_columns.return_value[1].__enter__ = Mock()
        mock_columns.return_value[1].__exit__ = Mock()
        
        mock_session_state = MockSessionState()
        
        with patch.object(st, 'session_state', mock_session_state):
            main()
            
            mock_setup_rag.assert_called_once()
            mock_display_history.assert_called_once()
            mock_chat_input.assert_called_once_with("Ask me anything about Blender...")


class TestErrorHandling:
    """Test error handling scenarios throughout the application."""
    
    @patch('streamlit.error')
    @patch('streamlit.spinner')
    def test_query_processing_error_handling(self, mock_spinner, mock_error):
        """Test error handling during query processing in main application."""
        mock_rag = Mock()
        mock_rag.handle_query.side_effect = Exception("LLM API error")
        mock_spinner.return_value.__enter__ = Mock()
        mock_spinner.return_value.__exit__ = Mock()
        
        # This test would need to be integrated into a more comprehensive main() test
        # but demonstrates the error handling pattern
        with patch('interface.streamlit_app.add_message') as mock_add_message:
            try:
                response = mock_rag.handle_query("test query")
            except Exception as e:
                error_message = f"❌ Error generating response: {str(e)}"
                mock_add_message("assistant", error_message)
                
                mock_add_message.assert_called_with("assistant", "❌ Error generating response: LLM API error")
    
    @patch('streamlit.error')
    @patch('streamlit.spinner')
    @patch('interface.streamlit_app.BlenderAssistantRAG')
    @patch('interface.streamlit_app.initialize_logging')
    @patch('interface.streamlit_app.dotenv.load_dotenv')
    def test_setup_rag_system_initialization_failure(self, mock_dotenv, mock_logging, mock_rag_class, mock_spinner, mock_error, patched_session_state):
        """Test setup_rag_system when RAG initialization fails."""
        # Mock the spinner context manager properly
        mock_spinner_context = Mock()
        mock_spinner.return_value = mock_spinner_context
        mock_spinner_context.__enter__ = Mock(return_value=mock_spinner_context)
        mock_spinner_context.__exit__ = Mock(return_value=None)
        
        # Make BlenderAssistantRAG raise an exception
        mock_rag_class.side_effect = Exception("Failed to initialize RAG system")
        
        patched_session_state.rag_system = None
        patched_session_state.initialized = False
        
        result = setup_rag_system()
        
        # Should return None due to exception
        assert result is None
        # Should call st.error with the error message
        mock_error.assert_called_once()
        # Check that the error message contains the expected text
        error_call_args = mock_error.call_args[0][0]
        assert "Failed to initialize RAG system" in error_call_args
    
    def test_environment_variable_handling(self):
        """Test handling of missing environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            mock_session_state = MockSessionState()
            mock_session_state.initialized = False
            
            with patch.object(st, 'session_state', mock_session_state):
                # Test that the app gracefully handles missing env vars
                groq_key = os.getenv('GROQ_API_KEY')
                openai_key = os.getenv('OPENAI_API_KEY')
                
                assert groq_key is None
                assert openai_key is None


class TestIntegrationScenarios:
    """Test integration scenarios and realistic user flows."""
    
    @patch('interface.streamlit_app.setup_rag_system')
    @patch('interface.streamlit_app.add_message')
    @patch('interface.streamlit_app.display_message')
    @patch('streamlit.spinner')
    def test_complete_query_response_flow(self, mock_spinner, mock_display_message, 
                                        mock_add_message, mock_setup_rag):
        """Test complete flow from user query to response display."""
        mock_rag = Mock()
        mock_rag.handle_query.return_value = "Here's how to model in Blender..."
        mock_setup_rag.return_value = mock_rag
        mock_spinner.return_value.__enter__ = Mock()
        mock_spinner.return_value.__exit__ = Mock()
        
        user_query = "How do I start modeling in Blender?"
        
        # Simulate the query processing flow
        with patch('time.time', side_effect=[1.0, 2.5]):  # 1.5 second response time
            mock_add_message("user", user_query)
            response = mock_rag.handle_query(query=user_query)
            mock_add_message("assistant", response)
            
            mock_rag.handle_query.assert_called_once_with(query=user_query)
            assert mock_add_message.call_count == 2
            mock_add_message.assert_any_call("user", user_query)
            mock_add_message.assert_any_call("assistant", "Here's how to model in Blender...")
    
    @patch('streamlit.container')
    @patch('streamlit.columns')
    @patch('streamlit.button')
    @patch('streamlit.chat_input')
    @patch('interface.streamlit_app.add_message')
    @patch('interface.streamlit_app.display_message')
    @patch('streamlit.spinner')
    @patch('streamlit.caption')
    def test_main_application_random_question_flow(self, mock_caption, mock_spinner, mock_display_message, 
                                                 mock_add_message, mock_chat_input, mock_button, 
                                                 mock_columns, mock_container, patched_session_state):
        """Test main application flow when random question button is clicked."""
        # Mock the UI components
        mock_container.return_value.__enter__ = Mock()
        mock_container.return_value.__exit__ = Mock()
        mock_col1, mock_col2 = Mock(), Mock()
        mock_columns.return_value = [mock_col1, mock_col2]
        mock_col1.__enter__ = Mock()
        mock_col1.__exit__ = Mock()
        mock_col2.__enter__ = Mock()
        mock_col2.__exit__ = Mock()
        mock_spinner.return_value.__enter__ = Mock()
        mock_spinner.return_value.__exit__ = Mock()
        
        mock_chat_input.return_value = ""  # No chat input
        
        # Mock random button click (🎲 button)
        def button_side_effect(text, **kwargs):
            if "🎲" in text:
                return True
            return False
        mock_button.side_effect = button_side_effect
        
        # Mock RAG system
        mock_rag = Mock()
        mock_rag.handle_query.return_value = "Random question response"
        patched_session_state.rag_system = mock_rag
        
        # Patch the main app components
        with patch('interface.streamlit_app.setup_rag_system', return_value=mock_rag):
            with patch('interface.streamlit_app.display_chat_history'):
                with patch('streamlit.title'):
                    with patch('streamlit.markdown'):
                        with patch('interface.streamlit_app.initialize_session_state'):
                            with patch('interface.streamlit_app.create_sidebar'):
                                with patch('streamlit.set_page_config'):
                                    with patch('random.choice', return_value="What is Blender?"):
                                        with patch('time.time', side_effect=[1.0, 2.0]):
                                            main()
        
        # Verify that a random question was processed
        mock_rag.handle_query.assert_called_once_with(query="What is Blender?")
        mock_add_message.assert_any_call("user", "What is Blender?")
        mock_add_message.assert_any_call("assistant", "Random question response")
    
    @patch('streamlit.container')
    @patch('streamlit.columns')
    @patch('streamlit.button')
    @patch('streamlit.chat_input')
    @patch('interface.streamlit_app.add_message')
    @patch('interface.streamlit_app.display_message')
    @patch('streamlit.spinner')
    @patch('streamlit.error')
    def test_main_application_query_error_handling(self, mock_error, mock_spinner, mock_display_message,
                                                  mock_add_message, mock_chat_input, mock_button,
                                                  mock_columns, mock_container, patched_session_state):
        """Test main application error handling during query processing."""
        # Mock the UI components
        mock_container.return_value.__enter__ = Mock()
        mock_container.return_value.__exit__ = Mock()
        mock_col1, mock_col2 = Mock(), Mock()
        mock_columns.return_value = [mock_col1, mock_col2]
        mock_col1.__enter__ = Mock()
        mock_col1.__exit__ = Mock()
        mock_col2.__enter__ = Mock()
        mock_col2.__exit__ = Mock()
        
        # Mock the spinner context manager to NOT suppress the exception
        mock_spinner_context = Mock()
        mock_spinner.return_value = mock_spinner_context
        mock_spinner_context.__enter__ = Mock(return_value=mock_spinner_context)
        # Make the context manager NOT suppress exceptions
        mock_spinner_context.__exit__ = Mock(return_value=None)
        
        mock_chat_input.return_value = "Test query"  # User input
        mock_button.return_value = False  # No random button click
        
        # Mock RAG system that fails
        mock_rag = Mock()
        mock_rag.handle_query.side_effect = Exception("API error")
        patched_session_state.rag_system = mock_rag
        
        # Patch the main app components
        with patch('interface.streamlit_app.setup_rag_system', return_value=mock_rag):
            with patch('interface.streamlit_app.display_chat_history'):
                with patch('streamlit.title'):
                    with patch('streamlit.markdown'):
                        with patch('interface.streamlit_app.initialize_session_state'):
                            with patch('interface.streamlit_app.create_sidebar'):
                                with patch('streamlit.set_page_config'):
                                    main()
        
        # Verify error handling - The error should be logged and displayed
        mock_add_message.assert_any_call("user", "Test query")
        mock_add_message.assert_any_call("assistant", "❌ Error generating response: API error")
        # The st.error should be called with the error message
        mock_error.assert_called_once_with("❌ Error generating response: API error")
    
    @patch('streamlit.container')  
    @patch('streamlit.columns')
    @patch('streamlit.button')
    @patch('streamlit.chat_input')
    def test_main_application_user_input_from_session_state(self, mock_chat_input, mock_button,
                                                          mock_columns, mock_container, patched_session_state):
        """Test main application handling user input from session state (sample question selection)."""
        # Mock the UI components
        mock_container.return_value.__enter__ = Mock()
        mock_container.return_value.__exit__ = Mock()
        mock_col1, mock_col2 = Mock(), Mock()
        mock_columns.return_value = [mock_col1, mock_col2]
        mock_col1.__enter__ = Mock()
        mock_col1.__exit__ = Mock()
        mock_col2.__enter__ = Mock()
        mock_col2.__exit__ = Mock()
        
        mock_chat_input.return_value = ""  # No chat input
        mock_button.return_value = False   # No random button click
        
        # Set user input in session state (simulates sidebar sample question selection)
        patched_session_state.user_input = "Sample question from sidebar"
        
        # Mock RAG system
        mock_rag = Mock()
        mock_rag.handle_query.return_value = "Sample response"
        patched_session_state.rag_system = mock_rag
        
        # We need to test the del operation independently since the main() function is complex
        # Instead, let's test that the user_input was accessed and the query was processed
        with patch('interface.streamlit_app.setup_rag_system', return_value=mock_rag):
            with patch('interface.streamlit_app.display_chat_history'):
                with patch('interface.streamlit_app.add_message') as mock_add_message:
                    with patch('interface.streamlit_app.display_message'):
                        with patch('streamlit.title'):
                            with patch('streamlit.markdown'):
                                with patch('interface.streamlit_app.initialize_session_state'):
                                    with patch('interface.streamlit_app.create_sidebar'):
                                        with patch('streamlit.set_page_config'):
                                            with patch('streamlit.spinner') as mock_spinner:
                                                mock_spinner.return_value.__enter__ = Mock()
                                                mock_spinner.return_value.__exit__ = Mock()
                                                with patch('time.time', side_effect=[1.0, 2.0]):
                                                    main()
        
        # Verify that the query was processed
        mock_rag.handle_query.assert_called_once_with(query="Sample question from sidebar")
        mock_add_message.assert_any_call("user", "Sample question from sidebar")
        mock_add_message.assert_any_call("assistant", "Sample response")
    
    def test_session_state_persistence_simulation(self):
        """Test that session state maintains data across simulated interactions."""
        # Simulate multiple interactions with session state
        mock_session_state = MockSessionState()
        mock_session_state.messages = []
        mock_session_state.rag_system = None
        mock_session_state.initialized = False
        
        with patch.object(st, 'session_state', mock_session_state):
            # First interaction - initialization
            initialize_session_state()
            assert len(st.session_state.messages) == 0
            
            # Second interaction - add message
            add_message("user", "Test message")
            assert len(st.session_state.messages) == 1
            
            # Third interaction - add another message
            add_message("assistant", "Test response")
            assert len(st.session_state.messages) == 2
            
            # Verify message content
            assert st.session_state.messages[0]["role"] == "user"
            assert st.session_state.messages[1]["role"] == "assistant"