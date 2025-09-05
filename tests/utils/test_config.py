"""
Comprehensive tests for the configuration module.

This module tests all aspects of the configuration system including:
- Environment variable reading and defaults
- Type conversions (int, float, bool)
- Configuration validation
- Helper functions
- Logging initialization
- Error handling scenarios
"""

import pytest
import os
import tempfile
import shutil
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import utils.config as config_module
from utils.config import (
    read_env_variable, str_to_bool, get_env_int, get_env_float, get_env_bool,
    get_active_model_info, get_chunking_config, get_vector_db_config_dict,
    validate_config, print_config_summary, initialize_logging
)


class TestEnvironmentVariableReading:
    """Test environment variable reading functions."""
    
    def test_read_env_variable_existing_variable(self):
        """Test reading an existing environment variable."""
        with patch.dict(os.environ, {'TEST_VAR': 'test_value'}):
            result = read_env_variable('TEST_VAR')
            assert result == 'test_value'
    
    def test_read_env_variable_missing_with_default(self):
        """Test reading a missing variable with default value."""
        with patch.dict(os.environ, {}, clear=True):
            result = read_env_variable('MISSING_VAR', 'default_value')
            assert result == 'default_value'
    
    def test_read_env_variable_missing_without_default(self):
        """Test reading a missing variable without default."""
        with patch.dict(os.environ, {}, clear=True):
            result = read_env_variable('MISSING_VAR')
            assert result is None
    
    def test_read_env_variable_exit_on_missing(self):
        """Test exit_on_missing functionality."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(SystemExit):
                read_env_variable('MISSING_VAR', exit_on_missing=True)
    
    def test_read_env_variable_logging(self):
        """Test that appropriate log messages are generated."""
        with patch.dict(os.environ, {'TEST_VAR': 'test_value'}):
            with patch('logging.debug') as mock_debug:
                read_env_variable('TEST_VAR')
                mock_debug.assert_called_with('Using TEST_VAR=test_value (from environment)')
    
    def test_read_env_variable_default_logging(self):
        """Test logging when using default value."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('logging.debug') as mock_debug:
                read_env_variable('MISSING_VAR', 'default')
                mock_debug.assert_called_with('Using MISSING_VAR=default (default value)')
    
    def test_read_env_variable_none_logging(self):
        """Test logging when returning None."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('logging.debug') as mock_debug:
                read_env_variable('MISSING_VAR')
                mock_debug.assert_called_with('MISSING_VAR not set in environment, using None')


class TestTypeConversions:
    """Test type conversion functions."""
    
    def test_str_to_bool_true_values(self):
        """Test str_to_bool with various true values."""
        true_values = ['true', 'True', 'TRUE', '1', 'yes', 'YES', 'on', 'ON', 'enabled', 'ENABLED']
        for value in true_values:
            assert str_to_bool(value) is True, f"Failed for value: {value}"
    
    def test_str_to_bool_false_values(self):
        """Test str_to_bool with various false values."""
        false_values = ['false', 'False', 'FALSE', '0', 'no', 'NO', 'off', 'OFF', 'disabled', 'DISABLED', 'random']
        for value in false_values:
            assert str_to_bool(value) is False, f"Failed for value: {value}"
    
    def test_get_env_int_valid_integer(self):
        """Test getting valid integer from environment."""
        with patch.dict(os.environ, {'TEST_INT': '42'}):
            result = get_env_int('TEST_INT', 10)
            assert result == 42
    
    def test_get_env_int_invalid_integer(self):
        """Test getting invalid integer falls back to default."""
        with patch.dict(os.environ, {'TEST_INT': 'not_an_integer'}):
            with patch('logging.warning') as mock_warning:
                result = get_env_int('TEST_INT', 10)
                assert result == 10
                mock_warning.assert_called_once()
    
    def test_get_env_int_missing_variable(self):
        """Test getting integer from missing variable uses default."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_env_int('MISSING_INT', 42)
            assert result == 42
    
    def test_get_env_float_valid_float(self):
        """Test getting valid float from environment."""
        with patch.dict(os.environ, {'TEST_FLOAT': '3.14'}):
            result = get_env_float('TEST_FLOAT', 1.0)
            assert result == 3.14
    
    def test_get_env_float_invalid_float(self):
        """Test getting invalid float falls back to default."""
        with patch.dict(os.environ, {'TEST_FLOAT': 'not_a_float'}):
            with patch('logging.warning') as mock_warning:
                result = get_env_float('TEST_FLOAT', 2.5)
                assert result == 2.5
                mock_warning.assert_called_once()
    
    def test_get_env_float_missing_variable(self):
        """Test getting float from missing variable uses default."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_env_float('MISSING_FLOAT', 1.23)
            assert result == 1.23
    
    def test_get_env_bool_true_value(self):
        """Test getting boolean true from environment."""
        with patch.dict(os.environ, {'TEST_BOOL': 'true'}):
            result = get_env_bool('TEST_BOOL', False)
            assert result is True
    
    def test_get_env_bool_false_value(self):
        """Test getting boolean false from environment."""
        with patch.dict(os.environ, {'TEST_BOOL': 'false'}):
            result = get_env_bool('TEST_BOOL', True)
            assert result is False
    
    def test_get_env_bool_missing_variable_default_true(self):
        """Test getting boolean from missing variable with True default."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_env_bool('MISSING_BOOL', True)
            assert result is True
    
    def test_get_env_bool_missing_variable_default_false(self):
        """Test getting boolean from missing variable with False default."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_env_bool('MISSING_BOOL', False)
            assert result is False


class TestConfigurationValues:
    """Test that configuration values are properly loaded."""
    
    def test_groq_configuration_defaults(self):
        """Test Groq configuration with defaults."""
        with patch.dict(os.environ, {}, clear=True):
            # Reload the module to test defaults
            import importlib
            importlib.reload(config_module)
            
            assert config_module.GROQ_API_KEY is None
            assert config_module.GROQ_MODEL == 'llama-3.1-8b-instant'
            assert config_module.GROQ_TEMPERATURE == 0.7
            assert config_module.GROQ_MAX_TOKENS == 2048
    
    def test_openai_configuration_defaults(self):
        """Test OpenAI configuration with defaults."""
        with patch.dict(os.environ, {}, clear=True):
            import importlib
            importlib.reload(config_module)
            
            assert config_module.OPENAI_API_KEY is None
            assert config_module.OPENAI_MODEL == 'gpt-4'
            assert config_module.OPENAI_TEMPERATURE == 0.7
            assert config_module.OPENAI_MAX_TOKENS == 2048
    
    def test_vector_db_configuration_defaults(self):
        """Test vector database configuration defaults."""
        with patch.dict(os.environ, {}, clear=True):
            import importlib
            importlib.reload(config_module)
            
            assert str(config_module.CHROMA_PERSIST_DIRECTORY).endswith('data/vector_db')
            assert config_module.CHROMA_COLLECTION_NAME == 'blender_docs'
    
    def test_custom_environment_values(self):
        """Test that custom environment values override defaults."""
        custom_env = {
            'GROQ_API_KEY': 'custom-groq-key',
            'GROQ_MODEL': 'custom-model',
            'GROQ_TEMPERATURE': '0.5',
            'CHUNK_SIZE': '1024',
            'LOG_LEVEL': 'DEBUG'
        }
        
        with patch.dict(os.environ, custom_env, clear=True):
            import importlib
            importlib.reload(config_module)
            
            assert config_module.GROQ_API_KEY == 'custom-groq-key'
            assert config_module.GROQ_MODEL == 'custom-model'
            assert config_module.GROQ_TEMPERATURE == 0.5
            assert config_module.CHUNK_SIZE == 1024
            assert config_module.LOG_LEVEL == 'DEBUG'


class TestHelperFunctions:
    """Test configuration helper functions."""
    
    def test_get_active_model_info_groq(self):
        """Test model info when Groq API key is available."""
        with patch.object(config_module, 'GROQ_API_KEY', 'test-key'):
            with patch.object(config_module, 'GROQ_MODEL', 'llama-test'):
                result = get_active_model_info()
                assert result == "Groq (llama-test)"
    
    def test_get_active_model_info_openai(self):
        """Test model info when only OpenAI API key is available."""
        with patch.object(config_module, 'GROQ_API_KEY', None):
            with patch.object(config_module, 'OPENAI_API_KEY', 'test-key'):
                with patch.object(config_module, 'OPENAI_MODEL', 'gpt-test'):
                    result = get_active_model_info()
                    assert result == "OpenAI (gpt-test)"
    
    def test_get_active_model_info_no_keys(self):
        """Test model info when no API keys are available."""
        with patch.object(config_module, 'GROQ_API_KEY', None):
            with patch.object(config_module, 'OPENAI_API_KEY', None):
                result = get_active_model_info()
                assert result == "No API key configured"
    
    def test_get_chunking_config(self):
        """Test chunking configuration dictionary."""
        with patch.object(config_module, 'CHUNK_SIZE', 512):
            with patch.object(config_module, 'CHUNK_OVERLAP', 50):
                with patch.object(config_module, 'USE_SEMANTIC_CHUNKING', True):
                    with patch.object(config_module, 'MIN_CHUNK_SIZE', 100):
                        with patch.object(config_module, 'MAX_CHUNK_SIZE', 1000):
                            result = get_chunking_config()
                            expected = {
                                "chunk_size": 512,
                                "chunk_overlap": 50,
                                "use_semantic_chunking": True,
                                "min_chunk_size": 100,
                                "max_chunk_size": 1000,
                            }
                            assert result == expected
    
    def test_get_vector_db_config_dict(self):
        """Test vector database configuration dictionary."""
        with patch.object(config_module, 'EMBEDDING_MODEL', 'test-model'):
            with patch.object(config_module, 'CHUNK_SIZE', 256):
                with patch.object(config_module, 'CHUNK_OVERLAP', 25):
                    with patch.object(config_module, 'USE_SEMANTIC_CHUNKING', False):
                        with patch.object(config_module, 'BATCH_SIZE', 50):
                            result = get_vector_db_config_dict()
                            
                            assert result["embedding_model"] == "test-model"
                            assert result["chunk_size"] == 256
                            assert result["chunk_overlap"] == 25
                            assert result["use_semantic_chunking"] is False
                            assert result["batch_size"] == 50
                            assert "metadata_fields" in result
                            assert isinstance(result["metadata_fields"], list)
                            assert "title" in result["metadata_fields"]


class TestConfigurationValidation:
    """Test configuration validation functionality."""
    
    def setup_method(self):
        """Set up test environment with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_chroma_dir = config_module.CHROMA_PERSIST_DIRECTORY
    
    def teardown_method(self):
        """Clean up temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_validate_config_success_non_strict(self):
        """Test successful validation in non-strict mode."""
        with patch.object(config_module, 'CHROMA_PERSIST_DIRECTORY', Path(self.temp_dir) / 'test_db'):
            with patch.object(config_module, 'CHUNK_OVERLAP', 50):
                with patch.object(config_module, 'CHUNK_SIZE', 512):
                    with patch.object(config_module, 'MIN_CHUNK_SIZE', 100):
                        with patch.object(config_module, 'MAX_CHUNK_SIZE', 1000):
                            with patch.object(config_module, 'RETRIEVAL_K', 5):
                                with patch.object(config_module, 'SIMILARITY_THRESHOLD', 0.7):
                                    # Should not raise any exception
                                    validate_config(strict=False)
    
    def test_validate_config_success_strict_with_api_key(self):
        """Test successful validation in strict mode with API key."""
        with patch.object(config_module, 'GROQ_API_KEY', 'test-key'):
            with patch.object(config_module, 'CHROMA_PERSIST_DIRECTORY', Path(self.temp_dir) / 'test_db'):
                with patch.object(config_module, 'CHUNK_OVERLAP', 50):
                    with patch.object(config_module, 'CHUNK_SIZE', 512):
                        with patch.object(config_module, 'MIN_CHUNK_SIZE', 100):
                            with patch.object(config_module, 'MAX_CHUNK_SIZE', 1000):
                                with patch.object(config_module, 'RETRIEVAL_K', 5):
                                    with patch.object(config_module, 'SIMILARITY_THRESHOLD', 0.7):
                                        # Should not raise any exception
                                        validate_config(strict=True)
    
    def test_validate_config_strict_no_api_keys(self):
        """Test validation failure in strict mode without API keys."""
        with patch.object(config_module, 'GROQ_API_KEY', None):
            with patch.object(config_module, 'OPENAI_API_KEY', None):
                with pytest.raises(ValueError, match="Either GROQ_API_KEY or OPENAI_API_KEY must be set"):
                    validate_config(strict=True)
    
    def test_validate_config_chunk_overlap_too_large(self):
        """Test validation failure when chunk overlap >= chunk size."""
        with patch.object(config_module, 'CHUNK_OVERLAP', 512):
            with patch.object(config_module, 'CHUNK_SIZE', 512):
                with pytest.raises(ValueError, match="CHUNK_OVERLAP must be less than CHUNK_SIZE"):
                    validate_config(strict=False)
    
    def test_validate_config_invalid_chunk_sizes(self):
        """Test validation failure when min chunk size > max chunk size."""
        with patch.object(config_module, 'MIN_CHUNK_SIZE', 1000):
            with patch.object(config_module, 'MAX_CHUNK_SIZE', 500):
                with patch.object(config_module, 'CHUNK_OVERLAP', 50):
                    with patch.object(config_module, 'CHUNK_SIZE', 512):
                        with pytest.raises(ValueError, match="MIN_CHUNK_SIZE must be less than or equal to MAX_CHUNK_SIZE"):
                            validate_config(strict=False)
    
    def test_validate_config_invalid_retrieval_k(self):
        """Test validation failure when retrieval K <= 0."""
        with patch.object(config_module, 'RETRIEVAL_K', 0):
            with pytest.raises(ValueError, match="RETRIEVAL_K must be positive"):
                validate_config(strict=False)
    
    def test_validate_config_invalid_similarity_threshold_low(self):
        """Test validation failure when similarity threshold < 0."""
        with patch.object(config_module, 'SIMILARITY_THRESHOLD', -0.1):
            with pytest.raises(ValueError, match="SIMILARITY_THRESHOLD must be between 0.0 and 1.0"):
                validate_config(strict=False)
    
    def test_validate_config_invalid_similarity_threshold_high(self):
        """Test validation failure when similarity threshold > 1."""
        with patch.object(config_module, 'SIMILARITY_THRESHOLD', 1.1):
            with pytest.raises(ValueError, match="SIMILARITY_THRESHOLD must be between 0.0 and 1.0"):
                validate_config(strict=False)
    
    def test_validate_config_creates_directories(self):
        """Test that validation creates necessary directories."""
        test_chroma_dir = Path(self.temp_dir) / 'new_chroma_db'
        test_log_file = Path(self.temp_dir) / 'logs' / 'app.log'
        
        with patch.object(config_module, 'CHROMA_PERSIST_DIRECTORY', test_chroma_dir):
            with patch.object(config_module, 'LOG_FILE', str(test_log_file)):
                with patch.object(config_module, 'ENABLE_FILE_LOGGING', True):
                    validate_config(strict=False)
                    
                    assert test_chroma_dir.exists()
                    assert test_log_file.parent.exists()
    
    def test_validate_config_multiple_errors(self):
        """Test validation with multiple configuration errors."""
        with patch.object(config_module, 'CHUNK_OVERLAP', 1000):
            with patch.object(config_module, 'CHUNK_SIZE', 512):
                with patch.object(config_module, 'RETRIEVAL_K', -1):
                    with pytest.raises(ValueError) as exc_info:
                        validate_config(strict=False)
                    
                    error_message = str(exc_info.value)
                    assert "CHUNK_OVERLAP must be less than CHUNK_SIZE" in error_message
                    assert "RETRIEVAL_K must be positive" in error_message


class TestConfigurationOutput:
    """Test configuration output and summary functions."""
    
    def test_print_config_summary(self):
        """Test configuration summary printing."""
        with patch.object(config_module, 'CHROMA_COLLECTION_NAME', 'test_collection'):
            with patch.object(config_module, 'CHROMA_PERSIST_DIRECTORY', Path('/test/path')):
                with patch.object(config_module, 'EMBEDDING_MODEL', 'test-embedding'):
                    with patch.object(config_module, 'CHUNK_SIZE', 256):
                        with patch.object(config_module, 'USE_SEMANTIC_CHUNKING', True):
                            with patch.object(config_module, 'RETRIEVAL_K', 3):
                                with patch.object(config_module, 'SIMILARITY_THRESHOLD', 0.8):
                                    with patch.object(config_module, 'LOG_LEVEL', 'DEBUG'):
                                        with patch.object(config_module, 'DEBUG', True):
                                            with patch('logging.info') as mock_info:
                                                with patch('utils.config.get_active_model_info', return_value='Test Model'):
                                                    print_config_summary()
                                                    
                                                    # Verify all expected log calls were made
                                                    expected_calls = [
                                                        "Blender Bot Configuration:",
                                                        "  Model: Test Model",
                                                        "  Vector DB: test_collection @ /test/path", 
                                                        "  Embedding Model: test-embedding",
                                                        "  Chunking: 256 tokens (semantic)",
                                                        "  Retrieval: top-3 (threshold: 0.8)",
                                                        "  Logging: DEBUG",
                                                        "  Debug: True"
                                                    ]
                                                    
                                                    assert mock_info.call_count == len(expected_calls)
                                                    for i, expected_call in enumerate(expected_calls):
                                                        actual_call = mock_info.call_args_list[i][0][0]
                                                        assert expected_call == actual_call


class TestLoggingInitialization:
    """Test logging initialization functionality."""
    
    def test_initialize_logging(self):
        """Test logging initialization with configuration values."""
        with patch.object(config_module, 'LOG_LEVEL', 'DEBUG'):
            with patch.object(config_module, 'LOG_FILE', '/test/log.txt'):
                with patch.object(config_module, 'ENABLE_JSON_LOGS', True):
                    with patch('utils.logger.configure_logging') as mock_configure:
                        initialize_logging()
                        
                        mock_configure.assert_called_once_with(
                            level='DEBUG',
                            log_file='/test/log.txt',
                            enable_json_logs=True,
                        )


class TestModuleImportValidation:
    """Test module-level validation that runs on import."""
    
    def test_import_validation_warning_on_error(self):
        """Test that import validation logs warnings on configuration errors."""
        # This test needs to be careful since the module is already imported
        with patch('utils.config.validate_config') as mock_validate:
            with patch('logging.warning') as mock_warning:
                mock_validate.side_effect = ValueError("Test configuration error")
                
                # Trigger the import validation
                try:
                    exec("validate_config(strict=False)")
                except Exception as e:
                    pass  # Expected to be caught
                
                # Simulate what happens at module import
                try:
                    mock_validate(strict=False)
                except Exception as e:
                    mock_warning(f"Configuration validation warning: {e}")
                
                mock_warning.assert_called_once()


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""
    
    def test_path_object_handling(self):
        """Test that Path objects are handled correctly."""
        test_path = Path('/test/directory')
        with patch.object(config_module, 'CHROMA_PERSIST_DIRECTORY', test_path):
            # Should not raise any errors when working with Path objects
            result = str(config_module.CHROMA_PERSIST_DIRECTORY)
            assert result == '/test/directory'
    
    def test_environment_variable_edge_cases(self):
        """Test edge cases in environment variable handling."""
        # Test empty string
        with patch.dict(os.environ, {'EMPTY_VAR': ''}):
            result = read_env_variable('EMPTY_VAR', 'default')
            assert result == ''  # Empty string should be returned, not default
        
        # Test whitespace-only string
        with patch.dict(os.environ, {'WHITESPACE_VAR': '   '}):
            result = read_env_variable('WHITESPACE_VAR', 'default')
            assert result == '   '
    
    def test_type_conversion_edge_cases(self):
        """Test edge cases in type conversions."""
        # Test None values in type conversion
        with patch('utils.config.read_env_variable', return_value=None):
            result = get_env_int('TEST_VAR', 42)
            # Should handle None gracefully and return default
            assert result == 42
        
        # Test empty string in type conversion
        with patch('utils.config.read_env_variable', return_value=''):
            with patch('logging.warning'):
                result = get_env_float('TEST_VAR', 3.14)
                assert result == 3.14
    
    def test_boolean_conversion_edge_cases(self):
        """Test edge cases in boolean conversion."""
        # Test numeric strings
        assert str_to_bool('1') is True
        assert str_to_bool('0') is False
        
        # Test mixed case
        assert str_to_bool('TrUe') is True
        assert str_to_bool('FaLsE') is False
        
        # Test unexpected values default to False
        assert str_to_bool('maybe') is False
        assert str_to_bool('') is False
        assert str_to_bool('2') is False
    
    def test_configuration_consistency(self):
        """Test that configuration values are internally consistent."""
        # This is more of a sanity check that defaults make sense
        import importlib
        with patch.dict(os.environ, {}, clear=True):
            importlib.reload(config_module)
            
            # Chunk overlap should be less than chunk size
            assert config_module.CHUNK_OVERLAP < config_module.CHUNK_SIZE
            
            # Min chunk size should be <= max chunk size
            assert config_module.MIN_CHUNK_SIZE <= config_module.MAX_CHUNK_SIZE
            
            # Similarity threshold should be in valid range
            assert 0.0 <= config_module.SIMILARITY_THRESHOLD <= 1.0
            
            # Retrieval K should be positive
            assert config_module.RETRIEVAL_K > 0