"""
Comprehensive tests for the logging configuration module.

This module tests all aspects of the logging system including:
- Basic logging configuration
- File logging setup
- JSON logging functionality
- Console formatting
- Logger retrieval
- Third-party logger silencing
"""

import pytest
import logging
import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

import structlog

# Add src to path for imports
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.logger import configure_logging, get_logger


class TestLoggingConfiguration:
    """Test basic logging configuration functionality."""
    
    def setup_method(self):
        """Reset logging configuration before each test."""
        # Clear existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        root_logger.setLevel(logging.WARNING)
    
    def teardown_method(self):
        """Clean up logging configuration after each test."""
        # Reset to default state
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        root_logger.setLevel(logging.WARNING)
    
    def test_configure_logging_default_settings(self):
        """Test logging configuration with default settings."""
        configure_logging()
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
        assert len(root_logger.handlers) == 1
        
        handler = root_logger.handlers[0]
        assert isinstance(handler, logging.StreamHandler)
        assert handler.stream == sys.stdout
    
    def test_configure_logging_custom_level(self):
        """Test logging configuration with custom level."""
        configure_logging(level="DEBUG")
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG
        
        handler = root_logger.handlers[0]
        assert handler.level == logging.DEBUG
    
    def test_configure_logging_invalid_level_raises_error(self):
        """Test that invalid logging level raises AttributeError."""
        with pytest.raises(AttributeError):
            configure_logging(level="INVALID_LEVEL")
    
    def test_configure_logging_case_insensitive_level(self):
        """Test that logging level is case insensitive."""
        configure_logging(level="warning")
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING
    
    def test_configure_logging_with_file_handler(self):
        """Test logging configuration with file output."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            log_file = tmp_file.name
        
        try:
            configure_logging(log_file=log_file)
            
            root_logger = logging.getLogger()
            assert len(root_logger.handlers) == 2
            
            # Console handler
            console_handler = root_logger.handlers[0]
            assert isinstance(console_handler, logging.StreamHandler)
            assert console_handler.stream == sys.stdout
            
            # File handler
            file_handler = root_logger.handlers[1]
            assert isinstance(file_handler, logging.FileHandler)
            assert file_handler.baseFilename == log_file
        finally:
            # Clean up
            if os.path.exists(log_file):
                os.unlink(log_file)
    
    def test_configure_logging_creates_log_directory(self):
        """Test that logging configuration creates log directory if needed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = os.path.join(tmp_dir, "subdir", "app.log")
            
            configure_logging(log_file=log_file)
            
            # Directory should be created
            assert os.path.exists(os.path.dirname(log_file))
            
            # File handler should be configured
            root_logger = logging.getLogger()
            file_handler = root_logger.handlers[1]
            assert file_handler.baseFilename == log_file
    
    def test_configure_logging_with_json_logs_enabled(self):
        """Test logging configuration with JSON formatting enabled."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            log_file = tmp_file.name
        
        try:
            configure_logging(log_file=log_file, enable_json_logs=True)
            
            root_logger = logging.getLogger()
            file_handler = root_logger.handlers[1]
            
            # Check that formatter is configured for JSON (minimal format)
            assert file_handler.formatter.format(logging.LogRecord(
                name="test", level=logging.INFO, pathname="", lineno=0,
                msg="test message", args=(), exc_info=None
            )) == "test message"
        finally:
            if os.path.exists(log_file):
                os.unlink(log_file)
    
    def test_configure_logging_without_json_logs(self):
        """Test logging configuration with standard text formatting."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            log_file = tmp_file.name
        
        try:
            configure_logging(log_file=log_file, enable_json_logs=False)
            
            root_logger = logging.getLogger()
            file_handler = root_logger.handlers[1]
            
            # Create a test log record
            record = logging.LogRecord(
                name="test_logger", level=logging.INFO, pathname="", lineno=0,
                msg="test message", args=(), exc_info=None
            )
            
            formatted = file_handler.formatter.format(record)
            assert "test_logger" in formatted
            assert "INFO" in formatted
            assert "test message" in formatted
        finally:
            if os.path.exists(log_file):
                os.unlink(log_file)
    
    def test_configure_logging_replaces_existing_handlers(self):
        """Test that configuration replaces existing handlers."""
        # Add a dummy handler first
        root_logger = logging.getLogger()
        dummy_handler = logging.StreamHandler()
        root_logger.addHandler(dummy_handler)
        
        initial_handler_count = len(root_logger.handlers)
        assert initial_handler_count >= 1
        
        configure_logging()
        
        # Should have exactly one handler (the console handler)
        assert len(root_logger.handlers) == 1
        assert dummy_handler not in root_logger.handlers


class TestConsoleFormatter:
    """Test console formatting and color configuration."""
    
    def setup_method(self):
        """Reset logging before each test."""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    def test_console_formatter_colors_configured(self):
        """Test that console formatter has color configuration."""
        configure_logging()
        
        root_logger = logging.getLogger()
        console_handler = root_logger.handlers[0]
        
        # Check that it's a ColoredFormatter (from colorlog)
        from colorlog import ColoredFormatter
        assert isinstance(console_handler.formatter, ColoredFormatter)
        
        # Check color configuration
        formatter = console_handler.formatter
        expected_colors = {
            "DEBUG": "cyan",
            "INFO": "green", 
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        }
        assert formatter.log_colors == expected_colors
    
    def test_console_formatter_datetime_format(self):
        """Test console formatter datetime format."""
        configure_logging()
        
        root_logger = logging.getLogger()
        console_handler = root_logger.handlers[0]
        formatter = console_handler.formatter
        
        assert formatter.datefmt == "%Y-%m-%d %H:%M:%S"


class TestStructlogConfiguration:
    """Test structlog configuration and integration."""
    
    def setup_method(self):
        """Reset logging before each test."""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    def test_get_logger_returns_bound_logger(self):
        """Test that get_logger returns a structlog BoundLogger."""
        configure_logging()
        
        logger = get_logger("test_logger")
        assert isinstance(logger, structlog.stdlib.BoundLogger)
    
    def test_get_logger_different_names(self):
        """Test that get_logger works with different logger names."""
        configure_logging()
        
        logger1 = get_logger("logger1")
        logger2 = get_logger("logger2")
        
        assert isinstance(logger1, structlog.stdlib.BoundLogger)
        assert isinstance(logger2, structlog.stdlib.BoundLogger)
        # They should be different instances but same type
        assert type(logger1) == type(logger2)
    
    @patch('structlog.configure')
    def test_structlog_processors_console_mode(self, mock_structlog_configure):
        """Test structlog processor configuration for console output."""
        configure_logging(enable_json_logs=False)
        
        mock_structlog_configure.assert_called_once()
        call_args = mock_structlog_configure.call_args
        
        processors = call_args[1]['processors']
        assert len(processors) >= 5  # Should have multiple processors
        
        # Check for expected processor types
        processor_types = [type(p).__name__ for p in processors]
        assert 'TimeStamper' in str(processors)
        assert 'ConsoleRenderer' in str(processors)
    
    @patch('structlog.configure')
    def test_structlog_processors_json_mode(self, mock_structlog_configure):
        """Test structlog processor configuration for JSON output."""
        with tempfile.NamedTemporaryFile() as tmp_file:
            configure_logging(log_file=tmp_file.name, enable_json_logs=True)
            
            mock_structlog_configure.assert_called_once()
            call_args = mock_structlog_configure.call_args
            
            processors = call_args[1]['processors']
            
            # Check for JSON-specific processors
            assert 'JSONRenderer' in str(processors)
            assert 'TimeStamper' in str(processors)


class TestThirdPartyLoggerSilencing:
    """Test that noisy third-party loggers are properly silenced."""
    
    def test_urllib3_logger_silenced(self):
        """Test that urllib3 logger is set to WARNING level."""
        # This is configured at module import time
        urllib3_logger = logging.getLogger("urllib3")
        assert urllib3_logger.level == logging.WARNING
    
    def test_httpx_logger_silenced(self):
        """Test that httpx logger is set to WARNING level."""
        httpx_logger = logging.getLogger("httpx")
        assert httpx_logger.level == logging.WARNING
    
    def test_chromadb_logger_silenced(self):
        """Test that chromadb logger is set to WARNING level."""
        chromadb_logger = logging.getLogger("chromadb")
        assert chromadb_logger.level == logging.WARNING
    
    def test_sentence_transformers_logger_silenced(self):
        """Test that sentence_transformers logger is set to WARNING level."""
        st_logger = logging.getLogger("sentence_transformers")
        assert st_logger.level == logging.WARNING


class TestLoggingIntegration:
    """Test integration scenarios and realistic usage patterns."""
    
    def setup_method(self):
        """Reset logging before each test."""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    def test_logging_output_to_console(self):
        """Test that log messages are properly output to console."""
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            configure_logging(level="INFO")
            
            logger = logging.getLogger("test_logger")
            logger.info("Test log message")
            
            output = mock_stdout.getvalue()
            assert "Test log message" in output
            assert "test_logger" in output
            assert "INFO" in output
    
    def test_logging_output_to_file(self):
        """Test that log messages are properly written to file."""
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp_file:
            log_file = tmp_file.name
        
        try:
            configure_logging(level="DEBUG", log_file=log_file)
            
            logger = logging.getLogger("test_logger")
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            
            # Force flush handlers
            for handler in logging.getLogger().handlers:
                handler.flush()
            
            # Read file content
            with open(log_file, 'r') as f:
                content = f.read()
            
            assert "Debug message" in content
            assert "Info message" in content  
            assert "Warning message" in content
            assert "test_logger" in content
        finally:
            if os.path.exists(log_file):
                os.unlink(log_file)
    
    def test_structlog_and_stdlib_integration(self):
        """Test that structlog and stdlib logging work together."""
        configure_logging(level="INFO")
        
        # Get both types of loggers
        stdlib_logger = logging.getLogger("stdlib_test")
        struct_logger = get_logger("struct_test")
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            stdlib_logger.info("Standard library message")
            struct_logger.info("Structlog message")
            
            output = mock_stdout.getvalue()
            assert "Standard library message" in output
            assert "Structlog message" in output
    
    def test_log_level_filtering(self):
        """Test that log level filtering works correctly."""
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            configure_logging(level="WARNING")
            
            logger = logging.getLogger("test_logger")
            logger.debug("Debug message - should not appear")
            logger.info("Info message - should not appear")
            logger.warning("Warning message - should appear")
            logger.error("Error message - should appear")
            
            output = mock_stdout.getvalue()
            assert "Debug message" not in output
            assert "Info message" not in output
            assert "Warning message" in output
            assert "Error message" in output
    
    def test_multiple_configuration_calls(self):
        """Test that multiple configure_logging calls work correctly."""
        # First configuration
        configure_logging(level="INFO")
        
        root_logger = logging.getLogger()
        first_handler_count = len(root_logger.handlers)
        
        # Second configuration - should replace handlers
        configure_logging(level="DEBUG")
        
        second_handler_count = len(root_logger.handlers)
        assert second_handler_count == first_handler_count
        assert root_logger.level == logging.DEBUG


class TestErrorHandling:
    """Test error handling in logging configuration."""
    
    def test_invalid_log_file_path_handling(self):
        """Test handling of invalid log file paths."""
        # Try to write to a directory that doesn't exist and can't be created
        invalid_path = "/root/nonexistent/cannot_create.log"
        
        # This should not raise an exception in normal circumstances
        # The mkdir with parents=True should handle most cases
        # But we test the directory creation logic
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Remove write permissions
            os.chmod(tmp_dir, 0o444)
            
            try:
                # This might fail due to permissions, but should be handled gracefully
                restricted_path = os.path.join(tmp_dir, "subdir", "test.log")
                
                # The function should handle this gracefully 
                # In a real scenario, we'd want to catch the exception
                try:
                    configure_logging(log_file=restricted_path)
                except (PermissionError, OSError):
                    # This is expected behavior for permission-restricted paths
                    pass
            finally:
                # Restore permissions for cleanup
                os.chmod(tmp_dir, 0o755)


class TestLoggerPerformance:
    """Test performance aspects of logging configuration."""
    
    def test_logger_caching(self):
        """Test that loggers are properly cached."""
        configure_logging()
        
        # Get same logger multiple times
        logger1 = get_logger("test_logger")
        logger2 = get_logger("test_logger")
        logger3 = get_logger("test_logger")
        
        # They should be the same instance due to caching
        assert logger1 is logger2
        assert logger2 is logger3
    
    def test_different_logger_instances(self):
        """Test that different logger names create different instances."""
        configure_logging()
        
        logger_a = get_logger("logger_a")
        logger_b = get_logger("logger_b")
        logger_c = get_logger("logger_c")
        
        # They should be different instances
        assert logger_a is not logger_b
        assert logger_b is not logger_c
        assert logger_a is not logger_c