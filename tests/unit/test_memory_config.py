#!/usr/bin/env python3
"""
Unit tests for memory configuration in utils/config.py.

Tests memory-related environment variable handling, validation,
and configuration loading.
"""

import pytest
import os
from unittest.mock import patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


class TestMemoryConfiguration:
    """Test memory configuration environment variables."""
    
    def test_default_memory_configuration(self):
        """Test default memory configuration values."""
        with patch.dict(os.environ, {}, clear=True):
            # Reload config to get fresh defaults
            import importlib
            import utils.config
            importlib.reload(utils.config)
            
            from utils.config import MEMORY_TYPE, MEMORY_WINDOW_SIZE, MEMORY_MAX_TOKEN_LIMIT
            
            assert MEMORY_TYPE == 'none'
            assert MEMORY_WINDOW_SIZE == 6
            assert MEMORY_MAX_TOKEN_LIMIT == 1000
    
    def test_memory_type_configuration_options(self):
        """Test all valid memory type configuration options."""
        valid_types = ['none', 'window', 'summary']
        
        for memory_type in valid_types:
            with patch.dict(os.environ, {'MEMORY_TYPE': memory_type}, clear=True):
                import importlib
                import utils.config
                importlib.reload(utils.config)
                
                from utils.config import MEMORY_TYPE
                assert MEMORY_TYPE == memory_type
    
    def test_memory_window_size_configuration(self):
        """Test memory window size configuration."""
        test_values = ['4', '8', '10', '16']
        
        for size_str in test_values:
            with patch.dict(os.environ, {'MEMORY_WINDOW_SIZE': size_str}, clear=True):
                import importlib
                import utils.config
                importlib.reload(utils.config)
                
                from utils.config import MEMORY_WINDOW_SIZE
                assert MEMORY_WINDOW_SIZE == int(size_str)
    
    def test_memory_max_token_limit_configuration(self):
        """Test memory max token limit configuration."""
        test_values = ['500', '1500', '2000', '3000']
        
        for limit_str in test_values:
            with patch.dict(os.environ, {'MEMORY_MAX_TOKEN_LIMIT': limit_str}, clear=True):
                import importlib
                import utils.config
                importlib.reload(utils.config)
                
                from utils.config import MEMORY_MAX_TOKEN_LIMIT
                assert MEMORY_MAX_TOKEN_LIMIT == int(limit_str)
    
    def test_invalid_memory_window_size_fallback(self):
        """Test fallback for invalid memory window size values."""
        invalid_values = ['abc', 'not_a_number', 'invalid_number']
        
        for invalid_value in invalid_values:
            with patch.dict(os.environ, {'MEMORY_WINDOW_SIZE': invalid_value}, clear=True):
                import importlib
                import utils.config
                importlib.reload(utils.config)
                
                from utils.config import MEMORY_WINDOW_SIZE
                # Should fall back to default value of 6
                assert MEMORY_WINDOW_SIZE == 6
        
        # Test edge cases with valid but potentially problematic values
        edge_cases = {'-5': -5, '0': 0}  # These are valid integers, just potentially problematic
        for value_str, expected in edge_cases.items():
            with patch.dict(os.environ, {'MEMORY_WINDOW_SIZE': value_str}, clear=True):
                import importlib
                import utils.config
                importlib.reload(utils.config)
                
                from utils.config import MEMORY_WINDOW_SIZE
                assert MEMORY_WINDOW_SIZE == expected
    
    def test_invalid_memory_max_token_limit_fallback(self):
        """Test fallback for invalid max token limit values."""
        invalid_values = ['xyz', 'invalid', 'not_a_number']
        
        for invalid_value in invalid_values:
            with patch.dict(os.environ, {'MEMORY_MAX_TOKEN_LIMIT': invalid_value}, clear=True):
                import importlib
                import utils.config
                importlib.reload(utils.config)
                
                from utils.config import MEMORY_MAX_TOKEN_LIMIT
                # Should fall back to default value of 1000
                assert MEMORY_MAX_TOKEN_LIMIT == 1000
        
        # Test edge cases with valid but potentially problematic values  
        edge_cases = {'-100': -100, '0': 0}  # These are valid integers, just potentially problematic
        for value_str, expected in edge_cases.items():
            with patch.dict(os.environ, {'MEMORY_MAX_TOKEN_LIMIT': value_str}, clear=True):
                import importlib
                import utils.config
                importlib.reload(utils.config)
                
                from utils.config import MEMORY_MAX_TOKEN_LIMIT
                assert MEMORY_MAX_TOKEN_LIMIT == expected
    
    def test_memory_configuration_with_other_settings(self):
        """Test memory configuration alongside other environment variables."""
        env_vars = {
            'MEMORY_TYPE': 'window',
            'MEMORY_WINDOW_SIZE': '8',
            'MEMORY_MAX_TOKEN_LIMIT': '1500',
            'GROQ_API_KEY': 'test-key',
            'CHUNK_SIZE': '1024'
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            import importlib
            import utils.config
            importlib.reload(utils.config)
            
            from utils.config import (
                MEMORY_TYPE, MEMORY_WINDOW_SIZE, MEMORY_MAX_TOKEN_LIMIT,
                GROQ_API_KEY, CHUNK_SIZE
            )
            
            assert MEMORY_TYPE == 'window'
            assert MEMORY_WINDOW_SIZE == 8
            assert MEMORY_MAX_TOKEN_LIMIT == 1500
            assert GROQ_API_KEY == 'test-key'
            assert CHUNK_SIZE == 1024
    
    def test_memory_configuration_case_sensitivity(self):
        """Test that memory type is case sensitive (as expected)."""
        test_cases = ['Window', 'WINDOW', 'Summary', 'SUMMARY', 'None', 'NONE']
        
        for memory_type in test_cases:
            with patch.dict(os.environ, {'MEMORY_TYPE': memory_type}, clear=True):
                import importlib
                import utils.config
                importlib.reload(utils.config)
                
                from utils.config import MEMORY_TYPE
                # Should preserve exact case
                assert MEMORY_TYPE == memory_type
    
    def test_edge_case_memory_window_size_values(self):
        """Test edge case values for memory window size."""
        edge_cases = {
            '1': 1,        # Minimum reasonable value
            '100': 100,    # Large value
            '999': 999,    # Very large value
        }
        
        for size_str, expected in edge_cases.items():
            with patch.dict(os.environ, {'MEMORY_WINDOW_SIZE': size_str}, clear=True):
                import importlib
                import utils.config
                importlib.reload(utils.config)
                
                from utils.config import MEMORY_WINDOW_SIZE
                assert MEMORY_WINDOW_SIZE == expected
    
    def test_edge_case_memory_max_token_limit_values(self):
        """Test edge case values for max token limit."""
        edge_cases = {
            '1': 1,          # Minimum value
            '10000': 10000,  # Large value
            '50000': 50000,  # Very large value
        }
        
        for limit_str, expected in edge_cases.items():
            with patch.dict(os.environ, {'MEMORY_MAX_TOKEN_LIMIT': limit_str}, clear=True):
                import importlib
                import utils.config
                importlib.reload(utils.config)
                
                from utils.config import MEMORY_MAX_TOKEN_LIMIT
                assert MEMORY_MAX_TOKEN_LIMIT == expected


class TestMemoryConfigurationIntegration:
    """Integration tests for memory configuration."""
    
    def test_full_memory_configuration_scenarios(self):
        """Test complete memory configuration scenarios."""
        scenarios = [
            {
                'name': 'No memory configuration',
                'env': {},
                'expected': {
                    'MEMORY_TYPE': 'none',
                    'MEMORY_WINDOW_SIZE': 6,
                    'MEMORY_MAX_TOKEN_LIMIT': 1000
                }
            },
            {
                'name': 'Window memory configuration',
                'env': {
                    'MEMORY_TYPE': 'window',
                    'MEMORY_WINDOW_SIZE': '4'
                },
                'expected': {
                    'MEMORY_TYPE': 'window',
                    'MEMORY_WINDOW_SIZE': 4,
                    'MEMORY_MAX_TOKEN_LIMIT': 1000  # Default
                }
            },
            {
                'name': 'Summary memory configuration',
                'env': {
                    'MEMORY_TYPE': 'summary',
                    'MEMORY_MAX_TOKEN_LIMIT': '2000'
                },
                'expected': {
                    'MEMORY_TYPE': 'summary',
                    'MEMORY_WINDOW_SIZE': 6,  # Default
                    'MEMORY_MAX_TOKEN_LIMIT': 2000
                }
            },
            {
                'name': 'Complete memory configuration',
                'env': {
                    'MEMORY_TYPE': 'window',
                    'MEMORY_WINDOW_SIZE': '8',
                    'MEMORY_MAX_TOKEN_LIMIT': '1500'
                },
                'expected': {
                    'MEMORY_TYPE': 'window',
                    'MEMORY_WINDOW_SIZE': 8,
                    'MEMORY_MAX_TOKEN_LIMIT': 1500
                }
            }
        ]
        
        for scenario in scenarios:
            with patch.dict(os.environ, scenario['env'], clear=True):
                import importlib
                import utils.config
                importlib.reload(utils.config)
                
                from utils.config import MEMORY_TYPE, MEMORY_WINDOW_SIZE, MEMORY_MAX_TOKEN_LIMIT
                
                assert MEMORY_TYPE == scenario['expected']['MEMORY_TYPE'], \
                    f"Failed for scenario: {scenario['name']}"
                assert MEMORY_WINDOW_SIZE == scenario['expected']['MEMORY_WINDOW_SIZE'], \
                    f"Failed for scenario: {scenario['name']}"
                assert MEMORY_MAX_TOKEN_LIMIT == scenario['expected']['MEMORY_MAX_TOKEN_LIMIT'], \
                    f"Failed for scenario: {scenario['name']}"
    
    def test_memory_configuration_validation_integration(self):
        """Test memory configuration with the validation system."""
        # Test that memory config doesn't break existing validation
        valid_env = {
            'MEMORY_TYPE': 'window',
            'MEMORY_WINDOW_SIZE': '6',
            'MEMORY_MAX_TOKEN_LIMIT': '1000',
            'CHUNK_SIZE': '512',
            'CHUNK_OVERLAP': '50',
            'RETRIEVAL_K': '5',
            'SIMILARITY_THRESHOLD': '0.7'
        }
        
        with patch.dict(os.environ, valid_env, clear=True):
            import importlib
            import utils.config
            importlib.reload(utils.config)
            
            # Should not raise any validation errors
            try:
                utils.config.validate_config(strict=False)
            except ValueError:
                pytest.fail("Memory configuration caused validation to fail")
    
    def test_memory_config_with_different_api_keys(self):
        """Test memory configuration with different API keys."""
        memory_config = {
            'MEMORY_TYPE': 'summary',
            'MEMORY_WINDOW_SIZE': '4',
            'MEMORY_MAX_TOKEN_LIMIT': '800'
        }
        
        # Test with Groq API key
        groq_env = {**memory_config, 'GROQ_API_KEY': 'test-groq-key'}
        with patch.dict(os.environ, groq_env, clear=True):
            import importlib
            import utils.config
            importlib.reload(utils.config)
            
            from utils.config import MEMORY_TYPE, GROQ_API_KEY
            assert MEMORY_TYPE == 'summary'
            assert GROQ_API_KEY == 'test-groq-key'
        
        # Test with OpenAI API key
        openai_env = {**memory_config, 'OPENAI_API_KEY': 'test-openai-key'}
        with patch.dict(os.environ, openai_env, clear=True):
            import importlib
            import utils.config
            importlib.reload(utils.config)
            
            from utils.config import MEMORY_TYPE, OPENAI_API_KEY
            assert MEMORY_TYPE == 'summary'
            assert OPENAI_API_KEY == 'test-openai-key'


@pytest.mark.unit
class TestMemoryConfigHelperFunctions:
    """Test memory-related helper functions if any are added."""
    
    def test_memory_configuration_helper_accessibility(self):
        """Test that memory configuration is accessible through config module."""
        with patch.dict(os.environ, {
            'MEMORY_TYPE': 'window',
            'MEMORY_WINDOW_SIZE': '8', 
            'MEMORY_MAX_TOKEN_LIMIT': '1200'
        }, clear=True):
            import importlib
            import utils.config
            importlib.reload(utils.config)
            
            # Test that all memory config variables are accessible
            assert hasattr(utils.config, 'MEMORY_TYPE')
            assert hasattr(utils.config, 'MEMORY_WINDOW_SIZE')
            assert hasattr(utils.config, 'MEMORY_MAX_TOKEN_LIMIT')
            
            # Test values are correct
            assert utils.config.MEMORY_TYPE == 'window'
            assert utils.config.MEMORY_WINDOW_SIZE == 8
            assert utils.config.MEMORY_MAX_TOKEN_LIMIT == 1200
    
    def test_config_summary_includes_memory_info(self):
        """Test that config summary includes memory information (future enhancement)."""
        # This is a placeholder for when we might add memory info to print_config_summary
        with patch.dict(os.environ, {'MEMORY_TYPE': 'window'}, clear=True):
            import importlib
            import utils.config
            importlib.reload(utils.config)
            
            # Just verify the function exists and doesn't crash
            try:
                utils.config.print_config_summary()
            except Exception as e:
                pytest.fail(f"print_config_summary failed with memory config: {e}")


class TestMemoryConfigurationSmokeTets:
    """Smoke tests for memory configuration."""
    
    def test_memory_config_smoke_test_all_types(self):
        """Basic smoke test for all memory types."""
        memory_types = ['none', 'window', 'summary']
        
        for memory_type in memory_types:
            with patch.dict(os.environ, {'MEMORY_TYPE': memory_type}, clear=True):
                import importlib
                import utils.config
                importlib.reload(utils.config)
                
                # Should be able to import without errors
                from utils.config import MEMORY_TYPE, MEMORY_WINDOW_SIZE, MEMORY_MAX_TOKEN_LIMIT
                
                assert MEMORY_TYPE == memory_type
                assert isinstance(MEMORY_WINDOW_SIZE, int)
                assert isinstance(MEMORY_MAX_TOKEN_LIMIT, int)
                assert MEMORY_WINDOW_SIZE > 0
                assert MEMORY_MAX_TOKEN_LIMIT > 0
    
    def test_memory_config_import_performance(self):
        """Test that memory configuration doesn't significantly slow imports."""
        import time
        
        env_vars = {
            'MEMORY_TYPE': 'window',
            'MEMORY_WINDOW_SIZE': '6',
            'MEMORY_MAX_TOKEN_LIMIT': '1000'
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            start_time = time.time()
            
            import importlib
            import utils.config
            importlib.reload(utils.config)
            
            from utils.config import MEMORY_TYPE, MEMORY_WINDOW_SIZE, MEMORY_MAX_TOKEN_LIMIT
            
            end_time = time.time()
            import_time = end_time - start_time
            
            # Should import quickly (less than 1 second)
            assert import_time < 1.0, f"Memory config import took too long: {import_time}s"