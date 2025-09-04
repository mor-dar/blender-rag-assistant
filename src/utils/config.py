#!/usr/bin/env python3
"""
Simplified Configuration System for Blender Bot

Loads configuration from environment variables with sensible defaults.
Supports both Groq and OpenAI models based on API key availability.
"""

import logging
import os
from pathlib import Path


def read_env_variable(variable_name: str, default_value=None, exit_on_missing: bool = False):
    """Read environment variable with logging and error handling.
    
    Args:
        variable_name: Name of the environment variable
        default_value: Default value if variable not found
        exit_on_missing: If True, exit the program when variable is missing and no default
        
    Returns:
        Environment variable value, default value, or None
    """
    if variable_name in os.environ:
        value = os.environ[variable_name]
        logging.debug(f'Using {variable_name}={value} (from environment)')
        return value
    elif default_value is not None:
        logging.debug(f'Using {variable_name}={default_value} (default value)')
        return default_value
    elif exit_on_missing:
        logging.error(f'{variable_name} not set in environment, exiting')
        exit(1)
    else:
        logging.debug(f'{variable_name} not set in environment, using None')
        return None


def str_to_bool(value: str) -> bool:
    """Convert string environment variable to boolean."""
    return str(value).lower() in ('true', '1', 'yes', 'on', 'enabled')


def get_env_int(variable_name: str, default: int) -> int:
    """Get integer from environment variable with default."""
    value = read_env_variable(variable_name, str(default))
    try:
        return int(value)
    except (ValueError, TypeError):
        logging.warning(f'Invalid integer value for {variable_name}={value}, using default {default}')
        return default


def get_env_float(variable_name: str, default: float) -> float:
    """Get float from environment variable with default."""
    value = read_env_variable(variable_name, str(default))
    try:
        return float(value)
    except (ValueError, TypeError):
        logging.warning(f'Invalid float value for {variable_name}={value}, using default {default}')
        return default


def get_env_bool(variable_name: str, default: bool) -> bool:
    """Get boolean from environment variable with default."""
    value = read_env_variable(variable_name, str(default).lower())
    return str_to_bool(value)


# === LLM Configuration ===
# Groq configuration
GROQ_API_KEY = read_env_variable('GROQ_API_KEY')
GROQ_MODEL = read_env_variable('GROQ_MODEL', 'llama-3.1-8b-instant')
GROQ_TEMPERATURE = get_env_float('GROQ_TEMPERATURE', 0.7)
GROQ_MAX_TOKENS = get_env_int('GROQ_MAX_TOKENS', 2048)

# OpenAI configuration
OPENAI_API_KEY = read_env_variable('OPENAI_API_KEY')
OPENAI_MODEL = read_env_variable('OPENAI_MODEL', 'gpt-4')
OPENAI_TEMPERATURE = get_env_float('OPENAI_TEMPERATURE', 0.7)
OPENAI_MAX_TOKENS = get_env_int('OPENAI_MAX_TOKENS', 2048)

# === Vector Database Configuration ===
CHROMA_PERSIST_DIRECTORY = Path(read_env_variable('CHROMA_PERSIST_DIRECTORY', './data/vector_db'))
CHROMA_COLLECTION_NAME = read_env_variable('CHROMA_COLLECTION_NAME', 'blender_docs')

# === Embedding Configuration ===
EMBEDDING_MODEL = read_env_variable('EMBEDDING_MODEL', 'multi-qa-MiniLM-L6-cos-v1')
EMBEDDING_DEVICE = read_env_variable('EMBEDDING_DEVICE', 'cpu')
EMBEDDING_BATCH_SIZE = get_env_int('EMBEDDING_BATCH_SIZE', 32)

# === Retrieval Configuration ===
RETRIEVAL_K = get_env_int('RETRIEVAL_K', 5)
SIMILARITY_THRESHOLD = get_env_float('SIMILARITY_THRESHOLD', 0.7)
ENABLE_RERANKING = get_env_bool('ENABLE_RERANKING', False)
RETRIEVAL_CONTEXT_WINDOW = get_env_int('RETRIEVAL_CONTEXT_WINDOW', 0)

# === Chunking Configuration ===
CHUNK_SIZE = get_env_int('CHUNK_SIZE', 512)
CHUNK_OVERLAP = get_env_int('CHUNK_OVERLAP', 50)
USE_SEMANTIC_CHUNKING = get_env_bool('USE_SEMANTIC_CHUNKING', True)
MIN_CHUNK_SIZE = get_env_int('MIN_CHUNK_SIZE', 50)
MAX_CHUNK_SIZE = get_env_int('MAX_CHUNK_SIZE', 2048)
TOKENIZER_ENCODING = read_env_variable('TOKENIZER_ENCODING', 'cl100k_base')

# === Logging Configuration ===
LOG_LEVEL = read_env_variable('LOG_LEVEL', 'INFO').upper()
LOG_FILE = read_env_variable('LOG_FILE')
ENABLE_JSON_LOGS = get_env_bool('ENABLE_JSON_LOGS', False)
ENABLE_FILE_LOGGING = get_env_bool('ENABLE_FILE_LOGGING', True)

# === Performance Configuration ===
ENABLE_CACHING = get_env_bool('ENABLE_CACHING', True)
CACHE_TTL = get_env_int('CACHE_TTL', 3600)
MAX_CACHE_SIZE = get_env_int('MAX_CACHE_SIZE', 1000)
BATCH_SIZE = get_env_int('BATCH_SIZE', 100)

# === Development Configuration ===
DEBUG = get_env_bool('DEBUG', False)
VERBOSE = get_env_bool('VERBOSE', False)
ENABLE_PROFILING = get_env_bool('ENABLE_PROFILING', False)

# === Memory Configuration ===
MEMORY_TYPE = read_env_variable('MEMORY_TYPE', 'none')  # none, window, summary
MEMORY_WINDOW_SIZE = get_env_int('MEMORY_WINDOW_SIZE', 6)
MEMORY_MAX_TOKEN_LIMIT = get_env_int('MEMORY_MAX_TOKEN_LIMIT', 1000)


# === Helper Functions ===

def get_active_model_info() -> str:
    """Get information about which model will be used based on available API keys."""
    if GROQ_API_KEY:
        return f"Groq ({GROQ_MODEL})"
    elif OPENAI_API_KEY:
        return f"OpenAI ({OPENAI_MODEL})"
    else:
        return "No API key configured"


def get_chunking_config() -> dict:
    """Get chunking configuration as dictionary for backward compatibility."""
    return {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "use_semantic_chunking": USE_SEMANTIC_CHUNKING,
        "min_chunk_size": MIN_CHUNK_SIZE,
        "max_chunk_size": MAX_CHUNK_SIZE,
    }


def get_vector_db_config_dict() -> dict:
    """Get vector database configuration in dictionary format for legacy compatibility."""
    return {
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "use_semantic_chunking": USE_SEMANTIC_CHUNKING,
        "metadata_fields": [
            "title", "section", "subsection", "heading_hierarchy_json", 
            "content_type", "url", "chunk_id", "token_count"
        ],
        "batch_size": BATCH_SIZE
    }


def validate_config(strict: bool = False) -> None:
    """Validate configuration settings.
    
    Args:
        strict: If True, enforce all validation rules. If False, only validate critical settings.
    """
    errors = []
    
    # Validate that at least one API key is present (only in strict mode)
    if strict:
        if not GROQ_API_KEY and not OPENAI_API_KEY:
            errors.append("Either GROQ_API_KEY or OPENAI_API_KEY must be set")
    
    # Ensure directories exist
    CHROMA_PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)
    
    if LOG_FILE and ENABLE_FILE_LOGGING:
        log_file_path = Path(LOG_FILE)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Validate chunking parameters
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        errors.append("CHUNK_OVERLAP must be less than CHUNK_SIZE")
    
    if MIN_CHUNK_SIZE > MAX_CHUNK_SIZE:
        errors.append("MIN_CHUNK_SIZE must be less than or equal to MAX_CHUNK_SIZE")
    
    # Validate retrieval parameters
    if RETRIEVAL_K <= 0:
        errors.append("RETRIEVAL_K must be positive")
    
    if not (0.0 <= SIMILARITY_THRESHOLD <= 1.0):
        errors.append("SIMILARITY_THRESHOLD must be between 0.0 and 1.0")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))


def print_config_summary() -> None:
    """Print a summary of the current configuration."""
    logging.info("Blender Bot Configuration:")
    logging.info(f"  Model: {get_active_model_info()}")
    logging.info(f"  Vector DB: {CHROMA_COLLECTION_NAME} @ {CHROMA_PERSIST_DIRECTORY}")
    logging.info(f"  Embedding Model: {EMBEDDING_MODEL}")
    logging.info(f"  Chunking: {CHUNK_SIZE} tokens ({'semantic' if USE_SEMANTIC_CHUNKING else 'legacy'})")
    logging.info(f"  Retrieval: top-{RETRIEVAL_K} (threshold: {SIMILARITY_THRESHOLD})")
    logging.info(f"  Logging: {LOG_LEVEL}")
    logging.info(f"  Debug: {DEBUG}")


def initialize_logging() -> None:
    """Initialize logging based on configuration."""
    from .logger import configure_logging
    
    configure_logging(
        level=LOG_LEVEL,
        log_file=LOG_FILE,
        enable_json_logs=ENABLE_JSON_LOGS,
    )


# Run validation on import (non-strict)
try:
    validate_config(strict=False)
except Exception as e:
    logging.warning(f"Configuration validation warning: {e}")