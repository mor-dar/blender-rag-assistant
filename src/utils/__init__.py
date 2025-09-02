"""Utility modules for the Blender RAG Assistant."""

from .logger import configure_logging, get_logger
from .config import (
    # Environment variables
    RAG_MODE,
    GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE, GROQ_MAX_TOKENS,
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS,
    CHROMA_PERSIST_DIRECTORY, CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL, EMBEDDING_DEVICE, EMBEDDING_BATCH_SIZE,
    RETRIEVAL_K, SIMILARITY_THRESHOLD, ENABLE_RERANKING, RETRIEVAL_CONTEXT_WINDOW,
    CHUNK_SIZE, CHUNK_OVERLAP, USE_SEMANTIC_CHUNKING, MIN_CHUNK_SIZE, MAX_CHUNK_SIZE,
    LOG_LEVEL, LOG_FILE, ENABLE_JSON_LOGS, ENABLE_FILE_LOGGING,
    ENABLE_CACHING, CACHE_TTL, MAX_CACHE_SIZE, BATCH_SIZE,
    DEBUG, VERBOSE, ENABLE_PROFILING,
    # Helper functions
    initialize_logging,
    print_config_summary,
    get_vector_db_config_dict,
    get_chunking_config,
    is_production_mode,
    get_active_api_key,
    get_active_model,
)

__all__ = [
    "configure_logging",
    "get_logger",
    # Environment variables 
    "RAG_MODE",
    "GROQ_API_KEY", "GROQ_MODEL", "GROQ_TEMPERATURE", "GROQ_MAX_TOKENS",
    "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_TEMPERATURE", "OPENAI_MAX_TOKENS",
    "CHROMA_PERSIST_DIRECTORY", "CHROMA_COLLECTION_NAME",
    "EMBEDDING_MODEL", "EMBEDDING_DEVICE", "EMBEDDING_BATCH_SIZE",
    "RETRIEVAL_K", "SIMILARITY_THRESHOLD", "ENABLE_RERANKING", "RETRIEVAL_CONTEXT_WINDOW",
    "CHUNK_SIZE", "CHUNK_OVERLAP", "USE_SEMANTIC_CHUNKING", "MIN_CHUNK_SIZE", "MAX_CHUNK_SIZE",
    "LOG_LEVEL", "LOG_FILE", "ENABLE_JSON_LOGS", "ENABLE_FILE_LOGGING",
    "ENABLE_CACHING", "CACHE_TTL", "MAX_CACHE_SIZE", "BATCH_SIZE",
    "DEBUG", "VERBOSE", "ENABLE_PROFILING",
    # Helper functions
    "initialize_logging",
    "print_config_summary", 
    "get_vector_db_config_dict",
    "get_chunking_config",
    "is_production_mode",
    "get_active_api_key",
    "get_active_model",
]