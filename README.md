# Blender RAG Assistant

An intelligent documentation assistant for Blender 3D software, built using Retrieval-Augmented Generation (RAG) architecture. This project demonstrates advanced agentic AI capabilities for Ready Tensor's Agentic AI Developer Certification Program (Module 1).

## Project Overview

The Blender RAG Assistant transforms the complex Blender documentation into an interactive, intelligent guide that can answer user questions with contextual, cited responses. Unlike simple chatbots, this system retrieves relevant documentation chunks and generates accurate, grounded answers while maintaining proper attribution to source materials.

## Architecture

This RAG system follows a modular, scalable architecture:

```
User Query → Query Processing → Vector Retrieval → Context Assembly → LLM Generation → Response Formatting
```

## 🛠️ Technical Implementation

### RAG Architecture

**Modular Design Principles:**
- **Single Responsibility**: Each module handles one concern
- **Clean Interfaces**: Well-defined APIs between components  
- **Comprehensive Testing**: 100% test coverage with unit and integration tests

### RAG Pipeline

1. **Document Processing** (`src/data/processing/`)
   - `document_processor.py`: HTML parsing and text extraction
   - `vector_builder.py`: Orchestrates end-to-end pipeline
   - Intelligent chunking preserves procedural steps

2. **Vector Operations** (`src/retrieval/`)
   - `embeddings.py`: HuggingFace sentence-transformer interface
   - `vector_store.py`: ChromaDB operations and persistence  
   - `retriever.py`: High-level semantic search API
   - Configurable relevance thresholds and metadata filtering

3. **Retrieval Strategy**
   - Query embedding and similarity matching
   - Context window optimization
   - Health checks and error handling

4. **Response Generation**
   - Context-aware prompt construction
   - LLM reasoning with retrieved knowledge
   - Structured response formatting with citations

### Key Technologies

- **Language Models**: Groq Llama3-8B
- **Embeddings**: HuggingFace sentence-transformers
- **Vector Database**: ChromaDB with persistence
- **Framework**: LangChain for orchestration
- **Documentation**: Blender 4.5 Manual (CC-BY-SA 4.0)

## License & Attribution

This project is licensed under the **Creative Commons Attribution-ShareAlike 4.0 International License (CC-BY-SA 4.0)** to maintain compatibility with Blender's documentation licensing.

### Blender Documentation Attribution

This project incorporates content from the **Blender Manual** by the **Blender Documentation Team**, licensed under CC-BY-SA 4.0. All responses that include Blender documentation content include proper attribution and source citations.

- **Source**: https://docs.blender.org/manual/en/latest/
- **License**: CC-BY-SA 4.0
- **Attribution**: Blender Documentation Team

### Development Attribution

This project was developed with assistance from **Anthropic's Claude AI**, which contributed to:
- Documentation structure and technical writing
- Comprehensive test suite development and coverage
- HTML to text conversion cleanup (strange characters)

## Configuration

The system is highly configurable through environment variables. Copy `.env.example_full` to `.env` and customize as needed.

### Environment Variables

#### RAG Mode Configuration
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `RAG_MODE` | string | "evaluation" | Operation mode: "evaluation" (Groq) or "production" (OpenAI) |
| `GROQ_API_KEY` | string | None | API key for Groq (required in evaluation mode) |
| `GROQ_MODEL` | string | "llama3-8b-8192" | Groq model name |
| `GROQ_TEMPERATURE` | float | 0.7 | Temperature for Groq generation |
| `GROQ_MAX_TOKENS` | int | 2048 | Max tokens for Groq responses |
| `OPENAI_API_KEY` | string | None | API key for OpenAI (required in production mode) |
| `OPENAI_MODEL` | string | "gpt-4" | OpenAI model name |
| `OPENAI_TEMPERATURE` | float | 0.7 | Temperature for OpenAI generation |
| `OPENAI_MAX_TOKENS` | int | 2048 | Max tokens for OpenAI responses |

#### Vector Database Configuration
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CHROMA_PERSIST_DIRECTORY` | string | "./data/vector_db" | ChromaDB persistence directory |
| `CHROMA_COLLECTION_NAME` | string | "blender_docs" | Default collection name |

#### Embedding Configuration
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `EMBEDDING_MODEL` | string | "all-MiniLM-L6-v2" | HuggingFace embedding model |
| `EMBEDDING_DEVICE` | string | "cpu" | Device for embeddings: "cpu" or "cuda" |
| `EMBEDDING_BATCH_SIZE` | int | 32 | Batch size for embedding generation |

#### Retrieval Configuration
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `RETRIEVAL_K` | int | 5 | Number of similar documents to retrieve |
| `SIMILARITY_THRESHOLD` | float | 0.7 | Minimum similarity score for results |
| `ENABLE_RERANKING` | bool | false | Enable semantic reranking of results |
| `RETRIEVAL_CONTEXT_WINDOW` | int | 0 | Context window for adjacent chunks |

#### Chunking Configuration
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CHUNK_SIZE` | int | 512 | Target chunk size in tokens |
| `CHUNK_OVERLAP` | int | 50 | Overlap between chunks in tokens |
| `USE_SEMANTIC_CHUNKING` | bool | true | Enable semantic-aware chunking |
| `MIN_CHUNK_SIZE` | int | 50 | Minimum chunk size in tokens |
| `MAX_CHUNK_SIZE` | int | 2048 | Maximum chunk size in tokens |

#### Logging Configuration
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LOG_LEVEL` | string | "INFO" | Logging level: DEBUG, INFO, WARNING, ERROR |
| `LOG_FILE` | string | None | Path to log file (optional) |
| `ENABLE_JSON_LOGS` | bool | false | Output logs in JSON format |
| `ENABLE_FILE_LOGGING` | bool | true | Enable logging to file |

#### Performance Configuration
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENABLE_CACHING` | bool | true | Enable response caching |
| `CACHE_TTL` | int | 3600 | Cache time-to-live in seconds |
| `MAX_CACHE_SIZE` | int | 1000 | Maximum cache entries |
| `BATCH_SIZE` | int | 100 | Default batch size for processing |

#### Development Configuration
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DEBUG` | bool | false | Enable debug mode |
| `VERBOSE` | bool | false | Enable verbose output |
| `ENABLE_PROFILING` | bool | false | Enable performance profiling |

### Usage Rights

You are free to:
- **Share**: Copy and redistribute this project
- **Adapt**: Remix, transform, and build upon the material
- **Commercial Use**: Use for any purpose, even commercially

Under the following terms:
- **Attribution**: You must give appropriate credit
- **ShareAlike**: If you modify this work, distribute under the same license
