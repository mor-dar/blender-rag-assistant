# Blender Bot

An intelligent documentation assistant for Blender 3D software, built using Retrieval-Augmented Generation (RAG) architecture. This project demonstrates advanced agentic AI capabilities for Ready Tensor's Agentic AI Developer Certification Program (Module 1).

## Project Overview

Blender Bot transforms the complex Blender documentation into an interactive, intelligent guide that can answer user questions with contextual, cited responses. Unlike simple chatbots, this system retrieves relevant documentation chunks and generates accurate, grounded answers while maintaining proper attribution to source materials.

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
   - Structured response formatting with numbered citations [1], [2], [3]
   - Clean source references with working Blender documentation URLs

5. **Conversation Memory** (Optional)
   - Window memory: Keep recent N messages for context
   - Summary memory: Summarize old conversations to manage token usage
   - Configurable via `MEMORY_TYPE` environment variable

### Key Technologies

- **Language Models**: Groq (Llama3.1-8B by default) or OpenAI GPT models
- **Embeddings**: HuggingFace sentence-transformers
- **Vector Database**: ChromaDB with persistence
- **Framework**: LangChain for orchestration
- **Citations**: Numbered references with working documentation URLs

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

The system is highly configurable through environment variables. Copy `.env.example` to `.env` and customize as needed.

### Environment Variables

#### LLM Configuration
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GROQ_API_KEY` | string | None | API key for Groq models |
| `GROQ_MODEL` | string | "llama-3.1-8b-instant" | Groq model name |
| `GROQ_TEMPERATURE` | float | 0.7 | Temperature for Groq generation |
| `GROQ_MAX_TOKENS` | int | 2048 | Max tokens for Groq responses |
| `OPENAI_API_KEY` | string | None | API key for OpenAI models |
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
| `EMBEDDING_MODEL` | string | "multi-qa-MiniLM-L6-cos-v1" | HuggingFace embedding model |
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

#### Memory Configuration
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MEMORY_TYPE` | string | "none" | Conversation memory type: "none", "window", or "summary" |
| `MEMORY_WINDOW_SIZE` | int | 6 | Number of recent messages to keep (window memory only) |
| `MEMORY_MAX_TOKEN_LIMIT` | int | 1000 | Max tokens before summarization (summary memory only) |

## Usage - Interface Options

Blender Bot provides two interaction modes to suit different preferences:

### Command Line Interface (CLI)

The default interface provides a simple terminal-based chat experience:

```bash
# Run CLI interface (default)
python main.py

# Or explicitly specify CLI
python main.py --cli
```

**Features:**
- Simple text-based interaction
- Perfect for development and scripting
- Lightweight with minimal dependencies
- Suitable for server environments

### Web Interface (Streamlit)

A modern, user-friendly web interface with rich features:

```bash
# Launch web interface
python main.py --web
```

**Features:**
- Configuration sidebar with system status and environment info
- Built-in sample questions for easy testing
- Random question generator for exploration
- Response time monitoring
- System reinitialization without restart

**Access:** Once launched, open your browser to `http://localhost:8501`

### Interface Selection

```bash
# Show all available options
python main.py --help

# CLI interface (default)
python main.py
python main.py --cli

# Web interface  
python main.py --web

# Check version
python main.py --version
```

Both interfaces use the same underlying RAG system and provide identical functionality - choose the one that best fits your workflow!

## Docker Deployment

Blender Bot provides a comprehensive Docker setup with custom commands for different deployment scenarios.

### Quick Start

#### Using Pre-built Image

```bash
# Pull the latest version
docker pull mdar/blender-rag-assistant:v1.0.5
```

#### Available Commands

The Docker image supports six custom commands for different deployment scenarios:

**Evaluation Commands (Demo Database):**
- `evaluate-web`: Download data, build demo DB, launch web interface
- `evaluate-cli`: Download data, build demo DB, launch CLI interface

> **Note:** Demo mode uses a database built from only 5 pages of the Blender manual for quick testing and evaluation.

**Setup Commands:**
- `build-demo`: Download data and build demo database only (5 pages)
- `build-full`: Download data and build full database only (complete manual)

**Runtime Commands (Assume Database Exists):**
- `run-web`: Launch web interface only
- `run-cli`: Launch CLI interface only

### Usage Examples

#### 1. Quick Evaluation (Web Interface)
Perfect for testing and evaluation - sets up demo database and launches web UI:

```bash
docker run -p 8501:8501 -v $(pwd)/data:/app/data mdar/blender-rag-assistant:v1.0.5 evaluate-web
```

Then open your browser to http://localhost:8501

#### 2. Quick Evaluation (CLI Interface)
Interactive command-line evaluation with demo database:

```bash
docker run -it -v $(pwd)/data:/app/data mdar/blender-rag-assistant:v1.0.5 evaluate-cli
```

#### 3. Build Demo Database Only
Useful for CI/CD pipelines or when you want to separate setup from runtime:

```bash
docker run -v $(pwd)/data:/app/data mdar/blender-rag-assistant:v1.0.5 build-demo
```

#### 4. Build Full Production Database
For production deployment with complete Blender documentation:

```bash
docker run -v $(pwd)/data:/app/data mdar/blender-rag-assistant:v1.0.5 build-full
```

#### 5. Run Web Interface (DB Pre-built)
When database is already built and persisted:

```bash
docker run -p 8501:8501 -v $(pwd)/data:/app/data mdar/blender-rag-assistant:v1.0.5 run-web
```

#### 6. Run CLI Interface (DB Pre-built)
Command-line interface with existing database:

```bash
docker run -it -v $(pwd)/data:/app/data mdar/blender-rag-assistant:v1.0.5 run-cli
```

### Docker Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key | - | For OpenAI models |
| `GROQ_API_KEY` | Groq API key | - | For Groq models |

### Volume Mounts

**Volume Mounts (Optional):**
- `-v $(pwd)/data:/app/data` - Persist downloaded docs and vector database (recommended for data persistence)
- `-v $(pwd)/.env:/app/.env` - Custom environment configuration

**Note:** Volume mounts are optional but recommended. Without data volume mount, the system will re-download documentation on each container restart.

### Production Deployment

#### Complete Production Setup

```bash
# Pull production image
docker pull mdar/blender-rag-assistant:v1.0.5

# Set up full database (one-time setup)
docker run -v $(pwd)/data:/app/data \
  mdar/blender-rag-assistant:v1.0.5 build-full

# Run web interface
docker run -d -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  --name blender-rag-prod \
  mdar/blender-rag-assistant:v1.0.5 run-web
```

#### Using Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  blender-rag:
    image: mdar/blender-rag-assistant:v1.0.5
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
    command: evaluate-web
```

Run with:
```bash
docker-compose up
```

### Development and Testing

```bash
# Run tests
docker run --rm mdar/blender-rag-assistant:v1.0.5 python -m pytest

# Interactive shell
docker run -it --entrypoint bash mdar/blender-rag-assistant:v1.0.5

# Custom commands
docker run --rm mdar/blender-rag-assistant:v1.0.5 python --version
```

### Troubleshooting

**Common Issues:**

1. **Permission Errors**: Ensure volumes are writable
   ```bash
   chmod 755 data/
   ```

2. **Port Already in Use**: Change port mapping
   ```bash
   docker run -p 8502:8501 ... run-web
   ```

3. **Memory Issues**: Increase Docker memory limit (4GB+ recommended)

4. **API Key Issues**: Verify environment variables
   ```bash
   docker run --rm -e OPENAI_API_KEY=test blender-rag-assistant bash -c "env | grep API"
   ```

## Evaluation Mode

When running in evaluation mode (default configuration), try these example questions to test the RAG system's capabilities:

### Suggested Evaluation Questions

Based on the foundational concepts from the [Ready Tensor Agentic AI Developer Certification Program](https://app.readytensor.ai/publications/aaidc-module-1-project-foundations-of-agentic-ai-your-first-rag-assistant-4n07ViGCey0l) and related [publications](https://app.readytensor.ai/publications/WsaE5uxLBqnH), these questions test different aspects of the Blender documentation knowledge:

1. **"How do I use the eyedropper tool to sample colors in Blender?"**
   - Common workflow question about color sampling

2. **"What are the different types of buttons in Blender's interface and what do they do?"**
   - Basic UI understanding question

3. **"How can I quickly enter exact values into number fields instead of dragging?"**
   - Practical workflow efficiency question

4. **"What do the small icons and decorators next to buttons mean?"**
   - Interface comprehension question about visual indicators

5. **"How do I access context menus and what options are usually available?"**
   - Right-click menu and navigation question

6. **"Can I customize or change the appearance of interface buttons and menus?"**
   - Customization and preferences question

These questions cover fundamental Blender interface concepts and demonstrate the RAG system's ability to retrieve and synthesize information from the documentation.

### Usage Rights

You are free to:
- **Share**: Copy and redistribute this project
- **Adapt**: Remix, transform, and build upon the material
- **Commercial Use**: Use for any purpose, even commercially

Under the following terms:
- **Attribution**: You must give appropriate credit
- **ShareAlike**: If you modify this work, distribute under the same license
