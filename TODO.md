# Blender RAG Assistant - Project TODO List

## Phase 1: Project Setup & Foundation
### Environment & Dependencies
- [x] Create and configure virtual environment (Python 3.9+) - *Using Docker container*
- [x] Create comprehensive `requirements.txt` with all dependencies
  - [x] LangChain and extensions
  - [x] ChromaDB for vector storage
  - [x] HuggingFace sentence-transformers
  - [x] Groq SDK for LLM API
  - [x] OpenAI SDK (optional scaling)
  - [x] Pydantic for data validation
  - [x] python-dotenv for configuration
  - [x] pytest for testing
  - [x] tiktoken for token counting
- [ ] Create `.env.example` template with all required variables
- [x] Set up `.gitignore` for Python project
- [ ] Configure logging system

### Project Structure
- [ ] Create directory structure following Ready Tensor patterns
  - [ ] `src/` with modular components
  - [ ] `data/` with raw/processed/vector_db subdirectories
  - [ ] `scripts/` for setup and evaluation
  - [ ] `tests/` for comprehensive test suite
  - [ ] `outputs/` for responses and logs
  - [ ] `docs/` for additional documentation

## Phase 2: Data Gathering and Preprocessing
### Data Collection
- [ ] Download Blender documentation (CC-BY-SA 4.0 licensed)
  - [ ] Identify core sections to keep assignment small-ish (~5 sections)

### Document Prep (`src/data/`)
- [ ] Implement document loader script (`document_loader.py`)
  - [ ] Extract metadata (title, section, subsection)
- [ ] Create text chunking system (`chunker.py`)
  - [ ] Configurable chunk size (512-1024 tokens)
  - [ ] Overlapping chunks for context preservation
  - [ ] Preserve document structure metadata
- [ ] Build preprocessing pipeline (`preprocessor.py`)
  - [ ] Text cleaning and normalization
  - [ ] Code block handling
  - [ ] Special character processing

## Phase 3: Vector Store & Retrieval System
### Embedding System (`src/retrieval/`)
- [ ] Implement embedding generator (`embeddings.py`)
  - [ ] HuggingFace sentence-transformers integration
  - [ ] Batch processing for efficiency
  - [ ] Caching mechanism for repeated queries
- [ ] Create vector store manager (`vector_store.py`)
  - [ ] ChromaDB initialization and configuration
  - [ ] Persistent storage in `data/vector_db/`
  - [ ] Collection management (create/update/delete)
  - [ ] Metadata filtering capabilities

### Retrieval Pipeline
- [ ] Build semantic search system (`retriever.py`)
  - [ ] Similarity search with configurable k
  - [ ] Hybrid search (semantic + keyword)
  - [ ] Relevance scoring and ranking

## Phase 4: RAG Implementation
### Core RAG (`src/rag/`)
- [ ] Create base RAG (`rag.py`)
  - [ ] Query processing and reformulation
  - [ ] Retrieval orchestration
  - [ ] Response generation pipeline
- [ ] Implement prompt templates (`prompts.py`)
  - [ ] System prompts for Blender expertise
  - [ ] Query enhancement prompts
  - [ ] Response formatting instructions

### LLM Integration (`src/rag/llm/`)
- [ ] Create LLM interface abstraction (`base_llm.py`)
- [ ] Implement Groq integration (`groq_llm.py`)
  - [ ] Llama3-8B model configuration
  - [ ] Rate limiting and retry logic
  - [ ] Token counting and management
- [ ] Add OpenAI integration (`openai_llm.py`) [Optional]
  - [ ] GPT model configuration
  - [ ] Streaming response support
- [ ] Build model selector (`model_selector.py`)
  - [ ] Environment-based model selection
  - [ ] Fallback mechanisms

## Phase 5: Configuration & Utilities
### Configuration Management (`src/config/`)
- [ ] Create configuration classes (`config.py`)
  - [ ] Evaluation mode settings
  - [ ] Production mode settings
  - [ ] Model-specific parameters
- [ ] Implement environment manager (`env_manager.py`)
  - [ ] API key validation
  - [ ] Mode detection (evaluation/production)
  - [ ] Dynamic configuration loading

### Utilities (`src/utils/`)
- [ ] Token counter utility (`token_utils.py`)
- [ ] File I/O helpers (`file_utils.py`)
- [ ] Performance monitoring (`metrics.py`)
- [ ] Error handling and logging (`logger.py`)

## Phase 6: Memory & Conversation Management
### Memory Systems (`src/rag/memory/`)
- [ ] Implement conversation buffer (`buffer_memory.py`)
  - [ ] Full conversation retention
  - [ ] Token-aware truncation
- [ ] Create sliding window memory (`window_memory.py`)
  - [ ] Recent message retention
  - [ ] Configurable window size
- [ ] Build summarization memory (`summary_memory.py`)
  - [ ] Conversation compression
  - [ ] Key point extraction

## Phase 7: Advanced Features
### Query Enhancement (`src/rag/query/`)
- [ ] Build query reformulator (`query_reformulator.py`)
  - [ ] Question expansion
  - [ ] Synonym generation

## Phase 8: Scripts
### Setup Scripts (`scripts/`)
- [ ] Create knowledge base setup (`setup_knowledge_base.py`)
  - [ ] Document downloading
  - [ ] Processing pipeline execution
  - [ ] Vector database initialization

## Phase 9: Testing
### Unit Tests (`tests/unit/`)
- [ ] Test document processing (`test_data_processing.py`)
- [ ] Test embedding generation (`test_embeddings.py`)
- [ ] Test retrieval system (`test_retrieval.py`)
- [ ] Test LLM integrations (`test_llm.py`)
- [ ] Test configuration management (`test_config.py`)

### Integration Tests (`tests/integration/`)
- [ ] Test end-to-end pipeline (`test_pipeline.py`)
- [ ] Test memory systems (`test_memory.py`)

## Phase 10: Main Application & Interface
### CLI Application
- [ ] Create main entry point (`main.py`)
  - [ ] Interactive terminal chat interface
  - [ ] Command-line arguments parsing
  - [ ] Mode selection (evaluation/production)
- [ ] Implement chat loop (`src/interface/chat.py`)
  - [ ] User input handling
  - [ ] Response streaming
  - [ ] Session management
  - [ ] Graceful exit handling

### Demo & Examples
- [ ] Create demo script (`demo.py`)
  - [ ] Predefined example queries
  - [ ] Showcase different capabilities
- [ ] Prepare example outputs (`outputs/examples/`)
  - [ ] Sample conversations
  - [ ] Performance metrics
  - [ ] Quality assessments

## Phase 11: Documentation & Polish
### Documentation
- [ ] Update README.md
  - [ ] Project overview
  - [ ] Installation instructions
  - [ ] Usage examples
  - [ ] Architecture diagram
- [ ] Create API documentation
  - [ ] Module documentation
  - [ ] Function docstrings
  - [ ] Type hints throughout
- [ ] Write evaluation report
  - [ ] Performance metrics
  - [ ] Quality assessment
  - [ ] Limitations and future work

### Code Quality
- [ ] Run code formatters (black, isort)
- [ ] Fix all linting issues (ruff)
- [ ] Ensure 80%+ test coverage
- [ ] Remove debug code and TODOs

## Cool Bonus Features (Nice to have ideas)
- [x] Create Docker containerization
- [ ] Add streaming response support
- [ ] Implement caching layer for common queries
- [ ] Create web interface with Streamlit
- [ ] Add multilingual support
