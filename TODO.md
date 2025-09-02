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
- [x] Create `.env.example` template with all required variables
- [x] Set up `.gitignore` for Python project
- [x] Configure logging system

### Project Structure
- [x] Create directory structure following Ready Tensor patterns
  - [x] `src/` with modular components (rags, config, data, interface, retrieval, utils)
  - [x] `data/` with raw/processed/vector_db subdirectories
  - [x] `scripts/` for setup and evaluation
  - [x] `tests/` for comprehensive test suite
  - [x] `outputs/` for responses and logs
  - [x] `docs/` for additional documentation

## Phase 2: Data Gathering and Preprocessing
### Data Collection
- [x] Download Blender documentation (CC-BY-SA 4.0 licensed)
  - [x] Identify core sections to keep assignment small-ish (~5 sections)

### Document Prep (`src/data/`)
- [x] Implement document loader script (`document_processor.py`)
  - [x] Extract metadata (title, section, subsection)
  - [x] Extract HTML hierarchy and heading structure
  - [x] Classify content type (procedural, reference, conceptual, etc.)
- [x] Create text chunking system (`document_processor.py`)
  - [x] Configurable chunk size (512-1024 tokens)
  - [x] Overlapping chunks for context preservation
  - [x] Preserve document structure metadata in chunks

## Phase 3: Vector Store & Retrieval System ✅ **COMPLETE - Elite Architecture**
### Embedding System (`src/retrieval/`)
- [x] Implement embedding generator (`embeddings.py`)
  - [x] HuggingFace sentence-transformers integration
  - [x] Batch processing for efficiency
  - [x] Caching mechanism for repeated queries
- [x] Create vector store manager (`vector_store.py`)
  - [x] ChromaDB initialization and configuration
  - [x] Persistent storage in `data/vector_db/`
  - [x] Collection management (create/update/delete)
  - [x] Metadata filtering capabilities

### Retrieval Pipeline
- [x] Build semantic search system (`retriever.py`)
  - [x] Similarity search with configurable k
  - [x] Semantic search with vector embeddings
  - [x] Relevance scoring and ranking

### Encoding Issue Fixes ✅ **IMPLEMENTATION COMPLETE - Database Rebuild Required**
- [x] Fix HTML encoding artifacts (`src/data/processing/text_cleaner.py`)
  - [x] HTML entity decoding (`Â¶` → `¶`, `â£` → `→`)
  - [x] Unicode normalization (NFKC) for mixed encodings
  - [x] Special character cleaning for malformed UTF-8
  - [x] Preserve technical symbols and code blocks properly
  - [x] **Comprehensive unit tests**: 31 tests covering all functionality
  - [x] **Integration with DocumentProcessor**: Automatic cleaning during processing
  - [x] **Search impact verified**: Current database still has artifacts, confirming need for rebuild
- [ ] Optimize chunking strategy for technical documentation
  - [ ] Increase chunk size from 512 to 1024+ tokens for better context
  - [ ] Adjust overlap ratio for technical content (25% instead of 10%)
  - [ ] Preserve procedural steps and code examples as complete units
  - [ ] Test chunk boundary detection to avoid splitting mid-sentence
- [ ] Improve embedding quality for technical content
  - [ ] Evaluate alternative embedding models (`all-mpnet-base-v2`, `text-embedding-ada-002`)
  - [ ] Test domain-specific preprocessing for Blender terminology
  - [ ] Implement embedding model comparison and benchmarking

## Phase 4: RAG Implementation
### Core RAG (`src/rag/`)
- [x] Create base RAG (`rag.py`)
  - [x] Query processing and reformulation
  - [x] Retrieval orchestration
  - [x] Response generation pipeline
- [x] Implement prompt templates (`prompts.py`)
  - [x] System prompts for Blender expertise
  - [x] Query enhancement prompts
  - [x] Response formatting instructions

### LLM Integration (`src/rag/llm/`)
- [ ] Create LLM interface abstraction (`base_llm.py`) *Not needed - direct imports work fine*
- [x] Implement Groq integration (`groq_llm.py`)
  - [x] Llama3-8B model configuration
  - [ ] Rate limiting and retry logic
  - [ ] Token counting and management
- [x] Add OpenAI integration (`openai_llm.py`) [Optional]
  - [x] GPT model configuration
  - [ ] Streaming response support
- [x] Build model selector (`model_selector.py`)
  - [x] Environment-based model selection
  - [x] Fallback mechanisms

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

### Advanced Retrieval (`src/retrieval/`)
- [ ] Implement hybrid search (`hybrid_retriever.py`)
  - [ ] Combine semantic similarity with keyword matching
  - [ ] Weighted scoring between semantic and lexical relevance
  - [ ] Query expansion using both approaches

## Phase 8: Scripts
### Setup Scripts (`scripts/`)
- [x] Create knowledge base setup (`setup_knowledge_base.py`)
  - [x] Document downloading
  - [x] Processing pipeline execution
  - [x] Vector database initialization

## Phase 9: Testing **IN PROGRESS - 119 Tests for Implemented Modules**
### Unit Tests (`tests/data/processing/` & `tests/retrieval/`)
- [x] Test document processing (`test_document_processor.py`) - 22 tests
- [x] Test embedding generation (`test_embeddings.py`) - 18 tests  
- [x] Test retrieval system (`test_retriever.py`) - 31 tests
- [x] Test vector store operations (`test_vector_store.py`) - 29 tests
- [x] Test vector builder orchestration (`test_vector_builder.py`) - 19 tests
- [ ] Test LLM integrations (`test_llm.py`) *awaiting Phase 4 implementation*
- [ ] Test configuration management (`test_config.py`) *awaiting Phase 5 implementation*

### Integration Tests
- [x] Test end-to-end pipeline (via `test_vector_builder.py`)
- [ ] Test full RAG pipeline (`test_pipeline.py`) *awaiting Phase 4 implementation*
- [ ] Test memory systems (`test_memory.py`) *awaiting Phase 6 implementation*

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

## Phase 12: Enhanced Chunking Strategy (Future Enhancement)
### Structural/Semantic Chunking Upgrade
- [ ] **Analysis & Planning**
  - [ ] Analyze current fixed-size token chunking limitations
  - [ ] Research hybrid structural + semantic chunking approaches
  - [ ] Design chunking strategy specific to technical documentation
  
- [ ] **HTML Structure Parser (`src/data/processing/structure_parser.py`)**
  - [ ] Parse HTML document hierarchy (h1, h2, h3, sections, lists)
  - [ ] Identify content blocks and their relationships
  - [ ] Extract procedural steps and maintain their sequence
  - [ ] Preserve code blocks and examples as units
  
- [ ] **Semantic Chunking Engine (`src/data/processing/semantic_chunker.py`)**
  - [ ] Implement topic boundary detection using embeddings
  - [ ] Create content coherence scoring for chunk quality
  - [ ] Handle Blender-specific terminology and concepts
  - [ ] Maintain context between related procedures
  
- [ ] **Hybrid Chunking Pipeline (`src/data/processing/hybrid_chunker.py`)**
  - [ ] **Primary Strategy**: Split on HTML structural boundaries (sections, headings)
  - [ ] **Secondary Strategy**: Use semantic similarity for large sections
  - [ ] **Size Constraints**: Ensure chunks fit embedding model limits (512-1024 tokens)
  - [ ] **Smart Overlap**: Preserve context at structural boundaries
  - [ ] **Procedural Integrity**: Keep step-by-step instructions together
  
- [ ] **Implementation Details**
  - [ ] Replace current `chunk_text()` method with structural approach
  - [ ] Add chunking strategy configuration options
  - [ ] Implement backward compatibility for existing vector databases
  - [ ] Create migration script for re-chunking existing data
  
- [ ] **Quality Improvements Expected**
  - [ ] Better retrieval: Complete procedural steps returned together
  - [ ] Context preservation: Tool explanations stay with usage instructions
  - [ ] Feature isolation: Modeling techniques don't mix with animation workflows
  - [ ] Natural boundaries: Sections like "Basic Operations" stay intact
  
- [ ] **Testing & Validation**
  - [ ] Create test suite for new chunking strategies
  - [ ] Compare retrieval quality: structural vs. fixed-size chunking
  - [ ] Measure chunk coherence and semantic integrity
  - [ ] Validate with Blender-specific use cases and queries

## Phase 13: Advanced Preprocessing (Future Enhancement)
- [ ] Build preprocessing pipeline (`preprocessor.py`)
  - [ ] Text cleaning and normalization
  - [ ] Code block handling
  - [ ] Special character processing
  
## Cool Bonus Features (Nice to have ideas)
- [x] Create Docker containerization
- [ ] Add streaming response support
- [ ] Implement caching layer for common queries
- [ ] Create web interface with Streamlit
- [ ] Add multilingual support
