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

### RAG Pipeline

1. **Document Ingestion**
   - Processes Blender manual markdown files
   - Intelligent chunking preserves procedural steps
   - Generates embeddings using HuggingFace transformers

2. **Vector Storage**
   - ChromaDB with persistent storage
   - Semantic similarity search
   - Configurable relevance thresholds

3. **Retrieval Strategy**
   - Query embedding and similarity matching
   - Context window optimization
   - Duplicate detection and filtering

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

### Usage Rights

You are free to:
- **Share**: Copy and redistribute this project
- **Adapt**: Remix, transform, and build upon the material
- **Commercial Use**: Use for any purpose, even commercially

Under the following terms:
- **Attribution**: You must give appropriate credit
- **ShareAlike**: If you modify this work, distribute under the same license
