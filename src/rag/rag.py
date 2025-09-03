import logging
from typing import List
from rag.prompts import blender_bot_template
from retrieval import SemanticRetriever
from retrieval.retriever import RetrievalResult
from utils.config import (
    CHROMA_PERSIST_DIRECTORY,
    EMBEDDING_MODEL,
    CHROMA_COLLECTION_NAME,
    GROQ_API_KEY,
    OPENAI_API_KEY
)

if GROQ_API_KEY is not None:
    from rag.llms.groq_llm import GroqLLM as LLM
elif OPENAI_API_KEY is not None:
    from rag.llms.openai_llm import OpenAILLM as LLM
else:
    logging.error("No API key found for Groq or OpenAI. Please set GROQ_API_KEY or OPENAI_API_KEY in environment.")


class BlenderAssistantRAG:
    """
    Main RAG orchestrator for the Blender documentation assistant.
    
    Coordinates between the semantic retriever and language model to provide
    accurate answers to Blender-related questions using retrieved context.
    """

    def __init__(self, collection_type: str = "demo"):
        """
        Initialize the RAG system with retriever and language model.
        
        Args:
            collection_type: Collection name suffix ("demo" or "full")
        """
        # Initialize semantic retriever for document search
        self.retriever = SemanticRetriever(
            db_path=CHROMA_PERSIST_DIRECTORY,
            embedding_model=EMBEDDING_MODEL,
            collection_name=f"{CHROMA_COLLECTION_NAME}_{collection_type}"
        )

        # Store prompt template for LLM formatting
        self.system_prompt = blender_bot_template
        
        # Initialize LLM with error handling
        try:
            self.llm = LLM()  # Instantiate the selected LLM (Groq or OpenAI)
        except Exception as e:
            logging.error(f"Failed to initialize LLM: {e}")
            self.llm = None

    def _format_context(self, results: List[RetrievalResult]) -> str:
        """
        Format retrieval results into a structured string context for the LLM.
        
        Args:
            results: List of RetrievalResult objects from semantic search
            
        Returns:
            Formatted context string with sources and content
        """
        if not results:
            return "No relevant documentation found for this query."
        
        context_parts = ["Retrieved Documentation Context:\n"]
        
        for i, result in enumerate(results, 1):
            # Extract source info from metadata if available
            source_info = "Unknown source"
            if result.metadata:
                if 'source' in result.metadata:
                    source_info = result.metadata['source']
                elif 'file_path' in result.metadata:
                    source_info = result.metadata['file_path']
            
            # Format each result with source and score
            context_parts.append(f"[Source {i} - Score: {result.score:.3f}]")
            context_parts.append(f"From: {source_info}")
            context_parts.append(f"Content: {result.text.strip()}")
            context_parts.append("")  # Empty line for readability
        
        return "\n".join(context_parts)

    def handle_query(self, query: str) -> str:
        """
        Process a user query through the complete RAG pipeline.
        
        Args:
            query: User's question about Blender
            
        Returns:
            Generated response based on retrieved documentation context
        """
        # Check if LLM is available
        if not self.llm:
            logging.error("LLM is not initialized, but query was received.")
            return "LLM not available. Please check API key configuration."
        
        # Retrieve relevant context from vector database
        retrieval_results = self.retriever.retrieve(query)
        
        # Format retrieval results into string context for LLM
        formatted_context = self._format_context(retrieval_results)
        
        # Generate response using LLM with formatted context
        response = self.llm.invoke(question=query, context=formatted_context)
        
        return response
