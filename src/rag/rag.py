import logging
from typing import List, Optional, Any
from rag.prompts import blender_bot_template
from retrieval import SemanticRetriever
from retrieval.retriever import RetrievalResult
from utils.config import (
    CHROMA_PERSIST_DIRECTORY,
    EMBEDDING_MODEL,
    CHROMA_COLLECTION_NAME,
    GROQ_API_KEY,
    OPENAI_API_KEY,
    MEMORY_TYPE,
    MEMORY_WINDOW_SIZE,
    MEMORY_MAX_TOKEN_LIMIT
)

if GROQ_API_KEY is not None:
    from rag.llms.groq_llm import GroqLLM as LLM
elif OPENAI_API_KEY is not None:
    from rag.llms.openai_llm import OpenAILLM as LLM
else:
    logging.error("No API key found for Groq or OpenAI. Please set GROQ_API_KEY or OPENAI_API_KEY in environment.")

# Import memory classes (always import, use conditionally)
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory


class BlenderAssistantRAG:
    """
    Main RAG orchestrator for the Blender documentation assistant.
    
    Coordinates between the semantic retriever and language model to provide
    accurate answers to Blender-related questions using retrieved context.
    """

    def __init__(self):
        """
        Initialize the RAG system with retriever, language model, and optional memory.
        """
        # Initialize semantic retriever for document search
        self.retriever = SemanticRetriever(
            db_path=CHROMA_PERSIST_DIRECTORY,
            embedding_model=EMBEDDING_MODEL,
            collection_name=CHROMA_COLLECTION_NAME
        )

        # Store prompt template for LLM formatting
        self.system_prompt = blender_bot_template
        
        # Initialize LLM with error handling
        try:
            self.llm = LLM()  # Instantiate the selected LLM (Groq or OpenAI)
        except Exception as e:
            logging.error(f"Failed to initialize LLM: {e}")
            self.llm = None
            
        # Initialize memory system based on configuration
        self.memory: Optional[Any] = None
        self._init_memory()
        
    def _init_memory(self) -> None:
        """Initialize conversation memory based on MEMORY_TYPE configuration."""
        if MEMORY_TYPE == 'window':
            try:
                self.memory = ConversationBufferWindowMemory(
                    k=MEMORY_WINDOW_SIZE,
                    return_messages=False  # Return as string for easier integration
                )
                logging.info(f"Initialized window memory (size: {MEMORY_WINDOW_SIZE})")
            except Exception as e:
                logging.error(f"Failed to initialize window memory: {e}")
                
        elif MEMORY_TYPE == 'summary':
            if self.llm and hasattr(self.llm, 'model'):
                try:
                    self.memory = ConversationSummaryMemory(
                        llm=self.llm.model,
                        max_token_limit=MEMORY_MAX_TOKEN_LIMIT,
                        return_messages=False  # Return as string for easier integration
                    )
                    logging.info(f"Initialized summary memory (max tokens: {MEMORY_MAX_TOKEN_LIMIT})")
                except Exception as e:
                    logging.error(f"Failed to initialize summary memory: {e}")
            else:
                logging.warning("Summary memory requires LLM, but LLM not available")
                
        else:
            logging.info("No conversation memory enabled (MEMORY_TYPE=none)")

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
        Process a user query through the complete RAG pipeline with optional memory.
        
        Args:
            query: User's question about Blender
            
        Returns:
            Generated response based on retrieved documentation context and conversation history
        """
        # Check if LLM is available
        if not self.llm:
            logging.error("LLM is not initialized, but query was received.")
            return "LLM not available. Please check API key configuration."
        
        # Retrieve relevant context from vector database
        retrieval_results = self.retriever.retrieve(query)
        
        # Format retrieval results into string context for LLM
        formatted_context = self._format_context(retrieval_results)
        
        # Add conversation history if memory is enabled
        if self.memory:
            try:
                # Get conversation history
                memory_vars = self.memory.load_memory_variables({})
                conversation_history = memory_vars.get('history', '')
                
                # Combine context with conversation history
                if conversation_history:
                    full_context = f"{formatted_context}\n\nConversation History:\n{conversation_history}"
                else:
                    full_context = formatted_context
                    
                # Generate response using LLM with full context
                response = self.llm.invoke(question=query, context=full_context)
                
                # Save the conversation to memory
                self.memory.save_context({"input": query}, {"output": response})
                
            except Exception as e:
                logging.error(f"Error using memory: {e}")
                # Fallback to normal operation without memory
                response = self.llm.invoke(question=query, context=formatted_context)
        else:
            # Generate response using LLM with formatted context (no memory)
            response = self.llm.invoke(question=query, context=formatted_context)
        
        return response
        
    def clear_memory(self) -> None:
        """Clear conversation memory if enabled."""
        if self.memory:
            try:
                self.memory.clear()
                logging.info("Conversation memory cleared")
            except Exception as e:
                logging.error(f"Error clearing memory: {e}")
        else:
            logging.info("No memory to clear")
            
    def get_memory_info(self) -> dict:
        """Get information about the current memory configuration and state."""
        info = {
            "memory_type": MEMORY_TYPE,
            "memory_enabled": self.memory is not None
        }
        
        if self.memory:
            try:
                if MEMORY_TYPE == 'window':
                    info.update({
                        "window_size": MEMORY_WINDOW_SIZE,
                        "current_messages": len(self.memory.chat_memory.messages) if hasattr(self.memory, 'chat_memory') else 0
                    })
                elif MEMORY_TYPE == 'summary':
                    info.update({
                        "max_token_limit": MEMORY_MAX_TOKEN_LIMIT,
                        "has_summary": hasattr(self.memory, 'summary') and bool(getattr(self.memory, 'summary', None))
                    })
            except Exception as e:
                logging.error(f"Error getting memory info: {e}")
                info["error"] = str(e)
                
        return info
