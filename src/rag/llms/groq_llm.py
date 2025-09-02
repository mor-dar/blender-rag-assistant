from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import logging
from utils.config import (
    GROQ_MODEL,
    GROQ_API_KEY,
    GROQ_TEMPERATURE,
)
from rag.prompts import blender_bot_template


class GroqLLM:
    """
    Groq LLM wrapper for the Blender RAG Assistant.
    
    Provides an interface to Groq's language models using LangChain,
    specifically configured for Blender documentation Q&A tasks.
    """
    
    def __init__(self):
        """Initialize the Groq LLM with configuration from environment variables."""
        self.model = ChatGroq(model=GROQ_MODEL, temperature=GROQ_TEMPERATURE, api_key=GROQ_API_KEY)
        logging.info(f"Initialized GroqLLM with model: {GROQ_MODEL}")

    def invoke(self, question: str, context: str) -> str:
        """
        Generate a response to a Blender-related question using retrieved context.
        
        Args:
            question: The user's question about Blender
            context: Relevant documentation context retrieved from vector store
            
        Returns:
            Generated response based on the provided context
            
        Raises:
            Returns error message string if API call fails
        """
        try:
            # Format the prompt template with context and question
            formatted_prompt = blender_bot_template.format(
                context=context,
                question=question
            )
            
            # Send formatted prompt to Groq model
            response = self.model.invoke([HumanMessage(content=formatted_prompt)])
            
            # Log for debugging purposes
            logging.debug(f"GroqLLM request: {formatted_prompt}")
            logging.debug(f"GroqLLM response: {response.content}")
            
            return response.content
        except Exception as e:
            # Log error and return user-friendly message
            logging.error(f"Error occurred while invoking GroqLLM: {e}")
            return "Error occurred while processing your request."
