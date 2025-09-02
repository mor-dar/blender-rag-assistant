from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import logging
from utils.config import (
    OPENAI_MODEL,
    OPENAI_API_KEY,
    OPENAI_TEMPERATURE,
    OPENAI_MAX_TOKENS,
)
from rag.prompts import blender_bot_template


class OpenAILLM:
    """
    OpenAI LLM wrapper for the Blender RAG Assistant.
    
    Provides an interface to OpenAI's language models using LangChain,
    specifically configured for Blender documentation Q&A tasks.
    """
    
    def __init__(self):
        """Initialize the OpenAI LLM with configuration from environment variables."""
        self.model = ChatOpenAI(
            model=OPENAI_MODEL, 
            temperature=OPENAI_TEMPERATURE, 
            api_key=OPENAI_API_KEY,
            max_tokens=OPENAI_MAX_TOKENS
        )
        logging.info(f"Initialized OpenAI LLM with model: {OPENAI_MODEL}")

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
            
            # Send formatted prompt to OpenAI model
            response = self.model.invoke([HumanMessage(content=formatted_prompt)])
            
            # Log for debugging purposes
            logging.debug(f"OpenAI LLM request: {formatted_prompt}")
            logging.debug(f"OpenAI LLM response: {response.content}")
            
            return response.content
        except Exception as e:
            # Log error and return user-friendly message
            logging.error(f"Error occurred while invoking OpenAI LLM: {e}")
            return "Error occurred while processing your request."
