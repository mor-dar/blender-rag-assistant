import logging
from rag.rag import BlenderAssistantRAG
import dotenv
from utils.config import (
    initialize_logging
)

def main() -> None:
    dotenv.load_dotenv()  # Load environment variables from .env file (if it exists)
    initialize_logging()
    logging.info("Blender RAG Assistant started.")
    # Start with a simple query input for now
    query = input("Enter your query: ")
    logging.info(f"User query: {query}")
    
    rag = BlenderAssistantRAG(collection_type="demo")
    response = rag.handle_query(query=query)
    logging.info(f"RAG response: {response}")

if __name__ == "__main__":
    main()