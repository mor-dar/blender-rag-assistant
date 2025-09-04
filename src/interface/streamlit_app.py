"""
Streamlit web interface for Blender Bot.

This module provides a user-friendly web interface using Streamlit,
allowing users to interact with the RAG system through a browser.
"""

import streamlit as st
import logging
from typing import List, Dict, Any, Optional
import time
import os
from datetime import datetime

# Import RAG system components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from rag.rag import BlenderAssistantRAG
from utils.config import initialize_logging
import dotenv


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False


def setup_rag_system() -> Optional[BlenderAssistantRAG]:
    """Initialize the RAG system with error handling."""
    try:
        if st.session_state.rag_system is None:
            with st.spinner("Initializing RAG system..."):
                dotenv.load_dotenv()
                initialize_logging()
                st.session_state.rag_system = BlenderAssistantRAG()
                st.session_state.initialized = True
                logging.info("Streamlit RAG system initialized successfully")
        
        return st.session_state.rag_system
    
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
        logging.error(f"RAG initialization error: {e}")
        return None


def display_message(role: str, content: str, timestamp: Optional[datetime] = None):
    """Display a chat message with proper formatting."""
    with st.chat_message(role):
        st.write(content)
        if timestamp:
            st.caption(f"*{timestamp.strftime('%H:%M:%S')}*")


def display_chat_history():
    """Display the conversation history."""
    for message in st.session_state.messages:
        display_message(
            role=message["role"],
            content=message["content"],
            timestamp=message.get("timestamp")
        )


def add_message(role: str, content: str):
    """Add a message to the chat history."""
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now()
    })


def clear_chat():
    """Clear the chat history."""
    st.session_state.messages = []
    st.rerun()


def create_sidebar():
    """Create the sidebar with configuration and information."""
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # RAG System Status
        st.subheader("System Status")
        if st.session_state.initialized:
            st.success("✅ RAG System Online")
        else:
            st.warning("⚠️ RAG System Not Initialized")
        
        # Environment Info
        st.subheader("Environment")
        rag_mode = os.getenv('RAG_MODE', 'evaluation')
        st.info(f"**Mode**: {rag_mode.title()}")
        
        # Model Info
        if rag_mode == 'evaluation':
            model = os.getenv('GROQ_MODEL', 'llama3-8b-8192')
            st.info(f"**Model**: {model}")
        else:
            model = os.getenv('OPENAI_MODEL', 'gpt-4')
            st.info(f"**Model**: {model}")
        
        # Actions
        st.subheader("Actions")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            clear_chat()
        
        if st.button("🔄 Reinitialize System", use_container_width=True):
            st.session_state.rag_system = None
            st.session_state.initialized = False
            st.rerun()
        
        # Help Section
        st.subheader("💡 Sample Questions")
        sample_questions = [
            "What are the different types of buttons in Blender's interface?",
            "How do I use the eyedropper tool to sample colors?",
            "What does it mean to extrude a face in Blender?",
            "How can I quickly enter exact values into number fields?",
            "Can I customize interface buttons and menus?"
        ]
        
        for question in sample_questions:
            if st.button(f"📝 {question[:30]}...", key=f"sample_{hash(question)}"):
                st.session_state.user_input = question
                st.rerun()
        
        # Documentation
        st.subheader("📚 About")
        st.markdown("""
        **Blender Bot**
        
        An intelligent documentation assistant for Blender 3D software, built using Retrieval-Augmented Generation (RAG) architecture.
        
        - **Vector Database**: ChromaDB
        - **Embeddings**: HuggingFace transformers  
        - **LLM**: Groq/OpenAI models
        - **Framework**: LangChain
        
        *Built for Ready Tensor's Agentic AI Developer Certification Program*
        """)


def main():
    """Main Streamlit application."""
    # Page configuration
    st.set_page_config(
        page_title="Blender Bot",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Create sidebar
    create_sidebar()
    
    # Main interface
    st.title("🎨 Blender Bot")
    st.markdown("*Your intelligent guide to Blender 3D software*")
    
    # Initialize RAG system
    rag_system = setup_rag_system()
    
    if not rag_system:
        st.error("❌ Unable to initialize RAG system. Please check your configuration and API keys.")
        st.stop()
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        display_chat_history()
    
    # Chat input
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_input = st.chat_input("Ask me anything about Blender...")
        
        # Handle sample question selection from sidebar
        if 'user_input' in st.session_state:
            user_input = st.session_state.user_input
            del st.session_state.user_input
    
    with col2:
        if st.button("🎲", help="Get a random question", use_container_width=True):
            import random
            sample_questions = [
                "What are the different types of buttons in Blender's interface?",
                "How do I use the eyedropper tool to sample colors?", 
                "What does it mean to extrude a face in Blender?",
                "How can I quickly enter exact values into number fields?",
                "Can I customize interface buttons and menus?",
                "How do I access context menus in Blender?",
                "What are the small icons next to buttons?",
                "How do I navigate the 3D viewport?",
                "What's the difference between Edit and Object mode?",
                "How do I save my Blender project?"
            ]
            user_input = random.choice(sample_questions)
    
    # Process user input
    if user_input:
        # Add user message to history
        add_message("user", user_input)
        
        # Display user message immediately
        with chat_container:
            display_message("user", user_input, datetime.now())
        
        # Generate response
        try:
            with st.spinner("🤔 Thinking..."):
                start_time = time.time()
                response = rag_system.handle_query(query=user_input)
                end_time = time.time()
                
                response_time = end_time - start_time
                logging.info(f"Response generated in {response_time:.2f} seconds")
            
            # Add assistant response to history
            add_message("assistant", response)
            
            # Display assistant response
            with chat_container:
                display_message("assistant", response, datetime.now())
                st.caption(f"*Response time: {response_time:.2f}s*")
        
        except Exception as e:
            error_message = f"❌ Error generating response: {str(e)}"
            st.error(error_message)
            logging.error(f"Query processing error: {e}")
            add_message("assistant", error_message)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Built with ❤️ using Streamlit • "
        "Blender Documentation © Blender Foundation (CC-BY-SA 4.0)"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()