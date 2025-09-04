#!/usr/bin/env python3
"""
Blender Bot - Main Entry Point

This script provides unified access to both CLI and web interfaces.
Run with --web flag for Streamlit interface, or default CLI interface.
"""

import argparse
import logging
import sys
from pathlib import Path
import dotenv

# Load environment variables first, before any other imports
dotenv.load_dotenv()

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from utils.config import initialize_logging
from rag.rag import BlenderAssistantRAG


def run_cli():
    """Run the Command Line Interface."""
    logging.info("\nBlender Bot (CLI Mode)")
    logging.info("=" * 50)
    logging.info("Ask me anything about Blender 3D software!")
    logging.info("Type 'quit', 'exit', or 'q' to stop.\n")
    logging.info("CLI mode started")
    
    try:
        rag = BlenderAssistantRAG()
        logging.info("RAG system initialized successfully!\n")
    except Exception as e:
        logging.error(f"Failed to initialize RAG system: {e}")
        return 1
    
    while True:
        try:
            # Get user input
            query = input("Your question: ").strip()
            
            # Handle quit commands
            if query.lower() in ['quit', 'exit', 'q', '']:
                logging.info("\nThanks for using Blender Bot!")
                break
            
            logging.info("🔍 Searching knowledge base...")
            logging.info(f"User query: {query}")
            
            # Generate response
            response = rag.handle_query(query=query)
            
            # Display response
            logging.info(f"\nAnswer:")
            logging.info("-" * 40)
            logging.info(response)
            logging.info("-" * 40)
            logging.info("")
            
            logging.info(f"RAG response: {response}")
            
        except KeyboardInterrupt:
            logging.info("\n\nThanks for using Blender Bot!")
            break
        except Exception as e:
            logging.error(f"Error: {e}")
    
    return 0


def run_web():
    """Run the Streamlit Web Interface."""
    try:
        import streamlit.web.cli as stcli
        import streamlit as st
        
        # Path to the streamlit app
        app_path = src_path / "interface" / "streamlit_app.py"
        
        if not app_path.exists():
            logging.error(f"Streamlit app not found at: {app_path}")
            return 1
        
        logging.info("Starting Streamlit web interface...")
        logging.info("Open your browser to: http://localhost:8501")
        
        # Run streamlit app
        sys.argv = ["streamlit", "run", str(app_path), "--server.port=8501"]
        stcli.main()
        
    except ImportError:
        logging.error("Streamlit not installed. Run: pip install streamlit")
        return 1
    except Exception as e:
        logging.error(f"Error starting web interface: {e}")
        return 1
    
    return 0


def main():
    """Main entry point with interface selection."""
    parser = argparse.ArgumentParser(
        description="Blender Bot - Intelligent Blender Documentation Guide",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py              # Run CLI interface (default)
  python main.py --cli        # Run CLI interface explicitly  
  python main.py --web        # Run Streamlit web interface
  python main.py --help       # Show this help message

For more information, visit: https://github.com/your-repo/blender-rag-assistant
        """
    )
    
    parser.add_argument(
        "--web", 
        action="store_true", 
        help="Launch Streamlit web interface"
    )
    
    parser.add_argument(
        "--cli", 
        action="store_true", 
        help="Launch CLI interface (default)"
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version="Blender Bot v1.0.0"
    )
    
    args = parser.parse_args()
    
    # Initialize logging (environment already loaded at top)
    initialize_logging()
    logging.info("Blender Bot started")
    
    # Determine interface
    if args.web:
        interface = "web"
    elif args.cli:
        interface = "cli"  
    else:
        # Default to CLI if no interface specified
        interface = "cli"
    
    # Run selected interface
    try:
        if interface == "web":
            return run_web()
        else:
            return run_cli()
            
    except KeyboardInterrupt:
        logging.info("\nGoodbye!")
        return 0
    except Exception as e:
        logging.error(f"Application error: {e}")
        return 1
    finally:
        logging.info("Blender Bot stopped")


if __name__ == "__main__":
    sys.exit(main())