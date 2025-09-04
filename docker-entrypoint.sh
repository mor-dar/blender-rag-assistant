#!/bin/bash
set -e

# Docker entrypoint script for Blender Bot
# Supports custom commands for evaluation and production deployment

# Function to download docs and setup demo database
setup_demo() {
    echo "Downloading Blender documentation..."
    python scripts/download_docs.py
    
    echo "Building demo database..."
    python scripts/setup_knowledge_base.py --tier demo
    
    echo "Demo setup complete!"
}

# Function to download docs and setup full database
setup_full() {
    echo "Downloading Blender documentation..."
    python scripts/download_docs.py
    
    echo "Building full database..."
    python scripts/setup_knowledge_base.py --tier full
    
    echo "Full setup complete!"
}

# Function to run web interface
run_web() {
    echo "Starting web interface..."
    python main.py --web
}

# Function to run CLI interface
run_cli() {
    echo "Starting CLI interface..."
    python main.py --cli
}

# Main command dispatcher
case "$1" in
    "evaluate-web")
        echo "Evaluation mode: Setting up demo database and launching web interface..."
        export RAG_MODE=evaluation
        setup_demo
        run_web
        ;;
    
    "evaluate-cli")
        echo "Evaluation mode: Setting up demo database and launching CLI interface..."
        export RAG_MODE=evaluation
        setup_demo
        run_cli
        ;;
    
    "build-demo")
        echo "Building demo database..."
        export RAG_MODE=evaluation
        setup_demo
        ;;
    
    "build-full")
        echo "Building full database..."
        export RAG_MODE=production
        setup_full
        ;;
    
    "run-web")
        echo "Running web interface (assuming database exists)..."
        run_web
        ;;
    
    "run-cli")
        echo "Running CLI interface (assuming database exists)..."
        run_cli
        ;;
    
    "help"|"--help"|"-h")
        echo "Blender Bot Docker Commands:"
        echo ""
        echo "Evaluation commands (demo database):"
        echo "  evaluate-web    Download data, build demo DB, run web interface"
        echo "  evaluate-cli    Download data, build demo DB, run CLI interface"
        echo ""
        echo "Setup commands:"
        echo "  build-demo      Download data and build demo database"
        echo "  build-full      Download data and build full database"
        echo ""
        echo "Runtime commands (assume DB exists):"
        echo "  run-web         Run web interface only"
        echo "  run-cli         Run CLI interface only"
        echo ""
        echo "Utility commands:"
        echo "  help            Show this help message"
        echo ""
        echo "Examples:"
        echo "  docker run -p 8501:8501 -v \$(pwd)/data:/app/data <image> evaluate-web"
        echo "  docker run -it -v \$(pwd)/data:/app/data <image> evaluate-cli"
        echo "  docker run -v \$(pwd)/data:/app/data <image> build-demo"
        echo "  docker run -p 8501:8501 -v \$(pwd)/data:/app/data <image> run-web"
        echo ""
        echo "Environment variables:"
        echo "  RAG_MODE=evaluation  Use demo dataset (default)"
        echo "  RAG_MODE=production  Use full dataset"
        echo "  OPENAI_API_KEY       Required for production mode"
        echo "  GROQ_API_KEY         Optional for Groq models"
        ;;
    
    *)
        # If no recognized command, pass through to the original command
        if [ "$1" = "python" ] || [ "$1" = "bash" ] || [ "$1" = "sh" ] || [ -x "$(command -v "$1")" ]; then
            echo "Running custom command: $*"
            exec "$@"
        else
            echo "Unknown command: $1"
            echo "Run 'help' to see available commands"
            exit 1
        fi
        ;;
esac