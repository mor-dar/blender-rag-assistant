#!/bin/bash
set -e

# Docker entrypoint script for Blender Bot
# Supports custom commands for evaluation and production deployment

# Function to check API keys
check_api_keys() {
    if [ -z "$GROQ_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
        echo "ERROR: No API keys found in environment variables"
        echo "Please set at least one of the following environment variables:"
        echo "  GROQ_API_KEY    - For Groq/Llama models (recommended for evaluation)"
        echo "  OPENAI_API_KEY  - For OpenAI models (required for production)"
        echo ""
        echo "Example:"
        echo "  docker run -e GROQ_API_KEY=your_key_here <image> <command>"
        echo "  docker run -e OPENAI_API_KEY=your_key_here <image> <command>"
        echo ""
        exit 1
    fi
}

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
        echo "Setting up demo database and launching web interface..."
        check_api_keys
        setup_demo
        run_web
        ;;
    
    "evaluate-cli")
        echo "Setting up demo database and launching CLI interface..."
        check_api_keys
        setup_demo
        run_cli
        ;;
    
    "build-demo")
        echo "Building demo database..."
        check_api_keys
        setup_demo
        ;;
    
    "build-full")
        echo "Building full database..."
        check_api_keys
        setup_full
        ;;
    
    "run-web")
        echo "Running web interface (assuming database exists)..."
        check_api_keys
        run_web
        ;;
    
    "run-cli")
        echo "Running CLI interface (assuming database exists)..."
        check_api_keys
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
        echo "  OPENAI_API_KEY       Required for OpenAI models"
        echo "  GROQ_API_KEY         Required for Groq models"
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