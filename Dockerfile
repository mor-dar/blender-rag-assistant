# Dockerfile for Blender RAG Assistant
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/vector_db outputs logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV RAG_MODE=evaluation

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port for optional web interface
EXPOSE 8501

# Default command
CMD ["python", "main.py"]

# Alternative commands:
# To run setup: docker run --rm -v $(pwd)/data:/app/data <image> python scripts/setup_knowledge_base.py
# To run tests: docker run --rm <image> python -m pytest
# To run with Streamlit: docker run --rm -p 8501:8501 <image> streamlit run src/interface/web.py