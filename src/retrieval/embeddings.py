#!/usr/bin/env python3
"""
Embedding Generation Module for Blender RAG Assistant

Handles text embedding generation using HuggingFace sentence-transformers.
Provides caching and batch processing capabilities for efficient operation.
"""

import logging
from typing import List, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(f"Missing required package: {e}")


class EmbeddingGenerator:
    """Generates embeddings for text using sentence-transformers models."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        logging.info(f"Initialized embedding generator with model: {model_name}")

    def encode_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding vector as numpy array
        """
        if not text.strip():
            raise ValueError("Cannot encode empty text")
            
        embedding = self.model.encode(text)
        return embedding  # type: ignore

    def encode_batch(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to encode
            batch_size: Optional batch size for processing (uses model default if None)
            
        Returns:
            List of embedding vectors as lists of floats
        """
        if not texts:
            return []
            
        # Filter out empty texts and keep track of indices
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text.strip():
                valid_texts.append(text)
                valid_indices.append(i)
        
        if not valid_texts:
            raise ValueError("Cannot encode batch with all empty texts")
        
        # Generate embeddings for valid texts
        if batch_size is not None:
            embeddings = self.model.encode(
                valid_texts, 
                batch_size=batch_size,
                show_progress_bar=len(valid_texts) > 100
            )
        else:
            embeddings = self.model.encode(
                valid_texts,
                show_progress_bar=len(valid_texts) > 100
            )
        
        # Convert to list of lists for ChromaDB compatibility
        embeddings_list = embeddings.tolist()
        
        # Reconstruct full list with None for empty texts
        result = [None] * len(texts)
        for i, embedding in zip(valid_indices, embeddings_list):
            result[i] = embedding
            
        # Filter out None values and return valid embeddings
        return [emb for emb in result if emb is not None]  # type: ignore

    def get_embedding_dimension(self) -> int:
        """Get the dimensionality of embeddings produced by this model.
        
        Returns:
            Embedding dimension size
        """
        return self.model.get_sentence_embedding_dimension()  # type: ignore

    def get_model_info(self) -> dict:
        """Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "max_seq_length": self.model.max_seq_length,
            "embedding_dimension": self.get_embedding_dimension(),
            "device": str(self.model.device)
        }