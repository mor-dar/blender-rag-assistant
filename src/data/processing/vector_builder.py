#!/usr/bin/env python3
"""
Vector Database Builder for Blender RAG Assistant

Orchestrates document processing and vector database creation.
Coordinates between document processing, embedding generation, and vector storage.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .document_processor import DocumentProcessor

# Import retrieval components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from retrieval import EmbeddingGenerator, VectorStore


class VectorDBBuilder:
    """Orchestrates vector database building from document processing to storage."""
    
    def __init__(self, config: Dict, db_path: Path):
        """Initialize the vector database builder.
        
        Args:
            config: Configuration dictionary with database settings
            db_path: Path to the ChromaDB persistence directory
        """
        self.config = config
        self.db_path = db_path
        
        # Initialize components
        self.processor = DocumentProcessor(config)
        self.embedding_generator = EmbeddingGenerator(config["embedding_model"])
        self.vector_store = VectorStore(db_path)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def build_collection(self, collection_name: str, raw_dir: Path, collection_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build vector database collection from documents.
        
        Args:
            collection_name: Name for the ChromaDB collection
            raw_dir: Directory containing raw HTML documentation
            collection_metadata: Optional metadata to attach to collection
            
        Returns:
            Build metadata dictionary
        """
        # Create collection using vector store
        collection = self.vector_store.create_collection(
            name=collection_name,
            metadata=collection_metadata,
            replace=True
        )
        
        # Process documents
        chunks = self.processor.process_documents(raw_dir)
        
        if not chunks:
            self.logger.warning("No chunks to process!")
            build_metadata = {
                "collection_name": collection_name,
                "chunks_added": 0,
                "config": self.config,
                "built_at": datetime.now().isoformat(),
                "source_dir": str(raw_dir),
                "collection_metadata": collection_metadata or {}
            }
            self.vector_store.save_build_metadata(collection_name, build_metadata)
            return build_metadata
        
        # Add chunks to database in batches
        batch_size = self.config["batch_size"]
        total_added = 0
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Prepare batch data
            texts = [chunk["text"] for chunk in batch]
            metadatas = [chunk["metadata"] for chunk in batch]
            ids = [chunk["metadata"]["chunk_id"] for chunk in batch]
            
            # Generate embeddings using embedding generator
            embeddings = self.embedding_generator.encode_batch(texts)
            
            # Add to vector store
            success = self.vector_store.add_documents(
                collection_name=collection_name,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            if success:
                total_added += len(batch)
                self.logger.info(f"Added batch {i//batch_size + 1}, total chunks: {total_added}")
            else:
                self.logger.error(f"Failed to add batch {i//batch_size + 1}")
        
        # Build and save metadata
        build_metadata = {
            "collection_name": collection_name,
            "chunks_added": total_added,
            "config": self.config,
            "built_at": datetime.now().isoformat(),
            "source_dir": str(raw_dir),
            "collection_metadata": collection_metadata or {}
        }
        
        self.vector_store.save_build_metadata(collection_name, build_metadata)
        return build_metadata

    def get_collection_info(self, collection_name: str) -> Dict:
        """Get information about a collection.
        
        Args:
            collection_name: Name of the collection to query
            
        Returns:
            Collection information dictionary
        """
        return self.vector_store.get_collection_info(collection_name)

    def list_collections(self) -> List[str]:
        """List all available collections.
        
        Returns:
            List of collection names
        """
        return self.vector_store.list_collections()