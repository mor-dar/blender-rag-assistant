#!/usr/bin/env python3
"""
Vector Database Builder for Blender RAG Assistant

Handles ChromaDB initialization and population with document chunks.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

try:
    import chromadb
    from chromadb.config import Settings
except ImportError as e:
    raise ImportError(f"Missing required package: {e}")

from .document_processor import DocumentProcessor


class VectorDBBuilder:
    """Builds and manages ChromaDB vector database from processed documents."""
    
    def __init__(self, config: Dict, db_path: Path):
        """Initialize the vector database builder.
        
        Args:
            config: Configuration dictionary with database settings
            db_path: Path to the ChromaDB persistence directory
        """
        self.config = config
        self.db_path = db_path
        self.processor = DocumentProcessor(config)
        
        # Initialize ChromaDB
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.logger = logging.getLogger(__name__)

    def build_collection(self, collection_name: str, raw_dir: Path, collection_metadata: Dict = None) -> Dict:
        """Build vector database collection from documents.
        
        Args:
            collection_name: Name for the ChromaDB collection
            raw_dir: Directory containing raw HTML documentation
            collection_metadata: Optional metadata to attach to collection
            
        Returns:
            Build metadata dictionary
        """
        # Delete existing collection if it exists
        try:
            self.client.delete_collection(collection_name)
            self.logger.info(f"Deleted existing collection: {collection_name}")
        except Exception:
            pass  # Collection doesn't exist
        
        # Prepare collection metadata
        metadata = {"created_at": datetime.now().isoformat()}
        if collection_metadata:
            metadata.update(collection_metadata)
        
        # Create new collection
        collection = self.client.create_collection(
            name=collection_name,
            metadata=metadata
        )
        
        # Process documents
        chunks = self.processor.process_documents(raw_dir)
        
        if not chunks:
            self.logger.warning("No chunks to process!")
            return {"chunks_added": 0, "collection_name": collection_name}
        
        # Add chunks to database in batches
        batch_size = self.config["batch_size"]
        total_added = 0
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Prepare batch data
            texts = [chunk["text"] for chunk in batch]
            metadatas = [chunk["metadata"] for chunk in batch]
            ids = [chunk["metadata"]["chunk_id"] for chunk in batch]
            
            # Generate embeddings
            embeddings = self.processor.embedding_model.encode(texts).tolist()
            
            # Add to collection
            collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            total_added += len(batch)
            self.logger.info(f"Added batch {i//batch_size + 1}, total chunks: {total_added}")
        
        # Save build metadata
        build_metadata = {
            "collection_name": collection_name,
            "chunks_added": total_added,
            "config": self.config,
            "built_at": datetime.now().isoformat(),
            "source_dir": str(raw_dir),
            "collection_metadata": collection_metadata or {}
        }
        
        metadata_path = self.db_path / f"build_metadata_{collection_name}.json"
        with open(metadata_path, 'w') as f:
            json.dump(build_metadata, f, indent=2)
        
        return build_metadata

    def get_collection_info(self, collection_name: str) -> Dict:
        """Get information about a collection.
        
        Args:
            collection_name: Name of the collection to query
            
        Returns:
            Collection information dictionary
        """
        try:
            collection = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "count": collection.count(),
                "metadata": collection.metadata
            }
        except Exception as e:
            return {"error": str(e)}

    def list_collections(self) -> List[str]:
        """List all available collections.
        
        Returns:
            List of collection names
        """
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            self.logger.error(f"Failed to list collections: {e}")
            return []