#!/usr/bin/env python3
"""
Vector Store Management Module for Blender RAG Assistant

Handles ChromaDB operations including collection management, document storage,
and persistence. Provides clean interface for vector database operations.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import chromadb
    from chromadb.config import Settings
except ImportError as e:
    raise ImportError(f"Missing required package: {e}")


class VectorStore:
    """Manages ChromaDB vector database operations."""
    
    def __init__(self, db_path: Path):
        """Initialize the vector store.
        
        Args:
            db_path: Path to the ChromaDB persistence directory
        """
        self.db_path = db_path
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initialized vector store at: {self.db_path}")

    def create_collection(self, name: str, metadata: Optional[Dict] = None, replace: bool = True) -> Any:
        """Create a new collection.
        
        Args:
            name: Collection name
            metadata: Optional metadata to attach to collection
            replace: If True, delete existing collection with same name
            
        Returns:
            ChromaDB collection object
        """
        if replace:
            try:
                self.client.delete_collection(name)
                self.logger.info(f"Deleted existing collection: {name}")
            except Exception:
                pass  # Collection doesn't exist
        
        # Prepare collection metadata
        collection_metadata = {"created_at": datetime.now().isoformat()}
        if metadata:
            collection_metadata.update(metadata)
        
        collection = self.client.create_collection(
            name=name,
            metadata=collection_metadata
        )
        
        self.logger.info(f"Created collection: {name}")
        return collection

    def get_collection(self, name: str) -> Optional[Any]:
        """Get an existing collection.
        
        Args:
            name: Collection name
            
        Returns:
            ChromaDB collection object or None if not found
        """
        try:
            return self.client.get_collection(name)
        except Exception as e:
            self.logger.warning(f"Collection '{name}' not found: {e}")
            return None

    def delete_collection(self, name: str) -> bool:
        """Delete a collection.
        
        Args:
            name: Collection name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete_collection(name)
            self.logger.info(f"Deleted collection: {name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete collection '{name}': {e}")
            return False

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

    def get_collection_info(self, name: str) -> Dict:
        """Get information about a collection.
        
        Args:
            name: Collection name
            
        Returns:
            Collection information dictionary
        """
        try:
            collection = self.client.get_collection(name)
            return {
                "name": name,
                "count": collection.count(),
                "metadata": collection.metadata
            }
        except Exception as e:
            return {"error": str(e)}

    def add_documents(self, 
                     collection_name: str,
                     documents: List[str],
                     embeddings: List[List[float]],
                     metadatas: List[Dict],
                     ids: List[str]) -> bool:
        """Add documents to a collection.
        
        Args:
            collection_name: Name of the collection
            documents: List of document texts
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            ids: List of document IDs
            
        Returns:
            True if successful, False otherwise
        """
        try:
            collection = self.get_collection(collection_name)
            if collection is None:
                self.logger.error(f"Collection '{collection_name}' not found")
                return False
            
            collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            self.logger.debug(f"Added {len(documents)} documents to collection '{collection_name}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add documents to collection '{collection_name}': {e}")
            return False

    def query_collection(self, 
                        collection_name: str,
                        query_embeddings: List[List[float]],
                        n_results: int = 5,
                        where: Optional[Dict] = None,
                        include: Optional[List[str]] = None) -> Optional[Dict]:
        """Query a collection for similar documents.
        
        Args:
            collection_name: Name of the collection to query
            query_embeddings: List of query embedding vectors
            n_results: Number of results to return per query
            where: Optional metadata filter
            include: Optional list of fields to include in results
            
        Returns:
            Query results dictionary or None if failed
        """
        try:
            collection = self.get_collection(collection_name)
            if collection is None:
                self.logger.error(f"Collection '{collection_name}' not found")
                return None
            
            # Set default include fields if not specified
            if include is None:
                include = ["documents", "metadatas", "distances"]
            
            results = collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                include=include
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to query collection '{collection_name}': {e}")
            return None

    def update_documents(self, 
                        collection_name: str,
                        ids: List[str],
                        documents: Optional[List[str]] = None,
                        embeddings: Optional[List[List[float]]] = None,
                        metadatas: Optional[List[Dict]] = None) -> bool:
        """Update existing documents in a collection.
        
        Args:
            collection_name: Name of the collection
            ids: List of document IDs to update
            documents: Optional new document texts
            embeddings: Optional new embedding vectors
            metadatas: Optional new metadata dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        try:
            collection = self.get_collection(collection_name)
            if collection is None:
                self.logger.error(f"Collection '{collection_name}' not found")
                return False
            
            collection.update(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            self.logger.debug(f"Updated {len(ids)} documents in collection '{collection_name}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update documents in collection '{collection_name}': {e}")
            return False

    def delete_documents(self, collection_name: str, ids: List[str]) -> bool:
        """Delete documents from a collection.
        
        Args:
            collection_name: Name of the collection
            ids: List of document IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            collection = self.get_collection(collection_name)
            if collection is None:
                self.logger.error(f"Collection '{collection_name}' not found")
                return False
            
            collection.delete(ids=ids)
            
            self.logger.debug(f"Deleted {len(ids)} documents from collection '{collection_name}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete documents from collection '{collection_name}': {e}")
            return False

    def save_build_metadata(self, collection_name: str, metadata: Dict) -> bool:
        """Save build metadata to a JSON file.
        
        Args:
            collection_name: Name of the collection
            metadata: Build metadata dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            metadata_path = self.db_path / f"build_metadata_{collection_name}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Saved build metadata for collection '{collection_name}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save build metadata for collection '{collection_name}': {e}")
            return False

    def load_build_metadata(self, collection_name: str) -> Optional[Dict]:
        """Load build metadata from a JSON file.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Build metadata dictionary or None if not found
        """
        try:
            metadata_path = self.db_path / f"build_metadata_{collection_name}.json"
            if not metadata_path.exists():
                return None
                
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load build metadata for collection '{collection_name}': {e}")
            return None

    def get_database_info(self) -> Dict:
        """Get overall database information.
        
        Returns:
            Database information dictionary
        """
        collections = self.list_collections()
        info = {
            "database_path": str(self.db_path),
            "total_collections": len(collections),
            "collections": {}
        }
        
        for collection_name in collections:
            collection_info = self.get_collection_info(collection_name)
            info["collections"][collection_name] = collection_info
        
        return info