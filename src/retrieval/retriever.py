#!/usr/bin/env python3
"""
Semantic Search Retriever Module for Blender RAG Assistant

Provides high-level interface for semantic document retrieval using 
vector similarity search. Handles query processing, relevance scoring,
and result formatting for RAG applications.
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore


@dataclass
class RetrievalResult:
    """Container for retrieval results."""
    text: str
    metadata: Dict
    score: float
    

class SemanticRetriever:
    """High-level interface for semantic document retrieval."""
    
    def __init__(self, 
                 db_path: Path,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 collection_name: str = "blender_docs_demo"):
        """Initialize the semantic retriever.
        
        Args:
            db_path: Path to the vector database
            embedding_model: Name of the embedding model to use
            collection_name: Default collection name for searches
        """
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Initialize components
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        self.vector_store = VectorStore(db_path)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Validate collection exists
        if not self._validate_collection():
            self.logger.warning(f"Default collection '{collection_name}' not found")

    def _validate_collection(self) -> bool:
        """Validate that the default collection exists."""
        try:
            collections = self.vector_store.list_collections()
            return self.collection_name in collections
        except Exception:
            return False

    def retrieve(self, 
                query: str,
                k: int = 5,
                collection_name: Optional[str] = None,
                metadata_filter: Optional[Dict] = None,
                min_score: float = 0.0) -> List[RetrievalResult]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: Search query text
            k: Number of results to return
            collection_name: Collection to search (uses default if None)
            metadata_filter: Optional metadata filter for results
            min_score: Minimum relevance score threshold
            
        Returns:
            List of RetrievalResult objects ordered by relevance
        """
        if not query.strip():
            return []
        
        collection = collection_name or self.collection_name
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.encode_single(query)
            
            # Search vector database
            results = self.vector_store.query_collection(
                collection_name=collection,
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                where=metadata_filter,
                include=["documents", "metadatas", "distances"]
            )
            
            if not results or not results.get("documents"):
                return []
            
            # Convert results to RetrievalResult objects
            retrieval_results = []
            documents = results["documents"][0]  # First query results
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            
            for doc, metadata, distance in zip(documents, metadatas, distances):
                # Convert distance to similarity score (higher is better)
                score = 1.0 - distance
                
                if score >= min_score:
                    retrieval_results.append(RetrievalResult(
                        text=doc,
                        metadata=metadata,
                        score=score
                    ))
            
            self.logger.debug(f"Retrieved {len(retrieval_results)} results for query")
            return retrieval_results
            
        except Exception as e:
            self.logger.error(f"Retrieval failed: {e}")
            return []

    def retrieve_with_context(self,
                            query: str,
                            k: int = 5,
                            context_window: int = 1,
                            collection_name: Optional[str] = None) -> List[RetrievalResult]:
        """Retrieve documents with surrounding context chunks.
        
        Args:
            query: Search query text
            k: Number of primary results to return
            context_window: Number of adjacent chunks to include (before and after)
            collection_name: Collection to search (uses default if None)
            
        Returns:
            List of RetrievalResult objects with expanded context
        """
        # Get primary results
        primary_results = self.retrieve(query, k, collection_name)
        
        if not primary_results or context_window <= 0:
            return primary_results
        
        # TODO: Implement context expansion by finding adjacent chunks
        # This would require additional metadata about chunk sequences
        # For now, return primary results
        return primary_results

    def search_by_metadata(self,
                          metadata_filter: Dict,
                          k: int = 10,
                          collection_name: Optional[str] = None) -> List[RetrievalResult]:
        """Search documents by metadata criteria only.
        
        Args:
            metadata_filter: Metadata filter criteria
            k: Maximum number of results to return
            collection_name: Collection to search (uses default if None)
            
        Returns:
            List of RetrievalResult objects matching metadata criteria
        """
        collection = collection_name or self.collection_name
        
        try:
            # Create a dummy query embedding (won't be used for filtering)
            dummy_embedding = [0.0] * self.embedding_generator.get_embedding_dimension()
            
            # Search with metadata filter only
            results = self.vector_store.query_collection(
                collection_name=collection,
                query_embeddings=[dummy_embedding],
                n_results=k,
                where=metadata_filter,
                include=["documents", "metadatas"]
            )
            
            if not results or not results.get("documents"):
                return []
            
            # Convert to RetrievalResult objects (no meaningful scores)
            retrieval_results = []
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            
            for doc, metadata in zip(documents, metadatas):
                retrieval_results.append(RetrievalResult(
                    text=doc,
                    metadata=metadata,
                    score=0.0  # No semantic scoring for metadata-only search
                ))
            
            return retrieval_results
            
        except Exception as e:
            self.logger.error(f"Metadata search failed: {e}")
            return []

    def get_similar_documents(self,
                            document_text: str,
                            k: int = 5,
                            collection_name: Optional[str] = None,
                            exclude_exact_match: bool = True) -> List[RetrievalResult]:
        """Find documents similar to a given document.
        
        Args:
            document_text: Text of the document to find similar documents for
            k: Number of similar documents to return
            collection_name: Collection to search (uses default if None)
            exclude_exact_match: Whether to exclude exact text matches
            
        Returns:
            List of RetrievalResult objects ordered by similarity
        """
        results = self.retrieve(document_text, k + 1, collection_name)
        
        if exclude_exact_match:
            # Filter out exact matches based on text similarity
            filtered_results = []
            for result in results:
                if result.text.strip() != document_text.strip():
                    filtered_results.append(result)
                if len(filtered_results) >= k:
                    break
            return filtered_results
        
        return results[:k]

    def get_retriever_info(self) -> Dict:
        """Get information about the retriever configuration.
        
        Returns:
            Dictionary with retriever information
        """
        embedding_info = self.embedding_generator.get_model_info()
        db_info = self.vector_store.get_database_info()
        
        return {
            "default_collection": self.collection_name,
            "database_path": str(self.db_path),
            "embedding_model": embedding_info,
            "database_info": db_info,
            "available_collections": self.vector_store.list_collections()
        }

    def set_default_collection(self, collection_name: str) -> bool:
        """Set the default collection for searches.
        
        Args:
            collection_name: Name of the collection to set as default
            
        Returns:
            True if successful, False if collection doesn't exist
        """
        if collection_name in self.vector_store.list_collections():
            self.collection_name = collection_name
            self.logger.info(f"Set default collection to: {collection_name}")
            return True
        else:
            self.logger.warning(f"Collection '{collection_name}' not found")
            return False

    def health_check(self) -> Dict:
        """Perform a health check of the retrieval system.
        
        Returns:
            Dictionary with health check results
        """
        health = {
            "status": "healthy",
            "issues": []
        }
        
        # Check if default collection exists
        if not self._validate_collection():
            health["issues"].append(f"Default collection '{self.collection_name}' not found")
        
        # Check if database is accessible
        try:
            collections = self.vector_store.list_collections()
            if not collections:
                health["issues"].append("No collections found in database")
        except Exception as e:
            health["issues"].append(f"Database access error: {e}")
        
        # Check embedding model
        try:
            test_embedding = self.embedding_generator.encode_single("test")
            if len(test_embedding) == 0:
                health["issues"].append("Embedding model not working properly")
        except Exception as e:
            health["issues"].append(f"Embedding model error: {e}")
        
        if health["issues"]:
            health["status"] = "unhealthy"
        
        return health