#!/usr/bin/env python3
"""
Build Vector Collections with Different Embedding Models

Creates separate ChromaDB collections for different embedding models to enable
fair comparison in evaluation framework.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.config import (
    initialize_logging, 
    CHROMA_PERSIST_DIRECTORY, 
    CHROMA_COLLECTION_NAME,
    get_vector_db_config_dict
)
from data.processing import VectorDBBuilder

# Available embedding models to build collections for
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": {
        "dimensions": 384,
        "description": "Fast and lightweight (current baseline)"
    },
    "all-mpnet-base-v2": {
        "dimensions": 768, 
        "description": "Higher accuracy, larger model"
    },
    "multi-qa-MiniLM-L6-cos-v1": {
        "dimensions": 384,
        "description": "Question-answering optimized"
    },
    "all-MiniLM-L12-v2": {
        "dimensions": 384,
        "description": "Larger MiniLM variant"
    }
}

def build_collection_for_model(model_name: str, raw_dir: Path, db_path: Path, 
                             tier: str = "full") -> bool:
    """Build a vector collection using a specific embedding model.
    
    Args:
        model_name: Name of embedding model to use
        raw_dir: Directory containing raw documentation
        db_path: Path to vector database directory
        tier: Dataset tier (demo or full)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logging.info(f"Building collection for embedding model: {model_name}")
        
        # Create configuration with specific embedding model
        config = get_vector_db_config_dict()
        config["embedding_model"] = model_name
        
        # Create unique collection name for this model
        collection_name = f"{CHROMA_COLLECTION_NAME}_{tier}_{model_name.replace('-', '_')}"
        
        # Collection metadata
        collection_metadata = {
            "tier": tier,
            "embedding_model": model_name,
            "embedding_dimensions": EMBEDDING_MODELS[model_name]["dimensions"],
            "description": f"{tier.title()} dataset with {model_name} embeddings"
        }
        
        # Build the collection
        builder = VectorDBBuilder(config, db_path)
        metadata = builder.build_collection(collection_name, raw_dir, collection_metadata)
        
        logging.info(f"Successfully built collection '{collection_name}'")
        logging.info(f"  Chunks added: {metadata['chunks_added']}")
        logging.info(f"  Embedding model: {model_name}")
        logging.info(f"  Dimensions: {EMBEDDING_MODELS[model_name]['dimensions']}")
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to build collection for {model_name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Build collections with different embedding models')
    parser.add_argument('--models', nargs='+', 
                        choices=list(EMBEDDING_MODELS.keys()),
                        default=list(EMBEDDING_MODELS.keys()),
                        help='Embedding models to build collections for')
    parser.add_argument('--tier', choices=['demo', 'full'], default='full',
                        help='Dataset tier to build collections for')
    parser.add_argument('--raw-dir', type=Path,
                        default=Path(__file__).parent.parent / 'data' / 'raw',
                        help='Directory containing raw documentation')
    parser.add_argument('--db-path', type=Path,
                        default=CHROMA_PERSIST_DIRECTORY,
                        help='Vector database directory')
    parser.add_argument('--force-rebuild', action='store_true',
                        help='Rebuild collections even if they already exist')
    parser.add_argument('--list-existing', action='store_true',
                        help='List existing collections and exit')
    
    args = parser.parse_args()
    
    # Initialize logging
    initialize_logging()
    
    # List existing collections if requested
    if args.list_existing:
        from retrieval.vector_store import VectorStore
        vector_store = VectorStore(args.db_path)
        collections = vector_store.list_collections()
        
        print("\nExisting Collections:")
        print("-" * 50)
        for collection_name in sorted(collections):
            info = vector_store.get_collection_info(collection_name)
            count = info.get('count', 'Unknown')
            metadata = info.get('metadata', {})
            embedding_model = metadata.get('embedding_model', 'Unknown')
            dimensions = metadata.get('embedding_dimensions', 'Unknown')
            
            print(f"  {collection_name}")
            print(f"    Documents: {count}")
            print(f"    Model: {embedding_model}")
            print(f"    Dimensions: {dimensions}")
            print()
        
        return 0
    
    # Validate raw directory
    if not args.raw_dir.exists():
        logging.error(f"Raw directory not found: {args.raw_dir}")
        logging.error("Run scripts/download_docs.py first to download documentation")
        return 1
    
    # Check which collections already exist
    from retrieval.vector_store import VectorStore
    vector_store = VectorStore(args.db_path)
    existing_collections = set(vector_store.list_collections())
    
    # Build collections for selected models
    success_count = 0
    total_count = len(args.models)
    
    logging.info(f"Building collections for {total_count} models with {args.tier} dataset")
    
    for model_name in args.models:
        collection_name = f"{CHROMA_COLLECTION_NAME}_{args.tier}_{model_name.replace('-', '_')}"
        
        # Skip if collection exists and not forcing rebuild
        if collection_name in existing_collections and not args.force_rebuild:
            logging.info(f"Collection '{collection_name}' already exists (use --force-rebuild to recreate)")
            success_count += 1
            continue
        
        # Build the collection
        if build_collection_for_model(model_name, args.raw_dir, args.db_path, args.tier):
            success_count += 1
        else:
            logging.error(f"Failed to build collection for {model_name}")
    
    # Summary
    logging.info(f"Collection building complete: {success_count}/{total_count} successful")
    
    if success_count == total_count:
        logging.info("All collections built successfully!")
        logging.info("You can now run embedding evaluation with:")
        logging.info(f"  python scripts/evaluate_embeddings.py --models {' '.join(args.models)}")
        return 0
    else:
        logging.error(f"Failed to build {total_count - success_count} collections")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)