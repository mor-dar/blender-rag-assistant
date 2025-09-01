#!/usr/bin/env python3
"""
Knowledge Base Setup Script

Processes downloaded Blender documentation and builds ChromaDB vector database.
Part of the Blender RAG Assistant knowledge base initialization.
Supports both demo and full dataset tiers.
"""

import argparse
import json
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from data.processing import VectorDBBuilder  # type: ignore[import-untyped]
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install -r requirements.txt")
    sys.exit(1)

# Configuration
VECTOR_DB_CONFIG = {
    "embedding_model": "all-MiniLM-L6-v2",  # Fast, good quality
    "chunk_size": 512,  # tokens
    "chunk_overlap": 50,  # tokens
    "metadata_fields": ["title", "section", "subsection", "heading_hierarchy_json", "content_type", "url", "chunk_id", "token_count"],
    "batch_size": 100  # For processing chunks in batches
}

# Tier configurations
TIER_CONFIGS = {
    "demo": {
        "collection_name": "blender_docs_demo",
        "description": "Demo dataset for evaluation and testing"
    },
    "full": {
        "collection_name": "blender_docs_full", 
        "description": "Complete Blender documentation dataset"
    }
}


def main():
    parser = argparse.ArgumentParser(description='Setup knowledge base from downloaded Blender documentation')
    parser.add_argument('--tier', choices=['demo', 'full'], required=True,
                        help='Which tier to build database for')
    parser.add_argument('--raw-dir', type=Path,
                        default=Path(__file__).parent.parent / 'data' / 'raw',
                        help='Directory containing downloaded HTML files')
    parser.add_argument('--db-path', type=Path,
                        default=Path(__file__).parent.parent / 'data' / 'vector_db',
                        help='Path for vector database')
    parser.add_argument('--config', type=Path,
                        help='Custom configuration file (JSON)')
    
    args = parser.parse_args()
    
    # Validate tier
    if args.tier not in TIER_CONFIGS:
        print(f"Invalid tier: {args.tier}")
        print(f"Available tiers: {list(TIER_CONFIGS.keys())}")
        sys.exit(1)
    
    # Load custom config if provided
    config = VECTOR_DB_CONFIG.copy()
    if args.config and args.config.exists():
        with open(args.config) as f:
            custom_config = json.load(f)
            config.update(custom_config)
    
    # Validate inputs
    if not args.raw_dir.exists():
        print(f"Raw directory not found: {args.raw_dir}")
        print("Run download_docs.py first to download the documentation")
        sys.exit(1)
    
    # Get tier configuration
    tier_config = TIER_CONFIGS[args.tier]
    collection_name = tier_config["collection_name"]
    collection_metadata = {
        "tier": args.tier,
        "description": tier_config["description"]
    }
    
    # Build vector database
    builder = VectorDBBuilder(config, args.db_path)
    
    try:
        metadata = builder.build_collection(collection_name, args.raw_dir, collection_metadata)
        print(f"Successfully built vector database for {args.tier} tier")
        print(f"Added {metadata['chunks_added']} chunks to collection '{metadata['collection_name']}'")
        print(f"Database saved to: {args.db_path}")
        
    except KeyboardInterrupt:
        print("\nBuild interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Build failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()