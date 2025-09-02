#!/usr/bin/env python3
"""
Knowledge Base Setup Script

Processes downloaded Blender documentation and builds ChromaDB vector database.
Part of the Blender RAG Assistant knowledge base initialization.
Supports both demo and full dataset tiers.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from utils.config import (  # type: ignore[import-not-found]
        CHROMA_COLLECTION_NAME,
        CHROMA_PERSIST_DIRECTORY,
        get_vector_db_config_dict,
        initialize_logging,
        print_config_summary,
    )

    from data.processing import VectorDBBuilder  # type: ignore[import-not-found]
except ImportError as e:
    logging.error(f"Missing required package: {e}")
    logging.error("Install with: pip install -r requirements.txt")
    sys.exit(1)

# Tier configurations
def get_tier_configs(base_collection_name: str) -> Dict[str, Dict[str, str]]:
    return {
        "demo": {
            "collection_name": f"{base_collection_name}_demo",
            "description": "Demo dataset for evaluation and testing"
        },
        "full": {
            "collection_name": f"{base_collection_name}_full", 
            "description": "Complete Blender documentation dataset"
        }
    }


def main() -> None:
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
    
    # Initialize logging and configuration
    initialize_logging()
    
    # Get tier configurations based on base collection name
    tier_configs = get_tier_configs(CHROMA_COLLECTION_NAME)
    
    # Validate tier
    if args.tier not in tier_configs:
        logging.info(f"Invalid tier: {args.tier}")
        logging.info(f"Available tiers: {list(tier_configs.keys())}")
        sys.exit(1)
    
    logging.info("Using configuration:")
    print_config_summary()
    
    # Get vector database configuration
    config = get_vector_db_config_dict()
    
    # Load custom config if provided (for backward compatibility)
    if args.config and args.config.exists():
        with open(args.config) as f:
            custom_config: Dict[str, Any] = json.load(f)
            config.update(custom_config)
    
    # Validate inputs
    if not args.raw_dir.exists():
        logging.info(f"Raw directory not found: {args.raw_dir}")
        logging.info("Run download_docs.py first to download the documentation")
        sys.exit(1)
    
    # Get tier configuration
    tier_config = tier_configs[args.tier]
    collection_name: str = tier_config["collection_name"]
    collection_metadata: Dict[str, str] = {
        "tier": args.tier,
        "description": tier_config["description"]
    }
    
    # Build vector database (use configured paths)
    builder = VectorDBBuilder(config, CHROMA_PERSIST_DIRECTORY)
    
    try:
        metadata = builder.build_collection(collection_name, args.raw_dir, collection_metadata)
        logging.info(f"Successfully built vector database for {args.tier} tier")
        logging.info(f"Added {metadata['chunks_added']} chunks to collection '{metadata['collection_name']}'")
        logging.info(f"Database saved to: {args.db_path}")
        
    except KeyboardInterrupt:
        logging.info("\nBuild interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Build failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()