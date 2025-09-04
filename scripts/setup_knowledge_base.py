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
from typing import Any, Dict, List

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

# Demo mode specific documents (Interface Controls - Buttons section)
DEMO_DOCUMENTS = [
    "interface/controls/buttons/buttons.html",
    "interface/controls/buttons/decorators.html",
    "interface/controls/buttons/eyedropper.html", 
    "interface/controls/buttons/fields.html",
    "interface/controls/buttons/menus.html"
]

# Tier configurations
def get_tier_configs(base_collection_name: str) -> Dict[str, Dict[str, str]]:
    return {
        "demo": {
            "collection_name": base_collection_name,
            "description": "Demo dataset for evaluation and testing"
        },
        "full": {
            "collection_name": base_collection_name, 
            "description": "Complete Blender documentation dataset"
        }
    }

def find_demo_documents(raw_dir: Path) -> List[Path]:
    """Find demo documents in the raw directory structure."""
    demo_files = []
    
    # Look for HTML manual directories (new flattened structure)
    html_dirs = list(raw_dir.glob("**/blender_manual_html"))
    if not html_dirs:
        # Fallback to old nested structure or direct structure
        html_dirs = list(raw_dir.glob("**/blender_manual_*html*"))
        if not html_dirs:
            html_dirs = [raw_dir]
    
    for html_dir in html_dirs:
        for doc_path in DEMO_DOCUMENTS:
            full_path = html_dir / doc_path
            if full_path.exists():
                demo_files.append(full_path)
                logging.info(f"Found demo document: {full_path}")
            else:
                # Try old nested structure for backward compatibility
                nested_path = html_dir / "blender_manual_v450_en.html" / doc_path
                if nested_path.exists():
                    demo_files.append(nested_path)
                    logging.info(f"Found demo document: {nested_path}")
                else:
                    logging.warning(f"Demo document not found: {doc_path}")
    
    return demo_files

def create_filtered_raw_dir(raw_dir: Path, demo_files: List[Path], temp_dir: Path) -> Path:
    """Create a temporary directory with only demo files maintaining structure."""
    import shutil
    
    # Clear and create temp directory
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    
    for demo_file in demo_files:
        # Calculate relative path from original raw_dir
        try:
            rel_path = demo_file.relative_to(raw_dir)
        except ValueError:
            # Handle nested structure case
            rel_path = Path(*demo_file.parts[-len(Path(DEMO_DOCUMENTS[0]).parts):])
        
        dest_file = temp_dir / rel_path
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(demo_file, dest_file)
        logging.info(f"Copied {demo_file} -> {dest_file}")
    
    return temp_dir


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
    
    # Prepare raw directory based on tier
    if args.tier == "demo":
        logging.info("Demo mode: filtering to specific documents")
        demo_files = find_demo_documents(args.raw_dir)
        if not demo_files:
            logging.error("No demo documents found. Check if HTML manual is properly downloaded.")
            sys.exit(1)
        
        # Create filtered temporary directory
        temp_dir = args.db_path.parent / "temp_demo_docs"
        filtered_raw_dir = create_filtered_raw_dir(args.raw_dir, demo_files, temp_dir)
        logging.info(f"Created filtered directory with {len(demo_files)} demo documents")
    else:
        filtered_raw_dir = args.raw_dir
        logging.info("Full mode: processing all documents")

    # Build vector database (use configured paths)
    builder = VectorDBBuilder(config, CHROMA_PERSIST_DIRECTORY)
    
    try:
        metadata = builder.build_collection(collection_name, filtered_raw_dir, collection_metadata)
        logging.info(f"Successfully built vector database for {args.tier} tier")
        logging.info(f"Added {metadata['chunks_added']} chunks to collection '{metadata['collection_name']}'")
        logging.info(f"Database saved to: {args.db_path}")
        
        # Clean up temp directory if created
        if args.tier == "demo" and temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)
            logging.info("Cleaned up temporary demo directory")
        
    except KeyboardInterrupt:
        logging.info("\nBuild interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Build failed: {e}")
        # Clean up temp directory if created
        if args.tier == "demo":
            temp_dir = args.db_path.parent / "temp_demo_docs"
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)
        sys.exit(1)

if __name__ == "__main__":
    main()