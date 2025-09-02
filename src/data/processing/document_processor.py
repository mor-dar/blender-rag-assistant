#!/usr/bin/env python3
"""
Document Processor for Blender RAG Assistant

Handles text extraction, cleaning, and chunking from HTML documentation.
"""

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

try:
    from bs4 import BeautifulSoup
    import tiktoken
except ImportError as e:
    raise ImportError(f"Missing required package: {e}")

from .text_cleaner import TextCleaner
from .semantic_chunker import SemanticChunker


class DocumentProcessor:
    """Processes HTML documents into chunks suitable for vector storage."""
    
    def __init__(self, config: Dict):
        """Initialize the document processor.
        
        Args:
            config: Configuration dictionary with chunk_size, etc.
        """
        self.config = config
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        self.text_cleaner = TextCleaner()
        self.semantic_chunker = SemanticChunker(config)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def extract_text_from_html(self, html_content: str, file_path: Path) -> Tuple[str, Dict]:
        """Extract clean text and metadata from HTML.
        
        Args:
            html_content: Raw HTML content
            file_path: Path to the source HTML file
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', '.navigation']):
                element.decompose()
            
            # Extract title
            title_elem = soup.find('title')
            title = title_elem.get_text().strip() if title_elem else file_path.stem
            title = self.text_cleaner.clean_text(title)
            
            # Extract main content
            content_elem = soup.find('main') or soup.find('article') or soup.find('body')
            if content_elem:
                text = content_elem.get_text(separator=' ', strip=True)
            else:
                text = soup.get_text(separator=' ', strip=True)
            
            # Clean up text using advanced text cleaner
            text = self.text_cleaner.clean_text(text)
            
            # Extract hierarchical structure from HTML
            hierarchy = self._extract_html_hierarchy(soup)
            
            # Extract metadata with ChromaDB-compatible serialization
            metadata = {
                "title": title,
                "section": self._extract_section_from_path(file_path),
                "subsection": hierarchy.get("subsection", ""),
                "heading_hierarchy_json": self._serialize_headings(hierarchy.get("headings", [])),
                "content_type": hierarchy.get("content_type", "general"),
                "url": self._path_to_url(file_path),
                "source_file": str(file_path),
                "extracted_at": datetime.now().isoformat()
            }
            
            return text, metadata
            
        except Exception as e:
            self.logger.error(f"Failed to extract text from {file_path}: {e}")
            return "", {}

    def _extract_section_from_path(self, file_path: Path) -> str:
        """Extract section name from file path.
        
        Args:
            file_path: Path to the source file
            
        Returns:
            Section name extracted from path
        """
        parts = file_path.parts
        if 'data' in parts and 'raw' in parts:
            try:
                raw_idx = parts.index('raw')
                if raw_idx + 1 < len(parts):
                    return parts[raw_idx + 1]
            except ValueError:
                pass
        return "unknown"

    def _serialize_headings(self, headings: List[Dict]) -> str:
        """Serialize headings list to JSON string for ChromaDB compatibility.
        
        Args:
            headings: List of heading dictionaries
            
        Returns:
            JSON string representation of headings
        """
        try:
            import json
            return json.dumps(headings)
        except Exception:
            return "[]"

    def _extract_html_hierarchy(self, soup) -> Dict[str, Any]:
        """Extract hierarchical structure and metadata from HTML document.
        
        Args:
            soup: BeautifulSoup parsed HTML
            
        Returns:
            Dictionary with hierarchy information
        """
        hierarchy: Dict[str, Any] = {
            "headings": [],
            "subsection": "",
            "content_type": "general"
        }
        
        try:
            # Extract heading hierarchy (h1, h2, h3, etc.)
            headings = []
            for level in range(1, 7):  # h1 through h6
                heading_tags = soup.find_all(f'h{level}')
                for tag in heading_tags:
                    heading_text = tag.get_text().strip()
                    if heading_text:
                        headings.append({
                            "level": level,
                            "text": heading_text,
                            "id": tag.get('id', ''),
                            "classes": tag.get('class', [])
                        })
            
            hierarchy["headings"] = headings
            
            # Determine subsection from first h2 or h3 heading
            if headings:
                # Look for first h2 or h3 as subsection
                for heading in headings:
                    if heading["level"] in [2, 3]:
                        hierarchy["subsection"] = heading["text"]
                        break
                
                # If no h2/h3, use first heading below h1
                if not hierarchy["subsection"]:
                    for heading in headings:
                        if heading["level"] > 1:
                            hierarchy["subsection"] = heading["text"]
                            break
            
            # Determine content type based on content analysis
            hierarchy["content_type"] = self._classify_content_type(soup, headings)
            
        except Exception as e:
            self.logger.warning(f"Failed to extract HTML hierarchy: {e}")
        
        return hierarchy
    
    def _classify_content_type(self, soup, headings) -> str:
        """Classify the type of content in the document.
        
        Args:
            soup: BeautifulSoup parsed HTML
            headings: List of extracted headings
            
        Returns:
            Content type classification
        """
        # Analyze heading text for content type indicators
        all_text = ' '.join([h["text"].lower() for h in headings])
        
        # Check for procedural content (tutorials, how-to)
        if any(keyword in all_text for keyword in [
            "how to", "tutorial", "step", "guide", "workflow", "process"
        ]):
            return "procedural"
        
        # Check for reference content (tools, properties, settings)
        if any(keyword in all_text for keyword in [
            "properties", "settings", "options", "parameters", "reference", "tool"
        ]):
            return "reference"
        
        # Check for conceptual content (theory, overview)
        if any(keyword in all_text for keyword in [
            "overview", "introduction", "concept", "theory", "principles"
        ]):
            return "conceptual"
        
        # Look for code/script content
        if soup.find('code') or soup.find('pre') or 'script' in all_text:
            return "code"
        
        # Look for lists (often procedural or reference)
        lists = soup.find_all(['ul', 'ol'])
        if len(lists) > 2:  # Multiple lists suggest structured content
            return "structured"
        
        return "general"

    def _path_to_url(self, file_path: Path) -> str:
        """Convert local file path to original URL.
        
        Args:
            file_path: Local file path
            
        Returns:
            Reconstructed URL for the documentation page
        """
        # This is a simplified conversion - in practice you'd want to map back to original URLs
        return f"https://docs.blender.org/manual/en/latest/{file_path.name}"

    def chunk_text(self, text: str, metadata: Dict, use_semantic: bool = True) -> List[Dict]:
        """Split text into chunks using either semantic or legacy strategy.
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk
            use_semantic: If True, use semantic chunking; otherwise use legacy token-based
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text.strip():
            return []
        
        if use_semantic:
            return self.semantic_chunker.chunk_text_semantically(text, metadata)
        else:
            return self._chunk_text_legacy(text, metadata)
    
    def _chunk_text_legacy(self, text: str, metadata: Dict) -> List[Dict]:
        """Legacy token-based chunking (for backward compatibility).
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text.strip():
            return []
        
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        chunk_size = self.config["chunk_size"]
        overlap = self.config["chunk_overlap"]
        
        start = 0
        chunk_id = 0
        
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Create chunk metadata
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_id": f"{hashlib.md5(metadata['source_file'].encode()).hexdigest()}_{chunk_id}",
                "token_count": len(chunk_tokens),
                "chunk_index": chunk_id,
                "start_token": start,
                "end_token": end,
                "chunking_strategy": "legacy"
            })
            
            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })
            
            # Move to next chunk with overlap
            start += chunk_size - overlap
            chunk_id += 1
        
        return chunks

    def process_documents(self, raw_dir: Path) -> List[Dict]:
        """Process all HTML documents in the raw directory.
        
        Args:
            raw_dir: Directory containing HTML files
            
        Returns:
            List of processed document chunks
        """
        html_files = list(raw_dir.rglob("*.html"))
        all_chunks = []
        
        self.logger.info(f"Processing {len(html_files)} HTML files")
        
        for i, html_file in enumerate(html_files, 1):
            try:
                with open(html_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                text, metadata = self.extract_text_from_html(html_content, html_file)
                
                if text:
                    # Use semantic chunking if enabled in config
                    use_semantic = self.config.get("use_semantic_chunking", False)
                    chunks = self.chunk_text(text, metadata, use_semantic=use_semantic)
                    all_chunks.extend(chunks)
                    
                    if i % 50 == 0:
                        self.logger.info(f"Processed {i}/{len(html_files)} files, {len(all_chunks)} chunks so far")
                
            except Exception as e:
                self.logger.error(f"Failed to process {html_file}: {e}")
        
        self.logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks