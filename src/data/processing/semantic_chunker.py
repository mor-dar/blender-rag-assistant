#!/usr/bin/env python3
"""
Semantic Chunker for Technical Documentation

Implements intelligent chunking strategies optimized for technical documentation
like Blender's manual, with heading preservation, adaptive sizing, and 
content-aware splitting.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

try:
    import tiktoken
    import spacy
    from spacy.lang.en import English
except ImportError as e:
    # Graceful degradation - will use simpler sentence splitting
    tiktoken = None  # type: ignore
    spacy = None  # type: ignore
    English = None  # type: ignore


class ContentType(Enum):
    """Content type classifications for adaptive chunking."""
    PROCEDURAL = "procedural"      # Step-by-step instructions
    REFERENCE = "reference"        # Tool descriptions, parameters
    CONCEPTUAL = "conceptual"      # Theory, overviews
    CODE = "code"                  # Scripts, expressions
    STRUCTURED = "structured"      # Lists, tables
    GENERAL = "general"           # Mixed content


@dataclass
class ChunkConfig:
    """Configuration for content-type-specific chunking."""
    max_tokens: int
    min_tokens: int
    overlap_tokens: int
    preserve_headings: bool
    preserve_sentences: bool
    preserve_lists: bool
    preserve_code_blocks: bool


class SemanticChunker:
    """Advanced chunking for technical documentation with semantic awareness."""
    
    def __init__(self, base_config: Dict):
        """Initialize the semantic chunker.
        
        Args:
            base_config: Base configuration with default chunk_size, etc.
        """
        self.base_config = base_config
        
        # Initialize tokenizer
        if tiktoken:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        else:
            # Fallback to simple word-based approximation
            self.tokenizer = None  # type: ignore
        
        # Initialize NLP pipeline for sentence detection
        self.nlp = self._init_nlp_pipeline()
        
        # Configure chunking strategies by content type
        self.chunk_configs = {
            ContentType.PROCEDURAL: ChunkConfig(
                max_tokens=768,         # Longer for step sequences
                min_tokens=256,         # Allow shorter steps
                overlap_tokens=100,     # More overlap to preserve context
                preserve_headings=True,
                preserve_sentences=True,
                preserve_lists=True,
                preserve_code_blocks=True
            ),
            ContentType.REFERENCE: ChunkConfig(
                max_tokens=512,         # Standard size for properties/tools
                min_tokens=128,         # Can be quite short
                overlap_tokens=50,      # Less overlap needed
                preserve_headings=True,
                preserve_sentences=True,
                preserve_lists=True,
                preserve_code_blocks=True
            ),
            ContentType.CONCEPTUAL: ChunkConfig(
                max_tokens=1024,        # Longer for concepts
                min_tokens=256,         # Need substantial content
                overlap_tokens=100,     # Good overlap for context
                preserve_headings=True,
                preserve_sentences=True,
                preserve_lists=False,   # Less critical
                preserve_code_blocks=True
            ),
            ContentType.CODE: ChunkConfig(
                max_tokens=512,         # Code can be dense
                min_tokens=64,          # Small code snippets OK
                overlap_tokens=25,      # Minimal overlap
                preserve_headings=True,
                preserve_sentences=False,  # Code has different structure
                preserve_lists=False,
                preserve_code_blocks=True  # Critical!
            ),
            ContentType.STRUCTURED: ChunkConfig(
                max_tokens=640,         # Medium size for lists/tables
                min_tokens=128,
                overlap_tokens=50,
                preserve_headings=True,
                preserve_sentences=True,
                preserve_lists=True,    # Critical!
                preserve_code_blocks=True
            ),
            ContentType.GENERAL: ChunkConfig(
                max_tokens=512,         # Default from base config
                min_tokens=128,
                overlap_tokens=50,
                preserve_headings=True,
                preserve_sentences=True,
                preserve_lists=True,
                preserve_code_blocks=True
            )
        }
        
        # Compile regex patterns for structure detection
        self.heading_pattern = re.compile(r'^\s*#+\s+(.+)$', re.MULTILINE)
        self.list_item_pattern = re.compile(r'^[\s]*[-*+]\s+|^\s*\d+\.\s+', re.MULTILINE)
        self.code_block_pattern = re.compile(r'```[\s\S]*?```|`[^`]+`')
        self.sentence_end_pattern = re.compile(r'[.!?]+[\s]+')
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def _init_nlp_pipeline(self):
        """Initialize NLP pipeline for sentence detection."""
        try:
            if spacy and English:
                # Use lightweight English model for sentence detection
                nlp = English()
                nlp.add_pipe("sentencizer")
                return nlp
        except Exception as e:
            self.logger.warning(f"Could not initialize spaCy pipeline: {e}")
        
        return None
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough approximation: 1 token ≈ 0.75 words
            return int(len(text.split()) / 0.75)
    
    def _classify_content_type(self, text: str, metadata: Dict) -> ContentType:
        """Classify content type for adaptive chunking."""
        # Use existing classification if available and valid
        if 'content_type' in metadata and metadata['content_type'] != 'general':
            content_type_str = metadata['content_type'].lower()
            for ct in ContentType:
                if ct.value == content_type_str:
                    return ct
        
        # Fallback classification based on text analysis
        text_lower = text.lower()
        
        # Check for code indicators (more comprehensive)
        if (self.code_block_pattern.search(text) or 
            any(keyword in text_lower for keyword in [
                'import ', 'def ', 'class ', 'function(', 'bpy.', 'print(', 
                'return ', 'if __name__', 'import bpy'
            ])):
            return ContentType.CODE
        
        # Check for structured content (lists) - check this before reference
        list_matches = len(self.list_item_pattern.findall(text))
        if list_matches > 2:  # Multiple list items (lowered threshold)
            return ContentType.STRUCTURED
        
        # Check for procedural indicators  
        if any(keyword in text_lower for keyword in [
            'step 1', 'step 2', 'first,', 'next,', 'then,', 'finally,', 
            'how to', 'tutorial', 'workflow', 'process'
        ]):
            return ContentType.PROCEDURAL
        
        # Check for reference indicators
        if any(keyword in text_lower for keyword in [
            'properties', 'settings', 'options', 'parameters', 'tool', 'menu',
            'reference', 'ctrl+', 'alt+', 'shift+'
        ]):
            return ContentType.REFERENCE
        
        # Check for conceptual content
        if any(keyword in text_lower for keyword in [
            'overview', 'introduction', 'concept', 'theory', 'principles',
            'understanding', 'basics'
        ]):
            return ContentType.CONCEPTUAL
        
        return ContentType.GENERAL
    
    def _find_semantic_boundaries(self, text: str, config: ChunkConfig) -> List[int]:
        """Find good semantic boundaries for splitting text."""
        boundaries = []
        
        # Always include start and end
        boundaries.append(0)
        boundaries.append(len(text))
        
        if config.preserve_headings:
            # Find heading positions
            for match in self.heading_pattern.finditer(text):
                boundaries.append(match.start())
        
        if config.preserve_sentences and self.nlp:
            # Use spaCy for accurate sentence boundaries
            doc = self.nlp(text)
            for sent in doc.sents:
                boundaries.append(sent.start_char)
                boundaries.append(sent.end_char)
        elif config.preserve_sentences:
            # Fallback to regex sentence detection
            for match in self.sentence_end_pattern.finditer(text):
                boundaries.append(match.end())
        
        if config.preserve_lists:
            # Find list item boundaries
            for match in self.list_item_pattern.finditer(text):
                boundaries.append(match.start())
        
        if config.preserve_code_blocks:
            # Find code block boundaries
            for match in self.code_block_pattern.finditer(text):
                boundaries.append(match.start())
                boundaries.append(match.end())
        
        # Remove duplicates and sort
        boundaries = sorted(set(boundaries))
        
        # Filter out boundaries that are too close together (< 20 chars)
        # But always keep start and end
        filtered_boundaries = []
        if boundaries:
            filtered_boundaries.append(boundaries[0])  # Always keep start
            
            for boundary in boundaries[1:-1]:  # Process middle boundaries
                if boundary - filtered_boundaries[-1] >= 20:
                    filtered_boundaries.append(boundary)
            
            # Always keep end if it's different from start
            if len(boundaries) > 1 and boundaries[-1] != boundaries[0]:
                filtered_boundaries.append(boundaries[-1])
        
        return filtered_boundaries
    
    def _find_optimal_split_point(self, text: str, target_pos: int, 
                                  boundaries: List[int], config: ChunkConfig) -> int:
        """Find the best split point near target position."""
        # Find the boundary closest to target position
        best_boundary = target_pos
        min_distance = float('inf')
        
        for boundary in boundaries:
            if boundary <= target_pos:
                distance = target_pos - boundary
                if distance < min_distance:
                    min_distance = distance
                    best_boundary = boundary
        
        # Ensure we don't split in the middle of preserved structures
        if config.preserve_code_blocks:
            # Check if split would be inside a code block
            for match in self.code_block_pattern.finditer(text):
                if match.start() < best_boundary < match.end():
                    # Move to start of code block
                    best_boundary = match.start()
                    break
        
        return best_boundary
    
    def chunk_text_semantically(self, text: str, metadata: Dict) -> List[Dict]:
        """Create semantic chunks optimized for the content type.
        
        Args:
            text: Text to chunk
            metadata: Document metadata
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text.strip():
            return []
        
        # Classify content type and get appropriate config
        content_type = self._classify_content_type(text, metadata)
        config = self.chunk_configs[content_type]
        
        self.logger.debug(f"Chunking as {content_type.value} with max_tokens={config.max_tokens}")
        
        # Find semantic boundaries
        boundaries = self._find_semantic_boundaries(text, config)
        
        chunks = []
        chunk_id = 0
        text_length = len(text)
        
        start_pos = 0
        
        while start_pos < text_length:
            # Calculate target end position based on token count
            current_text = text[start_pos:]
            
            # Binary search for optimal chunk size
            left, right = config.min_tokens, config.max_tokens
            best_end_pos = len(current_text)  # Fallback to end
            
            while left <= right:
                mid_tokens = (left + right) // 2
                
                # Estimate character position for token count
                # Rough approximation: 1 token ≈ 4 characters
                estimated_chars = mid_tokens * 4
                
                if estimated_chars < len(current_text):
                    # Check actual token count
                    candidate_text = current_text[:estimated_chars]
                    actual_tokens = self._count_tokens(candidate_text)
                    
                    if actual_tokens <= config.max_tokens:
                        left = mid_tokens + 1
                        best_end_pos = estimated_chars
                    else:
                        right = mid_tokens - 1
                else:
                    right = mid_tokens - 1
            
            # Convert relative to absolute position
            target_end_pos = min(start_pos + best_end_pos, text_length)
            
            # Find optimal split point at semantic boundary
            chunk_end = self._find_optimal_split_point(
                text, target_end_pos, boundaries, config
            )
            
            # Ensure minimum chunk size
            chunk_text = text[start_pos:chunk_end]
            if self._count_tokens(chunk_text) < config.min_tokens and chunk_end < text_length:
                # Try to extend chunk
                for boundary in boundaries:
                    if boundary > chunk_end:
                        extended_text = text[start_pos:boundary]
                        if self._count_tokens(extended_text) <= config.max_tokens:
                            chunk_end = boundary
                        break
            
            # Extract final chunk
            chunk_text = text[start_pos:chunk_end].strip()
            
            if chunk_text and self._count_tokens(chunk_text) >= 10:  # Lower minimum viable chunk
                # Create enhanced metadata
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_id": f"{metadata.get('source_file', 'unknown')}_semantic_{chunk_id}",
                    "token_count": self._count_tokens(chunk_text),
                    "chunk_index": chunk_id,
                    "content_type_detected": content_type.value,
                    "chunking_strategy": "semantic",
                    "chunk_start_pos": start_pos,
                    "chunk_end_pos": chunk_end
                })
                
                chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })
                
                chunk_id += 1
            
            # Calculate next start position with overlap
            if chunk_end >= text_length:
                break
                
            # Calculate overlap position
            overlap_chars = min(
                config.overlap_tokens * 4,  # Rough char estimate
                (chunk_end - start_pos) // 4  # Max 25% overlap
            )
            
            next_start = max(start_pos + 1, chunk_end - overlap_chars)
            
            # Align overlap with semantic boundary if possible
            for boundary in reversed(boundaries):
                if start_pos < boundary < chunk_end:
                    overlap_start = max(boundary, chunk_end - overlap_chars)
                    if overlap_start < chunk_end:
                        next_start = overlap_start
                        break
            
            start_pos = next_start
        
        self.logger.info(f"Created {len(chunks)} semantic chunks of type {content_type.value}")
        return chunks
    
    def get_chunking_stats(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Generate statistics about the chunking process."""
        if not chunks:
            return {}
        
        token_counts = [chunk["metadata"]["token_count"] for chunk in chunks]
        content_types = [chunk["metadata"].get("content_type_detected", "unknown") 
                        for chunk in chunks]
        
        # Count by content type
        type_counts: Dict[str, int] = {}
        for ct in content_types:
            type_counts[ct] = type_counts.get(ct, 0) + 1
        
        return {
            "total_chunks": len(chunks),
            "avg_tokens_per_chunk": sum(token_counts) / len(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "content_type_distribution": type_counts,
            "chunking_strategy": "semantic"
        }