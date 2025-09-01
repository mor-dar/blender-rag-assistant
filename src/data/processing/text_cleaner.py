#!/usr/bin/env python3
"""
Text Cleaner Module for Blender RAG Assistant

Handles HTML encoding artifacts, Unicode normalization, and special character
processing to improve text quality for embeddings and search.
"""

import html
import re
import unicodedata
from typing import Dict, List, Pattern, Any


class TextCleaner:
    """Cleans and normalizes text from HTML documents."""
    
    def __init__(self):
        """Initialize the text cleaner with encoding fix mappings."""
        
        # Common HTML encoding artifacts found in Blender docs
        self.encoding_fixes = {
            # Pilcrow signs (¶) - remove entirely as they don't help semantic search
            'Â¶': '',  # Remove broken pilcrow
            '¶': '',   # Remove clean pilcrow too
            
            # Arrow characters with encoding issues (these are multi-byte sequences)
            'â£': '→',
            'â\x80£': '→',  # 3-byte variant found in HTML
            'â†': '→',
            
            # Degree and other common symbols
            'Â°': '°',
            'Â±': '±',
            'Â²': '²',
            'Â³': '³',
            
            # Copyright and trademark
            'Â©': '©',
            'Â®': '®',
            
            # Mathematical symbols
            'Ã—': '×',
            'Ã·': '÷',
            
            # Remove or replace problematic zero-width characters
            '\u200b': '',  # zero-width space
            '\u200c': '',  # zero-width non-joiner
            '\u200d': '',  # zero-width joiner
            '\ufeff': '',  # byte order mark
        }
        
        # Compile regex patterns for efficient processing
        self.whitespace_pattern = re.compile(r'\s+')
        self.html_entity_pattern = re.compile(r'&[a-zA-Z][a-zA-Z0-9]+[^a-zA-Z0-9]')
        self.numeric_entity_pattern = re.compile(r'&#[0-9]+;')
        
        # Patterns for preserving technical content structure
        self.code_block_pattern = re.compile(r'```[\s\S]*?```|`[^`]+`')
        self.keyboard_shortcut_pattern = re.compile(r'\b[A-Z]+(?:\+[A-Z]+)*\b')
        self.file_path_pattern = re.compile(r'[/\\][\w\-./\\]+')
        
        # Blender-specific terminology patterns to preserve
        self.blender_terms_pattern = re.compile(r'\b(?:Ctrl|Alt|Shift|LMB|RMB|MMB|Tab)\b')

    def clean_text(self, text: str) -> str:
        """Main cleaning function that applies all text cleaning operations.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""
        
        # Step 1: HTML entity decoding
        text = self._decode_html_entities(text)
        
        # Step 2: Fix encoding artifacts
        text = self._fix_encoding_artifacts(text)
        
        # Step 3: Unicode normalization
        text = self._normalize_unicode(text)
        
        # Step 4: Clean whitespace while preserving structure
        text = self._normalize_whitespace(text)
        
        # Step 5: Remove problematic characters but preserve technical content
        text = self._clean_special_characters(text)
        
        return text.strip()

    def _decode_html_entities(self, text: str) -> str:
        """Decode HTML entities to proper Unicode characters.
        
        Args:
            text: Text containing HTML entities
            
        Returns:
            Text with decoded entities
        """
        try:
            # Decode named HTML entities (&amp; &lt; etc.)
            text = html.unescape(text)
            
            # Handle remaining numeric entities that might be malformed
            def decode_numeric_entity(match):
                try:
                    entity = match.group()
                    if entity.startswith('&#') and entity.endswith(';'):
                        num_str = entity[2:-1]
                        if num_str.isdigit():
                            return chr(int(num_str))
                except (ValueError, OverflowError):
                    pass
                return match.group()
            
            text = self.numeric_entity_pattern.sub(decode_numeric_entity, text)
            
        except Exception:
            # If decoding fails, return original text
            pass
        
        return text

    def _fix_encoding_artifacts(self, text: str) -> str:
        """Fix common encoding artifacts using the mapping table.
        
        Args:
            text: Text with encoding artifacts
            
        Returns:
            Text with fixed encodings
        """
        for artifact, replacement in self.encoding_fixes.items():
            text = text.replace(artifact, replacement)
        
        return text

    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters to consistent forms.
        
        Args:
            text: Text to normalize
            
        Returns:
            Unicode-normalized text
        """
        try:
            # Use NFKC normalization to handle compatibility characters
            # This converts things like ﬁ (ligature) to fi (separate chars)
            text = unicodedata.normalize('NFKC', text)
        except Exception:
            # If normalization fails, return original
            pass
        
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving document structure.
        
        Args:
            text: Text with irregular whitespace
            
        Returns:
            Text with normalized whitespace
        """
        # First preserve paragraph breaks (double newlines become single)
        text = re.sub(r'\n\s*\n', '\n', text)
        
        # Replace multiple consecutive whitespace (except newlines) with single space
        text = re.sub(r'[ \t\r\f\v]+', ' ', text)
        
        return text

    def _clean_special_characters(self, text: str) -> str:
        """Remove problematic characters while preserving technical content.
        
        Args:
            text: Text to clean
            
        Returns:
            Text with problematic characters removed
        """
        # Preserve code blocks and technical content during cleaning
        preserved_blocks = []
        
        def preserve_block(match):
            preserved_blocks.append(match.group())
            return f"__PRESERVED_BLOCK_{len(preserved_blocks)-1}__"
        
        # Preserve code blocks, shortcuts, and file paths
        text = self.code_block_pattern.sub(preserve_block, text)
        text = self.keyboard_shortcut_pattern.sub(preserve_block, text) 
        text = self.file_path_pattern.sub(preserve_block, text)
        
        # Remove problematic control characters (but keep printable ones)
        cleaned_chars = []
        for char in text:
            code = ord(char)
            
            # Keep normal ASCII and common Unicode ranges
            if (32 <= code <= 126 or          # ASCII printable
                160 <= code <= 255 or         # Latin-1 supplement
                8192 <= code <= 8303 or       # General punctuation  
                8592 <= code <= 8703 or       # Arrows (includes →)
                8704 <= code <= 8959 or       # Mathematical operators
                code in [9, 10, 13]):         # Tab, newline, carriage return
                cleaned_chars.append(char)
            
            # Keep preserved block markers
            elif char.isalnum() or char in '_':
                cleaned_chars.append(char)
        
        text = ''.join(cleaned_chars)
        
        # Restore preserved blocks
        for i, block in enumerate(preserved_blocks):
            text = text.replace(f"__PRESERVED_BLOCK_{i}__", block)
        
        return text

    def analyze_text_issues(self, text: str) -> Dict[str, Any]:
        """Analyze text for encoding and formatting issues.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with analysis results
        """
        issues: Dict[str, Any] = {
            'encoding_artifacts': [],
            'html_entities': [],
            'problematic_chars': [],
            'stats': {
                'total_chars': len(text),
                'non_ascii_chars': 0,
                'control_chars': 0
            }
        }
        
        # Find encoding artifacts
        for artifact in self.encoding_fixes.keys():
            if artifact in text:
                count = text.count(artifact)
                issues['encoding_artifacts'].append({
                    'artifact': artifact,
                    'count': count,
                    'should_be': self.encoding_fixes[artifact]
                })
        
        # Find HTML entities
        html_entities = self.html_entity_pattern.findall(text)
        html_entities.extend(self.numeric_entity_pattern.findall(text))
        issues['html_entities'] = list(set(html_entities))
        
        # Analyze character composition
        for char in text:
            code = ord(char)
            if code > 127:
                issues['stats']['non_ascii_chars'] += 1
            if code < 32 and code not in [9, 10, 13]:  # Control chars except tab/newline
                issues['stats']['control_chars'] += 1
                issues['problematic_chars'].append((char, hex(code)))
        
        return issues

    def clean_batch(self, texts: List[str]) -> List[str]:
        """Clean a batch of texts efficiently.
        
        Args:
            texts: List of texts to clean
            
        Returns:
            List of cleaned texts
        """
        return [self.clean_text(text) for text in texts]

    def get_cleaning_stats(self, original_text: str, cleaned_text: str) -> Dict[str, Any]:
        """Get statistics about the cleaning operation.
        
        Args:
            original_text: Text before cleaning
            cleaned_text: Text after cleaning
            
        Returns:
            Dictionary with cleaning statistics
        """
        return {
            'original_length': len(original_text),
            'cleaned_length': len(cleaned_text),
            'chars_removed': len(original_text) - len(cleaned_text),
            'non_ascii_before': sum(1 for c in original_text if ord(c) > 127),
            'non_ascii_after': sum(1 for c in cleaned_text if ord(c) > 127),
            'encoding_fixes_applied': sum(
                original_text.count(artifact) 
                for artifact in self.encoding_fixes.keys()
            )
        }