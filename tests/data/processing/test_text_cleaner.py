#!/usr/bin/env python3
"""
Unit tests for TextCleaner module.

Tests cover HTML encoding fixes, Unicode normalization, and edge cases.
"""

import pytest
import unicodedata
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from data.processing.text_cleaner import TextCleaner


class TestTextCleaner:
    """Test suite for TextCleaner class."""

    @pytest.fixture
    def cleaner(self):
        """Create a TextCleaner instance for testing."""
        return TextCleaner()

    def test_initialization(self, cleaner):
        """Test that TextCleaner initializes correctly."""
        assert cleaner is not None
        assert isinstance(cleaner.encoding_fixes, dict)
        assert len(cleaner.encoding_fixes) > 0
        assert cleaner.whitespace_pattern is not None
        assert cleaner.html_entity_pattern is not None

    # HTML Entity Decoding Tests
    def test_html_entity_decoding_named(self, cleaner):
        """Test decoding of named HTML entities."""
        test_cases = [
            ("&amp; &lt; &gt;", "& < >"),
            ("&quot;Hello&quot;", '"Hello"'),
            ("&nbsp;space", "\u00a0space"),  # Non-breaking space becomes U+00A0
            ("&copy; 2023", "© 2023"),
        ]
        
        for input_text, expected in test_cases:
            result = cleaner._decode_html_entities(input_text)
            assert result == expected, f"Failed for input: {input_text}"

    def test_html_entity_decoding_numeric(self, cleaner):
        """Test decoding of numeric HTML entities."""
        test_cases = [
            ("&#65;", "A"),  # Decimal
            ("&#8594;", "→"),  # Arrow
            ("&#169;", "©"),  # Copyright
        ]
        
        for input_text, expected in test_cases:
            result = cleaner._decode_html_entities(input_text)
            assert result == expected, f"Failed for input: {input_text}"

    def test_html_entity_decoding_malformed(self, cleaner):
        """Test handling of malformed HTML entities."""
        test_cases = [
            ("&#invalid;", "&#invalid;"),  # Should remain unchanged
            ("&incomplete", "&incomplete"),  # Missing semicolon
            ("&#99999999;", "�"),  # Out of range becomes replacement char
        ]
        
        for input_text, expected in test_cases:
            result = cleaner._decode_html_entities(input_text)
            assert result == expected, f"Failed for input: {input_text}"

    # Encoding Artifacts Tests
    def test_encoding_artifacts_pilcrow(self, cleaner):
        """Test removal of pilcrow signs (don't help semantic search)."""
        test_cases = [
            ("Add Cube Â¶", "Add Cube "),
            ("Usage Â¶ First step", "Usage  First step"),
            ("Multiple Â¶ sections ¶ here", "Multiple  sections  here"),
        ]
        
        for input_text, expected in test_cases:
            result = cleaner._fix_encoding_artifacts(input_text)
            assert result == expected, f"Failed for input: {input_text}"

    def test_encoding_artifacts_arrows(self, cleaner):
        """Test fixing arrow character encoding issues."""
        test_cases = [
            ("Toolbar â£ Add Cube", "Toolbar → Add Cube"),
            ("Menu â£ File", "Menu → File"),
        ]
        
        for input_text, expected in test_cases:
            result = cleaner._fix_encoding_artifacts(input_text)
            assert result == expected, f"Failed for input: {input_text}"

    def test_encoding_artifacts_symbols(self, cleaner):
        """Test fixing various symbol encoding issues."""
        test_cases = [
            ("Temperature 25Â°", "Temperature 25°"),
            ("Â± 5 units", "± 5 units"),
            ("Â© Copyright", "© Copyright"),
            ("Â® Registered", "® Registered"),
        ]
        
        for input_text, expected in test_cases:
            result = cleaner._fix_encoding_artifacts(input_text)
            assert result == expected, f"Failed for input: {input_text}"

    def test_encoding_artifacts_zero_width_chars(self, cleaner):
        """Test removal of zero-width characters."""
        test_cases = [
            ("Text\u200bwith\u200bzero\u200bwidth", "Textwithzerowidth"),
            ("Normal\ufeffwith\u200cbom", "Normalwithbom"),
            ("Join\u200dchar\u200dtest", "Joinchartest"),
        ]
        
        for input_text, expected in test_cases:
            result = cleaner._fix_encoding_artifacts(input_text)
            assert result == expected, f"Failed for input: {input_text}"

    # Unicode Normalization Tests
    def test_unicode_normalization_basic(self, cleaner):
        """Test basic Unicode normalization."""
        # Test with a ligature character that should be normalized
        input_text = "ﬁle"  # fi ligature
        result = cleaner._normalize_unicode(input_text)
        assert result == "file", "Ligature should be decomposed to separate chars"

    def test_unicode_normalization_accents(self, cleaner):
        """Test Unicode normalization with accented characters."""
        # Composed vs decomposed forms should be normalized to same result
        composed = "é"  # Single é character (U+00E9)
        decomposed = "é"  # e + combining acute (U+0065 + U+0301)
        
        result1 = cleaner._normalize_unicode(composed)
        result2 = cleaner._normalize_unicode(decomposed)
        
        assert result1 == result2, "Different Unicode forms should normalize identically"

    def test_unicode_normalization_edge_cases(self, cleaner):
        """Test Unicode normalization with edge cases."""
        test_cases = [
            ("", ""),  # Empty string
            ("simple", "simple"),  # ASCII only
            ("123", "123"),  # Numbers
        ]
        
        for input_text, expected in test_cases:
            result = cleaner._normalize_unicode(input_text)
            assert result == expected, f"Failed for input: {input_text}"

    # Whitespace Normalization Tests
    def test_whitespace_normalization_multiple_spaces(self, cleaner):
        """Test normalization of multiple consecutive spaces."""
        test_cases = [
            ("single  double   triple", "single double triple"),
            ("  leading spaces", " leading spaces"),
            ("trailing spaces  ", "trailing spaces "),
            ("mixed\t\tspaces\n\nhere", "mixed spaces\nhere"),  # Preserves single newline
        ]
        
        for input_text, expected in test_cases:
            result = cleaner._normalize_whitespace(input_text)
            assert result == expected, f"Failed for input: {input_text}"

    def test_whitespace_normalization_paragraph_breaks(self, cleaner):
        """Test preservation and normalization of paragraph breaks."""
        test_cases = [
            ("Para1\n\nPara2", "Para1\nPara2"),
            ("Para1\n  \n  Para2", "Para1\n Para2"),  # Space after newline preserved
            ("Single\nbreak", "Single\nbreak"),
        ]
        
        for input_text, expected in test_cases:
            result = cleaner._normalize_whitespace(input_text)
            assert result == expected, f"Failed for input: {input_text}"

    # Special Character Cleaning Tests
    def test_clean_special_characters_preserve_technical(self, cleaner):
        """Test that technical content is preserved during special character cleaning."""
        test_cases = [
            ("Press Ctrl+Alt+T", "Press Ctrl+Alt+T"),
            ("/usr/local/bin", "/usr/local/bin"),
            ("```code block```", "```code block```"),
            ("`inline code`", "`inline code`"),
        ]
        
        for input_text, expected in test_cases:
            result = cleaner._clean_special_characters(input_text)
            assert result == expected, f"Failed to preserve: {input_text}"

    def test_clean_special_characters_remove_problematic(self, cleaner):
        """Test removal of problematic control characters."""
        # Test with various control characters (excluding allowed ones)
        input_text = "Normal text\x00with\x01control\x02chars"
        result = cleaner._clean_special_characters(input_text)
        
        # Should remove control chars but keep normal text
        assert "Normal text" in result
        assert "with" in result
        assert "control" in result
        assert "chars" in result
        assert "\x00" not in result
        assert "\x01" not in result
        assert "\x02" not in result

    def test_clean_special_characters_preserve_arrows(self, cleaner):
        """Test that arrow characters are preserved."""
        test_cases = [
            ("Step 1 → Step 2", "Step 1 → Step 2"),
            ("Go → Menu → File", "Go → Menu → File"),
        ]
        
        for input_text, expected in test_cases:
            result = cleaner._clean_special_characters(input_text)
            assert result == expected, f"Failed to preserve arrows in: {input_text}"

    # Integration Tests (Full Pipeline)
    def test_clean_text_integration_blender_example(self, cleaner):
        """Test the complete cleaning pipeline with realistic Blender documentation."""
        input_text = """Add Cube Â¶ Reference Mode : Object Mode and Edit Mode Tool : Toolbar â£ Add Cube Interactively add a cube mesh object . Usage Â¶ First define the base of the object by dragging with LMB ."""
        
        result = cleaner.clean_text(input_text)
        
        # Check that encoding issues are fixed
        assert "Â¶" not in result, "Should remove pilcrow artifacts"
        assert "¶" not in result, "Should remove pilcrow symbols entirely" 
        assert "â£" not in result, "Should fix arrow encoding"
        assert "→" in result, "Should contain proper arrow"
        
        # Check that content is preserved
        assert "Add Cube" in result
        assert "Object Mode" in result
        assert "LMB" in result

    def test_clean_text_integration_empty_and_whitespace(self, cleaner):
        """Test the complete cleaning pipeline with edge cases."""
        test_cases = [
            ("", ""),
            ("   ", ""),
            ("\t\n\r", ""),
            ("single", "single"),
        ]
        
        for input_text, expected in test_cases:
            result = cleaner.clean_text(input_text)
            assert result == expected, f"Failed for input: '{input_text}'"

    def test_clean_text_integration_preserve_structure(self, cleaner):
        """Test that document structure is preserved during cleaning."""
        input_text = "Title\n\nParagraph 1 with Â¶ symbol.\n\nParagraph 2 with â£ arrow."
        result = cleaner.clean_text(input_text)
        
        # Should preserve paragraph structure (double newlines become single)
        assert result.count('\n') == 2, "Should preserve paragraph structure"
        assert "¶" not in result, "Should remove pilcrow symbols"
        assert "Â¶" not in result, "Should remove pilcrow artifacts"
        assert "→" in result, "Should fix arrow"

    # Analysis and Statistics Tests
    def test_analyze_text_issues_detection(self, cleaner):
        """Test detection of text issues."""
        test_text = "Add Cube Â¶ and Toolbar â£ with &amp; entity"
        issues = cleaner.analyze_text_issues(test_text)
        
        assert len(issues['encoding_artifacts']) >= 2, "Should detect encoding artifacts"
        assert len(issues['html_entities']) >= 1, "Should detect HTML entities"
        assert issues['stats']['total_chars'] == len(test_text)
        assert issues['stats']['non_ascii_chars'] > 0, "Should count non-ASCII chars"

    def test_analyze_text_issues_clean_text(self, cleaner):
        """Test analysis of already clean text."""
        test_text = "This is clean ASCII text with no issues."
        issues = cleaner.analyze_text_issues(test_text)
        
        assert len(issues['encoding_artifacts']) == 0, "Should find no encoding artifacts"
        assert len(issues['html_entities']) == 0, "Should find no HTML entities"
        assert issues['stats']['non_ascii_chars'] == 0, "Should find no non-ASCII chars"

    def test_get_cleaning_stats(self, cleaner):
        """Test cleaning statistics generation."""
        original = "Text with Â¶ and â£ issues"
        cleaned = cleaner.clean_text(original)
        stats = cleaner.get_cleaning_stats(original, cleaned)
        
        assert stats['original_length'] == len(original)
        assert stats['cleaned_length'] == len(cleaned)
        assert stats['chars_removed'] >= 0
        assert stats['encoding_fixes_applied'] > 0
        assert stats['non_ascii_before'] >= stats['non_ascii_after']

    # Batch Processing Tests
    def test_clean_batch_empty(self, cleaner):
        """Test batch cleaning with empty input."""
        result = cleaner.clean_batch([])
        assert result == []

    def test_clean_batch_multiple_texts(self, cleaner):
        """Test batch cleaning with multiple texts."""
        input_texts = [
            "Text 1 with Â¶ issue",
            "Text 2 with â£ problem", 
            "Clean text 3"
        ]
        
        result = cleaner.clean_batch(input_texts)
        
        assert len(result) == len(input_texts)
        assert "Â¶" not in result[0], "Should remove pilcrow artifacts"
        assert "→" in result[1], "Should fix second text"
        assert result[2] == "Clean text 3", "Should preserve clean text"

    def test_clean_batch_mixed_content(self, cleaner):
        """Test batch cleaning with various content types."""
        input_texts = [
            "",  # Empty
            "   ",  # Whitespace only
            "Â¶",  # Encoding issue only
            "Normal text",  # Clean text
        ]
        
        result = cleaner.clean_batch(input_texts)
        
        assert len(result) == 4
        assert result[0] == ""
        assert result[1] == ""
        assert result[2] == ""  # Pilcrow removed entirely
        assert result[3] == "Normal text"

    # Edge Cases and Error Handling
    def test_edge_case_none_input(self, cleaner):
        """Test handling of None input."""
        result = cleaner.clean_text(None)
        assert result == ""

    def test_edge_case_very_long_text(self, cleaner):
        """Test handling of very long text."""
        # Create a long text with encoding issues
        long_text = ("Text with Â¶ issue " * 1000)
        result = cleaner.clean_text(long_text)
        
        assert len(result) > 0
        assert "Â¶" not in result
        # Pilcrows are removed entirely, so check for clean text
        assert "Text with issue" in result

    def test_edge_case_unicode_edge_cases(self, cleaner):
        """Test handling of various Unicode edge cases."""
        test_cases = [
            "\U0001F600",  # Emoji
            "\u0000",      # Null character
            "\uFFFF",      # Max BMP character
        ]
        
        for test_char in test_cases:
            # Should not crash
            result = cleaner.clean_text(test_char)
            assert isinstance(result, str)

    def test_edge_case_mixed_encodings(self, cleaner):
        """Test handling of mixed encoding artifacts in same text."""
        mixed_text = "Â¶ pilcrow and â£ arrow and Â° degree all together"
        result = cleaner.clean_text(mixed_text)
        
        # All artifacts should be fixed
        assert "Â¶" not in result
        assert "â£" not in result
        assert "Â°" not in result
        # Pilcrow removed, arrow and degree fixed
        assert "¶" not in result  # Pilcrow removed entirely
        assert "→" in result
        assert "°" in result

    # Performance and Regression Tests
    def test_performance_no_infinite_loops(self, cleaner):
        """Test that cleaning operations don't cause infinite loops."""
        # Text that might cause issues if regex is poorly constructed
        test_text = "Recursive Â¶Â¶Â¶ patterns â£â£â£ here"
        
        # Should complete without hanging
        result = cleaner.clean_text(test_text)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_idempotency(self, cleaner):
        """Test that applying cleaning twice gives same result."""
        test_text = "Text with Â¶ and â£ issues"
        
        first_clean = cleaner.clean_text(test_text)
        second_clean = cleaner.clean_text(first_clean)
        
        assert first_clean == second_clean, "Cleaning should be idempotent"

    # Additional tests to cover missing lines
    def test_html_entity_decode_numeric_edge_cases(self, cleaner):
        """Test numeric HTML entity decoding edge cases and error handling.""" 
        # Test cases that would trigger the ValueError and OverflowError paths in decode_numeric_entity
        # These need to be malformed entities that html.unescape doesn't handle
        
        # Import patch to mock html.unescape to skip the first processing stage
        from unittest.mock import patch
        
        # Test the custom numeric entity handler directly by bypassing html.unescape
        with patch('html.unescape', side_effect=lambda x: x):  # Return input unchanged
            test_cases = [
                # Test ValueError in decode_numeric_entity (lines 116-117) 
                ("&#abc;", "&#abc;"),  # Non-numeric should remain unchanged
                ("&#999999999999999999999999999999999;", "&#999999999999999999999999999999999;"),  # Overflow case
                ("&#65;", "A"),  # Valid case
                ("&#-1;", "&#-1;"),  # Negative number (invalid)
                ("&#", "&#"),  # Incomplete entity
                ("&#;", "&#;"),  # Empty numeric entity
            ]
            
            for input_text, expected in test_cases:
                result = cleaner._decode_html_entities(input_text)
                assert result == expected, f"Failed for input: {input_text}"

    def test_html_entity_decode_exception_handling(self, cleaner):
        """Test HTML entity decoding exception handling (lines 122-124)."""
        # Import patch properly
        from unittest.mock import patch
        
        # Mock html.unescape to raise exception to test the exception handler
        with patch('html.unescape', side_effect=Exception("Mock error")):
            input_text = "&amp; test"
            result = cleaner._decode_html_entities(input_text)
            # Should return original text when exception occurs
            assert result == input_text

    def test_unicode_normalization_exception_handling(self, cleaner):
        """Test Unicode normalization exception handling (lines 155-157).""" 
        # Import patch properly
        from unittest.mock import patch
        
        # Mock unicodedata.normalize to raise exception
        with patch('unicodedata.normalize', side_effect=Exception("Mock error")):
            input_text = "test text"
            result = cleaner._normalize_unicode(input_text)
            # Should return original text when exception occurs
            assert result == input_text

    def test_clean_special_characters_control_char_handling(self, cleaner):
        """Test control character handling in _clean_special_characters (line 215)."""
        # Test character that should be kept (alphanumeric or underscore)
        test_text = "valid_text123"
        result = cleaner._clean_special_characters(test_text)
        assert result == test_text
        
        # Test with control characters mixed with preserved blocks
        test_text_with_control = "`code\x01block`\x02normal_text\x03"
        result = cleaner._clean_special_characters(test_text_with_control)
        # Should preserve code block and normal text but remove control chars
        assert "`code\x01block`" in result  # Preserved block should remain
        assert "normal_text" in result
        assert "\x02" not in result
        assert "\x03" not in result

    def test_analyze_text_issues_control_character_detection(self, cleaner):
        """Test detection of control characters in analyze_text_issues (lines 266-267)."""
        # Test with control characters (excluding allowed tab, newline, carriage return)
        test_text = "Normal text\x00with\x01control\x1fchars"
        issues = cleaner.analyze_text_issues(test_text)
        
        # Should detect control characters and add to problematic_chars
        assert issues['stats']['control_chars'] == 3  # \x00, \x01, \x1f
        assert len(issues['problematic_chars']) == 3
        
        # Check that problematic chars are recorded with hex codes
        problematic_chars = [char for char, hex_code in issues['problematic_chars']]
        assert '\x00' in problematic_chars
        assert '\x01' in problematic_chars  
        assert '\x1f' in problematic_chars
        
        # Check hex codes are correct
        hex_codes = [hex_code for char, hex_code in issues['problematic_chars']]
        assert '0x0' in hex_codes
        assert '0x1' in hex_codes
        assert '0x1f' in hex_codes

    def test_analyze_text_issues_allowed_control_chars(self, cleaner):
        """Test that allowed control characters are not flagged as problematic."""
        # Test with allowed control characters (tab=9, newline=10, carriage return=13)
        test_text = "Normal\ttext\nwith\rallowed\ncontrol\tchars"
        issues = cleaner.analyze_text_issues(test_text)
        
        # Should not count allowed control chars as problematic
        assert issues['stats']['control_chars'] == 0
        assert len(issues['problematic_chars']) == 0