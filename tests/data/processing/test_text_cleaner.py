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
        """Test fixing pilcrow sign encoding issues."""
        test_cases = [
            ("Add Cube Â¶", "Add Cube ¶"),
            ("Usage Â¶ First step", "Usage ¶ First step"),
            ("Multiple Â¶ sections Â¶ here", "Multiple ¶ sections ¶ here"),
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
        assert "Â¶" not in result, "Should fix pilcrow encoding"
        assert "¶" in result, "Should contain proper pilcrow"
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
        assert "¶" in result, "Should fix pilcrow"
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
        assert "¶" in result[0], "Should fix first text"
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
        assert result[2] == "¶"
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
        assert "¶" in result

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
        
        # All should be fixed
        assert "Â¶" not in result
        assert "â£" not in result
        assert "Â°" not in result
        assert "¶" in result
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