"""
Tests for RAG prompts module.
"""

import pytest
from langchain.prompts import PromptTemplate

from src.rag.prompts import blender_bot_template


class TestBlenderBotTemplate:
    """Test the blender_bot_template PromptTemplate."""

    def test_template_is_prompt_template(self):
        """Test that blender_bot_template is a PromptTemplate instance."""
        assert isinstance(blender_bot_template, PromptTemplate)

    def test_template_input_variables(self):
        """Test that template has the correct input variables."""
        expected_variables = ["context", "question"]
        assert blender_bot_template.input_variables == expected_variables

    def test_template_format_basic(self):
        """Test that template can be formatted with basic inputs."""
        test_context = "Test context information [1]"
        test_question = "What is Blender?"
        
        formatted = blender_bot_template.format(
            context=test_context,
            question=test_question
        )
        
        assert test_context in formatted
        assert test_question in formatted
        assert "BlenderBot" in formatted

    def test_template_contains_key_instructions(self):
        """Test that template contains expected instruction text."""
        template_text = blender_bot_template.template
        
        # Key instruction elements
        assert "BlenderBot" in template_text
        assert "expert assistant for Blender 3D software" in template_text
        assert "only use the information provided" in template_text
        assert "citations" in template_text
        assert "{context}" in template_text
        assert "{question}" in template_text

    def test_template_format_with_citations(self):
        """Test that template formatting preserves citation format."""
        test_context = "Blender is a 3D software [1]. It has modeling tools [2]."
        test_question = "What can Blender do?"
        
        formatted = blender_bot_template.format(
            context=test_context,
            question=test_question
        )
        
        # Verify citation format is preserved
        assert "[1]" in formatted
        assert "[2]" in formatted
        assert "citations" in formatted

    def test_template_format_empty_inputs(self):
        """Test template behavior with empty inputs."""
        formatted = blender_bot_template.format(
            context="",
            question=""
        )
        
        # Should still contain the base template structure
        assert "BlenderBot" in formatted
        assert "Question:" in formatted
        assert "Answer:" in formatted