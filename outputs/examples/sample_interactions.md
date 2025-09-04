# Sample Blender Bot Interactions

This file provides example inputs and expected outputs for Blender Bot, demonstrating the system's capabilities for Ready Tensor Module 1 submission.

## Example 1: Basic UI Question

**Input:**
```
What are the different types of buttons in Blender's interface?
```

**Expected Output Type:**
- Retrieval from Blender documentation about interface elements
- Detailed explanation of button types (tool buttons, menu buttons, toggles, etc.)
- Proper attribution to Blender Documentation Team
- 3-5 relevant source citations

## Example 2: Tool Usage Question

**Input:**
```
How do I use the eyedropper tool to sample colors in Blender?
```

**Expected Output Type:**
- Step-by-step procedural instructions
- Context about where the eyedropper tool is found
- Usage scenarios and practical examples
- Source citations from relevant Blender manual sections

## Example 3: Interface Controls Question

**Input:**
```
What are the different types of button decorators and field controls I can find in Blender's interface?
```

**Expected Output Type:**
- Explanation of button decorators (icons, badges, status indicators)
- Description of field controls (number fields, text fields, dropdowns)
- Information about eyedropper functionality for value sampling
- Details about menu button interactions and expand options
- References to interface/controls documentation sections

## Example 4: Workflow Efficiency Question

**Input:**
```
How can I quickly enter exact values into number fields instead of dragging?
```

**Expected Output Type:**
- Direct input methods (clicking, tabbing, keyboard shortcuts)
- Alternative interaction techniques
- Tips for precision work
- Interface documentation references

## Example 5: Interface Customization Question

**Input:**
```
Can I customize or change the appearance of interface buttons and menus?
```

**Expected Output Type:**
- Information about theme settings and preferences
- Customization options available
- Interface scaling and accessibility features
- Preferences documentation citations

## System Capabilities Demonstrated

1. **Document Retrieval**: Semantic search through 500+ Blender documentation chunks
2. **Context Assembly**: Relevant information synthesis from multiple sources  
3. **Response Generation**: Natural language answers with proper attribution
4. **Source Citation**: CC-BY-SA 4.0 compliant attribution to Blender Documentation Team
5. **Error Handling**: Graceful handling of unclear queries or missing information

## Technical Details

- **Vector Database**: ChromaDB with persistent storage
- **Embeddings**: HuggingFace sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: Groq Llama3-8B (evaluation mode) or OpenAI GPT models (production mode)
- **Framework**: LangChain for orchestration
- **Response Time**: Typically 2-5 seconds per query
- **Accuracy**: High relevance scores (>0.7 similarity threshold)

## Setup for Testing

1. Copy `.env.example` to `.env` and add your Groq API key
2. Run `python main.py` to start the interactive chat
3. Try the sample questions above to verify functionality
4. Expect detailed, well-sourced responses about Blender workflows

## Attribution

All responses containing Blender documentation content include proper attribution:
*Source: Blender Manual by the Blender Documentation Team, licensed under CC-BY-SA 4.0*