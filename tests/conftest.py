#!/usr/bin/env python3
"""
Shared test configuration and fixtures for Blender RAG Assistant tests.
"""

import tempfile
import shutil
from pathlib import Path
import pytest
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory with test data for entire test session."""
    temp_dir = tempfile.mkdtemp()
    test_dir = Path(temp_dir)
    
    # Create sample HTML files
    raw_dir = test_dir / "raw"
    raw_dir.mkdir()
    
    sample_docs = {
        "modeling.html": """
        <html>
        <head><title>Blender Modeling Basics</title></head>
        <body>
            <main>
                <h1>Modeling in Blender</h1>
                <p>Learn the fundamentals of 3D modeling.</p>
                <section>
                    <h2>Edit Mode</h2>
                    <p>Press Tab to enter Edit mode for mesh editing.</p>
                    <p>In Edit mode, you can select vertices, edges, or faces.</p>
                </section>
                <section>
                    <h2>Basic Tools</h2>
                    <p>Extrude (E): Add new geometry by extending selected elements.</p>
                    <p>Inset (I): Create inset faces for detailed modeling.</p>
                    <p>Loop Cut (Ctrl+R): Add edge loops to control geometry flow.</p>
                </section>
            </main>
        </body>
        </html>
        """,
        
        "animation.html": """
        <html>
        <head><title>Blender Animation Workflow</title></head>
        <body>
            <main>
                <h1>Animation in Blender</h1>
                <p>Create stunning animations with Blender's powerful tools.</p>
                <section>
                    <h2>Keyframes</h2>
                    <p>Press I to insert keyframes for object properties.</p>
                    <p>Use the Graph Editor to fine-tune animation curves.</p>
                </section>
                <section>
                    <h2>Timeline</h2>
                    <p>The Timeline shows your animation sequence.</p>
                    <p>Scrub through frames to preview your animation.</p>
                </section>
            </main>
        </body>
        </html>
        """,
        
        "rendering.html": """
        <html>
        <head><title>Rendering in Blender</title></head>
        <body>
            <main>
                <h1>Render Engines</h1>
                <p>Blender offers multiple rendering options.</p>
                <section>
                    <h2>Cycles</h2>
                    <p>Physically-based path tracing renderer.</p>
                    <p>Excellent for photorealistic results.</p>
                    <p>Supports GPU acceleration for faster rendering.</p>
                </section>
                <section>
                    <h2>Eevee</h2>
                    <p>Real-time rendering engine built into Blender.</p>
                    <p>Great for viewport preview and stylized renders.</p>
                    <p>Uses rasterization for fast feedback.</p>
                </section>
            </main>
        </body>
        </html>
        """
    }
    
    # Write sample files
    for filename, content in sample_docs.items():
        (raw_dir / filename).write_text(content)
    
    yield test_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_db():
    """Create temporary database directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def standard_config():
    """Standard test configuration for VectorDBBuilder."""
    return {
        "embedding_model": "all-MiniLM-L6-v2",
        "chunk_size": 256,  # Reasonable size for testing
        "chunk_overlap": 32,
        "batch_size": 10
    }


@pytest.fixture
def minimal_config():
    """Minimal configuration for quick tests."""
    return {
        "embedding_model": "all-MiniLM-L6-v2",
        "chunk_size": 50,
        "chunk_overlap": 10,
        "batch_size": 5
    }


@pytest.fixture
def sample_html():
    """Sample HTML content for individual tests."""
    return """
    <html>
    <head>
        <title>Test Document</title>
    </head>
    <body>
        <nav>Skip navigation</nav>
        <main>
            <h1>Test Content</h1>
            <p>This is a test document for unit testing.</p>
            <p>It contains multiple paragraphs to test chunking.</p>
            <section>
                <h2>Subsection</h2>
                <p>More content in a subsection.</p>
                <ul>
                    <li>List item 1</li>
                    <li>List item 2</li>
                </ul>
            </section>
        </main>
        <footer>Footer content to ignore</footer>
        <script>JavaScript to remove</script>
    </body>
    </html>
    """


# Test markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )