"""
Pytest configuration for model tests.
"""


def pytest_addoption(parser):
    """Add custom command line options for pytest."""
    parser.addoption(
        "--client",
        action="store",
        default="all",
        help="Client type to test: 'ollama', 'hf'/'huggingface', 'aisuite', or 'all' (default: all)",
    )

    parser.addoption(
        "--model",
        action="store",
        default="all",
        help="Model name to test (default: all)",
    )

    parser.addoption(
        "--model-path",
        action="store",
        default=None,
        help="Path to a local GGUF model file (required for --client=llamacpp)",
    )
