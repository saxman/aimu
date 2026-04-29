"""
Pytest configuration for model tests.

Shared helpers (MockModelClient, resolve_model_params, create_real_model_client)
live in ``tests/helpers.py`` so they can be imported as a normal module.
"""


def pytest_addoption(parser):
    """Add custom command line options for pytest."""
    parser.addoption(
        "--client",
        action="store",
        default="all",
        help=(
            "Client type to test: 'ollama', 'hf'/'huggingface', "
            "'openai', 'anthropic', 'gemini', or 'all' (default: all)"
        ),
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
