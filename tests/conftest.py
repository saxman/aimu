"""
Pytest configuration for model tests.
"""

import pytest


def pytest_addoption(parser):
    """Add custom command line options for pytest."""
    parser.addoption(
        "--client",
        action="store",
        default="all",
        help="Client type to test: 'ollama', 'hf'/'huggingface', or 'all' (default: all)"
    )