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
        default=None,
        help=(
            "Client type to test: 'ollama', 'hf'/'huggingface', "
            "'openai', 'anthropic', 'gemini', or 'all'. Omitted (default) skips "
            "live model tests; pass 'all' for the full cross-provider matrix."
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

    parser.addoption(
        "--image-client",
        action="store",
        default=None,
        help=(
            "Image client to test live: 'hf', 'gemini', or 'all'. Required to enable "
            "live tests in tests/test_images.py; omit to skip them."
        ),
    )

    parser.addoption(
        "--image-model",
        action="store",
        default="all",
        help=(
            "Image model enum member name (e.g. SD_1_5, NANO_BANANA). Defaults to 'all' "
            "within the selected --image-client."
        ),
    )

    parser.addoption(
        "--audio-client",
        action="store",
        default=None,
        help=(
            "Audio client to test live: 'hf' or 'all'. Required to enable "
            "live tests in tests/test_audio.py; omit to skip them."
        ),
    )

    parser.addoption(
        "--audio-model",
        action="store",
        default="all",
        help=(
            "Audio model enum member name (e.g. MUSICGEN_SMALL, AUDIOLDM2). "
            "Defaults to 'all' within the selected --audio-client."
        ),
    )

    parser.addoption(
        "--speech-client",
        action="store",
        default=None,
        help=(
            "Speech client to test live: 'hf', 'openai', or 'all'. Required to enable "
            "live tests in tests/test_speech.py; omit to skip them."
        ),
    )

    parser.addoption(
        "--speech-model",
        action="store",
        default="all",
        help=(
            "Speech model enum member name (e.g. MMS_TTS_ENG, TTS_1). "
            "Defaults to 'all' within the selected --speech-client."
        ),
    )

    parser.addoption(
        "--embedding-client",
        action="store",
        default=None,
        help=(
            "Embedding client to test live: 'openai', 'ollama', 'hf', or 'all'. Required to "
            "enable live tests in tests/test_embeddings.py; omit to skip them."
        ),
    )

    parser.addoption(
        "--embedding-model",
        action="store",
        default="all",
        help=(
            "Embedding model enum member name (e.g. TEXT_EMBEDDING_3_SMALL, NOMIC_EMBED_TEXT). "
            "Defaults to 'all' within the selected --embedding-client."
        ),
    )
