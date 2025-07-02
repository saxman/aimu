# Always available
from .core import ModelClient

# Optional imports with graceful fallbacks
try:
    from .hf import HuggingFaceClient

    HAS_HF = True
except ImportError:
    HAS_HF = False
    HuggingFaceClient = None

try:
    from .ollama import OllamaClient

    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False
    OllamaClient = None

try:
    from .aisuite import AisuiteClient

    HAS_AISUITE = True
except ImportError:
    HAS_AISUITE = False
    AisuiteClient = None

# Expose what's available
__all__ = ["ModelClient"]
if HAS_HF:
    __all__.append("HuggingFaceClient")
if HAS_OLLAMA:
    __all__.append("OllamaClient")
if HAS_AISUITE:
    __all__.append("AisuiteClient")
