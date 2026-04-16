# Always available
from .base import ModelClient, StreamingContentType, StreamChunk, StreamPhase

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
    from .anthropic import AnthropicClient, AnthropicModel

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    AnthropicClient = None
    AnthropicModel = None

try:
    from .openai_compat import (
        OpenAICompatClient,
        OpenAIClient,
        OpenAIModel,
        GeminiClient,
        GeminiModel,
        LMStudioOpenAIClient,
        OllamaOpenAIClient,
        HFOpenAIClient,
        VLLMOpenAIClient,
        LlamaServerOpenAIClient,
        SGLangOpenAIClient,
    )

    HAS_OPENAI_COMPAT = True
except ImportError:
    HAS_OPENAI_COMPAT = False
    OpenAICompatClient = None
    OpenAIClient = None
    OpenAIModel = None
    GeminiClient = None
    GeminiModel = None
    LMStudioOpenAIClient = None
    OllamaOpenAIClient = None
    HFOpenAIClient = None
    VLLMOpenAIClient = None
    LlamaServerOpenAIClient = None
    SGLangOpenAIClient = None

try:
    from .llamacpp import LlamaCppClient

    HAS_LLAMACPP = True
except ImportError:
    HAS_LLAMACPP = False
    LlamaCppClient = None

# Expose what's available
__all__ = ["ModelClient", "StreamingContentType", "StreamChunk", "StreamPhase"]
if HAS_HF:
    __all__.append("HuggingFaceClient")
if HAS_OLLAMA:
    __all__.append("OllamaClient")
if HAS_ANTHROPIC:
    __all__.extend(["AnthropicClient", "AnthropicModel"])
if HAS_OPENAI_COMPAT:
    __all__.extend(
        [
            "OpenAICompatClient",
            "OpenAIClient",
            "OpenAIModel",
            "GeminiClient",
            "GeminiModel",
            "LMStudioOpenAIClient",
            "OllamaOpenAIClient",
            "HFOpenAIClient",
            "VLLMOpenAIClient",
            "LlamaServerOpenAIClient",
            "SGLangOpenAIClient",
        ]
    )
if HAS_LLAMACPP:
    __all__.append("LlamaCppClient")
