# Always available
from .base import (
    BaseModelClient,
    Chunk,
    Model,
    ModelSpec,
    StreamChunk,
    StreamPhase,
    StreamingContentType,
)
from .model_client import ModelClient, resolve_model_string

# Optional imports with graceful fallbacks
try:
    from .hf import HuggingFaceClient, HuggingFaceModel, ToolCallFormat

    HAS_HF = True
except ImportError:
    HAS_HF = False
    HuggingFaceClient = None
    HuggingFaceModel = None
    ToolCallFormat = None

try:
    from .ollama import OllamaClient, OllamaModel

    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False
    OllamaClient = None
    OllamaModel = None

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
        LMStudioOpenAIModel,
        OllamaOpenAIClient,
        OllamaOpenAIModel,
        HFOpenAIClient,
        HFOpenAIModel,
        VLLMOpenAIClient,
        VLLMOpenAIModel,
        LlamaServerOpenAIClient,
        LlamaServerOpenAIModel,
        SGLangOpenAIClient,
        SGLangOpenAIModel,
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
    LMStudioOpenAIModel = None
    OllamaOpenAIClient = None
    OllamaOpenAIModel = None
    HFOpenAIClient = None
    HFOpenAIModel = None
    VLLMOpenAIClient = None
    VLLMOpenAIModel = None
    LlamaServerOpenAIClient = None
    LlamaServerOpenAIModel = None
    SGLangOpenAIClient = None
    SGLangOpenAIModel = None

try:
    from .llamacpp import LlamaCppClient, LlamaCppModel

    HAS_LLAMACPP = True
except ImportError:
    HAS_LLAMACPP = False
    LlamaCppClient = None
    LlamaCppModel = None

# Expose what's available
__all__ = [
    "BaseModelClient",
    "Chunk",
    "Model",
    "ModelClient",
    "ModelSpec",
    "StreamChunk",
    "StreamingContentType",
    "StreamPhase",
    "resolve_model_string",
]
if HAS_HF:
    __all__.extend(["HuggingFaceClient", "HuggingFaceModel", "ToolCallFormat"])
if HAS_OLLAMA:
    __all__.extend(["OllamaClient", "OllamaModel"])
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
            "LMStudioOpenAIModel",
            "OllamaOpenAIClient",
            "OllamaOpenAIModel",
            "HFOpenAIClient",
            "HFOpenAIModel",
            "VLLMOpenAIClient",
            "VLLMOpenAIModel",
            "LlamaServerOpenAIClient",
            "LlamaServerOpenAIModel",
            "SGLangOpenAIClient",
            "SGLangOpenAIModel",
        ]
    )
if HAS_LLAMACPP:
    __all__.extend(["LlamaCppClient", "LlamaCppModel"])
