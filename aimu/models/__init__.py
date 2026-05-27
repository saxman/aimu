# Always available
from .base import (
    BaseImageClient,
    BaseModelClient,
    GeminiImageSpec,
    HuggingFaceImageSpec,
    ImageModel,
    ImageSpec,
    Model,
    ModelSpec,
    StreamChunk,
    StreamingContentType,
)
from .image_client import ImageClient, resolve_image_model_string
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

try:
    from .hf_image import HuggingFaceImageClient, HuggingFaceImageModel

    HAS_HF_IMAGE = True
except ImportError:
    HAS_HF_IMAGE = False
    HuggingFaceImageClient = None
    HuggingFaceImageModel = None

try:
    from .gemini_image import GeminiImageClient, GeminiImageModel

    HAS_GEMINI_IMAGE = True
except ImportError:
    HAS_GEMINI_IMAGE = False
    GeminiImageClient = None
    GeminiImageModel = None

# Expose what's available
__all__ = [
    "BaseImageClient",
    "BaseModelClient",
    "GeminiImageSpec",
    "HuggingFaceImageSpec",
    "ImageClient",
    "ImageModel",
    "ImageSpec",
    "Model",
    "ModelClient",
    "ModelSpec",
    "StreamChunk",
    "StreamingContentType",
    "available_image_clients",
    "available_text_clients",
    "resolve_image_model_string",
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
if HAS_HF_IMAGE:
    __all__.extend(["HuggingFaceImageClient", "HuggingFaceImageModel"])
if HAS_GEMINI_IMAGE:
    __all__.extend(["GeminiImageClient", "GeminiImageModel"])


def available_text_clients() -> list[type[BaseModelClient]]:
    """Return client classes for all installed text providers, in display order."""
    clients: list[type[BaseModelClient]] = []
    if HAS_OLLAMA:
        clients.append(OllamaClient)
    if HAS_HF:
        clients.append(HuggingFaceClient)
    if HAS_ANTHROPIC:
        clients.append(AnthropicClient)
    if HAS_OPENAI_COMPAT:
        clients.extend([
            OpenAIClient, GeminiClient, LMStudioOpenAIClient,
            OllamaOpenAIClient, HFOpenAIClient, VLLMOpenAIClient,
            LlamaServerOpenAIClient, SGLangOpenAIClient,
        ])
    if HAS_LLAMACPP:
        clients.append(LlamaCppClient)
    return clients


def available_image_clients() -> list[type[BaseImageClient]]:
    """Return client classes for all installed image providers, in display order."""
    clients: list[type[BaseImageClient]] = []
    if HAS_HF_IMAGE:
        clients.append(HuggingFaceImageClient)
    if HAS_GEMINI_IMAGE:
        clients.append(GeminiImageClient)
    return clients
