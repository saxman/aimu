# Always available
from ._internal.json import extract_tool_calls, generate_json, parse_json_response
from .audio_client import AudioClient, resolve_audio_model_string
from .base import (
    AudioModel,
    AudioSpec,
    BaseAudioClient,
    BaseEmbeddingClient,
    BaseImageClient,
    BaseModelClient,
    BaseSpeechClient,
    BaseTranscriptionClient,
    EmbeddingModel,
    EmbeddingSpec,
    GeminiImageSpec,
    HuggingFaceAudioSpec,
    HuggingFaceEmbeddingSpec,
    HuggingFaceImageSpec,
    HuggingFaceSpeechSpec,
    HuggingFaceTranscriptionSpec,
    ImageModel,
    ImageSpec,
    Model,
    ModelSpec,
    OllamaEmbeddingSpec,
    OpenAIEmbeddingSpec,
    OpenAISpeechSpec,
    OpenAITranscriptionSpec,
    SpeechModel,
    SpeechSpec,
    StreamChunk,
    StreamingContentType,
    TranscriptionModel,
    TranscriptionSpec,
)
from ._internal.model_defaults import (
    available_audio_models,
    available_embedding_models,
    available_image_models,
    available_speech_models,
    available_transcription_models,
)
from .embedding_client import EmbeddingClient, resolve_embedding_model_string
from .fallback import FallbackClient, FallbackExhaustedError
from .image_client import ImageClient, resolve_image_model_enum, resolve_image_model_string
from .model_client import (
    ModelClient,
    available_text_models,
    resolve_default_text_model_enum,
    resolve_model_enum,
    resolve_model_string,
)
from .speech_client import SpeechClient, resolve_speech_model_string
from .transcription_client import TranscriptionClient, resolve_transcription_model_string

# Optional imports with graceful fallbacks
try:
    from .providers.hf.text import HuggingFaceClient, HuggingFaceModel, ToolCallFormat

    HAS_HF = True
except ImportError:
    HAS_HF = False
    HuggingFaceClient = None
    HuggingFaceModel = None
    ToolCallFormat = None

try:
    from .providers.ollama import OllamaClient, OllamaModel

    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False
    OllamaClient = None
    OllamaModel = None

try:
    from .providers.anthropic import AnthropicClient, AnthropicModel

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    AnthropicClient = None
    AnthropicModel = None

try:
    from .providers.gemini.text import GeminiClient, GeminiModel
    from .providers.openai.text import OpenAIClient, OpenAIModel
    from .providers.openai_compat import (
        HFOpenAIClient,
        HFOpenAIModel,
        LlamaServerOpenAIClient,
        LlamaServerOpenAIModel,
        LMStudioOpenAIClient,
        LMStudioOpenAIModel,
        OllamaOpenAIClient,
        OllamaOpenAIModel,
        OpenAICompatClient,
        SGLangOpenAIClient,
        SGLangOpenAIModel,
        VLLMOpenAIClient,
        VLLMOpenAIModel,
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
    from .providers.llamacpp import LlamaCppClient, LlamaCppModel

    HAS_LLAMACPP = True
except ImportError:
    HAS_LLAMACPP = False
    LlamaCppClient = None
    LlamaCppModel = None

try:
    from .providers.hf.image import HuggingFaceImageClient, HuggingFaceImageModel

    HAS_HF_IMAGE = True
except ImportError:
    HAS_HF_IMAGE = False
    HuggingFaceImageClient = None
    HuggingFaceImageModel = None

try:
    from .providers.gemini.image import GeminiImageClient, GeminiImageModel

    HAS_GEMINI_IMAGE = True
except ImportError:
    HAS_GEMINI_IMAGE = False
    GeminiImageClient = None
    GeminiImageModel = None

try:
    from .providers.hf.audio import HuggingFaceAudioClient, HuggingFaceAudioModel

    HAS_HF_AUDIO = True
except ImportError:
    HAS_HF_AUDIO = False
    HuggingFaceAudioClient = None
    HuggingFaceAudioModel = None

try:
    from .providers.hf.speech import HuggingFaceSpeechClient, HuggingFaceSpeechModel

    HAS_HF_SPEECH = True
except ImportError:
    HAS_HF_SPEECH = False
    HuggingFaceSpeechClient = None
    HuggingFaceSpeechModel = None

try:
    from .providers.openai.speech import OpenAISpeechClient, OpenAISpeechModel

    HAS_OPENAI_SPEECH = True
except ImportError:
    HAS_OPENAI_SPEECH = False
    OpenAISpeechClient = None
    OpenAISpeechModel = None

try:
    from .providers.hf.transcription import HuggingFaceTranscriptionClient, HuggingFaceTranscriptionModel

    HAS_HF_TRANSCRIPTION = True
except ImportError:
    HAS_HF_TRANSCRIPTION = False
    HuggingFaceTranscriptionClient = None
    HuggingFaceTranscriptionModel = None

try:
    from .providers.openai.transcription import OpenAITranscriptionClient, OpenAITranscriptionModel

    HAS_OPENAI_TRANSCRIPTION = True
except ImportError:
    HAS_OPENAI_TRANSCRIPTION = False
    OpenAITranscriptionClient = None
    OpenAITranscriptionModel = None

try:
    from .providers.openai.embedding import OpenAIEmbeddingClient, OpenAIEmbeddingModel

    HAS_OPENAI_EMBEDDING = True
except ImportError:
    HAS_OPENAI_EMBEDDING = False
    OpenAIEmbeddingClient = None
    OpenAIEmbeddingModel = None

try:
    from .providers.ollama import OllamaEmbeddingClient, OllamaEmbeddingModel

    HAS_OLLAMA_EMBEDDING = True
except ImportError:
    HAS_OLLAMA_EMBEDDING = False
    OllamaEmbeddingClient = None
    OllamaEmbeddingModel = None

try:
    from .providers.hf.embedding import HuggingFaceEmbeddingClient, HuggingFaceEmbeddingModel

    HAS_HF_EMBEDDING = True
except ImportError:
    HAS_HF_EMBEDDING = False
    HuggingFaceEmbeddingClient = None
    HuggingFaceEmbeddingModel = None

# Expose what's available
__all__ = [
    "extract_tool_calls",
    "generate_json",
    "parse_json_response",
    "AudioClient",
    "AudioModel",
    "AudioSpec",
    "BaseAudioClient",
    "BaseImageClient",
    "BaseModelClient",
    "BaseSpeechClient",
    "FallbackClient",
    "FallbackExhaustedError",
    "GeminiImageSpec",
    "HuggingFaceAudioSpec",
    "HuggingFaceImageSpec",
    "HuggingFaceSpeechSpec",
    "ImageClient",
    "ImageModel",
    "ImageSpec",
    "Model",
    "ModelClient",
    "ModelSpec",
    "OpenAISpeechSpec",
    "SpeechClient",
    "SpeechModel",
    "SpeechSpec",
    "StreamChunk",
    "StreamingContentType",
    "available_audio_clients",
    "available_audio_models",
    "available_embedding_models",
    "available_image_clients",
    "available_image_models",
    "available_speech_clients",
    "available_speech_models",
    "available_text_clients",
    "available_text_models",
    "available_transcription_models",
    "resolve_audio_model_string",
    "resolve_default_text_model_enum",
    "resolve_image_model_enum",
    "resolve_image_model_string",
    "resolve_model_enum",
    "resolve_model_string",
    "resolve_speech_model_string",
    "BaseTranscriptionClient",
    "HuggingFaceTranscriptionSpec",
    "OpenAITranscriptionSpec",
    "TranscriptionClient",
    "TranscriptionModel",
    "TranscriptionSpec",
    "resolve_transcription_model_string",
    "BaseEmbeddingClient",
    "EmbeddingClient",
    "EmbeddingModel",
    "EmbeddingSpec",
    "OpenAIEmbeddingSpec",
    "OllamaEmbeddingSpec",
    "HuggingFaceEmbeddingSpec",
    "available_embedding_clients",
    "resolve_embedding_model_string",
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
if HAS_HF_AUDIO:
    __all__.extend(["HuggingFaceAudioClient", "HuggingFaceAudioModel"])
if HAS_HF_SPEECH:
    __all__.extend(["HuggingFaceSpeechClient", "HuggingFaceSpeechModel"])
if HAS_OPENAI_SPEECH:
    __all__.extend(["OpenAISpeechClient", "OpenAISpeechModel"])
if HAS_HF_TRANSCRIPTION:
    __all__.extend(["HuggingFaceTranscriptionClient", "HuggingFaceTranscriptionModel"])
if HAS_OPENAI_TRANSCRIPTION:
    __all__.extend(["OpenAITranscriptionClient", "OpenAITranscriptionModel"])
if HAS_OPENAI_EMBEDDING:
    __all__.extend(["OpenAIEmbeddingClient", "OpenAIEmbeddingModel"])
if HAS_OLLAMA_EMBEDDING:
    __all__.extend(["OllamaEmbeddingClient", "OllamaEmbeddingModel"])
if HAS_HF_EMBEDDING:
    __all__.extend(["HuggingFaceEmbeddingClient", "HuggingFaceEmbeddingModel"])


def available_embedding_clients() -> list[type[BaseEmbeddingClient]]:
    """Return client classes for all installed embedding providers, in display order."""
    clients: list[type[BaseEmbeddingClient]] = []
    if HAS_OLLAMA_EMBEDDING:
        clients.append(OllamaEmbeddingClient)
    if HAS_HF_EMBEDDING:
        clients.append(HuggingFaceEmbeddingClient)
    if HAS_OPENAI_EMBEDDING:
        clients.append(OpenAIEmbeddingClient)
    return clients


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
        clients.extend(
            [
                OpenAIClient,
                GeminiClient,
                LMStudioOpenAIClient,
                OllamaOpenAIClient,
                HFOpenAIClient,
                VLLMOpenAIClient,
                LlamaServerOpenAIClient,
                SGLangOpenAIClient,
            ]
        )
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


def available_audio_clients() -> list[type[BaseAudioClient]]:
    """Return client classes for all installed audio providers, in display order."""
    clients: list[type[BaseAudioClient]] = []
    if HAS_HF_AUDIO:
        clients.append(HuggingFaceAudioClient)
    return clients


def available_speech_clients() -> list:
    """Return client classes for all installed speech (TTS) providers, in display order."""
    clients = []
    if HAS_OPENAI_SPEECH:
        clients.append(OpenAISpeechClient)
    if HAS_HF_SPEECH:
        clients.append(HuggingFaceSpeechClient)
    return clients
