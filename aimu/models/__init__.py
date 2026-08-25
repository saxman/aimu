# Always available
from typing import TYPE_CHECKING

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
    ContextOverflowError,
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
    ModelConnectionError,
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
from ._internal.message_meta import (
    INERT_MESSAGE_KEYS,
    PROVENANCE_CONTINUATION,
    PROVENANCE_FINAL_ANSWER,
    PROVENANCE_KEY,
    PROVENANCE_PROACTIVE,
    encode_tool_call_arguments,
    strip_inert_keys,
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
    resolve_default_text_model,
    resolve_default_text_model_enum,
    resolve_model_enum,
    resolve_model_string,
)
from .speech_client import SpeechClient, resolve_speech_model_string
from .transcription_client import TranscriptionClient, resolve_transcription_model_string

from importlib import import_module as _import_module

from ._internal.factory import installed as _installed

# symbol name -> (module, third-party dependency probed for availability)
_LAZY_PROVIDER_SYMBOLS: dict[str, tuple[str, str]] = {}

if TYPE_CHECKING:  # pragma: no cover
    # Static-analysis-only bindings for names __getattr__ resolves at runtime.
    # PEP 562 lookup is invisible to anything reading the source without importing
    # it, so griffe (behind mkdocstrings) cannot collect these and the docs build
    # aborts on the first one -- being listed in __all__ is not enough, since there
    # is no assignment for a static reader to follow. These imports never execute,
    # so the lazy resolution below still owns runtime behaviour and the optional
    # dependencies stay uninstalled-safe. Type checkers get the same benefit.
    # `X as X` is the redundant-alias re-export convention (PEP 484): it tells ruff and
    # type checkers these are deliberate re-exports, which a plain import cannot convey
    # here because this module builds __all__ dynamically inside the HAS_* guards below.
    from .providers.anthropic import AnthropicClient as AnthropicClient
    from .providers.gemini.text import GeminiClient as GeminiClient
    from .providers.hf.embedding import HuggingFaceEmbeddingClient as HuggingFaceEmbeddingClient
    from .providers.hf.text import HuggingFaceClient as HuggingFaceClient
    from .providers.llamacpp import LlamaCppClient as LlamaCppClient
    from .providers.ollama import OllamaClient as OllamaClient
    from .providers.ollama import OllamaEmbeddingClient as OllamaEmbeddingClient
    from .providers.openai.embedding import OpenAIEmbeddingClient as OpenAIEmbeddingClient
    from .providers.openai.text import OpenAIClient as OpenAIClient
    from .providers.openai_compat import HFOpenAIClient as HFOpenAIClient
    from .providers.openai_compat import LlamaServerOpenAIClient as LlamaServerOpenAIClient
    from .providers.openai_compat import LMStudioOpenAIClient as LMStudioOpenAIClient
    from .providers.openai_compat import OllamaOpenAIClient as OllamaOpenAIClient
    from .providers.openai_compat import OMLXOpenAIClient as OMLXOpenAIClient
    from .providers.openai_compat import OpenAICompatClient as OpenAICompatClient
    from .providers.openai_compat import SGLangOpenAIClient as SGLangOpenAIClient
    from .providers.openai_compat import VLLMOpenAIClient as VLLMOpenAIClient


def _register(module: str, requires: str, *symbols: str) -> None:
    for symbol in symbols:
        _LAZY_PROVIDER_SYMBOLS[symbol] = (module, requires)


_register("aimu.models.providers.hf.text", "transformers", "HuggingFaceClient", "HuggingFaceModel", "ToolCallFormat")
_register("aimu.models.providers.ollama", "ollama", "OllamaClient", "OllamaModel")
_register("aimu.models.providers.ollama", "ollama", "OllamaEmbeddingClient", "OllamaEmbeddingModel")
_register("aimu.models.providers.anthropic", "anthropic", "AnthropicClient", "AnthropicModel")
_register("aimu.models.providers.openai.text", "openai", "OpenAIClient", "OpenAIModel")
_register("aimu.models.providers.gemini.text", "openai", "GeminiClient", "GeminiModel")
_register(
    "aimu.models.providers.openai_compat",
    "openai",
    "OpenAICompatClient",
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
    "OMLXOpenAIClient",
    "OMLXOpenAIModel",
)
_register("aimu.models.providers.llamacpp", "llama_cpp", "LlamaCppClient", "LlamaCppModel")
_register("aimu.models.providers.hf.image", "diffusers", "HuggingFaceImageClient", "HuggingFaceImageModel")
_register("aimu.models.providers.gemini.image", "google.genai", "GeminiImageClient", "GeminiImageModel")
_register("aimu.models.providers.hf.audio", "soundfile", "HuggingFaceAudioClient", "HuggingFaceAudioModel")
_register("aimu.models.providers.hf.speech", "soundfile", "HuggingFaceSpeechClient", "HuggingFaceSpeechModel")
_register("aimu.models.providers.openai.speech", "openai", "OpenAISpeechClient", "OpenAISpeechModel")
_register(
    "aimu.models.providers.hf.transcription",
    "transformers",
    "HuggingFaceTranscriptionClient",
    "HuggingFaceTranscriptionModel",
)
_register(
    "aimu.models.providers.openai.transcription",
    "openai",
    "OpenAITranscriptionClient",
    "OpenAITranscriptionModel",
)
_register("aimu.models.providers.openai.embedding", "openai", "OpenAIEmbeddingClient", "OpenAIEmbeddingModel")
_register(
    "aimu.models.providers.hf.embedding",
    "sentence_transformers",
    "HuggingFaceEmbeddingClient",
    "HuggingFaceEmbeddingModel",
)

# HAS_* flags are ordinary module globals, answered by find_spec rather than by a
# completed import. They must be real globals, not __getattr__ entries: the 15
# `if HAS_X: __all__.extend([...])` blocks further down this module, and the
# available_*_clients() bodies, read them as bare names -- and PEP 562 __getattr__
# fires only for attribute access from *outside* a module, never for a bare-name
# lookup inside it. Routing them through __getattr__ raises NameError at import.
#
# Note the semantic change: these say "installed", where the previous try/except said
# "imported cleanly". A dependency that is present but broken now reports True and
# raises at first use instead of silently reporting False -- the failure moves to
# where it is actionable.
HAS_HF = _installed("transformers")
HAS_OLLAMA = _installed("ollama")
HAS_ANTHROPIC = _installed("anthropic")
HAS_OPENAI_COMPAT = _installed("openai")
HAS_LLAMACPP = _installed("llama_cpp")
HAS_HF_IMAGE = _installed("diffusers")
HAS_GEMINI_IMAGE = _installed("google.genai")
HAS_HF_AUDIO = _installed("soundfile")
HAS_HF_SPEECH = _installed("soundfile")
HAS_OPENAI_SPEECH = _installed("openai")
HAS_HF_TRANSCRIPTION = _installed("transformers")
HAS_OPENAI_TRANSCRIPTION = _installed("openai")
HAS_OPENAI_EMBEDDING = _installed("openai")
HAS_OLLAMA_EMBEDDING = _installed("ollama")
HAS_HF_EMBEDDING = _installed("sentence_transformers")


def __getattr__(name: str):
    """Resolve provider symbols on first access (PEP 562).

    A symbol whose dependency is absent evaluates to ``None``, preserving the contract
    the previous ``except ImportError`` blocks established. A symbol whose dependency is
    installed but fails to import raises, with the original cause chained.
    """
    entry = _LAZY_PROVIDER_SYMBOLS.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, requires = entry
    if not _installed(requires):
        return None
    try:
        return getattr(_import_module(module_name), name)
    except ImportError as exc:
        raise ImportError(
            f"{name} could not be loaded from {module_name!r} ({requires!r} is installed): {exc}"
        ) from exc


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_PROVIDER_SYMBOLS})


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
    "ModelConnectionError",
    "ContextOverflowError",
    "ModelSpec",
    "OpenAISpeechSpec",
    "SpeechClient",
    "SpeechModel",
    "SpeechSpec",
    "StreamChunk",
    "StreamingContentType",
    "INERT_MESSAGE_KEYS",
    "PROVENANCE_KEY",
    "PROVENANCE_CONTINUATION",
    "PROVENANCE_FINAL_ANSWER",
    "PROVENANCE_PROACTIVE",
    "encode_tool_call_arguments",
    "strip_inert_keys",
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
    "resolve_default_text_model",
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
            "OMLXOpenAIClient",
            "OMLXOpenAIModel",
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
    """Return client classes for all installed embedding providers, in display order.

    Explicit discovery, so importing the installed providers is the point rather than a
    cost to avoid.
    """
    from .embedding_client import _entries

    entries = {e.prefix: e for e in _entries()}
    order = ["ollama", "hf", "openai"]
    return [entries[prefix].load()[1] for prefix in order if entries[prefix].available]


def available_text_clients() -> list[type[BaseModelClient]]:
    """Return client classes for all installed text providers, in display order."""
    from .model_client import _TEXT_PROVIDERS

    return [entry.load()[1] for entry in _TEXT_PROVIDERS if entry.available]


def available_image_clients() -> list[type[BaseImageClient]]:
    """Return client classes for all installed image providers, in display order."""
    from .image_client import _entries

    return [entry.load()[1] for entry in _entries() if entry.available]


def available_audio_clients() -> list[type[BaseAudioClient]]:
    """Return client classes for all installed audio providers, in display order."""
    from .audio_client import _entries

    return [entry.load()[1] for entry in _entries() if entry.available]


def available_speech_clients() -> list:
    """Return client classes for all installed speech (TTS) providers, in display order."""
    from .speech_client import _entries

    entries = {e.prefix: e for e in _entries()}
    order = ["openai", "hf"]
    return [entries[prefix].load()[1] for prefix in order if entries[prefix].available]
