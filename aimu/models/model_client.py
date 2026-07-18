from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Union

from .base import AdHocModel, BaseModelClient, Model, ModelSpec, StreamChunk
from ._internal.model_defaults import available_text_models, resolve_default_text_model_enum
from ._internal.model_string import parse_model_string

log = logging.getLogger(__name__)

__all__ = [
    "ModelClient",
    "available_text_models",
    "resolve_default_text_model_enum",
    "resolve_model_enum",
    "resolve_model_string",
]

# --- Optional provider imports ---

try:
    from .providers.ollama import OllamaClient, OllamaModel

    _HAS_OLLAMA = True
except ImportError:
    _HAS_OLLAMA = False
    OllamaClient = None  # type: ignore[assignment,misc]
    OllamaModel = None  # type: ignore[assignment,misc]

try:
    from .providers.hf.text import HuggingFaceClient, HuggingFaceModel

    _HAS_HF = True
except ImportError:
    _HAS_HF = False
    HuggingFaceClient = None  # type: ignore[assignment,misc]
    HuggingFaceModel = None  # type: ignore[assignment,misc]

try:
    from .providers.anthropic import AnthropicClient, AnthropicModel

    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False
    AnthropicClient = None  # type: ignore[assignment,misc]
    AnthropicModel = None  # type: ignore[assignment,misc]

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

    _HAS_OPENAI_COMPAT = True
except ImportError:
    _HAS_OPENAI_COMPAT = False
    OpenAIClient = GeminiClient = LMStudioOpenAIClient = OllamaOpenAIClient = None  # type: ignore[assignment,misc]
    HFOpenAIClient = VLLMOpenAIClient = LlamaServerOpenAIClient = SGLangOpenAIClient = None  # type: ignore[assignment,misc]
    OpenAICompatClient = None  # type: ignore[assignment,misc]
    OpenAIModel = GeminiModel = LMStudioOpenAIModel = OllamaOpenAIModel = None  # type: ignore[assignment,misc]
    HFOpenAIModel = VLLMOpenAIModel = LlamaServerOpenAIModel = SGLangOpenAIModel = None  # type: ignore[assignment,misc]

try:
    from .providers.llamacpp import LlamaCppClient, LlamaCppModel

    _HAS_LLAMACPP = True
except ImportError:
    _HAS_LLAMACPP = False
    LlamaCppClient = None  # type: ignore[assignment,misc]
    LlamaCppModel = None  # type: ignore[assignment,misc]


def _provider_registry() -> dict[str, tuple[Any, Any]]:
    """Map ``"provider"`` strings to ``(ModelEnum, ClientClass)`` pairs.

    Only available providers (whose optional dependency is installed) appear in the table.
    Used by :func:`resolve_model_string` to look up a model by string identifier.
    """
    registry: dict[str, tuple[Any, Any]] = {}
    if _HAS_OLLAMA:
        registry["ollama"] = (OllamaModel, OllamaClient)
    if _HAS_HF:
        registry["hf"] = (HuggingFaceModel, HuggingFaceClient)
    if _HAS_ANTHROPIC:
        registry["anthropic"] = (AnthropicModel, AnthropicClient)
    if _HAS_OPENAI_COMPAT:
        registry["openai"] = (OpenAIModel, OpenAIClient)
        registry["gemini"] = (GeminiModel, GeminiClient)
        registry["lmstudio"] = (LMStudioOpenAIModel, LMStudioOpenAIClient)
        registry["ollama-openai"] = (OllamaOpenAIModel, OllamaOpenAIClient)
        registry["hf-openai"] = (HFOpenAIModel, HFOpenAIClient)
        registry["vllm"] = (VLLMOpenAIModel, VLLMOpenAIClient)
        registry["llamaserver"] = (LlamaServerOpenAIModel, LlamaServerOpenAIClient)
        registry["sglang"] = (SGLangOpenAIModel, SGLangOpenAIClient)
    if _HAS_LLAMACPP:
        registry["llamacpp"] = (LlamaCppModel, LlamaCppClient)
    return registry


def resolve_model_string(model_str: str) -> Model:
    """Look up a provider model enum from a ``"provider:model_id"`` string.

    Examples::

        resolve_model_string("anthropic:claude-sonnet-4-6")
        resolve_model_string("ollama:qwen3.5:9b")          # colons in the id are fine
        resolve_model_string("openai:gpt-4o-mini")

    Raises ``ValueError`` with the list of valid ids when the provider or id is unknown.
    """
    if ":" not in model_str:
        raise ValueError(
            f"Model string must be in 'provider:model_id' form, got: {model_str!r}. "
            f"Available providers: {sorted(_provider_registry())}"
        )
    provider, _, model_id = model_str.partition(":")
    registry = _provider_registry()
    if provider not in registry:
        raise ValueError(
            f"Unknown provider {provider!r}. Available providers (with installed deps): {sorted(registry)}"
        )
    model_enum, _ = registry[provider]
    for member in model_enum:
        if member.value == model_id:
            return member
    available = sorted(m.value for m in model_enum)
    raise ValueError(f"Provider {provider!r} has no model id {model_id!r}. Available: {available}")


_BASE_URL_PROVIDERS = {"llamaserver", "lmstudio", "vllm", "hf-openai", "sglang", "ollama-openai"}
_GENERIC_COMPAT_PROVIDER = "openai-compat"
_CAPABILITY_FLAGS = {"tools", "thinking", "vision", "audio", "structured"}


@dataclass
class ResolvedModel:
    """Outcome of resolving an (extended) model string.

    ``model`` is a ``Model`` enum member for a known id, or an ``AdHocModel`` for an id
    not in any provider enum. ``base_url`` is an endpoint override (or ``None`` to use the
    provider default). ``provider`` is the parsed prefix, used to route an ad-hoc model to
    its client class.
    """

    model: Any
    provider: str
    base_url: Optional[str]


def _build_adhoc_spec(model_id: str, flags: tuple[str, ...]) -> ModelSpec:
    unknown = sorted(f for f in flags if f not in _CAPABILITY_FLAGS)
    if unknown:
        raise ValueError(f"Unknown capability flag(s) {unknown}. Valid flags: {sorted(_CAPABILITY_FLAGS)}.")
    chosen = set(flags)
    return ModelSpec(
        id=model_id,
        tools="tools" in chosen,
        thinking="thinking" in chosen,
        vision="vision" in chosen,
        audio="audio" in chosen,
        structured_output="structured" in chosen,
    )


def resolve_model(model_str: str) -> ResolvedModel:
    """Resolve an extended ``provider:model_id[@base_url][;flags]`` string.

    Known ids resolve to their ``Model`` enum member (with an optional ``base_url``
    override for the OpenAI-compatible local-server providers). Ids not in any enum resolve
    to an ``AdHocModel`` whose capabilities come from ``flags``; these are allowed only for
    the base-url providers and the generic ``openai-compat`` prefix (which requires a
    ``base_url``). See ``docs/superpowers/specs/2026-07-17-model-string-base-url-design.md``.
    """
    parsed = parse_model_string(model_str)
    provider, model_id, base_url, flags = parsed.provider, parsed.model_id, parsed.base_url, parsed.flags

    if provider == _GENERIC_COMPAT_PROVIDER:
        if not _HAS_OPENAI_COMPAT:
            raise ImportError("The 'openai-compat' provider requires the openai-compatible extra.")
        if base_url is None:
            raise ValueError(f"Provider {provider!r} requires an endpoint: use 'openai-compat:<model_id>@<base_url>'.")
        return ResolvedModel(AdHocModel(_build_adhoc_spec(model_id, flags)), provider, base_url)

    registry = _provider_registry()
    if provider not in registry:
        available = sorted(list(registry) + [_GENERIC_COMPAT_PROVIDER])
        raise ValueError(f"Unknown provider {provider!r}. Available providers (with installed deps): {available}")

    if base_url is not None and provider not in _BASE_URL_PROVIDERS:
        supported = sorted(_BASE_URL_PROVIDERS | {_GENERIC_COMPAT_PROVIDER})
        raise ValueError(f"Provider {provider!r} does not accept an @base_url. Supported: {supported}.")

    enum_cls, _client_cls = registry[provider]
    match = next((member for member in enum_cls if member.value == model_id), None)
    if match is not None:
        if flags:
            raise ValueError(
                f"Capability flags are not allowed with the known model id {model_id!r}; it already "
                "declares its capabilities. Use a different id to define an ad-hoc model."
            )
        return ResolvedModel(match, provider, base_url)

    if provider not in _BASE_URL_PROVIDERS:
        available = sorted(member.value for member in enum_cls)
        raise ValueError(f"Provider {provider!r} has no model id {model_id!r}. Available: {available}")
    return ResolvedModel(AdHocModel(_build_adhoc_spec(model_id, flags)), provider, base_url)


def _sync_compat_client(provider: str):
    """The sync OpenAI-compat client class for an ad-hoc model's provider prefix."""
    if provider == _GENERIC_COMPAT_PROVIDER:
        return OpenAICompatClient
    return _provider_registry()[provider][1]


def resolve_model_enum(model: Union[Model, str]) -> Model:
    """Resolve a text model to a ``Model`` enum member.

    Accepts, in order:

    - a ``Model`` enum member (returned unchanged);
    - a ``"provider:model_id"`` string (delegates to :func:`resolve_model_string`);
    - a bare enum-member *name* (e.g. ``"QWEN_3_8B"``), looked up across every installed
      provider's ``Model`` enum.

    A bare name that ships under more than one provider's enum (common for text, where the same
    id is offered by many providers) is **ambiguous**. Rather than blindly picking the
    highest-priority provider, ambiguity is resolved the way the omitted-``model`` default is:
    prefer a provider where the model is *actually available locally* (running Ollama →
    cached HuggingFace → reachable local OpenAI-compat server, tool-capable first), logged at
    WARNING. If the ambiguous name is not available under any provider, a ``ValueError`` is
    raised listing the ``"provider:model_id"`` options; picking a provider for a model that
    isn't even loadable would be a blind guess. This availability tiebreaker only runs on the
    ambiguous path; enum / ``"provider:model_id"`` / unambiguous-name inputs do no I/O.

    Parallel to :func:`aimu.resolve_image_model_enum` for the image modality (which has no
    local-availability notion, so it always raises on the rare ambiguity).
    """
    if isinstance(model, Model):
        return model
    if not isinstance(model, str):
        raise TypeError(f"Expected a Model enum member or a string, got {type(model).__name__}.")
    if ":" in model:
        return resolve_model_string(model)

    registry = _provider_registry()
    matches = [
        (prefix, enum_cls[model]) for prefix, (enum_cls, _client) in registry.items() if model in enum_cls.__members__
    ]
    if len(matches) == 1:
        return matches[0][1]
    if not matches:
        names = sorted({name for _, (enum_cls, _client) in registry.items() for name in enum_cls.__members__})
        raise ValueError(f"Unknown model name {model!r}. Pass a 'provider:model_id' string or one of: {names}")

    # Ambiguous bare name: disambiguate by local availability (same probe order as the
    # omitted-model default), tool-capable first. ``available_text_models()`` already returns
    # members in provider-priority order; the I/O happens only here, on the ambiguous path.
    available = [m for m in available_text_models() if m.name == model]
    if available:
        picked = next((m for m in available if getattr(m, "supports_tools", False)), available[0])
        log.warning(
            "aimu: bare model name %r is ambiguous across providers; resolved to %r (locally available). "
            "Pass a 'provider:model_id' string to pin a specific provider.",
            model,
            picked,
        )
        return picked

    providers = ", ".join(prefix for prefix, _ in matches)
    options = ", ".join(f"{prefix}:{member.value}" for prefix, member in matches)
    raise ValueError(
        f"Model name {model!r} is ambiguous across providers ({providers}) and is not available "
        f"locally under any of them. Disambiguate with a 'provider:model_id' string, e.g. one of: {options}"
    )


class ModelClient(BaseModelClient):
    """Public factory for provider-backed model clients.

    Accepts either a provider's ``Model`` enum member or a ``"provider:model_id"`` string::

        from aimu.models import ModelClient, OllamaModel

        # Enum form
        client = ModelClient(OllamaModel.QWEN_3_8B)

        # String form (no enum import needed)
        client = ModelClient("anthropic:claude-sonnet-4-6")
        client = ModelClient("ollama:qwen3.5:9b")

    Provider-specific kwargs are forwarded to the concrete client::

        ModelClient(LlamaCppModel.QWEN_3_8B, model_path="/path/to/model.gguf")
        ModelClient(OllamaModel.LLAMA_3_1_8B, model_keep_alive_seconds=120)
        ModelClient(LMStudioOpenAIModel.LLAMA_3_2_3B, base_url="http://myserver:1234/v1")
    """

    def __init__(self, model: Union[Model, ModelSpec, str], **kwargs: Any) -> None:
        if isinstance(model, str):
            resolved = resolve_model(model)
            if resolved.base_url is not None:
                kwargs["base_url"] = resolved.base_url
            if isinstance(resolved.model, AdHocModel):
                client_cls = _sync_compat_client(resolved.provider)
                self._client = client_cls(resolved.model, **kwargs)
                self.model = self._client.model
                self.model_kwargs = self._client.model_kwargs
                self.default_generate_kwargs = self._client.default_generate_kwargs
                return
            model = resolved.model
        elif isinstance(model, ModelSpec):
            raise TypeError(
                "Pass a Model enum member (e.g. OllamaModel.QWEN_3_8B) or a "
                "'provider:model_id' string. ModelSpec is the value type held by enum members."
            )

        # Dispatch to concrete client by model enum type.
        # isinstance checks are guarded by _HAS_* flags (short-circuit and) so
        # that a None model class from a missing optional dep never reaches isinstance().
        if _HAS_OLLAMA and isinstance(model, OllamaModel):
            self._client: BaseModelClient = OllamaClient(model, **kwargs)
        elif _HAS_HF and isinstance(model, HuggingFaceModel):
            self._client = HuggingFaceClient(model, **kwargs)
        elif _HAS_ANTHROPIC and isinstance(model, AnthropicModel):
            self._client = AnthropicClient(model, **kwargs)
        elif _HAS_OPENAI_COMPAT and isinstance(model, OpenAIModel):
            self._client = OpenAIClient(model, **kwargs)
        elif _HAS_OPENAI_COMPAT and isinstance(model, GeminiModel):
            self._client = GeminiClient(model, **kwargs)
        elif _HAS_OPENAI_COMPAT and isinstance(model, LMStudioOpenAIModel):
            self._client = LMStudioOpenAIClient(model, **kwargs)
        elif _HAS_OPENAI_COMPAT and isinstance(model, OllamaOpenAIModel):
            self._client = OllamaOpenAIClient(model, **kwargs)
        elif _HAS_OPENAI_COMPAT and isinstance(model, HFOpenAIModel):
            self._client = HFOpenAIClient(model, **kwargs)
        elif _HAS_OPENAI_COMPAT and isinstance(model, VLLMOpenAIModel):
            self._client = VLLMOpenAIClient(model, **kwargs)
        elif _HAS_OPENAI_COMPAT and isinstance(model, LlamaServerOpenAIModel):
            self._client = LlamaServerOpenAIClient(model, **kwargs)
        elif _HAS_OPENAI_COMPAT and isinstance(model, SGLangOpenAIModel):
            self._client = SGLangOpenAIClient(model, **kwargs)
        elif _HAS_LLAMACPP and isinstance(model, LlamaCppModel):
            self._client = LlamaCppClient(model, **kwargs)
        else:
            raise ValueError(
                f"No available client for model type {type(model).__name__!r}. "
                "Ensure the required optional dependency is installed (e.g. pip install aimu[ollama])."
            )

        # Mirror non-mutable attributes so callers can read them directly on this wrapper.
        # super().__init__() is intentionally not called; it would reset inner client state.
        self.model = self._client.model
        self.model_kwargs = self._client.model_kwargs
        self.default_generate_kwargs = self._client.default_generate_kwargs

    # --- Delegate mutable state to inner client so both stay in sync ---

    @property
    def messages(self) -> list[dict]:
        return self._client.messages

    @messages.setter
    def messages(self, value: list[dict]) -> None:
        self._client.messages = value

    @property
    def tools(self) -> list:
        return self._client.tools

    @tools.setter
    def tools(self, value: list) -> None:
        self._client.tools = value

    @property
    def system_message(self) -> Optional[str]:
        return self._client.system_message

    @system_message.setter
    def system_message(self, message: Optional[str]) -> None:
        self._client.system_message = message

    @property
    def last_thinking(self) -> str:
        return self._client.last_thinking

    @last_thinking.setter
    def last_thinking(self, value: str) -> None:
        self._client.last_thinking = value

    @property
    def last_usage(self) -> Optional[dict]:
        """Token usage of the most recent non-streaming response, or ``None``.

        Shape: ``{"input_tokens", "output_tokens", "total_tokens"}``. ``None`` after a
        streaming call or when the provider/server did not report usage.
        """
        return self._client.last_usage

    @last_usage.setter
    def last_usage(self, value: Optional[dict]) -> None:
        self._client.last_usage = value

    @property
    def last_structured(self):
        """Validated object from the most recent ``schema=`` call, or ``None``.

        For a streamed structured-output call it is populated only after the stream is
        fully consumed (mirrors :attr:`last_usage`).
        """
        return self._client.last_structured

    @last_structured.setter
    def last_structured(self, value) -> None:
        self._client.last_structured = value

    def reset(self, system_message: Optional[str] = "__keep__") -> None:
        self._client.reset(system_message)

    # --- Implement abstract _chat / _generate by delegating to inner client ---

    def _generate(
        self,
        prompt: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
        audio: Optional[list] = None,
        response_format: Optional[dict] = None,
    ) -> Union[str, Iterator[StreamChunk]]:
        # Forward response_format only when set: it's non-None only on the native structured
        # path, where the inner client is structured-capable and accepts the param. Parse-path
        # inner clients (HuggingFace, LlamaCpp) never receive it.
        extra = {"response_format": response_format} if response_format is not None else {}
        return self._client._generate(prompt, generate_kwargs, stream=stream, images=images, audio=audio, **extra)

    def _chat(
        self,
        user_message: Optional[str] = None,
        generate_kwargs: Optional[dict[str, Any]] = None,
        use_tools: bool = True,
        stream: bool = False,
        images: Optional[list] = None,
        audio: Optional[list] = None,
        response_format: Optional[dict] = None,
    ) -> Union[str, Iterator[StreamChunk]]:
        extra = {"response_format": response_format} if response_format is not None else {}
        return self._client._chat(
            user_message, generate_kwargs, use_tools=use_tools, stream=stream, images=images, audio=audio, **extra
        )

    def _update_generate_kwargs(self, generate_kwargs: Optional[dict[str, Any]] = None) -> dict:
        return self._client._update_generate_kwargs(generate_kwargs)
