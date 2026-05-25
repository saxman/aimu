from __future__ import annotations

from typing import Any, Iterator, Optional, Union

from .base import BaseModelClient, Model, ModelSpec, StreamChunk

# --- Optional provider imports ---

try:
    from .ollama import OllamaClient, OllamaModel

    _HAS_OLLAMA = True
except ImportError:
    _HAS_OLLAMA = False
    OllamaClient = None  # type: ignore[assignment,misc]
    OllamaModel = None  # type: ignore[assignment,misc]

try:
    from .hf import HuggingFaceClient, HuggingFaceModel

    _HAS_HF = True
except ImportError:
    _HAS_HF = False
    HuggingFaceClient = None  # type: ignore[assignment,misc]
    HuggingFaceModel = None  # type: ignore[assignment,misc]

try:
    from .anthropic import AnthropicClient, AnthropicModel

    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False
    AnthropicClient = None  # type: ignore[assignment,misc]
    AnthropicModel = None  # type: ignore[assignment,misc]

try:
    from .openai_compat import (
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

    _HAS_OPENAI_COMPAT = True
except ImportError:
    _HAS_OPENAI_COMPAT = False
    OpenAIClient = GeminiClient = LMStudioOpenAIClient = OllamaOpenAIClient = None  # type: ignore[assignment,misc]
    HFOpenAIClient = VLLMOpenAIClient = LlamaServerOpenAIClient = SGLangOpenAIClient = None  # type: ignore[assignment,misc]
    OpenAIModel = GeminiModel = LMStudioOpenAIModel = OllamaOpenAIModel = None  # type: ignore[assignment,misc]
    HFOpenAIModel = VLLMOpenAIModel = LlamaServerOpenAIModel = SGLangOpenAIModel = None  # type: ignore[assignment,misc]

try:
    from .llamacpp import LlamaCppClient, LlamaCppModel

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
            model = resolve_model_string(model)
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
        # super().__init__() is intentionally not called — it would reset inner client state.
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
    def mcp_client(self) -> Any:
        return self._client.mcp_client

    @mcp_client.setter
    def mcp_client(self, value: Any) -> None:
        self._client.mcp_client = value

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

    def reset(self, system_message: Optional[str] = "__keep__") -> None:
        self._client.reset(system_message)

    # --- Implement abstract _chat / _generate by delegating to inner client ---

    def _generate(
        self,
        prompt: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        include_thinking: bool = True,
    ) -> Union[str, Iterator[StreamChunk]]:
        return self._client._generate(prompt, generate_kwargs, stream=stream, include_thinking=include_thinking)

    def _chat(
        self,
        user_message: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        use_tools: bool = True,
        stream: bool = False,
        images: Optional[list] = None,
    ) -> Union[str, Iterator[StreamChunk]]:
        return self._client._chat(user_message, generate_kwargs, use_tools=use_tools, stream=stream, images=images)

    def _update_generate_kwargs(self, generate_kwargs: Optional[dict[str, Any]] = None) -> dict:
        return self._client._update_generate_kwargs(generate_kwargs)
