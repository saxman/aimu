"""Async ``ModelClient`` factory mirroring ``aimu.models.model_client.ModelClient``.

Accepts either a provider ``Model`` enum member, a ``"provider:model_id"`` string,
or — for in-process providers (HuggingFace, LlamaCpp) — an existing sync client to
wrap (so model weights are loaded only once). See Decision 7 in the plan.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Iterable, Optional, Union

from aimu.models.base import Model, ModelSpec, StreamChunk, StreamingContentType
from aimu.models.model_client import resolve_model_string

from ._base import AsyncBaseModelClient

# --- Optional provider imports (mirror the sync registry) ---

try:
    from aimu.models.providers.ollama import OllamaModel

    from .providers.ollama import AsyncOllamaClient

    _HAS_OLLAMA = True
except Exception:
    _HAS_OLLAMA = False
    AsyncOllamaClient = None  # type: ignore[assignment,misc]
    OllamaModel = None  # type: ignore[assignment,misc]

try:
    from aimu.models.providers.hf.text import HuggingFaceClient, HuggingFaceModel

    from .providers.hf.text import AsyncHuggingFaceClient

    _HAS_HF = True
except Exception:
    _HAS_HF = False
    AsyncHuggingFaceClient = None  # type: ignore[assignment,misc]
    HuggingFaceClient = None  # type: ignore[assignment,misc]
    HuggingFaceModel = None  # type: ignore[assignment,misc]

try:
    from aimu.models.providers.anthropic import AnthropicModel

    from .providers.anthropic import AsyncAnthropicClient

    _HAS_ANTHROPIC = True
except Exception:
    _HAS_ANTHROPIC = False
    AsyncAnthropicClient = None  # type: ignore[assignment,misc]
    AnthropicModel = None  # type: ignore[assignment,misc]

try:
    from aimu.models.providers.gemini.text import GeminiModel
    from aimu.models.providers.openai.text import OpenAIModel
    from aimu.models.providers.openai_compat import (
        HFOpenAIModel,
        LlamaServerOpenAIModel,
        LMStudioOpenAIModel,
        OllamaOpenAIModel,
        SGLangOpenAIModel,
        VLLMOpenAIModel,
    )

    from .providers.gemini.text import AsyncGeminiClient
    from .providers.openai.text import AsyncOpenAIClient
    from .providers.openai_compat import (
        AsyncHFOpenAIClient,
        AsyncLlamaServerOpenAIClient,
        AsyncLMStudioOpenAIClient,
        AsyncOllamaOpenAIClient,
        AsyncSGLangOpenAIClient,
        AsyncVLLMOpenAIClient,
    )

    _HAS_OPENAI_COMPAT = True
except Exception:
    _HAS_OPENAI_COMPAT = False
    AsyncOpenAIClient = AsyncGeminiClient = AsyncLMStudioOpenAIClient = AsyncOllamaOpenAIClient = None  # type: ignore[assignment,misc]
    AsyncHFOpenAIClient = AsyncVLLMOpenAIClient = AsyncLlamaServerOpenAIClient = AsyncSGLangOpenAIClient = None  # type: ignore[assignment,misc]
    OpenAIModel = GeminiModel = LMStudioOpenAIModel = OllamaOpenAIModel = None  # type: ignore[assignment,misc]
    HFOpenAIModel = VLLMOpenAIModel = LlamaServerOpenAIModel = SGLangOpenAIModel = None  # type: ignore[assignment,misc]

try:
    from aimu.models.providers.llamacpp import LlamaCppClient, LlamaCppModel

    from .providers.llamacpp import AsyncLlamaCppClient

    _HAS_LLAMACPP = True
except Exception:
    _HAS_LLAMACPP = False
    AsyncLlamaCppClient = None  # type: ignore[assignment,misc]
    LlamaCppClient = None  # type: ignore[assignment,misc]
    LlamaCppModel = None  # type: ignore[assignment,misc]


_IN_PROCESS_MODEL_GUIDANCE = (
    "In-process providers (HuggingFace, LlamaCpp) load model weights into memory; "
    "constructing both a sync and async client for the same model would load weights twice. "
    "Build a sync client first and pass it to aio.client():\n"
    "    sync_client = aimu.client({model})\n"
    "    async_client = aio.client(sync_client)"
)


class AsyncModelClient(AsyncBaseModelClient):
    """Public factory for async provider-backed model clients.

    Accepts a provider ``Model`` enum member, a ``"provider:model_id"`` string, or
    — for in-process providers — an existing sync client to wrap.

    Examples::

        # Cloud providers (separate sync/async clients are cheap)
        client = AsyncModelClient("anthropic:claude-sonnet-4-6")
        client = AsyncModelClient(OllamaModel.QWEN_3_8B)

        # In-process providers — wrap an existing sync client to share weights
        sync_client = aimu.client(HuggingFaceModel.LLAMA_70B)
        async_client = AsyncModelClient(sync_client)
    """

    def __init__(self, model: Union[Model, ModelSpec, str, Any], **kwargs: Any) -> None:
        # In-process wrapping path: passed an existing sync client.
        if _HAS_HF and HuggingFaceClient is not None and isinstance(model, HuggingFaceClient):
            self._client: AsyncBaseModelClient = AsyncHuggingFaceClient(model, **kwargs)
        elif _HAS_LLAMACPP and LlamaCppClient is not None and isinstance(model, LlamaCppClient):
            self._client = AsyncLlamaCppClient(model, **kwargs)
        else:
            # Normal construction path: model enum or string.
            if isinstance(model, str):
                model = resolve_model_string(model)
            elif isinstance(model, ModelSpec):
                raise TypeError(
                    "Pass a Model enum member or a 'provider:model_id' string. "
                    "ModelSpec is the value type held by enum members."
                )

            # Guard in-process model types: refuse direct construction; point to wrapping pattern.
            if _HAS_HF and HuggingFaceModel is not None and isinstance(model, HuggingFaceModel):
                raise ValueError(_IN_PROCESS_MODEL_GUIDANCE.format(model=f"HuggingFaceModel.{model.name}"))
            if _HAS_LLAMACPP and LlamaCppModel is not None and isinstance(model, LlamaCppModel):
                raise ValueError(_IN_PROCESS_MODEL_GUIDANCE.format(model=f"LlamaCppModel.{model.name}"))

            if _HAS_OLLAMA and isinstance(model, OllamaModel):
                self._client = AsyncOllamaClient(model, **kwargs)
            elif _HAS_ANTHROPIC and isinstance(model, AnthropicModel):
                self._client = AsyncAnthropicClient(model, **kwargs)
            elif _HAS_OPENAI_COMPAT and isinstance(model, OpenAIModel):
                self._client = AsyncOpenAIClient(model, **kwargs)
            elif _HAS_OPENAI_COMPAT and isinstance(model, GeminiModel):
                self._client = AsyncGeminiClient(model, **kwargs)
            elif _HAS_OPENAI_COMPAT and isinstance(model, LMStudioOpenAIModel):
                self._client = AsyncLMStudioOpenAIClient(model, **kwargs)
            elif _HAS_OPENAI_COMPAT and isinstance(model, OllamaOpenAIModel):
                self._client = AsyncOllamaOpenAIClient(model, **kwargs)
            elif _HAS_OPENAI_COMPAT and isinstance(model, HFOpenAIModel):
                self._client = AsyncHFOpenAIClient(model, **kwargs)
            elif _HAS_OPENAI_COMPAT and isinstance(model, VLLMOpenAIModel):
                self._client = AsyncVLLMOpenAIClient(model, **kwargs)
            elif _HAS_OPENAI_COMPAT and isinstance(model, LlamaServerOpenAIModel):
                self._client = AsyncLlamaServerOpenAIClient(model, **kwargs)
            elif _HAS_OPENAI_COMPAT and isinstance(model, SGLangOpenAIModel):
                self._client = AsyncSGLangOpenAIClient(model, **kwargs)
            else:
                raise ValueError(
                    f"No available async client for model type {type(model).__name__!r}. "
                    "Ensure the required optional dependency is installed."
                )

        # Mirror attributes (super().__init__ would clobber inner client state).
        self.model = self._client.model
        self.model_kwargs = self._client.model_kwargs
        self.default_generate_kwargs = self._client.default_generate_kwargs

    # --- Delegate mutable state to inner client ---

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
        """Token usage of the most recent non-streaming response, or ``None``."""
        return self._client.last_usage

    @last_usage.setter
    def last_usage(self, value: Optional[dict]) -> None:
        self._client.last_usage = value

    @property
    def concurrent_tool_calls(self) -> bool:
        return self._client.concurrent_tool_calls

    @concurrent_tool_calls.setter
    def concurrent_tool_calls(self, value: bool) -> None:
        self._client.concurrent_tool_calls = value

    @property
    def tool_context_deps(self) -> Any:
        return getattr(self._client, "tool_context_deps", None)

    @tool_context_deps.setter
    def tool_context_deps(self, value: Any) -> None:
        self._client.tool_context_deps = value

    def reset(self, system_message: Optional[str] = "__keep__") -> None:
        self._client.reset(system_message)

    # --- Delegate abstract methods ---

    async def _generate(
        self,
        prompt: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
        audio: Optional[list] = None,
        response_format: Optional[dict] = None,
    ) -> Union[str, AsyncIterator[StreamChunk]]:
        # Forward response_format only when set (non-None only on the native structured path,
        # where the inner client accepts it); parse-path inner clients never receive it.
        extra = {"response_format": response_format} if response_format is not None else {}
        return await self._client._generate(prompt, generate_kwargs, stream=stream, images=images, audio=audio, **extra)

    async def _chat(
        self,
        user_message: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        use_tools: bool = True,
        stream: bool = False,
        images: Optional[list] = None,
        audio: Optional[list] = None,
        response_format: Optional[dict] = None,
    ) -> Union[str, AsyncIterator[StreamChunk]]:
        extra = {"response_format": response_format} if response_format is not None else {}
        return await self._client._chat(
            user_message, generate_kwargs, use_tools=use_tools, stream=stream, images=images, audio=audio, **extra
        )

    def _update_generate_kwargs(self, generate_kwargs: Optional[dict[str, Any]] = None) -> dict:
        return self._client._update_generate_kwargs(generate_kwargs)


def client(
    model: Union[str, Model, Any, None] = None, *, system: Optional[str] = None, **kwargs: Any
) -> AsyncModelClient:
    """Construct an :class:`AsyncModelClient` from a model string, enum, or existing sync client.

    For in-process providers (HuggingFace, LlamaCpp), pass an existing sync client to
    avoid loading model weights twice::

        sync_client = aimu.client(HuggingFaceModel.LLAMA_70B)
        async_client = aio.client(sync_client)

    When ``model`` is omitted, a default is resolved from ``AIMU_LANGUAGE_MODEL`` or an
    already-available local model. The async path probes only Ollama and local
    OpenAI-compatible servers (an ``hf:`` default would need an explicit sync-client wrap).
    """
    if model is None:
        from aimu.models._internal.model_defaults import resolve_default_text_model

        model = resolve_default_text_model(include_hf_cache=False)
    if system is not None:
        kwargs["system_message"] = system
    return AsyncModelClient(model, **kwargs)


async def chat(
    user_message: str,
    *,
    model: Union[str, Model, None] = None,
    system: Optional[str] = None,
    generate_kwargs: Optional[dict] = None,
    stream: bool = False,
    images: Optional[list] = None,
    include: Optional[Iterable[Union[str, StreamingContentType]]] = None,
) -> Union[str, AsyncIterator[StreamChunk]]:
    """One-shot async chat — builds a fresh client, sends one message, returns the response.

    Example::

        text = await aio.chat("Summarize this", model="anthropic:claude-sonnet-4-6")

        async for chunk in await aio.chat("Tell me a story", model="ollama:qwen3.5:9b", stream=True):
            if chunk.is_text():
                print(chunk.content, end="")
    """
    c = client(model, system=system)
    return await c.chat(
        user_message,
        generate_kwargs=generate_kwargs,
        stream=stream,
        images=images,
        include=include,
    )
