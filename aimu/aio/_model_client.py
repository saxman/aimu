"""Async ``ModelClient`` factory mirroring ``aimu.models.model_client.ModelClient``.

Accepts either a provider ``Model`` enum member, a ``"provider:model_id"`` string,
or, for in-process providers (HuggingFace, LlamaCpp), an existing sync client to
wrap (so model weights are loaded only once). See Decision 7 in the plan.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterable, Optional, Union

from aimu.models.base import AdHocModel, Model, ModelSpec, StreamChunk, StreamingContentType
from aimu.models.model_client import _GENERIC_COMPAT_PROVIDER, _TEXT_PROVIDERS, endpoint_kwargs, resolve_model

from ._base import AsyncBaseModelClient

if TYPE_CHECKING:
    from aimu.events import EventSink

# Provider prefix -> (async module, async client class name). Described, not imported:
# nothing here loads until a specific row is requested. The enum side of dispatch is
# the *sync* _TEXT_PROVIDERS table (the enum classes are shared with the sync surface;
# only the client classes differ), so this table only needs to cover the client half.
# `hf` and `llamacpp` are deliberately absent: those two prefixes have no
# async-from-enum path (aio.client() refuses direct construction and points at the
# sync-wrap pattern instead; see _IN_PROCESS_MODEL_GUIDANCE below).
_ASYNC_CLIENTS: dict[str, tuple[str, str]] = {
    "ollama": ("aimu.aio.providers.ollama", "AsyncOllamaClient"),
    "anthropic": ("aimu.aio.providers.anthropic", "AsyncAnthropicClient"),
    "openai": ("aimu.aio.providers.openai.text", "AsyncOpenAIClient"),
    "gemini": ("aimu.aio.providers.gemini.text", "AsyncGeminiClient"),
    "lmstudio": ("aimu.aio.providers.openai_compat", "AsyncLMStudioOpenAIClient"),
    "ollama-openai": ("aimu.aio.providers.openai_compat", "AsyncOllamaOpenAIClient"),
    "hf-openai": ("aimu.aio.providers.openai_compat", "AsyncHFOpenAIClient"),
    "vllm": ("aimu.aio.providers.openai_compat", "AsyncVLLMOpenAIClient"),
    "llamaserver": ("aimu.aio.providers.openai_compat", "AsyncLlamaServerOpenAIClient"),
    "sglang": ("aimu.aio.providers.openai_compat", "AsyncSGLangOpenAIClient"),
    "omlx": ("aimu.aio.providers.openai_compat", "AsyncOMLXOpenAIClient"),
}
_GENERIC_ASYNC_COMPAT: tuple[str, str] = ("aimu.aio.providers.openai_compat", "AsyncOpenAICompatClient")

_IN_PROCESS_PREFIXES = frozenset({"hf", "llamacpp"})

_IN_PROCESS_MODEL_GUIDANCE = (
    "In-process providers (HuggingFace, LlamaCpp) load model weights into memory; "
    "constructing both a sync and async client for the same model would load weights twice. "
    "Build a sync client first and pass it to aio.client():\n"
    "    sync_client = aimu.client({model})\n"
    "    async_client = aio.client(sync_client)"
)


def _load_async_client(module_name: str, class_name: str):
    return getattr(import_module(module_name), class_name)


def _wrap_target(model: Any) -> Optional[tuple[str, str]]:
    """``(async_module, async_class_name)`` for wrapping a sync in-process client instance.

    A sync ``HuggingFaceClient``/``LlamaCppClient`` instance can only exist if its
    defining module was already imported by the caller, so matching on
    ``type(model).__module__`` + class name identifies it without importing anything.
    """
    module_name = type(model).__module__
    class_name = type(model).__name__
    if module_name == "aimu.models.providers.hf.text" and class_name == "HuggingFaceClient":
        return "aimu.aio.providers.hf.text", "AsyncHuggingFaceClient"
    if module_name == "aimu.models.providers.llamacpp" and class_name == "LlamaCppClient":
        return "aimu.aio.providers.llamacpp", "AsyncLlamaCppClient"
    return None


class AsyncModelClient(AsyncBaseModelClient):
    """Public factory for async provider-backed model clients.

    Accepts a provider ``Model`` enum member, a ``"provider:model_id"`` string, or
    for in-process providers, an existing sync client to wrap.

    Examples::

        # Cloud providers (separate sync/async clients are cheap)
        client = AsyncModelClient("anthropic:claude-sonnet-4-6")
        client = AsyncModelClient(OllamaModel.QWEN_3_8B)

        # In-process providers: wrap an existing sync client to share weights
        sync_client = aimu.client(HuggingFaceModel.LLAMA_70B)
        async_client = AsyncModelClient(sync_client)
    """

    def __init__(self, model: Union[Model, ModelSpec, str, Any], **kwargs: Any) -> None:
        # Popped rather than left in kwargs: the concrete provider constructors don't accept
        # events= (it's set on the inner client afterward, like the other mutable-state
        # properties below), so forwarding it verbatim would raise a TypeError.
        events = kwargs.pop("events", None)
        # In-process wrapping path: passed an existing sync client.
        wrap_target = _wrap_target(model)
        if wrap_target is not None:
            client_cls = _load_async_client(*wrap_target)
            self._client: AsyncBaseModelClient = client_cls(model, **kwargs)
        else:
            # Normal construction path: model enum or string.
            if isinstance(model, str):
                resolved = resolve_model(model)
                kwargs.update(endpoint_kwargs(resolved.provider, resolved.base_url))
                if isinstance(resolved.model, AdHocModel):
                    async_entry = (
                        _GENERIC_ASYNC_COMPAT
                        if resolved.provider == _GENERIC_COMPAT_PROVIDER
                        else _ASYNC_CLIENTS[resolved.provider]
                    )
                    client_cls = _load_async_client(*async_entry)
                    self._client = client_cls(resolved.model, **kwargs)
                    self._client.events = events
                    self.model = self._client.model
                    self.model_kwargs = self._client.model_kwargs
                    return
                model = resolved.model
            elif isinstance(model, ModelSpec):
                raise TypeError(
                    "Pass a Model enum member or a 'provider:model_id' string. "
                    "ModelSpec is the value type held by enum members."
                )

            # Dispatch by the model enum's defining module + class name (mirrors the sync
            # ModelClient). The enum classes are shared with the sync surface, so matching
            # against the sync _TEXT_PROVIDERS table costs no import.
            member_module = type(model).__module__
            member_enum = type(model).__name__
            entry = next(
                (e for e in _TEXT_PROVIDERS if e.module == member_module and e.enum_name == member_enum),
                None,
            )
            if entry is None:
                raise ValueError(
                    f"No available async client for model type {type(model).__name__!r}. "
                    "Ensure the required optional dependency is installed."
                )

            # Guard in-process model types: refuse direct construction; point to wrapping pattern.
            if entry.prefix in _IN_PROCESS_PREFIXES:
                raise ValueError(_IN_PROCESS_MODEL_GUIDANCE.format(model=f"{entry.enum_name}.{model.name}"))

            async_entry = _ASYNC_CLIENTS.get(entry.prefix)
            if async_entry is None:
                raise ValueError(
                    f"No available async client for model type {type(model).__name__!r}. "
                    "Ensure the required optional dependency is installed."
                )
            client_cls = _load_async_client(*async_entry)
            self._client = client_cls(model, **kwargs)

        self._client.events = events

        # Mirror attributes (super().__init__ would clobber inner client state).
        self.model = self._client.model
        self.model_kwargs = self._client.model_kwargs

    # --- Delegate mutable state to inner client ---

    @property
    def events(self) -> Optional["EventSink"]:
        return self._client.events

    @events.setter
    def events(self, value: Optional["EventSink"]) -> None:
        self._client.events = value

    @property
    def default_generate_kwargs(self) -> dict:
        return self._client.default_generate_kwargs

    @default_generate_kwargs.setter
    def default_generate_kwargs(self, value: dict) -> None:
        self._client.default_generate_kwargs = value

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
    def last_output_truncated(self) -> bool:
        """Whether the most recent response was cut off at an output limit rather than finishing."""
        return self._client.last_output_truncated

    @last_output_truncated.setter
    def last_output_truncated(self, value: bool) -> None:
        self._client.last_output_truncated = value

    @property
    def last_structured(self):
        """Validated object from the most recent ``schema=`` call, or ``None`` (populated
        after a streamed structured call is fully consumed; mirrors :attr:`last_usage`)."""
        return self._client.last_structured

    @last_structured.setter
    def last_structured(self, value) -> None:
        self._client.last_structured = value

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
        # Forward response_format only when set (non-None only on the native structured
        # path, where the inner client accepts it); parse-path inner clients never receive it.
        extra = {"response_format": response_format} if response_format is not None else {}
        return await self._client._generate(prompt, generate_kwargs, stream=stream, images=images, audio=audio, **extra)

    async def _chat(
        self,
        user_message: Optional[str] = None,
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

    def _resolve_generate_kwargs(self, generate_kwargs: Optional[dict[str, Any]] = None) -> dict:
        return self._client._resolve_generate_kwargs(generate_kwargs)


def client(
    model: Union[str, Model, Any, None] = None,
    *,
    system: Optional[str] = None,
    events: Optional["EventSink"] = None,
    **kwargs: Any,
) -> AsyncModelClient:
    """Construct an :class:`AsyncModelClient` from a model string, enum, or existing sync client.

    For in-process providers (HuggingFace, LlamaCpp), pass an existing sync client to
    avoid loading model weights twice::

        sync_client = aimu.client(HuggingFaceModel.LLAMA_70B)
        async_client = aio.client(sync_client)

    When ``model`` is omitted, a default is resolved from ``AIMU_LANGUAGE_MODEL`` or an
    already-available local model. The async path probes only Ollama and local
    OpenAI-compatible servers (an ``hf:`` default would need an explicit sync-client wrap).

    Args:
        events: Optional event sink (see :mod:`aimu.events`). Attach it to see the
            ``ModelTurnStarted`` / ``ModelTurnFinished`` events every ``chat()`` /
            ``generate()`` call on the returned client emits.
    """
    if model is None:
        from aimu.models._internal.model_defaults import resolve_default_text_model

        model = resolve_default_text_model(include_hf_cache=False)
    if system is not None:
        kwargs["system_message"] = system
    kwargs["events"] = events
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
    thinking: Optional[Union[bool, str]] = None,
    events: Optional["EventSink"] = None,
) -> Union[str, AsyncIterator[StreamChunk]]:
    """One-shot async chat: builds a fresh client, sends one message, returns the response.

    Example::

        text = await aio.chat("Summarize this", model="anthropic:claude-sonnet-4-6")

        async for chunk in await aio.chat("Tell me a story", model="ollama:qwen3.5:9b", stream=True):
            if chunk.is_text():
                print(chunk.content, end="")

    Args:
        thinking: Optional thinking control. ``None`` (default) leaves the provider's
            own behavior untouched. ``False`` disables reasoning and selects the model's
            instruct-mode sampling profile; ``True`` enables it at the model's default
            effort; ``"low"``/``"medium"``/``"high"`` sets the effort level. A model that
            cannot honour the request logs a warning and continues, so models stay
            swappable; an unrecognised value raises ``ValueError``.
        events: Optional event sink (see :mod:`aimu.events`); attach it to see the
            ``ModelTurnStarted`` / ``ModelTurnFinished`` events this one-shot call emits.
    """
    c = client(model, system=system, events=events)
    return await c.chat(
        user_message,
        generate_kwargs=generate_kwargs,
        stream=stream,
        images=images,
        include=include,
        thinking=thinking,
    )
