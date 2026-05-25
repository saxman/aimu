import logging
import random
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Iterator, NamedTuple, Optional, Union

logger = logging.getLogger(__name__)


class StreamingContentType(str, Enum):
    THINKING = "thinking"
    TOOL_CALLING = "tool_calling"
    GENERATING = "generating"
    DONE = "done"


class StreamChunk(NamedTuple):
    """A single chunk yielded by ``client.chat(stream=True)``, ``Agent.run(stream=True)``,
    or any workflow ``run(stream=True)``.

    Fields:
        phase:     content type of this chunk (THINKING, TOOL_CALLING, GENERATING, DONE)
        content:   ``str`` for THINKING/GENERATING; ``dict {"name", "response"}`` for TOOL_CALLING
        agent:     name of the agent that produced this chunk, or ``None`` for a plain
                   ``client.chat()`` call. Set automatically by ``Agent`` and workflow runners.
        iteration: zero-based iteration index inside the agent loop, or ``0`` for plain chat.

    Use ``chunk.is_text()`` / ``chunk.is_tool_call()`` to dispatch on phase without
    repeating the equality check in user code.
    """

    phase: StreamingContentType
    content: Union[str, dict]
    agent: Optional[str] = None
    iteration: int = 0

    def is_text(self) -> bool:
        """True if this chunk carries text (THINKING or GENERATING)."""
        return self.phase in (StreamingContentType.THINKING, StreamingContentType.GENERATING)

    def is_tool_call(self) -> bool:
        """True if this chunk carries a tool-call result."""
        return self.phase == StreamingContentType.TOOL_CALLING


@dataclass
class ModelSpec:
    """Capability descriptor for a single model.

    Holds the provider-side model id plus universal capability flags. Provider-specific
    extras (e.g. HuggingFace tool-call format) live on the provider's ``Model`` subclass,
    not here.

    Equality and hash are by ``id`` only, so a ``ModelSpec`` can be used directly as an
    enum value even when ``generation_kwargs`` is a dict.
    """

    id: str
    tools: bool = False
    thinking: bool = False
    vision: bool = False
    generation_kwargs: Optional[dict] = field(default=None)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ModelSpec):
            return self.id == other.id
        return NotImplemented


class Model(Enum):
    """Base enum for provider model catalogs.

    Each member's value is a ``ModelSpec``; capability flags are mirrored as plain
    attributes (``supports_tools``, ``supports_thinking``, ``supports_vision``,
    ``generation_kwargs``) for direct read access. ``.value`` returns the provider id
    string so code can call e.g. ``ollama.pull(model.value)``.
    """

    def __init__(self, spec: ModelSpec):
        self._value_ = spec.id
        self.spec = spec
        self.supports_tools = spec.tools
        self.supports_thinking = spec.thinking
        self.supports_vision = spec.vision
        self.generation_kwargs = dict(spec.generation_kwargs or {})


class classproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, cls=None):
        if cls is None:
            cls = type(obj)
        return self.func(cls)


class BaseModelClient(ABC):
    """Abstract base for all provider clients.

    Subclasses implement :meth:`generate`, :meth:`chat`, and :meth:`_update_generate_kwargs`.
    Tool calling, message history, vision input, and streaming filters are handled here
    once for every provider.
    """

    MODELS = Model

    model: Model
    model_kwargs: Optional[dict]
    _system_message: Optional[str]
    _system_message_locked: bool
    default_generate_kwargs: dict
    messages: list[dict]
    mcp_client: Optional[Any]  # Avoid circular imports by not referencing MCPClient directly
    last_thinking: str | None

    @abstractmethod
    def __init__(self, model: Model, model_kwargs: Optional[dict] = None, system_message: Optional[str] = None):
        self.model = model
        self.model_kwargs = model_kwargs
        self._system_message = system_message
        self._system_message_locked = False
        self.default_generate_kwargs = {}
        self.messages = []
        self.mcp_client = None
        self.tools: list = []
        self.last_thinking = ""
        self.concurrent_tool_calls = False

    def __deepcopy__(self, memo):
        # BaseModelClient manages stateful conversation history and non-copyable backend resources.
        memo[id(self)] = self
        return self

    @classproperty
    def THINKING_MODELS(cls) -> list[Model]:  # noqa: N805
        raise NotImplementedError

    @classproperty
    def TOOL_MODELS(cls) -> list[Model]:  # noqa: N805
        raise NotImplementedError

    @classproperty
    def VISION_MODELS(cls) -> list[Model]:  # noqa: N805
        raise NotImplementedError

    @property
    def is_thinking_model(self) -> bool:
        return self.model.supports_thinking

    @property
    def is_tool_using_model(self) -> bool:
        return self.model.supports_tools

    @property
    def is_vision_model(self) -> bool:
        return self.model.supports_vision

    @property
    def system_message(self):
        return self._system_message

    @system_message.setter
    def system_message(self, message: Optional[str]):
        if self._system_message_locked:
            raise RuntimeError(
                "system_message is immutable after the conversation starts. "
                "Call client.reset() to clear messages, then assign a new system_message."
            )
        self._system_message = message
        if self.messages:
            if self.messages[0]["role"] == "system":
                if message is None:
                    self.messages.pop(0)
                else:
                    self.messages[0]["content"] = message
            elif message is not None:
                self.messages.insert(0, {"role": "system", "content": message})

    def reset(self, system_message: Optional[str] = "__keep__") -> None:
        """Clear the conversation history and unlock ``system_message``.

        By default the existing ``system_message`` is preserved. Pass ``None`` to clear it,
        or a string to replace it.
        """
        self.messages = []
        self._system_message_locked = False
        if system_message != "__keep__":
            self._system_message = system_message
        self.last_thinking = ""

    @abstractmethod
    def _generate(
        self,
        prompt: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
    ) -> Union[str, Iterator[StreamChunk]]:
        """Provider-specific generate implementation. Use :meth:`generate`."""
        pass

    @abstractmethod
    def _chat(
        self,
        user_message: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        use_tools: bool = True,
        stream: bool = False,
        images: Optional[list] = None,
    ) -> Union[str, Iterator[StreamChunk]]:
        """Provider-specific chat implementation. Use :meth:`chat`."""
        pass

    def generate(
        self,
        prompt: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        include: Optional[Iterable[Union[str, StreamingContentType]]] = None,
    ) -> Union[str, Iterator[StreamChunk]]:
        """Single-turn generation. See :meth:`chat` for the ``include`` filter semantics."""
        result = self._generate(prompt, generate_kwargs, stream=stream)
        if stream and include is not None:
            return self._filter_chunks(result, self._resolve_include(include))
        return result

    def chat(
        self,
        user_message: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        use_tools: bool = True,
        stream: bool = False,
        images: Optional[list] = None,
        include: Optional[Iterable[Union[str, StreamingContentType]]] = None,
    ) -> Union[str, Iterator[StreamChunk]]:
        """Multi-turn chat with persistent message history.

        Args:
            user_message: The text the user is sending this turn.
            generate_kwargs: Provider-specific generation parameters. Unknown keys are
                dropped per-provider; see each client for accepted names.
            use_tools: If False, suppress tool calls even when the model supports tools.
            stream: If True, return an iterator of :class:`StreamChunk` instead of a string.
            images: Optional list of images for vision-capable models. Each entry may be a
                file path (str or ``pathlib.Path``), raw ``bytes``, an ``http(s)://`` URL,
                or a ``data:image/...;base64,...`` URL. Raises ``ValueError`` if the model
                does not support vision. Only used on the initial user turn.
            include: Optional iterable of stream phases to yield. Defaults to all phases
                (THINKING, TOOL_CALLING, GENERATING, DONE). Has no effect when ``stream=False``.
                Values may be :class:`StreamingContentType` members or their string equivalents
                (``"thinking"``, ``"tool_calling"``, ``"generating"``, ``"done"``).
        """
        result = self._chat(user_message, generate_kwargs, use_tools=use_tools, stream=stream, images=images)
        if stream and include is not None:
            return self._filter_chunks(result, self._resolve_include(include))
        return result

    @abstractmethod
    def _update_generate_kwargs(self, generate_kwargs: Optional[dict[str, Any]] = None) -> dict:
        pass

    @staticmethod
    def _resolve_include(
        include: Iterable[Union[str, StreamingContentType]],
    ) -> set[StreamingContentType]:
        """Normalise an ``include=`` argument to a set of :class:`StreamingContentType`."""
        return {StreamingContentType(p) if not isinstance(p, StreamingContentType) else p for p in include}

    @staticmethod
    def _filter_chunks(
        chunks: Iterator[StreamChunk],
        include: set[StreamingContentType],
    ) -> Iterator[StreamChunk]:
        """Drop chunks whose phase isn't in the include set."""
        for chunk in chunks:
            if chunk.phase in include:
                yield chunk

    def _chat_setup(
        self,
        user_message: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        use_tools: bool = True,
        images: Optional[list] = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        # Add the system message if we're processing the first user message and system_message is set
        if len(self.messages) == 0 and self.system_message:
            self.messages.append({"role": "system", "content": self.system_message})

        if images:
            if not self.model.supports_vision:
                raise ValueError(
                    f"Model {self.model.name} does not support vision input. Use a model with supports_vision=True."
                )
            from ._images import _build_user_content_blocks

            self.messages.append({"role": "user", "content": _build_user_content_blocks(user_message, images)})
        else:
            self.messages.append({"role": "user", "content": user_message})

        # First user message of the conversation locks system_message against further mutation.
        self._system_message_locked = True

        tools = []
        if self.model.supports_tools and use_tools:
            if self.mcp_client:
                tools.extend(self.mcp_client.get_tools())
            for fn in self.tools:
                spec = getattr(fn, "__tool_spec__", None)
                if spec is None:
                    raise ValueError(
                        f"Tool '{getattr(fn, '__name__', fn)}' is missing __tool_spec__. "
                        "Decorate it with @aimu.tools.tool."
                    )
                tools.append(spec)

        return generate_kwargs, tools

    def _handle_tool_calls(self, tool_calls: list[dict], tools: list) -> None:
        # Normalize arguments and assign IDs upfront so concurrent execution
        # can use pre-assigned IDs and still append results in original order.
        prepared = []
        for tc in tool_calls:
            # llama 3.1 uses 'parameters' instead of 'arguments'
            if "arguments" not in tc and "parameters" in tc:
                tc["arguments"] = tc.pop("parameters")
            tc_id = "".join(random.choices(string.ascii_letters + string.digits, k=9))
            prepared.append((tc, tc_id))

        self.messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {"type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}, "id": tc_id}
                    for tc, tc_id in prepared
                ],
            }
        )

        python_tools_by_name = {fn.__name__: fn for fn in self.tools}

        def _call_one(tc: dict, tc_id: str) -> dict:
            # Python-function tools (registered via @tool) take precedence over MCP.
            fn = python_tools_by_name.get(tc["name"])
            if fn is not None:
                try:
                    response = fn(**tc["arguments"])
                    content = str(response)
                except Exception as exc:
                    content = f"Tool '{tc['name']}' raised an error: {exc}"
                    logger.warning("Tool call '%s' failed: %s", tc["name"], exc)
                return {"role": "tool", "name": tc["name"], "content": content, "tool_call_id": tc_id}

            for tool in tools:
                if tool["type"] == "function" and tool["function"]["name"] == tc["name"]:
                    if self.mcp_client is None:
                        raise ValueError(
                            "MCP client not initialized. Please initialize and assign an MCP client before using MCP tools."
                        )
                    try:
                        tool_response = self.mcp_client.call_tool(tool["function"]["name"], tc["arguments"])
                        # FastMCP call_tool returns a list of content objects directly
                        response_content = tool_response if isinstance(tool_response, list) else tool_response.content
                        content = ""
                        for part in response_content:
                            if part.type == "text":
                                content += part.text
                            else:
                                logger.debug("Skipping unsupported tool response part type: %s", part.type)
                    except Exception as exc:
                        content = f"Tool '{tc['name']}' raised an error: {exc}"
                        logger.warning("Tool call '%s' failed: %s", tc["name"], exc)
                    return {"role": "tool", "name": tool["function"]["name"], "content": content, "tool_call_id": tc_id}

            return {
                "role": "tool",
                "name": tc["name"],
                "content": f"Tool '{tc['name']}' not found.",
                "tool_call_id": tc_id,
            }

        if self.concurrent_tool_calls and len(prepared) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(_call_one, tc, tc_id) for tc, tc_id in prepared]
                # Collect in original order (not arrival order) to keep history consistent.
                results = [f.result() for f in futures]
        else:
            results = [_call_one(tc, tc_id) for tc, tc_id in prepared]

        self.messages.extend(results)
