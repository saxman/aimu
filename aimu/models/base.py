import logging
from enum import Enum
from typing import Optional, Iterator, Any, NamedTuple, Union
from abc import ABC, abstractmethod
import random
import string

logger = logging.getLogger(__name__)


class StreamingContentType(str, Enum):
    THINKING = "thinking"
    TOOL_CALLING = "tool_calling"
    GENERATING = "generating"
    DONE = "done"


# Alias used in notebooks and user-facing code
StreamPhase = StreamingContentType


class StreamChunk(NamedTuple):
    """A single chunk from a streamed chat or generation response.

    phase:   the content type of this chunk (THINKING, TOOL_CALLING, GENERATING)
    content: str for THINKING/GENERATING; dict {"name": ..., "response": ...} for TOOL_CALLING
    """

    phase: StreamingContentType
    content: Union[str, dict]


class Model(Enum):
    def __init__(
        self,
        value: str,
        supports_tools: bool = False,
        supports_thinking: bool = False,
        supports_vision: bool = False,
        generation_kwargs: Optional[dict] = None,
    ):
        self._value_ = value
        self.supports_tools = supports_tools
        self.supports_thinking = supports_thinking
        self.supports_vision = supports_vision
        self.generation_kwargs = generation_kwargs or {}


class classproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, cls=None):
        if cls is None:
            cls = type(obj)
        return self.func(cls)


class BaseModelClient(ABC):
    MODELS = Model

    model: Model
    model_kwargs: Optional[dict]
    _system_message: Optional[str]
    default_generate_kwargs: dict
    messages: list[dict]
    mcp_client: Optional[Any]  # Avoid circular imports by not referencing MCPClient directly
    last_thinking: str | None

    @abstractmethod
    def __init__(self, model: Model, model_kwargs: Optional[dict] = None, system_message: Optional[str] = None):
        self.model = model
        self.model_kwargs = model_kwargs
        self._system_message = system_message
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
    def system_message(self, message: str):
        self._system_message = message

        # TODO: add support for models that don't have a system message
        # TODO: move system message handling out of the sub classes
        if self.messages:
            if self.messages[0]["role"] == "system":
                self.messages[0]["content"] = message
            else:
                self.messages.insert(0, {"role": "system", "content": message})

    @abstractmethod
    def generate(
        self,
        prompt: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        include_thinking: bool = True,
    ) -> Union[str, Iterator[StreamChunk]]:
        pass

    @abstractmethod
    def chat(
        self,
        user_message: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        use_tools: bool = True,
        stream: bool = False,
        images: Optional[list] = None,
    ) -> Union[str, Iterator[StreamChunk]]:
        pass

    @abstractmethod
    def _update_generate_kwargs(self, generate_kwargs: Optional[dict[str, Any]] = None) -> dict:
        pass

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
                    f"Model {self.model.name} does not support vision input. "
                    "Use a model with supports_vision=True."
                )
            from ._images import _build_user_content_blocks

            self.messages.append(
                {"role": "user", "content": _build_user_content_blocks(user_message, images)}
            )
        else:
            self.messages.append({"role": "user", "content": user_message})

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

        self.messages.append({
            "role": "assistant",
            "tool_calls": [
                {"type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}, "id": tc_id}
                for tc, tc_id in prepared
            ],
        })

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

            return {"role": "tool", "name": tc["name"], "content": f"Tool '{tc['name']}' not found.", "tool_call_id": tc_id}

        if self.concurrent_tool_calls and len(prepared) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(_call_one, tc, tc_id) for tc, tc_id in prepared]
                # Collect in original order (not arrival order) to keep history consistent.
                results = [f.result() for f in futures]
        else:
            results = [_call_one(tc, tc_id) for tc, tc_id in prepared]

        self.messages.extend(results)
