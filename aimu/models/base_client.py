import logging
from enum import Enum
from typing import Optional, Iterator, Any
from abc import ABC, abstractmethod
import random
import string

logger = logging.getLogger(__name__)


class StreamingContentType(str, Enum):
    THINKING = "thinking"
    TOOL_CALLING = "tool_calling"
    GENERATING = "generating"
    DONE = "done"


class Model(Enum):
    def __init__(self, value: str, supports_tools: bool = False, supports_thinking: bool = False):
        self._value_ = value
        self.supports_tools = supports_tools
        self.supports_thinking = supports_thinking


class classproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, cls=None):
        if cls is None:
            cls = type(obj)
        return self.func(cls)


class ModelClient(ABC):
    MODELS = Model

    @abstractmethod
    def __init__(self, model: Model, model_kwargs: Optional[dict] = None, system_message: Optional[str] = None):
        self.model = model
        self.model_kwargs = model_kwargs
        self._system_message = system_message
        self.default_generate_kwargs = {}
        self.messages = []
        self.mcp_client = None
        self.last_thinking = ""
        self._streaming_content_type = StreamingContentType.DONE

    def __deepcopy__(self, memo):
        # ModelClient manages stateful conversation history and non-copyable backend resources.
        memo[id(self)] = self
        return self

    @classproperty
    def THINKING_MODELS(cls) -> list[Model]:  # noqa: N805
        raise NotImplementedError

    @classproperty
    def TOOL_MODELS(cls) -> list[Model]:  # noqa: N805
        raise NotImplementedError

    @property
    def streaming_content_type(self) -> StreamingContentType:
        return self._streaming_content_type

    @property
    def is_thinking_model(self) -> bool:
        return self.model.supports_thinking

    @property
    def is_tool_using_model(self) -> bool:
        return self.model.supports_tools

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
    def generate(self, prompt: str, generate_kwargs: Optional[dict[str, Any]] = None) -> str:
        pass

    @abstractmethod
    def generate_streamed(
        self,
        prompt: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        include_thinking: bool = True,
    ) -> Iterator[str]:
        pass

    @abstractmethod
    def chat(self, user_message: str, generate_kwargs: Optional[dict[str, Any]] = None, use_tools: bool = True) -> str:
        pass

    @abstractmethod
    def chat_streamed(
        self, user_message: str, generate_kwargs: Optional[dict[str, Any]] = None, use_tools: bool = True
    ) -> Iterator[str]:
        pass

    @abstractmethod
    def _update_generate_kwargs(self, generate_kwargs: Optional[dict[str, Any]] = None) -> dict:
        pass

    def _chat_setup(
        self, user_message: str, generate_kwargs: Optional[dict[str, Any]] = None, use_tools: bool = True
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        # Add the system message if we're processing the first user message and system_message is set
        if len(self.messages) == 0 and self.system_message:
            self.messages.append({"role": "system", "content": self.system_message})

        # Add the user message
        self.messages.append({"role": "user", "content": user_message})

        tools = []
        if self.model.supports_tools and use_tools and self.mcp_client:
            tools = self.mcp_client.get_tools()

        return generate_kwargs, tools

    def _handle_tool_calls(self, tool_calls: list[dict], tools: list) -> None:
        message = {"role": "assistant", "tool_calls": []}
        self.messages.append(message)

        for tool_call in tool_calls:
            # llama 3.1 uses 'parameters' instead of 'arguments'
            if "arguments" not in tool_call and "parameters" in tool_call:
                tool_call["arguments"] = tool_call.pop("parameters")

            id = "".join(random.choices(string.ascii_letters + string.digits, k=9))

            message["tool_calls"].append(
                {
                    "type": "function",
                    "function": {"name": tool_call["name"], "arguments": tool_call["arguments"]},
                    "id": id,
                }
            )

            for tool in tools:
                # If the tool is a call-able python function, call it directly. Otherwise, use the MCP client to call the tool.
                if hasattr(tool, "__call__") and tool.__name__ == tool_call["name"]:
                    tool_response = tool(**tool_call["arguments"])

                    self.messages.append(
                        {"role": "tool", "name": tool_call["name"], "content": str(tool_response), "tool_call_id": id}
                    )

                    break
                elif tool["type"] == "function" and tool["function"]["name"] == tool_call["name"]:
                    if self.mcp_client is None:
                        raise ValueError(
                            "MCP client not initialized. Please initialize and assign an MCP client before using MCP tools."
                        )

                    tool_response = self.mcp_client.call_tool(tool["function"]["name"], tool_call["arguments"])

                    # FastMCP call_tool returns a list of content objects directly
                    response_content = tool_response if isinstance(tool_response, list) else tool_response.content

                    content = ""
                    if len(response_content) > 0:
                        # TODO: handle different tool response types, errors, and multiple responses
                        if response_content[0].type != "text":
                            raise ValueError(
                                f"Tool response type {response_content[0].type} not supported. Supported types: text"
                            )

                        content = response_content[0].text

                    self.messages.append(
                        {"role": "tool", "name": tool["function"]["name"], "content": content, "tool_call_id": id}
                    )

                    break
