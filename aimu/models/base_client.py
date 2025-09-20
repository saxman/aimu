import logging
from enum import Enum
from typing import Optional, Iterator, Any
from abc import ABC, abstractmethod
import random
import string

logger = logging.getLogger(__name__)


class Model(Enum):
    pass


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
    def generate_streamed(self, prompt: str, generate_kwargs: Optional[dict[str, Any]] = None) -> Iterator[str]:
        pass

    @abstractmethod
    def chat(
        self, user_message: str, generate_kwargs: Optional[dict[str, Any]] = None, use_tools: Optional[bool] = True
    ) -> str:
        pass

    @abstractmethod
    def chat_streamed(
        self, user_message: str, generate_kwargs: Optional[dict[str, Any]] = None, use_tools: Optional[bool] = True
    ) -> Iterator[str]:
        pass

    @abstractmethod
    def _update_generate_kwargs(self, generate_kwargs: Optional[dict[str, Any]] = None) -> dict:
        pass

    def _chat_setup(
        self, user_message: str, generate_kwargs: Optional[dict[str, Any]] = None, use_tools: Optional[bool] = True
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        # Add the system message if we're processing the first user message and system_message is set
        if len(self.messages) == 0 and self.system_message:
            self.messages.append({"role": "system", "content": self.system_message})

        # Add the user message
        self.messages.append({"role": "user", "content": user_message})

        tools = []
        if use_tools and self.mcp_client:
            tools = self.mcp_client.get_tools()

        return generate_kwargs, tools

    def _handle_tool_calls(self, tool_calls, tools: list) -> None:
        message = {"role": "assistant", "tool_calls": []}
        self.messages.append(message)

        # If we're processing tool calls from Ollama, we need to convert the calls to a dictionary
        if hasattr(tool_calls[0], "function"):
            calls = []
            for tool_call in tool_calls:
                calls.append(
                    {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    }
                )
            tool_calls = calls

        for tool_call in tool_calls:
            # llama 3.1 uses 'parameters' instead of 'arguments'
            if "arguments" not in tool_call and "parameters" in tool_call:
                tool_call["arguments"] = tool_call["parameters"]

            id = "".join(random.choices(string.ascii_letters + string.digits, k=9))

            message["tool_calls"].append(
                {
                    "type": "function",
                    "function": {"name": tool_call["name"], "arguments": tool_call["arguments"]},
                    "id": id,
                }
            )

            for tool in tools:
                # If the tool is a callable python function, call it directly. Otherwise, use the MCP client to call the tool.
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

                    content = ""
                    if len(tool_response.content) > 0:
                        # TODO: handle different tool response types, errors, and multiple responses
                        if tool_response.content[0].type != "text":
                            raise ValueError(
                                f"Tool response type {tool_response.content[0].type} not supported. Supported types: text"
                            )

                        content = tool_response.content[0].text

                    self.messages.append(
                        {"role": "tool", "name": tool["function"]["name"], "content": content, "tool_call_id": id}
                    )

                    break

        return
