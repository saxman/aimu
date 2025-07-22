import logging
import enum

logger = logging.getLogger(__name__)


class Model(enum.Enum):
    pass


class ModelClient:
    MODELS = Model

    def __init__(self, model: Model, model_kwargs: dict = None, system_message: str = None):
        self.model = model
        self.model_kwargs = model_kwargs
        self._system_message = system_message

        self.messages = []

        self.mcp_client = None

    @property
    def system_message(self):
        return self._system_message
    
    @system_message.setter
    def system_message(self, message: str):
        self._system_message = message

        # TODO: add support for models that don't have a system message
        if self.messages:
            if self.messages[0]["role"] == "system":
                self.messages[0]["content"] = message
            else:
                self.messages.insert(0, {"role": "system", "content": message})

    def _handle_tool_calls(self, tool_calls, tools: dict) -> None:
        message = {"role": "assistant"}
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

        message["tool_calls"] = []
        for tool_call in tool_calls:
            message["tool_calls"].append(
                {
                    "type": "function",
                    "function": {"name": tool_call["name"], "arguments": tool_call["arguments"]},
                }
            )

            for tool in tools:
                # If the tool is a callable python function, call it directly. Otherwise, use the MCP client to call the tool.
                if hasattr(tool, "__call__") and tool.__name__ == tool_call["name"]:
                    tool_response = tool(**tool_call["arguments"])

                    self.messages.append({"role": "tool", "name": tool_call["name"], "content": str(tool_response)})

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

                    self.messages.append({"role": "tool", "name": tool["function"]["name"], "content": content})

                    break

        return
