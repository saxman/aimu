from ..models import ModelClient

import logging
from typing import Iterator

logger = logging.getLogger(__name__)

try:
    import ollama
except ImportError:
    ollama = None


class OllamaClient(ModelClient):
    MODEL_LLAMA_3_1_8B = "llama3.1:8b"
    MODEL_LLAMA_3_2_3B = "llama3.2:3b"
    MODEL_LLAMA_3_3_70B = "llama3.3:70b"

    MODEL_GEMMA_3_12B = "gemma3:12b"

    MODEL_PHI_4_14B = "phi4:14b"
    MODEL_PHI_4_MINI_3_8B = "phi4-mini:3.8b"

    MODEL_DEEPSEEK_R1_8B = "deepseek-r1:8b"

    MODEL_MISTRAL_7B = "mistral:7b"
    MODEL_MISTRAL_NEMO_12B = "mistral-nemo:12b"
    MODEL_MISTRAL_SMALL_3_2_24B = "mistral-small3.2:24b"

    MODEL_QWEN_3_8B = "qwen3:8b"

    TOOL_MODELS = [
        MODEL_MISTRAL_SMALL_3_2_24B,
        MODEL_MISTRAL_NEMO_12B,
        MODEL_QWEN_3_8B,
        # MODEL_LLAMA_3_1_8B, ## Tools not fully supported by model
        MODEL_LLAMA_3_2_3B,
        # MODEL_PHI_4_MINI_3_8B, ## Tools not fully supported by Ollama
    ]

    THINKING_MODELS = [
        MODEL_DEEPSEEK_R1_8B,
        MODEL_QWEN_3_8B,
    ]

    def __init__(self, model_id: str):
        super().__init__(model_id, None)

        self.thinking = True if model_id in self.THINKING_MODELS else False

        ollama.pull(model_id)

    @property
    def system_role(self) -> str:
        return "system"

    def _update_generate_kwargs(self, generate_kwargs: dict) -> None:
        if generate_kwargs and "max_tokens" in generate_kwargs:
            generate_kwargs["num_predict"] = generate_kwargs.pop("max_tokens")

        return generate_kwargs

    def generate(self, prompt: str, generate_kwargs: dict = None) -> str:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        response = ollama.generate(model=self.model_id, prompt=prompt, options=generate_kwargs, think=self.thinking)

        return response["response"] if not self.thinking else response.response

    def generate_streamed(self, prompt: str, generate_kwargs: dict = None) -> Iterator[str]:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        response = ollama.generate(
            model=self.model_id, prompt=prompt, options=generate_kwargs, stream=True, think=self.thinking
        )

        for response_part in response:
            yield response_part["response"]

    def _chat(self, message: dict, generate_kwargs: dict = None, tools: dict = None) -> None:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        self.messages.append(message)

        if tools:
            if self.model_id == OllamaClient.MODEL_LLAMA_3_1_8B:
                logger.warning(
                    "Tool usage with Llama 3.1 8B is not fully supported, ref: https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/"
                )

            if len(self.messages) == 0 and ("llama" in self.model_id or "mistral" in self.model_id):
                logger.warning("Llama and Mistral models require a system message to be sent before using tools.")

            if self.model_id not in OllamaClient.TOOL_MODELS:
                raise ValueError(
                    f"Model {self.model_id} does not support tools. Supported models: {OllamaClient.TOOL_MODELS}"
                )

    def chat(self, message: dict, generate_kwargs: dict = None, tools: dict = None) -> str:
        self._chat(message, generate_kwargs, tools)

        response = ollama.chat(
            model=self.model_id, messages=self.messages, options=generate_kwargs, tools=tools, think=self.thinking
        )

        if response["message"].tool_calls:
            self._handle_tool_calls(response, tools)
            response = ollama.chat(
                model=self.model_id, messages=self.messages, options=generate_kwargs, tools=tools, think=self.thinking
            )

        self.messages.append({"role": response["message"].role, "content": response["message"].content})

        return response["message"].content

    def chat_streamed(self, message: dict, generate_kwargs: dict = None, tools: dict = None) -> Iterator[str]:
        self._chat(message, generate_kwargs, tools)

        response = ollama.chat(
            model=self.model_id,
            messages=self.messages,
            options=generate_kwargs,
            tools=tools,
            stream=True,
            think=self.thinking,
        )

        # Streaming with thinking *and* tools, the thinking needs to complete before the response can be inspected for tool calls.
        thinking = ""
        if self.thinking and tools:
            for response_part in response:
                if response_part["message"].thinking:
                    thinking += response_part["message"].thinking
                else:
                    break

        ## TODO need to assign thinking to the response, not the last user message
        self.messages[-1]["thinking"] = thinking

        response_part = next(response)
        print(response_part)

        if response_part["message"].tool_calls:
            self._handle_tool_calls(response_part, tools)

            response = ollama.chat(
                model=self.model_id,
                messages=self.messages,
                options=generate_kwargs,
                tools=tools,
                stream=True,
                think=self.thinking,
            )

            response_part = next(response)

        content = response_part["message"].content
        yield content

        for response_part in response:
            content += response_part["message"].content
            yield response_part["message"].content

        self.messages.append({"role": response_part["message"].role, "content": content})

    def _handle_tool_calls(self, response, tools: dict) -> None:
        message = {"role": response.message.role}
        self.messages.append(message)

        message["tool_calls"] = []
        for tool_call in response["message"].tool_calls:
            message["tool_calls"].append(
                {
                    "type": "function",
                    "function": {"name": tool_call.function.name, "arguments": tool_call.function.arguments},
                }
            )

            for tool in tools:
                # If the tool is a callable python function, call it directly. Otherwise, use the MCP client to call the tool.
                if hasattr(tool, "__call__") and tool.__name__ == tool_call.function.name:
                    tool_response = tool(**tool_call.function.arguments)

                    self.messages.append(
                        {"role": "tool", "name": tool_call.function.name, "content": str(tool_response)}
                    )

                    break
                elif tool["type"] == "function" and tool["function"]["name"] == tool_call.function.name:
                    if self.mcp_client is None:
                        raise ValueError(
                            "MCP client not initialized. Please initialize and assign an MCP client before using MCP tools."
                        )

                    tool_response = self.mcp_client.call_tool(tool["function"]["name"], tool_call.function.arguments)

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
