from ..models import Model, ModelClient

import logging
from typing import Iterator

logger = logging.getLogger(__name__)

try:
    import ollama
except ImportError:
    ollama = None


class OllamaModel(Model):
    LLAMA_3_1_8B = "llama3.1:8b"
    LLAMA_3_2_3B = "llama3.2:3b"
    LLAMA_3_3_70B = "llama3.3:70b"

    GEMMA_3_12B = "gemma3:12b"

    PHI_4_14B = "phi4:14b"
    PHI_4_MINI_3_8B = "phi4-mini:3.8b"

    DEEPSEEK_R1_8B = "deepseek-r1:8b"

    MISTRAL_7B = "mistral:7b"
    MISTRAL_NEMO_12B = "mistral-nemo:12b"
    MISTRAL_SMALL_3_2_24B = "mistral-small3.2:24b"

    QWEN_3_8B = "qwen3:8b"

    SMOLLM2_1_7B = "smollm2:latest"  # "smollm2:1.7b" error downloading model, using latest for now


class OllamaClient(ModelClient):
    MODELS = OllamaModel

    TOOL_MODELS = [
        MODELS.QWEN_3_8B,
        # MODELS.MISTRAL_7B, # issue with tool usage
        # MODELS.MISTRAL_NEMO_12B, # issue with tool usage
        MODELS.MISTRAL_SMALL_3_2_24B,
        MODELS.LLAMA_3_1_8B,
        MODELS.LLAMA_3_2_3B,
        MODELS.LLAMA_3_3_70B,
        MODELS.SMOLLM2_1_7B,
    ]

    THINKING_MODELS = [
        MODELS.DEEPSEEK_R1_8B,
        MODELS.QWEN_3_8B,
    ]

    def __init__(self, model: OllamaModel, system_message: str = None, model_keep_alive_seconds: int = 60):
        super().__init__(model, None, system_message)

        # TODO extend model_keep_alive_seconds to other model clients
        self.model_keep_alive_seconds = model_keep_alive_seconds

        self.thinking = True if model in self.THINKING_MODELS else False

        ollama.pull(model.value)

    def _update_generate_kwargs(self, generate_kwargs: dict) -> dict[str, str]:
        if generate_kwargs and "max_tokens" in generate_kwargs:
            generate_kwargs["num_predict"] = generate_kwargs.pop("max_tokens")

        return generate_kwargs

    def generate(self, prompt: str, generate_kwargs: dict = None) -> str:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options=generate_kwargs,
            think=self.thinking,
            keep_alive=self.model_keep_alive_seconds,
        )

        return response["response"] if not self.thinking else response.response

    def generate_streamed(self, prompt: str, generate_kwargs: dict = None) -> Iterator[str]:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options=generate_kwargs,
            stream=True,
            think=self.thinking,
            keep_alive=self.model_keep_alive_seconds,
        )

        for response_part in response:
            yield response_part["response"]

    def _chat(self, user_message: str, generate_kwargs: dict, use_tools: bool) -> None:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        if use_tools and self.model not in OllamaClient.TOOL_MODELS:
            raise ValueError(f"Model {self.model} does not support tools. Supported models: {OllamaClient.TOOL_MODELS}")

        # Add system message if it's the first user message and system_message is set
        if len(self.messages) == 0 and self.system_message:
            self.messages.append({"role": "system", "content": self.system_message})

        # Add user message
        self.messages.append({"role": "user", "content": user_message})

    def chat(self, user_message: str, generate_kwargs: dict = None, use_tools: bool = True) -> str:
        self._chat(user_message, generate_kwargs, use_tools)

        tools = []
        if use_tools and self.mcp_client:
            tools = self.mcp_client.get_tools()

        response = ollama.chat(
            model=self.model,
            messages=self.messages,
            options=generate_kwargs,
            tools=tools,
            think=self.thinking,
            keep_alive=self.model_keep_alive_seconds,
        )

        if response["message"].tool_calls:
            self._handle_tool_calls(response["message"].tool_calls, tools)

            if response["message"].thinking:
                self.messages[-2]["thinking"] = response["message"].thinking

            response = ollama.chat(
                model=self.model,
                messages=self.messages,
                options=generate_kwargs,
                tools=tools,
                think=self.thinking,
                keep_alive=self.model_keep_alive_seconds,
            )

        self.messages.append({"role": response["message"].role, "content": response["message"].content})

        if response["message"].thinking:
            self.messages[-1]["thinking"] = response["message"].thinking

        return response["message"].content

    def chat_streamed(self, user_message: str, generate_kwargs: dict = None, use_tools: bool = True) -> Iterator[str]:
        self._chat(user_message, generate_kwargs, use_tools)

        tools = []
        if use_tools and self.mcp_client:
            tools = self.mcp_client.get_tools()

        response = ollama.chat(
            model=self.model,
            messages=self.messages,
            options=generate_kwargs,
            tools=tools,
            stream=True,
            think=self.thinking,
            keep_alive=self.model_keep_alive_seconds,
        )

        response_part = next(response)

        # If the model is thinking, we need to capture the thinking before processing tools and streaming the response.
        thinking = ""
        if response_part["message"].thinking:
            thinking = response_part["message"].thinking
            for response_part in response:
                if response_part["message"].thinking:
                    thinking += response_part["message"].thinking
                else:
                    break

        if response_part["message"].tool_calls:
            self._handle_tool_calls(response_part["message"].tool_calls, tools)

            if thinking:
                self.messages[-2]["thinking"] = thinking  # TODO: correct functionality if there are multiple tool calls
                thinking = ""

            response = ollama.chat(
                model=self.model,
                messages=self.messages,
                options=generate_kwargs,
                tools=tools,
                stream=True,
                think=self.thinking,
                keep_alive=self.model_keep_alive_seconds,
            )

            response_part = next(response)

            # For the response after the tool call, we need to capture the thinking again if it exists.
            thinking = ""
            if response_part["message"].thinking:
                thinking = response_part["message"].thinking
                for response_part in response:
                    if response_part["message"].thinking:
                        thinking += response_part["message"].thinking
                    else:
                        break

        content = response_part["message"].content
        yield content

        for response_part in response:
            content += response_part["message"].content
            yield response_part["message"].content

        message = {"role": response_part["message"].role, "content": content}

        if thinking:
            message["thinking"] = thinking

        self.messages.append(message)
