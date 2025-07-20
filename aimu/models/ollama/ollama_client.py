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
        MODEL_PHI_4_MINI_3_8B,
    ]

    THINKING_MODELS = [
        MODEL_DEEPSEEK_R1_8B,
        MODEL_QWEN_3_8B,
    ]

    def __init__(self, model_id: str, system_message: str = None):
        super().__init__(model_id, None, system_message)

        self.thinking = True if model_id in self.THINKING_MODELS else False

        ollama.pull(model_id)

    def _update_generate_kwargs(self, generate_kwargs: dict) -> dict[str, str]:
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

    def _chat(self, user_message: str, generate_kwargs: dict = None, tools: dict = None) -> None:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        if tools and self.model_id not in OllamaClient.TOOL_MODELS:
            raise ValueError(
                f"Model {self.model_id} does not support tools. Supported models: {OllamaClient.TOOL_MODELS}"
            )

        # Add system message if it's the first user message and system_message is set
        if len(self.messages) == 0 and self.system_message:
            self.messages.append({"role": "system", "content": self.system_message})

        # Add user message
        self.messages.append({"role": "user", "content": user_message})

    def chat(self, user_message: str, generate_kwargs: dict = None, tools: dict = None) -> str:
        self._chat(user_message, generate_kwargs, tools)

        response = ollama.chat(
            model=self.model_id, messages=self.messages, options=generate_kwargs, tools=tools, think=self.thinking
        )

        if response["message"].tool_calls:
            self._handle_tool_calls(response["message"].tool_calls, tools)

            if response["message"].thinking:
                self.messages[-2]["thinking"] = response["message"].thinking

            response = ollama.chat(
                model=self.model_id, messages=self.messages, options=generate_kwargs, tools=tools, think=self.thinking
            )

        self.messages.append({"role": response["message"].role, "content": response["message"].content})

        if response["message"].thinking:
            self.messages[-1]["thinking"] = response["message"].thinking

        return response["message"].content

    def chat_streamed(self, user_message: str, generate_kwargs: dict = None, tools: dict = None) -> Iterator[str]:
        self._chat(user_message, generate_kwargs, tools)

        response = ollama.chat(
            model=self.model_id,
            messages=self.messages,
            options=generate_kwargs,
            tools=tools,
            stream=True,
            think=self.thinking,
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
                model=self.model_id,
                messages=self.messages,
                options=generate_kwargs,
                tools=tools,
                stream=True,
                think=self.thinking,
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
