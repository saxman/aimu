from ..base_client import Model, ModelClient

import ollama
import logging
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


class OllamaModel(Model):
    GPT_OSS_20B = "gpt-oss:20b"

    LLAMA_3_1_8B = "llama3.1:8b"
    LLAMA_3_2_3B = "llama3.2:3b"
    # LLAMA_3_3_70B = "llama3.3:70b"

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
        MODELS.GPT_OSS_20B,
        MODELS.QWEN_3_8B,
        # MODELS.MISTRAL_7B, # issue with tool usage
        # MODELS.MISTRAL_NEMO_12B, # issue with tool usage
        MODELS.MISTRAL_SMALL_3_2_24B,
        MODELS.LLAMA_3_1_8B,
        MODELS.LLAMA_3_2_3B,
        # MODELS.LLAMA_3_3_70B,
        MODELS.SMOLLM2_1_7B,
    ]

    THINKING_MODELS = [
        MODELS.GPT_OSS_20B,
        MODELS.DEEPSEEK_R1_8B,
        MODELS.QWEN_3_8B,
    ]

    def __init__(self, model: OllamaModel, system_message: Optional[str] = None, model_keep_alive_seconds: int = 60):
        super().__init__(model, None, system_message)

        # TODO extend model_keep_alive_seconds to other model clients
        self.model_keep_alive_seconds = model_keep_alive_seconds

        self.thinking = True if model in self.THINKING_MODELS else False

        ollama.pull(model.value)

    def _update_generate_kwargs(self, generate_kwargs: Optional[dict] = None) -> dict[str, str]:
        if not generate_kwargs:
            generate_kwargs = self.default_generate_kwargs

        if "max_tokens" in generate_kwargs:
            generate_kwargs["num_predict"] = generate_kwargs.pop("max_tokens")

        return generate_kwargs

    def generate(self, prompt: str, generate_kwargs: Optional[dict] = None) -> str:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        response = ollama.generate(
            model=self.model.value,
            prompt=prompt,
            options=generate_kwargs,
            think=self.thinking,
            keep_alive=self.model_keep_alive_seconds,
        )

        self.last_thinking = ""

        if not self.thinking:
            return response["response"]
        
        self.last_thinking = response.thinking

        return response.response

    def generate_streamed(self, prompt: str, generate_kwargs: Optional[dict] = None) -> Iterator[str]:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        response = ollama.generate(
            model=self.model.value,
            prompt=prompt,
            options=generate_kwargs,
            stream=True,
            think=self.thinking,
            keep_alive=self.model_keep_alive_seconds,
        )

        self.last_thinking = ""

        if self.thinking:
            next(response)
            next(response)
            response_part = next(response)

            if response_part.thinking:
                self.last_thinking = response_part.thinking
                for response_part in response:
                    if response_part.thinking:
                        self.last_thinking += response_part.thinking
                    else:
                        break
        
        for response_part in response:
            yield response_part["response"]

    def chat(self, user_message: str, generate_kwargs: Optional[dict] = None, use_tools: bool = True) -> str:
        generate_kwargs, tools = self._chat_setup(user_message, generate_kwargs, use_tools)

        response = ollama.chat(
            model=self.model.value,
            messages=self.messages,
            options=generate_kwargs,
            tools=tools,
            think=self.thinking,
            keep_alive=self.model_keep_alive_seconds,
        )

        if response["message"].tool_calls:
            tool_calls = []
            for tool_call in response["message"].tool_calls:
                tool_calls.append(
                    {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    }
                )

            self._handle_tool_calls(tool_calls, tools)

            if response["message"].thinking:
                self.messages[-1 - len(tool_calls)]["thinking"] = response["message"].thinking

            response = ollama.chat(
                model=self.model.value,
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

    def chat_streamed(
        self, user_message: str, generate_kwargs: Optional[dict] = None, use_tools: bool = True
    ) -> Iterator[str]:
        generate_kwargs, tools = self._chat_setup(user_message, generate_kwargs, use_tools)

        response = ollama.chat(
            model=self.model.value,
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
            tool_calls = []
            for tool_call in response_part["message"].tool_calls:
                tool_calls.append(
                    {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    }
                )

            self._handle_tool_calls(tool_calls, tools)

            if thinking:
                self.messages[-1 - len(tool_calls)]["thinking"] = thinking
                thinking = ""

            response = ollama.chat(
                model=self.model.value,
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
