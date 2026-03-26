from ..base_client import Model, ModelCapabilities, ModelClient, StreamChunk, StreamPhase

import ollama
import logging
from typing import Iterator, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OllamaCapabilities(ModelCapabilities):
    pass


class OllamaModel(str, Model):
    GPT_OSS_20B = "gpt-oss:20b"
    LLAMA_3_1_8B = "llama3.1:8b"
    LLAMA_3_2_3B = "llama3.2:3b"
    # LLAMA_3_3_70B = "llama3.3:70b"  # model too big
    GEMMA_3_12B = "gemma3:12b"
    PHI_4_14B = "phi4:14b"
    PHI_4_MINI_3_8B = "phi4-mini:3.8b"
    DEEPSEEK_R1_8B = "deepseek-r1:8b"
    MISTRAL_7B = "mistral:7b"  # tool support disabled: issue with tool usage
    MISTRAL_NEMO_12B = "mistral-nemo:12b"  # tool support disabled: issue with tool usage
    MISTRAL_SMALL_3_2_24B = "mistral-small3.2:24b"
    MAGISTRAL_SMALL_24B = "magistral:24b"  # tool/thinking support disabled: test issues
    MINISTRAL_3_14B = "ministral-3:14b"
    QWEN_3_5_9B = "qwen3.5:9b"
    GLM_4_7_FLASH_31B_Q4 = "glm-4.7-flash:q4_K_M"
    SMOLLM2_1_7B = "smollm2:latest"  # "smollm2:1.7b" error downloading model, using latest for now


OLLAMA_CAPABILITIES: dict[OllamaModel, OllamaCapabilities] = {
    OllamaModel.GPT_OSS_20B: OllamaCapabilities(supports_tools=True, supports_thinking=True),
    OllamaModel.LLAMA_3_1_8B: OllamaCapabilities(supports_tools=True),
    OllamaModel.LLAMA_3_2_3B: OllamaCapabilities(supports_tools=True),
    OllamaModel.DEEPSEEK_R1_8B: OllamaCapabilities(supports_thinking=True),
    OllamaModel.MISTRAL_SMALL_3_2_24B: OllamaCapabilities(supports_tools=True),
    OllamaModel.MINISTRAL_3_14B: OllamaCapabilities(supports_tools=True),
    OllamaModel.QWEN_3_5_9B: OllamaCapabilities(supports_tools=True, supports_thinking=True),
    OllamaModel.GLM_4_7_FLASH_31B_Q4: OllamaCapabilities(supports_tools=True, supports_thinking=True),
    OllamaModel.SMOLLM2_1_7B: OllamaCapabilities(supports_tools=True),
}


class OllamaClient(ModelClient):
    MODELS = OllamaModel

    def __init__(self, model: OllamaModel, system_message: Optional[str] = None, model_keep_alive_seconds: int = 60):
        super().__init__(model, None, system_message)

        self.capabilities = OLLAMA_CAPABILITIES.get(model, OllamaCapabilities())

        # TODO extend model_keep_alive_seconds to other model clients
        self.model_keep_alive_seconds = model_keep_alive_seconds

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
            think=self.is_thinking_model,
            keep_alive=self.model_keep_alive_seconds,
        )

        logger.debug("LLM raw response: %s", response)

        self.last_thinking = ""

        if not self.is_thinking_model:
            return response["response"]

        self.last_thinking = response.thinking

        return response.response

    def generate_streamed(
        self,
        prompt: str,
        generate_kwargs: Optional[dict] = None,
        include_thinking: bool = False,
    ) -> Iterator[StreamChunk]:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        response = ollama.generate(
            model=self.model.value,
            prompt=prompt,
            options=generate_kwargs,
            stream=True,
            think=self.is_thinking_model,
            keep_alive=self.model_keep_alive_seconds,
        )

        self.last_thinking = ""

        for response_part in response:
            logger.debug("LLM raw response part: %s", response_part)

            if response_part.thinking:
                self.last_thinking += response_part.thinking
                if include_thinking:
                    yield StreamChunk(StreamPhase.THINKING, response_part.thinking)
                else:
                    continue

            yield StreamChunk(StreamPhase.GENERATING, response_part["response"])

        yield StreamChunk(StreamPhase.DONE, "")

    def chat(self, user_message: str, generate_kwargs: Optional[dict] = None, use_tools: bool = True) -> str:
        generate_kwargs, tools = self._chat_setup(user_message, generate_kwargs, use_tools)

        response = ollama.chat(
            model=self.model.value,
            messages=self.messages,
            options=generate_kwargs,
            tools=tools,
            think=self.is_thinking_model,
            keep_alive=self.model_keep_alive_seconds,
        )

        logger.debug("LLM raw response: %s", response)

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
                think=self.is_thinking_model,
                keep_alive=self.model_keep_alive_seconds,
            )

            logger.debug("LLM raw response: %s", response)

        self.messages.append({"role": response["message"].role, "content": response["message"].content})

        if response["message"].thinking:
            self.messages[-1]["thinking"] = response["message"].thinking

        return response["message"].content

    def chat_streamed(
        self, user_message: str, generate_kwargs: Optional[dict] = None, use_tools: bool = True
    ) -> Iterator[StreamChunk]:
        generate_kwargs, tools = self._chat_setup(user_message, generate_kwargs, use_tools)

        response = ollama.chat(
            model=self.model.value,
            messages=self.messages,
            options=generate_kwargs,
            tools=tools,
            stream=True,
            think=self.is_thinking_model,
            keep_alive=self.model_keep_alive_seconds,
        )

        response_part = next(response)
        logger.debug("LLM raw response part: %s", response_part)

        thinking = ""
        if response_part["message"].thinking:
            thinking = response_part["message"].thinking
            yield StreamChunk(StreamPhase.THINKING, thinking)
            for response_part in response:
                logger.debug("LLM raw response part: %s", response_part)
                if response_part["message"].thinking:
                    thinking += response_part["message"].thinking
                    yield StreamChunk(StreamPhase.THINKING, response_part["message"].thinking)
                else:
                    break

        if response_part["message"].tool_calls:
            tool_calls = [
                {"name": tc.function.name, "arguments": tc.function.arguments}
                for tc in response_part["message"].tool_calls
            ]

            msgs_before = len(self.messages)
            self._handle_tool_calls(tool_calls, tools)

            if thinking:
                self.messages[msgs_before]["thinking"] = thinking
                thinking = ""

            for tc, tr in zip(self.messages[msgs_before]["tool_calls"], self.messages[msgs_before + 1 :]):
                yield StreamChunk(StreamPhase.TOOL_CALLING, {"name": tc["function"]["name"], "response": tr["content"]})

            response = ollama.chat(
                model=self.model.value,
                messages=self.messages,
                options=generate_kwargs,
                tools=tools,
                stream=True,
                think=self.is_thinking_model,
                keep_alive=self.model_keep_alive_seconds,
            )

            response_part = next(response)
            logger.debug("LLM raw response part: %s", response_part)

            thinking = ""
            if response_part["message"].thinking:
                thinking = response_part["message"].thinking
                yield StreamChunk(StreamPhase.THINKING, thinking)
                for response_part in response:
                    logger.debug("LLM raw response part: %s", response_part)
                    if response_part["message"].thinking:
                        thinking += response_part["message"].thinking
                        yield StreamChunk(StreamPhase.THINKING, response_part["message"].thinking)
                    else:
                        break

        content = response_part["message"].content
        yield StreamChunk(StreamPhase.GENERATING, content)

        for response_part in response:
            logger.debug("LLM raw response part: %s", response_part)
            content += response_part["message"].content
            yield StreamChunk(StreamPhase.GENERATING, response_part["message"].content)

        message = {"role": response_part["message"].role, "content": content}
        if thinking:
            message["thinking"] = thinking
        self.messages.append(message)

        yield StreamChunk(StreamPhase.DONE, "")
