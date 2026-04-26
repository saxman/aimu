from ..base import StreamingContentType, StreamChunk, Model, BaseModelClient, classproperty

import ollama
import logging
from typing import Iterator, Optional, Union

logger = logging.getLogger(__name__)


class OllamaModel(Model):
    def __init__(self, value, supports_tools=False, supports_thinking=False, generation_kwargs=None):
        super().__init__(value, supports_tools, supports_thinking, generation_kwargs)

    # Alibaba
    QWEN_3_6_35B = ("qwen3.6:35b", True, True)
    QWEN_3_5_9B = ("qwen3.5:9b", True, True)
    QWEN_3_32B = ("qwen3:32b", True, True)
    QWEN_3_8B = ("qwen3:8b", True, True)
    # Google
    GEMMA_4_E4B = ("gemma4:e4b", True, True, {"temperature": 1.0, "top_p": 0.95, "top_k": 64})
    GEMMA_4_26B = ("gemma4:26b", True, True, {"temperature": 1.0, "top_p": 0.95, "top_k": 64})
    GEMMA_4_31B = ("gemma4:31b", True, True, {"temperature": 1.0, "top_p": 0.95, "top_k": 64})
    GEMMA_3_12B = "gemma3:12b"
    # NVIDIA
    NEMOTRON_CASCADE_2_30B = ("nemotron-cascade-2:30b", True, True)
    NEMOTRON_3_NANO_30B = ("nemotron-3-nano:30b", True, True)
    # Zhipu AI
    GLM_4_7_FLASH_31B_Q4 = ("glm-4.7-flash:q4_K_M", False, True)  # doesn't use tools when expected
    # OpenAI
    GPT_OSS_20B = ("gpt-oss:20b", True, True)
    # Mistral
    MAGISTRAL_SMALL_24B = ("magistral:24b", True, True)
    MINISTRAL_3_14B = ("ministral-3:14b", True)
    # Microsoft
    PHI_4_MINI_3_8B = "phi4-mini:3.8b"
    PHI_4_14B = "phi4:14b"
    # DeepSeek
    DEEPSEEK_R1_8B = ("deepseek-r1:8b", False, True)
    # HuggingFace
    SMOLLM2_1_7B = ("smollm2:1.7b", False)  # tool call responses don't always look correct
    # Meta
    LLAMA_3_2_3B = ("llama3.2:3b", False)  # doesn't use tools when expected
    LLAMA_3_1_8B = ("llama3.1:8b", False)  # doesn't use tools when expected


class OllamaClient(BaseModelClient):
    MODELS = OllamaModel

    def __init__(self, model: OllamaModel, system_message: Optional[str] = None, model_keep_alive_seconds: int = 60):
        super().__init__(model, None, system_message)

        # TODO extend model_keep_alive_seconds to other model clients
        self.model_keep_alive_seconds = model_keep_alive_seconds
        self.default_generate_kwargs = dict(model.generation_kwargs)

        ollama.pull(model.value)

    @classproperty
    def THINKING_MODELS(cls) -> list[Model]:  # noqa: N805
        return [m for m in cls.MODELS if m.supports_thinking]

    @classproperty
    def TOOL_MODELS(cls) -> list[Model]:  # noqa: N805
        return [m for m in cls.MODELS if m.supports_tools]

    def _update_generate_kwargs(self, generate_kwargs: Optional[dict] = None) -> dict[str, str]:
        if not generate_kwargs:
            generate_kwargs = self.default_generate_kwargs

        if "max_tokens" in generate_kwargs:
            generate_kwargs["num_predict"] = generate_kwargs.pop("max_tokens")

        return generate_kwargs

    def generate(
        self,
        prompt: str,
        generate_kwargs: Optional[dict] = None,
        stream: bool = False,
        include_thinking: bool = True,
    ) -> Union[str, Iterator[StreamChunk]]:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        if stream:
            return self._generate_streamed(prompt, generate_kwargs, include_thinking)

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

    def _generate_streamed(
        self,
        prompt: str,
        generate_kwargs: dict,
        include_thinking: bool,
    ) -> Iterator[StreamChunk]:
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
                    yield StreamChunk(StreamingContentType.THINKING, response_part.thinking)
                else:
                    continue

            yield StreamChunk(StreamingContentType.GENERATING, response_part["response"])

    def chat(
        self,
        user_message: str,
        generate_kwargs: Optional[dict] = None,
        use_tools: bool = True,
        stream: bool = False,
    ) -> Union[str, Iterator[StreamChunk]]:
        generate_kwargs, tools = self._chat_setup(user_message, generate_kwargs, use_tools)

        if stream:
            return self._chat_streamed(generate_kwargs, tools)

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

    def _chat_streamed(self, generate_kwargs: dict, tools: list) -> Iterator[StreamChunk]:
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
            yield StreamChunk(StreamingContentType.THINKING, thinking)
            for response_part in response:
                logger.debug("LLM raw response part: %s", response_part)
                if response_part["message"].thinking:
                    thinking += response_part["message"].thinking
                    yield StreamChunk(StreamingContentType.THINKING, response_part["message"].thinking)
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
                yield StreamChunk(
                    StreamingContentType.TOOL_CALLING,
                    {"name": tc["function"]["name"], "response": tr["content"]},
                )

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
                yield StreamChunk(StreamingContentType.THINKING, thinking)
                for response_part in response:
                    logger.debug("LLM raw response part: %s", response_part)
                    if response_part["message"].thinking:
                        thinking += response_part["message"].thinking
                        yield StreamChunk(StreamingContentType.THINKING, response_part["message"].thinking)
                    else:
                        break

        content = response_part["message"].content
        yield StreamChunk(StreamingContentType.GENERATING, content)

        for response_part in response:
            logger.debug("LLM raw response part: %s", response_part)
            content += response_part["message"].content
            yield StreamChunk(StreamingContentType.GENERATING, response_part["message"].content)

        message = {"role": response_part["message"].role, "content": content}
        if thinking:
            message["thinking"] = thinking
        self.messages.append(message)
