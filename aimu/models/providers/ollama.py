from ..base import (
    StreamingContentType,
    StreamChunk,
    Model,
    ModelSpec,
    BaseModelClient,
    BaseEmbeddingClient,
    EmbeddingModel,
    OllamaEmbeddingSpec,
    classproperty,
)
from .._internal.image_input import _adapt_messages_for_ollama, _build_user_content_blocks, _ollama_split_message
from .._internal.usage import usage_from_ollama

import ollama
import logging
from typing import Any, Iterator, Optional, Union

logger = logging.getLogger(__name__)


_GEMMA_KWARGS = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
_QWEN_3_6_KWARGS = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.9,
    "repetition_penalty": 1.0,
}
_QWEN_3_5_KWARGS = {**_QWEN_3_6_KWARGS, "presence_penalty": 1.5}


class OllamaModel(Model):
    # Alibaba
    QWEN_3_6_35B = ModelSpec(
        "qwen3.6:35b", tools=True, thinking=True, generation_kwargs=_QWEN_3_6_KWARGS, structured_output=True
    )
    QWEN_3_6_27B = ModelSpec(
        "qwen3.6:27b", tools=True, thinking=True, generation_kwargs=_QWEN_3_6_KWARGS, structured_output=True
    )
    QWEN_3_5_9B = ModelSpec(
        "qwen3.5:9b", tools=True, thinking=True, generation_kwargs=_QWEN_3_5_KWARGS, structured_output=True
    )
    QWEN_3_32B = ModelSpec("qwen3:32b", tools=True, thinking=True, structured_output=True)
    QWEN_3_8B = ModelSpec("qwen3:8b", tools=True, thinking=True, structured_output=True)
    # Google: these weights support audio; add audio=True once Ollama API exposes audio input
    GEMMA_4_E4B = ModelSpec(
        "gemma4:e4b", tools=True, thinking=True, vision=True, generation_kwargs=_GEMMA_KWARGS, structured_output=True
    )
    GEMMA_4_12B = ModelSpec(
        "gemma4:12b", tools=True, thinking=True, vision=True, generation_kwargs=_GEMMA_KWARGS, structured_output=True
    )
    GEMMA_4_26B = ModelSpec(
        "gemma4:26b", tools=True, thinking=True, vision=True, generation_kwargs=_GEMMA_KWARGS, structured_output=True
    )
    GEMMA_4_31B = ModelSpec(
        "gemma4:31b", tools=True, thinking=True, vision=True, generation_kwargs=_GEMMA_KWARGS, structured_output=True
    )
    GEMMA_3_12B = ModelSpec("gemma3:12b", vision=True, structured_output=True)
    # NVIDIA
    NEMOTRON_CASCADE_2_30B = ModelSpec("nemotron-cascade-2:30b", tools=True, thinking=True, structured_output=True)
    NEMOTRON_3_NANO_30B = ModelSpec("nemotron-3-nano:30b", tools=True, thinking=True, structured_output=True)
    # Zhipu AI: doesn't use tools when expected
    GLM_4_7_FLASH_31B_Q4 = ModelSpec("glm-4.7-flash:q4_K_M", thinking=True, structured_output=True)
    # OpenAI
    GPT_OSS_20B = ModelSpec("gpt-oss:20b", tools=True, thinking=True, structured_output=True)
    # Mistral
    MAGISTRAL_SMALL_24B = ModelSpec("magistral:24b", tools=True, thinking=True, structured_output=True)
    MINISTRAL_3_14B = ModelSpec("ministral-3:14b", tools=True, structured_output=True)
    # Microsoft
    PHI_4_MINI_3_8B = ModelSpec("phi4-mini:3.8b", structured_output=True)
    PHI_4_14B = ModelSpec("phi4:14b", structured_output=True)
    # DeepSeek
    DEEPSEEK_R1_8B = ModelSpec("deepseek-r1:8b", thinking=True, structured_output=True)
    # HuggingFace: tool call responses don't always look correct
    SMOLLM2_1_7B = ModelSpec("smollm2:1.7b", structured_output=True)
    # Meta: don't reliably use tools when expected
    LLAMA_3_2_3B = ModelSpec("llama3.2:3b", structured_output=True)
    LLAMA_3_1_8B = ModelSpec("llama3.1:8b", structured_output=True)


class OllamaClient(BaseModelClient):
    MODELS = OllamaModel

    def __init__(
        self,
        model: OllamaModel,
        system_message: Optional[str] = None,
        model_keep_alive_seconds: int = 60,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        super().__init__(model, None, system_message)

        if max_retries is not None:
            raise ValueError(
                "Ollama's native client has no request-retry support. Omit max_retries, "
                "or use the 'ollama-openai' provider, which routes through the OpenAI SDK and supports it."
            )

        # TODO extend model_keep_alive_seconds to other model clients
        self.model_keep_alive_seconds = model_keep_alive_seconds
        self.default_generate_kwargs = dict(model.generation_kwargs)

        self._client = ollama.Client(**({"timeout": timeout} if timeout is not None else {}))
        self._client.pull(model.value)

    @classproperty
    def THINKING_MODELS(cls) -> list[Model]:  # noqa: N805
        return [m for m in cls.MODELS if m.supports_thinking]

    @classproperty
    def TOOL_MODELS(cls) -> list[Model]:  # noqa: N805
        return [m for m in cls.MODELS if m.supports_tools]

    @classproperty
    def VISION_MODELS(cls) -> list[Model]:  # noqa: N805
        return [m for m in cls.MODELS if m.supports_vision]

    @classproperty
    def AUDIO_MODELS(cls) -> list[Model]:  # noqa: N805
        return [m for m in cls.MODELS if m.supports_audio]

    @classproperty
    def STRUCTURED_MODELS(cls) -> list[Model]:  # noqa: N805
        return [m for m in cls.MODELS if m.supports_structured_output]

    def _update_generate_kwargs(self, generate_kwargs: Optional[dict] = None) -> dict[str, str]:
        if not generate_kwargs:
            generate_kwargs = self.default_generate_kwargs

        if "max_tokens" in generate_kwargs:
            generate_kwargs["num_predict"] = generate_kwargs.pop("max_tokens")

        return generate_kwargs

    def _generate(
        self,
        prompt: str,
        generate_kwargs: Optional[dict] = None,
        stream: bool = False,
        images: Optional[list] = None,
        audio: Optional[list] = None,
        response_format: Optional[dict] = None,
    ) -> Union[str, Iterator[StreamChunk]]:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)
        gen_images = self._extract_ollama_images(images)

        if stream:
            return self._generate_streamed(prompt, generate_kwargs, images=gen_images, response_format=response_format)

        response = self._client.generate(
            model=self.model.value,
            prompt=prompt,
            images=gen_images,
            options=generate_kwargs,
            think=self.is_thinking_model,
            keep_alive=self.model_keep_alive_seconds,
            format=response_format,
        )

        logger.debug("LLM raw response: %s", response)
        self.last_usage = usage_from_ollama(response)

        self.last_thinking = ""

        if not self.is_thinking_model:
            return response["response"]

        self.last_thinking = response.thinking

        return response.response

    @staticmethod
    def _extract_ollama_images(images: Optional[list]) -> Optional[list]:
        """Normalise vision inputs to Ollama's bare-base64 list (the generate endpoint's ``images=``).

        Reuses ``_ollama_split_message`` so the http(s)-URL rejection matches the chat path.
        """
        if not images:
            return None
        adapted = _ollama_split_message({"role": "user", "content": _build_user_content_blocks("", images)})
        return adapted.get("images")

    def _generate_streamed(
        self,
        prompt: str,
        generate_kwargs: dict,
        images: Optional[list] = None,
        response_format: Optional[dict] = None,
    ) -> Iterator[StreamChunk]:
        response = self._client.generate(
            model=self.model.value,
            prompt=prompt,
            images=images,
            options=generate_kwargs,
            stream=True,
            think=self.is_thinking_model,
            keep_alive=self.model_keep_alive_seconds,
            format=response_format,
        )

        self.last_thinking = ""
        self.last_usage = None

        response_part = None
        for response_part in response:
            logger.debug("LLM raw response part: %s", response_part)

            if response_part.thinking:
                self.last_thinking += response_part.thinking
                yield StreamChunk(StreamingContentType.THINKING, response_part.thinking)

            yield StreamChunk(StreamingContentType.GENERATING, response_part["response"])

        # The final part (done=true) carries the eval counts.
        if response_part is not None:
            self.last_usage = usage_from_ollama(response_part)

    def _chat(
        self,
        user_message: str,
        generate_kwargs: Optional[dict] = None,
        use_tools: bool = True,
        stream: bool = False,
        images: Optional[list] = None,
        audio: Optional[list] = None,
        response_format: Optional[dict] = None,
    ) -> Union[str, Iterator[StreamChunk]]:
        generate_kwargs, tools = self._chat_setup(user_message, generate_kwargs, use_tools, images=images, audio=audio)

        if stream:
            return self._chat_streamed(generate_kwargs, tools, response_format=response_format)

        response = self._client.chat(
            model=self.model.value,
            messages=_adapt_messages_for_ollama(self.messages),
            options=generate_kwargs,
            tools=tools,
            think=self.is_thinking_model,
            keep_alive=self.model_keep_alive_seconds,
            format=response_format,
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

            self._handle_tool_calls(tool_calls)

            if response["message"].thinking:
                self.messages[-1 - len(tool_calls)]["thinking"] = response["message"].thinking

            response = self._client.chat(
                model=self.model.value,
                messages=_adapt_messages_for_ollama(self.messages),
                options=generate_kwargs,
                tools=tools,
                think=self.is_thinking_model,
                keep_alive=self.model_keep_alive_seconds,
                format=response_format,
            )

            logger.debug("LLM raw response: %s", response)

        self.last_usage = usage_from_ollama(response)
        self.messages.append({"role": response["message"].role, "content": response["message"].content})

        if response["message"].thinking:
            self.messages[-1]["thinking"] = response["message"].thinking

        return response["message"].content

    def _chat_streamed(
        self, generate_kwargs: dict, tools: list, response_format: Optional[dict] = None
    ) -> Iterator[StreamChunk]:
        self.last_usage = None

        def _open():
            return self._client.chat(
                model=self.model.value,
                messages=_adapt_messages_for_ollama(self.messages),
                options=generate_kwargs,
                tools=tools,
                stream=True,
                think=self.is_thinking_model,
                keep_alive=self.model_keep_alive_seconds,
                format=response_format,
            )

        # First turn. Consume the whole stream, collecting tool calls from ANY part (Ollama can
        # emit an empty transitional part before the one carrying tool_calls) and streaming only
        # non-empty content (so empty/transitional/done parts never surface as stray output).
        turn = {}
        yield from self._consume_turn(_open(), turn)

        if turn["tool_calls"]:
            msgs_before = len(self.messages)
            yield from self._handle_tool_calls_streamed(turn["tool_calls"])
            if turn["thinking"]:
                self.messages[msgs_before]["thinking"] = turn["thinking"]
            if turn["last_part"] is not None:
                self.last_usage = usage_from_ollama(turn["last_part"])

            # Follow-up turn after the tool results (the Agent loop drives any further rounds).
            turn = {}
            yield from self._consume_turn(_open(), turn)

        message = {"role": turn["role"], "content": turn["content"]}
        if turn["thinking"]:
            message["thinking"] = turn["thinking"]
        self.messages.append(message)
        if turn["last_part"] is not None:
            self.last_usage = usage_from_ollama(turn["last_part"])

    @staticmethod
    def _consume_turn(response, out: dict) -> Iterator[StreamChunk]:
        """Stream one Ollama turn: yield THINKING and non-empty GENERATING chunks, and record
        ``thinking`` / ``content`` / ``tool_calls`` / ``role`` / ``last_part`` into ``out``.

        Collecting tool_calls across every part (not just the first non-thinking one) is what
        makes a tool call survive an empty transitional part; skipping empty content keeps the
        per-part empty ``done`` chunk from rendering as a stray response.
        """
        thinking = content = ""
        role = "assistant"
        tool_calls: list = []
        last_part = None
        for part in response:
            logger.debug("LLM raw response part: %s", part)
            msg = part["message"]
            last_part = part
            if msg.role:
                role = msg.role
            if msg.thinking:
                thinking += msg.thinking
                yield StreamChunk(StreamingContentType.THINKING, msg.thinking)
            if msg.tool_calls:
                tool_calls.extend(
                    {"name": tc.function.name, "arguments": tc.function.arguments} for tc in msg.tool_calls
                )
            if msg.content:
                content += msg.content
                yield StreamChunk(StreamingContentType.GENERATING, msg.content)
        out.update(thinking=thinking, content=content, tool_calls=tool_calls, role=role, last_part=last_part)


class OllamaEmbeddingModel(EmbeddingModel):
    """Catalog of Ollama embedding models (pull with ``ollama pull <id>``).

    Each member's value is an :class:`OllamaEmbeddingSpec`. ``.value`` is the Ollama
    model tag; ``.spec`` returns the full spec.
    """

    NOMIC_EMBED_TEXT = OllamaEmbeddingSpec("nomic-embed-text", dimensions=768, max_input_tokens=8192)
    MXBAI_EMBED_LARGE = OllamaEmbeddingSpec("mxbai-embed-large", dimensions=1024, max_input_tokens=512)
    BGE_M3 = OllamaEmbeddingSpec("bge-m3", dimensions=1024, max_input_tokens=8192)
    ALL_MINILM = OllamaEmbeddingSpec("all-minilm", dimensions=384, max_input_tokens=512)


def _parse_embedding_model_string(s: str) -> OllamaEmbeddingSpec:
    """Resolve an ``"ollama:<model_id>"`` string to a known :class:`OllamaEmbeddingModel` spec."""
    if ":" not in s:
        raise ValueError(
            f"Ollama embedding model string must be in 'provider:model_id' form "
            f"(e.g. 'ollama:nomic-embed-text'). Got: {s!r}"
        )
    provider, model_id = s.split(":", 1)
    if provider != "ollama":
        raise ValueError(f"Only 'ollama:' provider is supported for OllamaEmbeddingClient. Got provider: {provider!r}")
    for member in OllamaEmbeddingModel:
        if member.value == model_id:
            return member.spec
    available = sorted(m.value for m in OllamaEmbeddingModel)
    raise ValueError(
        f"Unknown Ollama embedding model id {model_id!r}. AIMU supports curated models only; "
        f"pass a known id, an OllamaEmbeddingModel member, or a hand-built OllamaEmbeddingSpec. "
        f"Available ids: {available}"
    )


class OllamaEmbeddingClient(BaseEmbeddingClient):
    """Text-embedding client for a local Ollama server.

    Pass an :class:`OllamaEmbeddingModel` member, an :class:`OllamaEmbeddingSpec`, or an
    ``"ollama:<model_id>"`` string. The model is pulled on construction (same as
    :class:`OllamaClient`).
    """

    MODELS = OllamaEmbeddingModel

    def __init__(
        self,
        model: "OllamaEmbeddingModel | OllamaEmbeddingSpec | str",
        model_kwargs: Optional[dict] = None,
    ):
        if isinstance(model, str):
            spec = _parse_embedding_model_string(model)
        elif isinstance(model, OllamaEmbeddingModel):
            spec = model.spec
        elif isinstance(model, OllamaEmbeddingSpec):
            spec = model
        else:
            raise TypeError(
                f"OllamaEmbeddingClient expects an OllamaEmbeddingModel member, OllamaEmbeddingSpec, "
                f"or 'ollama:<model_id>' string. Got: {type(model).__name__}"
            )
        super().__init__(model=model, model_kwargs=model_kwargs)
        self.spec = spec
        ollama.pull(spec.id)

    def _embed(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        response = ollama.embed(model=self.spec.id, input=texts, **kwargs)
        return [list(vector) for vector in response["embeddings"]]

    def __repr__(self) -> str:
        return f"OllamaEmbeddingClient(model={self.spec.id!r})"
