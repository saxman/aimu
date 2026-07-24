import json
import logging
import threading
from typing import Iterator, Optional, Any, Union

# Hard module-load import so HAS_LLAMACPP accurately reflects that llama-cpp-python is
# installed (matches the diffusers/soundfile convention in the HuggingFace clients). The
# Llama model itself is still constructed lazily in __init__ to defer weight loading.
import llama_cpp

from ..base import StreamingContentType, StreamChunk, Model, ModelSpec, BaseModelClient, classproperty
from .._internal.image_input import _build_user_content_blocks
from ._thinking import _split_thinking, _ThinkingParser

logger = logging.getLogger(__name__)

_model_registry: dict[tuple, Any] = {}  # cache_key → Llama instance
_registry_lock = threading.Lock()


def _make_cache_key(model_path: str, n_ctx: int, n_gpu_layers: int, chat_format: str | None) -> tuple:
    return (model_path, n_ctx, n_gpu_layers, str(chat_format))


class LlamaCppModel(Model):
    LLAMA_3_1_8B = ModelSpec("llama-3.1-8b")
    LLAMA_3_2_3B = ModelSpec("llama-3.2-3b")
    MISTRAL_7B = ModelSpec("mistral-7b", tools=True)
    QWEN_3_4B = ModelSpec("qwen3-4b", tools=True, thinking=True)
    QWEN_3_8B = ModelSpec("qwen3-8b", tools=True, thinking=True)
    DEEPSEEK_R1_7B = ModelSpec("deepseek-r1-7b", thinking=True)
    PHI_4_MINI = ModelSpec("phi-4-mini", tools=True)
    GEMMA_4_12B = ModelSpec("gemma-4-12b", tools=True)


class LlamaCppClient(BaseModelClient):
    MODELS = LlamaCppModel

    DEFAULT_GENERATE_KWARGS = {
        "max_tokens": 1024,
        "temperature": 0.1,
    }

    def __init__(
        self,
        model: LlamaCppModel,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        chat_format: Optional[str] = None,
        chat_handler: Optional[Any] = None,
        verbose: bool = False,
        system_message: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
    ):
        super().__init__(model, model_kwargs, system_message)
        self.default_generate_kwargs = self.DEFAULT_GENERATE_KWARGS.copy()

        self._cache_key = _make_cache_key(model_path, n_ctx, n_gpu_layers, chat_format)
        with _registry_lock:
            if self._cache_key in _model_registry:
                self._llm = _model_registry[self._cache_key]
                return

        self._llm = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            chat_format=chat_format,
            chat_handler=chat_handler,
            verbose=verbose,
        )
        with _registry_lock:
            _model_registry[self._cache_key] = self._llm

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

    def _update_generate_kwargs(self, generate_kwargs: Optional[dict[str, Any]] = None) -> dict:
        if not generate_kwargs:
            return self.default_generate_kwargs.copy()
        return {**self.default_generate_kwargs, **generate_kwargs}

    def _iter_stream(self, stream) -> Iterator[StreamChunk]:
        """Iterate a completion stream, yielding StreamChunks and updating self.last_thinking."""
        self.last_thinking = ""
        parser = _ThinkingParser() if self.is_thinking_model else None

        for chunk in stream:
            delta = chunk["choices"][0]["delta"]
            reasoning = delta.get("reasoning_content")
            if reasoning:
                self.last_thinking += reasoning
                yield StreamChunk(StreamingContentType.THINKING, reasoning)
            text = delta.get("content") or ""
            if not text:
                continue
            logger.debug("LLM raw chunk: %s", chunk)
            if parser:
                for phase, part in parser.feed(text):
                    if phase == StreamingContentType.THINKING:
                        self.last_thinking += part
                        yield StreamChunk(StreamingContentType.THINKING, part)
                    else:
                        yield StreamChunk(StreamingContentType.GENERATING, part)
            else:
                yield StreamChunk(StreamingContentType.GENERATING, text)

    def _generate(
        self,
        prompt: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
        audio: Optional[list] = None,
    ) -> Union[str, Iterator[StreamChunk]]:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        if stream:
            return self._generate_streamed(prompt, generate_kwargs, images=images, audio=audio)

        content_in = _build_user_content_blocks(prompt, images) if images else prompt
        response = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": content_in}],
            **generate_kwargs,
        )
        logger.debug("LLM raw response: %s", response)
        msg = response["choices"][0]["message"]
        content = msg["content"] or ""

        self.last_thinking = ""
        reasoning = msg.get("reasoning_content")
        if reasoning:
            self.last_thinking = reasoning
        elif self.is_thinking_model:
            self.last_thinking, content = _split_thinking(content)

        return content

    def _generate_streamed(
        self,
        prompt: str,
        generate_kwargs: dict[str, Any],
        images: Optional[list] = None,
        audio: Optional[list] = None,
    ) -> Iterator[StreamChunk]:
        content_in = _build_user_content_blocks(prompt, images) if images else prompt
        stream = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": content_in}],
            stream=True,
            **generate_kwargs,
        )
        yield from self._iter_stream(stream)

    def _chat(
        self,
        user_message: Optional[str] = None,
        generate_kwargs: Optional[dict[str, Any]] = None,
        use_tools: bool = True,
        stream: bool = False,
        images: Optional[list] = None,
        audio: Optional[list] = None,
    ) -> Union[str, Iterator[StreamChunk]]:
        generate_kwargs, tools = self._chat_setup(user_message, generate_kwargs, use_tools, images=images, audio=audio)

        if stream:
            return self._chat_streamed(generate_kwargs, tools)

        response = self._llm.create_chat_completion(
            messages=self.messages,
            tools=tools if tools else None,
            **generate_kwargs,
        )
        logger.debug("LLM raw response: %s", response)
        msg = response["choices"][0]["message"]

        self.last_thinking = ""
        # Prefer a server-provided reasoning_content field over parsing inline <think> tags.
        reasoning = msg.get("reasoning_content")

        # Single turn: if the model called tools, execute them and return. The model's response
        # to the tool results comes on the next chat() call (the loop lives in Agent).
        if msg.get("tool_calls"):
            tool_calls = [
                {"name": tc["function"]["name"], "arguments": json.loads(tc["function"]["arguments"])}
                for tc in msg["tool_calls"]
            ]
            text = msg.get("content") or ""
            if reasoning:
                self.last_thinking = reasoning
            elif self.is_thinking_model:
                self.last_thinking, text = _split_thinking(text)
            msgs_before = len(self.messages)
            self._record_tool_calls(tool_calls, content=text)
            if self.last_thinking:
                self.messages[msgs_before]["thinking"] = self.last_thinking
            return text

        content = msg.get("content") or ""
        if reasoning:
            self.last_thinking = reasoning
        elif self.is_thinking_model:
            self.last_thinking, content = _split_thinking(content)

        self._append_message({"role": "assistant", "content": content})
        if self.last_thinking:
            self.messages[-1]["thinking"] = self.last_thinking
        return content

    def _chat_streamed(self, generate_kwargs: dict[str, Any], tools: list) -> Iterator[StreamChunk]:
        stream = self._llm.create_chat_completion(
            messages=self.messages,
            stream=True,
            tools=tools if tools else None,
            **generate_kwargs,
        )

        # Yield content/thinking chunks as they arrive (incremental streaming) while accumulating
        # any tool-call deltas separately; content and tool_call deltas don't require buffering.
        tool_calls_acc: dict[int, dict] = {}
        full_content = ""
        parser = _ThinkingParser() if self.is_thinking_model else None
        self.last_thinking = ""

        for chunk in stream:
            delta = chunk["choices"][0]["delta"]
            logger.debug("LLM raw chunk: %s", chunk)
            reasoning = delta.get("reasoning_content")
            if reasoning:
                self.last_thinking += reasoning
                yield StreamChunk(StreamingContentType.THINKING, reasoning)
            if delta.get("tool_calls"):
                for tc_delta in delta["tool_calls"]:
                    acc = tool_calls_acc.setdefault(tc_delta["index"], {"name": "", "arguments": ""})
                    fn = tc_delta.get("function") or {}
                    if fn.get("name"):
                        acc["name"] += fn["name"]
                    if fn.get("arguments"):
                        acc["arguments"] += fn["arguments"]
            elif delta.get("content"):
                text = delta["content"]
                if parser:
                    for phase, part in parser.feed(text):
                        if phase == StreamingContentType.THINKING:
                            self.last_thinking += part
                        else:
                            full_content += part
                        yield StreamChunk(phase, part)
                else:
                    full_content += text
                    yield StreamChunk(StreamingContentType.GENERATING, text)

        if not tool_calls_acc:
            self._append_message({"role": "assistant", "content": full_content})
            if self.last_thinking:
                self.messages[-1]["thinking"] = self.last_thinking
            return

        # Single turn: prose/thinking already streamed above; now dispatch the tools (yields
        # TOOL_CALLING chunks via streaming-tool support in the base) and return. The model's
        # response to the tool results comes on the next chat() call (loop lives in Agent).
        tool_calls = [{"name": tc["name"], "arguments": json.loads(tc["arguments"])} for tc in tool_calls_acc.values()]
        tool_turn_thinking = self.last_thinking
        msgs_before = len(self.messages)
        self._record_tool_calls(tool_calls, content=full_content)
        if tool_turn_thinking:
            self.messages[msgs_before]["thinking"] = tool_turn_thinking
