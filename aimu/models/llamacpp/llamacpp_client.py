import json
import logging
from typing import Iterator, Optional, Any, Union

from ..base import StreamingContentType, StreamChunk, Model, BaseModelClient, classproperty
from .._thinking import _split_thinking, _ThinkingParser

logger = logging.getLogger(__name__)


class LlamaCppModel(Model):
    def __init__(self, value, supports_tools=False, supports_thinking=False):
        super().__init__(value, supports_tools, supports_thinking)

    LLAMA_3_1_8B = ("llama-3.1-8b", False)
    LLAMA_3_2_3B = ("llama-3.2-3b", False)
    MISTRAL_7B = ("mistral-7b", True)
    QWEN_3_4B = ("qwen3-4b", True, True)
    QWEN_3_8B = ("qwen3-8b", True, True)
    DEEPSEEK_R1_7B = ("deepseek-r1-7b", False, True)
    PHI_4_MINI = ("phi-4-mini", True)


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
        verbose: bool = False,
        system_message: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
    ):
        super().__init__(model, model_kwargs, system_message)
        self.default_generate_kwargs = self.DEFAULT_GENERATE_KWARGS.copy()

        from llama_cpp import Llama

        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            chat_format=chat_format,
            verbose=verbose,
        )

    @classproperty
    def THINKING_MODELS(cls) -> list[Model]:  # noqa: N805
        return [m for m in cls.MODELS if m.supports_thinking]

    @classproperty
    def TOOL_MODELS(cls) -> list[Model]:  # noqa: N805
        return [m for m in cls.MODELS if m.supports_tools]

    def _update_generate_kwargs(self, generate_kwargs: Optional[dict[str, Any]] = None) -> dict:
        if not generate_kwargs:
            return self.default_generate_kwargs.copy()
        return {**self.default_generate_kwargs, **generate_kwargs}

    def _iter_stream(self, stream, include_thinking: bool = True) -> Iterator[StreamChunk]:
        """Iterate a completion stream, yielding StreamChunks and updating self.last_thinking."""
        self.last_thinking = ""
        parser = _ThinkingParser() if self.is_thinking_model else None

        for chunk in stream:
            delta = chunk["choices"][0]["delta"]
            text = delta.get("content") or ""
            if not text:
                continue
            logger.debug("LLM raw chunk: %s", chunk)
            if parser:
                for phase, part in parser.feed(text):
                    if phase == StreamingContentType.THINKING:
                        self.last_thinking += part
                        if include_thinking:
                            yield StreamChunk(StreamingContentType.THINKING, part)
                    else:
                        yield StreamChunk(StreamingContentType.GENERATING, part)
            else:
                yield StreamChunk(StreamingContentType.GENERATING, text)

    def generate(
        self,
        prompt: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        include_thinking: bool = True,
    ) -> Union[str, Iterator[StreamChunk]]:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        if stream:
            return self._generate_streamed(prompt, generate_kwargs, include_thinking)

        response = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            **generate_kwargs,
        )
        logger.debug("LLM raw response: %s", response)
        content = response["choices"][0]["message"]["content"] or ""

        self.last_thinking = ""
        if self.is_thinking_model:
            self.last_thinking, content = _split_thinking(content)

        return content

    def _generate_streamed(
        self,
        prompt: str,
        generate_kwargs: dict[str, Any],
        include_thinking: bool,
    ) -> Iterator[StreamChunk]:
        stream = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **generate_kwargs,
        )
        yield from self._iter_stream(stream, include_thinking)

    def chat(
        self,
        user_message: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        use_tools: bool = True,
        stream: bool = False,
    ) -> Union[str, Iterator[StreamChunk]]:
        generate_kwargs, tools = self._chat_setup(user_message, generate_kwargs, use_tools)

        if stream:
            return self._chat_streamed(generate_kwargs, tools)

        response = self._llm.create_chat_completion(
            messages=self.messages,
            tools=tools if tools else None,
            **generate_kwargs,
        )
        logger.debug("LLM raw response: %s", response)
        msg = response["choices"][0]["message"]

        if msg.get("tool_calls"):
            tool_calls = [
                {"name": tc["function"]["name"], "arguments": json.loads(tc["function"]["arguments"])}
                for tc in msg["tool_calls"]
            ]
            self._handle_tool_calls(tool_calls, tools)

            response = self._llm.create_chat_completion(
                messages=self.messages,
                tools=tools if tools else None,
                **generate_kwargs,
            )
            logger.debug("LLM raw response (after tools): %s", response)
            msg = response["choices"][0]["message"]

        content = msg.get("content") or ""
        self.last_thinking = ""
        if self.is_thinking_model:
            self.last_thinking, content = _split_thinking(content)

        self.messages.append({"role": "assistant", "content": content})
        return content

    def _chat_streamed(self, generate_kwargs: dict[str, Any], tools: list) -> Iterator[StreamChunk]:
        stream = self._llm.create_chat_completion(
            messages=self.messages,
            stream=True,
            tools=tools if tools else None,
            **generate_kwargs,
        )

        # Buffer the first stream to detect tool calls vs. content.
        tool_calls_acc: dict[int, dict] = {}
        first_pass_chunks: list[StreamChunk] = []
        parser = _ThinkingParser() if self.is_thinking_model else None
        self.last_thinking = ""

        for chunk in stream:
            delta = chunk["choices"][0]["delta"]
            logger.debug("LLM raw chunk: %s", chunk)
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
                        first_pass_chunks.append(StreamChunk(phase, part))
                else:
                    first_pass_chunks.append(StreamChunk(StreamingContentType.GENERATING, text))

        if not tool_calls_acc:
            full_content = ""
            for sc in first_pass_chunks:
                if sc.phase == StreamingContentType.GENERATING:
                    full_content += sc.content
                yield sc
            self.messages.append({"role": "assistant", "content": full_content})
            return

        # Tool call path: dispatch calls, emit TOOL_CALLING chunks, then stream second response.
        tool_calls = [{"name": tc["name"], "arguments": json.loads(tc["arguments"])} for tc in tool_calls_acc.values()]
        msgs_before = len(self.messages)
        self._handle_tool_calls(tool_calls, tools)

        for i, tc in enumerate(self.messages[msgs_before]["tool_calls"]):
            yield StreamChunk(
                StreamingContentType.TOOL_CALLING,
                {"name": tc["function"]["name"], "response": self.messages[msgs_before + 1 + i]["content"]},
            )

        stream2 = self._llm.create_chat_completion(
            messages=self.messages,
            stream=True,
            tools=tools if tools else None,
            **generate_kwargs,
        )

        full_content = ""
        for sc in self._iter_stream(stream2):
            if sc.phase == StreamingContentType.GENERATING:
                full_content += sc.content
            yield sc
        self.messages.append({"role": "assistant", "content": full_content})
