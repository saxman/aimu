import json
import logging
import re
from typing import Iterator, Optional, Any

import openai

from ..base_client import StreamingContentType, StreamChunk, Model, ModelClient, classproperty

logger = logging.getLogger(__name__)


def _split_thinking(content: str) -> tuple[str, str]:
    """Extract <think>...</think> block from content. Returns (thinking, clean_content)."""
    match = re.match(r"<think>(.*?)</think>(.*)", content, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    # Unclosed think tag
    match = re.match(r"<think>(.*)", content, re.DOTALL)
    if match:
        return match.group(1).strip(), ""
    return "", content


class _ThinkingParser:
    """Stateful streaming parser that separates <think>...</think> from content across chunk boundaries."""

    def __init__(self):
        self._in_thinking = False
        self._buffer = ""

    def feed(self, text: str) -> list[tuple[StreamingContentType, str]]:
        results = []
        self._buffer += text
        while True:
            tag = "</think>" if self._in_thinking else "<think>"
            phase = StreamingContentType.THINKING if self._in_thinking else StreamingContentType.GENERATING
            idx = self._buffer.find(tag)

            if idx == -1:
                # Hold back any partial tag at the end of the buffer
                safe_len = self._safe_emit_length(self._buffer, tag)
                if safe_len > 0:
                    results.append((phase, self._buffer[:safe_len]))
                    self._buffer = self._buffer[safe_len:]
                break
            else:
                if idx > 0:
                    results.append((phase, self._buffer[:idx]))
                self._buffer = self._buffer[idx + len(tag) :]
                self._in_thinking = not self._in_thinking

        return results

    @staticmethod
    def _safe_emit_length(buffer: str, tag: str) -> int:
        """Return how many characters at the start of buffer can be safely emitted without risking a partial tag at the end."""
        for i in range(1, min(len(tag), len(buffer)) + 1):
            if buffer.endswith(tag[:i]):
                return len(buffer) - i
        return len(buffer)


class OpenAICompatClient(ModelClient):
    MODELS = Model

    DEFAULT_GENERATE_KWARGS = {
        "max_tokens": 1024,
        "temperature": 0.1,
    }

    def __init__(
        self,
        model: Model,
        base_url: str,
        api_key: str = "not-needed",
        system_message: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
    ):
        super().__init__(model, model_kwargs, system_message)
        self.default_generate_kwargs = self.DEFAULT_GENERATE_KWARGS.copy()
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key)

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

    def generate(self, prompt: str, generate_kwargs: Optional[dict[str, Any]] = None) -> str:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)
        messages = [{"role": "user", "content": prompt}]

        response = self._client.chat.completions.create(
            model=self.model.value,
            messages=messages,
            **generate_kwargs,
        )
        logger.debug("LLM raw response: %s", response)
        content = response.choices[0].message.content or ""

        self.last_thinking = ""
        if self.is_thinking_model:
            thinking, content = _split_thinking(content)
            self.last_thinking = thinking

        return content

    def generate_streamed(
        self,
        prompt: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        include_thinking: bool = True,
    ) -> Iterator[StreamChunk]:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)
        messages = [{"role": "user", "content": prompt}]

        stream = self._client.chat.completions.create(
            model=self.model.value,
            messages=messages,
            stream=True,
            **generate_kwargs,
        )

        self.last_thinking = ""
        parser = _ThinkingParser() if self.is_thinking_model else None

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content is None:
                continue
            logger.debug("LLM raw chunk: %s", chunk)

            if parser:
                for phase, text in parser.feed(delta.content):
                    if phase == StreamingContentType.THINKING:
                        self.last_thinking += text
                        if include_thinking:
                            yield StreamChunk(StreamingContentType.THINKING, text)
                    else:
                        yield StreamChunk(StreamingContentType.GENERATING, text)
            else:
                yield StreamChunk(StreamingContentType.GENERATING, delta.content)

    def chat(self, user_message: str, generate_kwargs: Optional[dict[str, Any]] = None, use_tools: bool = True) -> str:
        generate_kwargs, tools = self._chat_setup(user_message, generate_kwargs, use_tools)

        response = self._client.chat.completions.create(
            model=self.model.value,
            messages=self.messages,
            tools=tools if tools else openai.NOT_GIVEN,
            **generate_kwargs,
        )
        logger.debug("LLM raw response: %s", response)
        msg = response.choices[0].message

        if msg.tool_calls:
            tool_calls = [
                {"name": tc.function.name, "arguments": json.loads(tc.function.arguments)} for tc in msg.tool_calls
            ]
            self._handle_tool_calls(tool_calls, tools)

            response = self._client.chat.completions.create(
                model=self.model.value,
                messages=self.messages,
                tools=tools if tools else openai.NOT_GIVEN,
                **generate_kwargs,
            )
            logger.debug("LLM raw response (after tools): %s", response)
            msg = response.choices[0].message

        content = msg.content or ""
        self.last_thinking = ""

        if self.is_thinking_model:
            thinking, content = _split_thinking(content)
            self.last_thinking = thinking

        self.messages.append({"role": "assistant", "content": content})
        return content

    def chat_streamed(
        self, user_message: str, generate_kwargs: Optional[dict[str, Any]] = None, use_tools: bool = True
    ) -> Iterator[StreamChunk]:
        generate_kwargs, tools = self._chat_setup(user_message, generate_kwargs, use_tools)

        stream = self._client.chat.completions.create(
            model=self.model.value,
            messages=self.messages,
            stream=True,
            tools=tools if tools else openai.NOT_GIVEN,
            **generate_kwargs,
        )

        # Accumulate tool calls (keyed by index) and content chunks
        tool_calls_acc: dict[int, dict] = {}
        parser = _ThinkingParser() if self.is_thinking_model else None
        pending_chunks: list[StreamChunk] = []

        for chunk in stream:
            choice = chunk.choices[0]
            delta = choice.delta
            logger.debug("LLM raw chunk: %s", chunk)

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {"name": "", "arguments": ""}
                    if tc_delta.function and tc_delta.function.name:
                        tool_calls_acc[idx]["name"] += tc_delta.function.name
                    if tc_delta.function and tc_delta.function.arguments:
                        tool_calls_acc[idx]["arguments"] += tc_delta.function.arguments
                continue

            if delta.content is not None:
                if parser:
                    for phase, text in parser.feed(delta.content):
                        pending_chunks.append(StreamChunk(phase, text))
                else:
                    pending_chunks.append(StreamChunk(StreamingContentType.GENERATING, delta.content))

        # Non-tool-call path: emit accumulated chunks and store assistant message
        if not tool_calls_acc:
            self.last_thinking = ""
            full_content = ""
            for sc in pending_chunks:
                if sc.phase == StreamingContentType.THINKING:
                    self.last_thinking += sc.content
                else:
                    full_content += sc.content
                yield sc
            self.messages.append({"role": "assistant", "content": full_content})
            return

        # Tool call path: dispatch tool calls, emit TOOL_CALLING chunks, then stream second response
        tool_calls = [{"name": tc["name"], "arguments": json.loads(tc["arguments"])} for tc in tool_calls_acc.values()]
        msgs_before = len(self.messages)
        self._handle_tool_calls(tool_calls, tools)

        assistant_msg = self.messages[msgs_before]
        for i, tc in enumerate(assistant_msg["tool_calls"]):
            tool_result_msg = self.messages[msgs_before + 1 + i]
            yield StreamChunk(
                StreamingContentType.TOOL_CALLING,
                {"name": tc["function"]["name"], "response": tool_result_msg["content"]},
            )

        stream2 = self._client.chat.completions.create(
            model=self.model.value,
            messages=self.messages,
            stream=True,
            tools=tools if tools else openai.NOT_GIVEN,
            **generate_kwargs,
        )

        self.last_thinking = ""
        parser2 = _ThinkingParser() if self.is_thinking_model else None
        full_content = ""

        for chunk in stream2:
            delta = chunk.choices[0].delta
            if delta.content is None:
                continue
            logger.debug("LLM raw chunk (after tools): %s", chunk)

            if parser2:
                for phase, text in parser2.feed(delta.content):
                    if phase == StreamingContentType.THINKING:
                        self.last_thinking += text
                        yield StreamChunk(StreamingContentType.THINKING, text)
                    else:
                        full_content += text
                        yield StreamChunk(StreamingContentType.GENERATING, text)
            else:
                full_content += delta.content
                yield StreamChunk(StreamingContentType.GENERATING, delta.content)

        self.messages.append({"role": "assistant", "content": full_content})
