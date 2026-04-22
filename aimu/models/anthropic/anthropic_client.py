import json
import logging
import os
from types import SimpleNamespace
from typing import Any, Iterator, Optional

import anthropic
from dotenv import load_dotenv

from ..base import ModelClient, Model, StreamingContentType, StreamChunk, classproperty

logger = logging.getLogger(__name__)

# Default thinking budget in tokens (must be < max_tokens)
_DEFAULT_THINKING_BUDGET = 8000
_THINKING_MAX_TOKENS_FLOOR = _DEFAULT_THINKING_BUDGET + 1024


class AnthropicModel(Model):
    def __init__(self, value, supports_tools=False, supports_thinking=False):
        super().__init__(value, supports_tools, supports_thinking)

    CLAUDE_SONNET_4_6 = ("claude-sonnet-4-6", True, True)
    CLAUDE_OPUS_4_6 = ("claude-opus-4-6", True, True)
    CLAUDE_HAIKU_4_5 = ("claude-haiku-4-5", True, False)


class AnthropicClient(ModelClient):
    """Client for Anthropic Claude models using the native anthropic SDK.

    Reads ANTHROPIC_API_KEY from the environment (or a .env file).
    self.messages is always stored in OpenAI format; conversion to the
    Anthropic API format happens at call time.
    """

    MODELS = AnthropicModel

    DEFAULT_GENERATE_KWARGS = {
        "max_tokens": 1024,
        "temperature": 0.1,
    }

    def __init__(
        self,
        model: AnthropicModel,
        model_kwargs: Optional[dict] = None,
        system_message: Optional[str] = None,
    ):
        super().__init__(model, model_kwargs, system_message)
        self.default_generate_kwargs = self.DEFAULT_GENERATE_KWARGS.copy()
        load_dotenv()
        self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # ------------------------------------------------------------------ #
    # Capability class properties                                          #
    # ------------------------------------------------------------------ #

    @classproperty
    def THINKING_MODELS(cls) -> list[Model]:  # noqa: N805
        return [m for m in cls.MODELS if m.supports_thinking]

    @classproperty
    def TOOL_MODELS(cls) -> list[Model]:  # noqa: N805
        return [m for m in cls.MODELS if m.supports_tools]

    # ------------------------------------------------------------------ #
    # generate_kwargs helpers                                              #
    # ------------------------------------------------------------------ #

    # Parameters not accepted by the Anthropic Messages API (e.g. HuggingFace-specific)
    _UNSUPPORTED_KWARGS = frozenset({"max_new_tokens", "do_sample", "num_return_sequences"})

    def _update_generate_kwargs(self, generate_kwargs: Optional[dict[str, Any]] = None) -> dict:
        if not generate_kwargs:
            kwargs = self.default_generate_kwargs.copy()
        else:
            kwargs = {**self.default_generate_kwargs, **generate_kwargs}
        # Strip HuggingFace / other framework-specific keys the Anthropic API rejects
        for key in self._UNSUPPORTED_KWARGS:
            kwargs.pop(key, None)
        # Anthropic rejects top_p alongside thinking; drop it unconditionally
        kwargs.pop("top_p", None)
        # Thinking models require temperature=1
        if self.is_thinking_model:
            kwargs["temperature"] = 1
        return kwargs

    def _thinking_kwargs(self, generate_kwargs: dict) -> dict:
        """Inject the thinking parameter for thinking-capable models."""
        if not self.is_thinking_model:
            return generate_kwargs
        kwargs = generate_kwargs.copy()
        budget = kwargs.pop("thinking_budget_tokens", _DEFAULT_THINKING_BUDGET)
        # max_tokens must exceed the thinking budget
        if kwargs.get("max_tokens", 0) <= budget:
            kwargs["max_tokens"] = budget + 1024
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
        # temperature must be omitted (defaults to 1) when thinking is enabled
        kwargs.pop("temperature", None)
        return kwargs

    # ------------------------------------------------------------------ #
    # Message / tool format conversion                                     #
    # ------------------------------------------------------------------ #

    def _openai_messages_to_anthropic(self, messages: list[dict]) -> tuple[str, list[dict]]:
        """Convert self.messages (OpenAI format) to Anthropic API format.

        Returns (system_str, anthropic_messages) where system_str is the
        content of the system message (empty string if none).

        OpenAI → Anthropic mapping:
          system message          → extracted to system= param
          user text               → {"role": "user", "content": "..."}  (unchanged)
          assistant text          → {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
          assistant tool_calls    → {"role": "assistant", "content": [{"type": "tool_use", ...}]}
          run of tool results     → single {"role": "user", "content": [{"type": "tool_result", ...}]}
        """
        system_str = ""
        ant_messages = []
        i = 0

        while i < len(messages):
            msg = messages[i]
            role = msg["role"]

            if role == "system":
                system_str = msg["content"] or ""
                i += 1

            elif role == "user":
                ant_messages.append({"role": "user", "content": msg["content"]})
                i += 1

            elif role == "assistant":
                if "tool_calls" in msg:
                    content_blocks = []
                    for tc in msg["tool_calls"]:
                        args = tc["function"]["arguments"]
                        if isinstance(args, str):
                            args = json.loads(args)
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc["id"],
                                "name": tc["function"]["name"],
                                "input": args,
                            }
                        )
                    ant_messages.append({"role": "assistant", "content": content_blocks})
                else:
                    text = msg.get("content") or ""
                    ant_messages.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": text}],
                        }
                    )
                i += 1

            elif role == "tool":
                # Collect all consecutive tool-result messages into one user message
                tool_results = []
                while i < len(messages) and messages[i]["role"] == "tool":
                    tm = messages[i]
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tm["tool_call_id"],
                            "content": tm["content"],
                        }
                    )
                    i += 1
                ant_messages.append({"role": "user", "content": tool_results})

            else:
                i += 1  # skip unknown roles

        return system_str, ant_messages

    def _openai_tools_to_anthropic(self, tools: list[dict]) -> list[dict]:
        """Convert OpenAI function-calling format to Anthropic tool format.

        OpenAI: [{"type": "function", "function": {"name", "description", "parameters"}}]
        Anthropic: [{"name", "description", "input_schema"}]
        """
        return [
            {
                "name": t["function"]["name"],
                "description": t["function"].get("description", ""),
                "input_schema": t["function"].get("parameters", {"type": "object", "properties": {}}),
            }
            for t in tools
        ]

    def _patch_tool_ids(self, msgs_before: int, tool_use_blocks: list) -> None:
        """Replace the random IDs assigned by _handle_tool_calls() with the
        original IDs from Anthropic's tool_use blocks, so that the tool_result
        tool_use_id values match when re-converting to Anthropic format.
        """
        assistant_msg = self.messages[msgs_before]
        for i, (tc, tub) in enumerate(zip(assistant_msg["tool_calls"], tool_use_blocks)):
            tc["id"] = tub.id
            self.messages[msgs_before + 1 + i]["tool_call_id"] = tub.id

    # ------------------------------------------------------------------ #
    # ModelClient abstract method implementations                          #
    # ------------------------------------------------------------------ #

    def generate(self, prompt: str, generate_kwargs: Optional[dict[str, Any]] = None) -> str:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)
        generate_kwargs = self._thinking_kwargs(generate_kwargs)

        response = self._client.messages.create(
            model=self.model.value,
            messages=[{"role": "user", "content": prompt}],
            **generate_kwargs,
        )
        logger.debug("Anthropic raw response: %s", response)

        self.last_thinking = ""
        content = ""
        for block in response.content:
            if block.type == "thinking":
                self.last_thinking = block.thinking
            elif block.type == "text":
                content = block.text
        return content

    def generate_streamed(
        self,
        prompt: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        include_thinking: bool = True,
    ) -> Iterator[StreamChunk]:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)
        generate_kwargs = self._thinking_kwargs(generate_kwargs)
        self.last_thinking = ""

        with self._client.messages.stream(
            model=self.model.value,
            messages=[{"role": "user", "content": prompt}],
            **generate_kwargs,
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "thinking_delta":
                        self.last_thinking += delta.thinking
                        if include_thinking:
                            yield StreamChunk(StreamingContentType.THINKING, delta.thinking)
                    elif delta.type == "text_delta":
                        yield StreamChunk(StreamingContentType.GENERATING, delta.text)

    def chat(
        self,
        user_message: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        use_tools: bool = True,
    ) -> str:
        generate_kwargs, tools = self._chat_setup(user_message, generate_kwargs, use_tools)
        generate_kwargs = self._thinking_kwargs(generate_kwargs)

        system_str, ant_messages = self._openai_messages_to_anthropic(self.messages)
        ant_tools = self._openai_tools_to_anthropic(tools) if tools else anthropic.NOT_GIVEN

        response = self._client.messages.create(
            model=self.model.value,
            system=system_str if system_str else anthropic.NOT_GIVEN,
            messages=ant_messages,
            tools=ant_tools,
            **generate_kwargs,
        )
        logger.debug("Anthropic raw response: %s", response)

        self.last_thinking = ""
        tool_use_blocks = []
        text_content = ""

        for block in response.content:
            if block.type == "thinking":
                self.last_thinking = block.thinking
            elif block.type == "text":
                text_content = block.text
            elif block.type == "tool_use":
                tool_use_blocks.append(block)

        if tool_use_blocks:
            tool_calls = [{"name": b.name, "arguments": b.input} for b in tool_use_blocks]
            msgs_before = len(self.messages)
            self._handle_tool_calls(tool_calls, tools)
            self._patch_tool_ids(msgs_before, tool_use_blocks)

            # Store thinking from first call in the tool-call assistant message
            if self.last_thinking:
                self.messages[msgs_before]["thinking"] = self.last_thinking

            # Preserve thinking from first call; second call may not emit thinking
            first_call_thinking = self.last_thinking

            # Re-convert with tool results included and make the follow-up call
            system_str, ant_messages = self._openai_messages_to_anthropic(self.messages)
            response = self._client.messages.create(
                model=self.model.value,
                system=system_str if system_str else anthropic.NOT_GIVEN,
                messages=ant_messages,
                tools=ant_tools,
                **generate_kwargs,
            )
            logger.debug("Anthropic raw response (after tools): %s", response)

            self.last_thinking = ""
            text_content = ""
            for block in response.content:
                if block.type == "thinking":
                    self.last_thinking = block.thinking
                elif block.type == "text":
                    text_content = block.text

            # Fall back to first-call thinking if second call produced none
            if not self.last_thinking:
                self.last_thinking = first_call_thinking

        assistant_msg: dict = {"role": "assistant", "content": text_content}
        if self.last_thinking:
            assistant_msg["thinking"] = self.last_thinking
        self.messages.append(assistant_msg)
        return text_content

    def chat_streamed(
        self,
        user_message: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        use_tools: bool = True,
    ) -> Iterator[StreamChunk]:
        generate_kwargs, tools = self._chat_setup(user_message, generate_kwargs, use_tools)
        generate_kwargs = self._thinking_kwargs(generate_kwargs)

        system_str, ant_messages = self._openai_messages_to_anthropic(self.messages)
        ant_tools = self._openai_tools_to_anthropic(tools) if tools else anthropic.NOT_GIVEN

        # Accumulated state from first stream pass
        tool_use_acc: list[dict] = []  # {"id": str, "name": str, "input_json": str}
        first_pass_chunks: list[StreamChunk] = []
        self.last_thinking = ""

        with self._client.messages.stream(
            model=self.model.value,
            system=system_str if system_str else anthropic.NOT_GIVEN,
            messages=ant_messages,
            tools=ant_tools,
            **generate_kwargs,
        ) as stream:
            for event in stream:
                if event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        tool_use_acc.append({"id": block.id, "name": block.name, "input_json": ""})
                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "thinking_delta":
                        self.last_thinking += delta.thinking
                        first_pass_chunks.append(StreamChunk(StreamingContentType.THINKING, delta.thinking))
                    elif delta.type == "text_delta":
                        first_pass_chunks.append(StreamChunk(StreamingContentType.GENERATING, delta.text))
                    elif delta.type == "input_json_delta" and tool_use_acc:
                        tool_use_acc[-1]["input_json"] += delta.partial_json

        if not tool_use_acc:
            # No tool calls; yield buffered chunks and store assistant message
            full_content = ""
            for sc in first_pass_chunks:
                if sc.phase == StreamingContentType.GENERATING:
                    full_content += sc.content
                yield sc
            self.messages.append({"role": "assistant", "content": full_content})
            return

        # Tool call path: parse accumulated JSON, dispatch, yield TOOL_CALLING chunks
        parsed_blocks = [
            SimpleNamespace(
                id=tub["id"],
                name=tub["name"],
                input=json.loads(tub["input_json"]) if tub["input_json"] else {},
            )
            for tub in tool_use_acc
        ]

        tool_calls = [{"name": b.name, "arguments": b.input} for b in parsed_blocks]
        msgs_before = len(self.messages)
        self._handle_tool_calls(tool_calls, tools)
        self._patch_tool_ids(msgs_before, parsed_blocks)

        # Store thinking from first stream pass in the tool-call assistant message
        if self.last_thinking:
            self.messages[msgs_before]["thinking"] = self.last_thinking

        for i, tc in enumerate(self.messages[msgs_before]["tool_calls"]):
            yield StreamChunk(
                StreamingContentType.TOOL_CALLING,
                {"name": tc["function"]["name"], "response": self.messages[msgs_before + 1 + i]["content"]},
            )

        # Second streaming call with tool results
        system_str, ant_messages = self._openai_messages_to_anthropic(self.messages)
        full_content = ""
        # Preserve thinking from first stream pass; second call may not emit thinking
        first_pass_thinking = self.last_thinking
        self.last_thinking = ""

        with self._client.messages.stream(
            model=self.model.value,
            system=system_str if system_str else anthropic.NOT_GIVEN,
            messages=ant_messages,
            tools=ant_tools,
            **generate_kwargs,
        ) as stream2:
            for event in stream2:
                if event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "thinking_delta":
                        self.last_thinking += delta.thinking
                        yield StreamChunk(StreamingContentType.THINKING, delta.thinking)
                    elif delta.type == "text_delta":
                        full_content += delta.text
                        yield StreamChunk(StreamingContentType.GENERATING, delta.text)

        # Fall back to first-pass thinking if second call produced none
        if not self.last_thinking:
            self.last_thinking = first_pass_thinking

        assistant_msg: dict = {"role": "assistant", "content": full_content}
        if self.last_thinking:
            assistant_msg["thinking"] = self.last_thinking
        self.messages.append(assistant_msg)
