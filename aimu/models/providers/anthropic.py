import json
import logging
import os
import re
from enum import Enum
from types import SimpleNamespace
from typing import Any, Iterator, Optional, Union

import anthropic
from dotenv import load_dotenv

from .._internal.sdk_config import sdk_client_kwargs
from ..base import BaseModelClient, Model, ModelSpec, StreamingContentType, StreamChunk, classproperty
from .._internal.image_input import _build_user_content_blocks, _openai_blocks_to_anthropic
from .._internal.usage import usage_from_anthropic

logger = logging.getLogger(__name__)

# Default thinking budget in tokens (must be < max_tokens); used by the ENABLED style only.
_DEFAULT_THINKING_BUDGET = 8000
_THINKING_MAX_TOKENS_FLOOR = _DEFAULT_THINKING_BUDGET + 1024
# Adaptive thinking shares max_tokens with the answer, so give it room to avoid truncation.
_ADAPTIVE_THINKING_MAX_TOKENS_FLOOR = 4096


class ThinkingStyle(Enum):
    """How a model's thinking parameter is expressed in the Anthropic Messages API.

    ENABLED  -> ``{"type": "enabled", "budget_tokens": N}``; the model always thinks up
                to the budget. Used by Opus <= 4.6, Sonnet 4.6, and Haiku 4.5.
    ADAPTIVE -> ``{"type": "adaptive", "display": "summarized"}``; the model decides per
                request whether and how much to think (it may not think at all on simple
                prompts). Required by Opus 4.7+ and Fable 5 -- the ENABLED form returns a
                400 on those models, which also reject temperature/top_p/top_k. ``display``
                defaults to ``"omitted"`` (empty thinking text), so we request ``"summarized"``
                to surface thinking as StreamChunks.
    """

    ENABLED = "enabled"
    ADAPTIVE = "adaptive"


class AnthropicModel(Model):
    """Anthropic Claude model catalog.

    Each member's value is a ``ModelSpec`` or a ``(ModelSpec, ThinkingStyle)`` tuple
    (the style defaults to ``ENABLED`` when omitted). See :class:`ThinkingStyle`.
    """

    def __init__(self, spec: ModelSpec, thinking_style: ThinkingStyle = ThinkingStyle.ENABLED):
        super().__init__(spec)
        self.thinking_style = thinking_style

    CLAUDE_FABLE_5 = (
        ModelSpec("claude-fable-5", tools=True, thinking=True, vision=True, structured_output=True),
        ThinkingStyle.ADAPTIVE,
    )
    CLAUDE_OPUS_4_8 = (
        ModelSpec("claude-opus-4-8", tools=True, thinking=True, vision=True, structured_output=True),
        ThinkingStyle.ADAPTIVE,
    )
    CLAUDE_OPUS_4_7 = (
        ModelSpec("claude-opus-4-7", tools=True, thinking=True, vision=True, structured_output=True),
        ThinkingStyle.ADAPTIVE,
    )
    CLAUDE_OPUS_4_6 = ModelSpec("claude-opus-4-6", tools=True, thinking=True, vision=True, structured_output=True)
    CLAUDE_SONNET_4_6 = ModelSpec("claude-sonnet-4-6", tools=True, thinking=True, vision=True, structured_output=True)
    CLAUDE_HAIKU_4_5 = ModelSpec("claude-haiku-4-5", tools=True, thinking=True, vision=True, structured_output=True)


class AnthropicClient(BaseModelClient):
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
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        cache_prompt: bool = False,
    ):
        super().__init__(model, model_kwargs, system_message)
        self.default_generate_kwargs = self.DEFAULT_GENERATE_KWARGS.copy()
        # Opt-in Anthropic prompt caching: marks the system prompt and tools with ephemeral
        # cache_control breakpoints at request time (see the format adapters). Below the
        # provider's minimum cacheable size the API silently skips caching, so it's safe on.
        self.cache_prompt = cache_prompt
        load_dotenv()
        self._client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"), **sdk_client_kwargs(timeout, max_retries)
        )

    # ------------------------------------------------------------------ #
    # Capability class properties                                          #
    # ------------------------------------------------------------------ #

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

    def _structured_call(self, system_str, ant_messages: list, generate_kwargs: dict, response_format: dict) -> str:
        """Anthropic structured output via a forced single tool.

        Anthropic has no ``response_format`` param; the idiomatic enforcement is to expose
        one tool whose ``input_schema`` is the JSON Schema and force it with ``tool_choice``.
        Extended thinking is incompatible with a forced ``tool_choice``, so ``generate_kwargs``
        here must NOT carry the thinking param (callers route around ``_thinking_kwargs``).
        Returns the tool input as a JSON string so the base coerces it like every other provider.
        """
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", str(response_format.get("title", "Response")))[:64] or "Response"
        tool = {"name": name, "description": f"Emit the answer as a {name} object.", "input_schema": response_format}
        response = self._client.messages.create(
            model=self.model.value,
            system=system_str,
            messages=ant_messages,
            tools=[tool],
            tool_choice={"type": "tool", "name": name},
            **generate_kwargs,
        )
        logger.debug("Anthropic raw response (structured): %s", response)
        self.last_usage = usage_from_anthropic(response)
        for block in response.content:
            if block.type == "tool_use":
                return json.dumps(block.input)
        return "{}"

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
        style = getattr(self.model, "thinking_style", ThinkingStyle.ENABLED)

        if style is ThinkingStyle.ADAPTIVE:
            # Adaptive models reject budget_tokens and all sampling params; the model
            # decides whether to think. Give thinking room within the shared max_tokens.
            kwargs.pop("thinking_budget_tokens", None)
            for key in ("temperature", "top_p", "top_k"):
                kwargs.pop(key, None)
            if kwargs.get("max_tokens", 0) < _ADAPTIVE_THINKING_MAX_TOKENS_FLOOR:
                kwargs["max_tokens"] = _ADAPTIVE_THINKING_MAX_TOKENS_FLOOR
            kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
            return kwargs

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
        # Each entry is (role, content_blocks). Built as block lists so empty turns can be
        # dropped and adjacent same-role turns merged before returning (Anthropic rejects
        # empty text blocks and requires alternating roles).
        turns: list[tuple[str, list]] = []
        i = 0

        while i < len(messages):
            msg = messages[i]
            role = msg["role"]

            if role == "system":
                system_str = msg["content"] or ""
                i += 1

            elif role == "user":
                content = msg["content"]
                if isinstance(content, list):
                    blocks = _openai_blocks_to_anthropic(content)
                else:
                    blocks = [{"type": "text", "text": content}] if content else []
                turns.append(("user", blocks))
                i += 1

            elif role == "assistant":
                if "tool_calls" in msg:
                    blocks = []
                    for tc in msg["tool_calls"]:
                        args = tc["function"]["arguments"]
                        if isinstance(args, str):
                            args = json.loads(args)
                        blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc["id"],
                                "name": tc["function"]["name"],
                                "input": args,
                            }
                        )
                    turns.append(("assistant", blocks))
                else:
                    # Drop empty/whitespace-only assistant turns: they carry no content and
                    # would serialize to an empty text block the API rejects.
                    text = msg.get("content") or ""
                    turns.append(("assistant", [{"type": "text", "text": text}] if text.strip() else []))
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
                turns.append(("user", tool_results))

            else:
                i += 1  # skip unknown roles

        ant_messages: list[dict] = []
        for role, blocks in turns:
            if not blocks:
                continue
            if ant_messages and ant_messages[-1]["role"] == role:
                ant_messages[-1]["content"].extend(blocks)
            else:
                ant_messages.append({"role": role, "content": list(blocks)})

        # Opt-in prompt caching: mark the system prompt with an ephemeral cache breakpoint.
        # Returned as a text-block list (the API accepts str or list for system=); a
        # non-empty list stays truthy so the request sites' NOT_GIVEN guard is unaffected.
        if getattr(self, "cache_prompt", False) and system_str:
            return [{"type": "text", "text": system_str, "cache_control": {"type": "ephemeral"}}], ant_messages

        return system_str, ant_messages

    def _openai_tools_to_anthropic(self, tools: list[dict]) -> list[dict]:
        """Convert OpenAI function-calling format to Anthropic tool format.

        OpenAI: [{"type": "function", "function": {"name", "description", "parameters"}}]
        Anthropic: [{"name", "description", "input_schema"}]
        """
        ant_tools = [
            {
                "name": t["function"]["name"],
                "description": t["function"].get("description", ""),
                "input_schema": t["function"].get("parameters", {"type": "object", "properties": {}}),
            }
            for t in tools
        ]
        # Opt-in prompt caching: one ephemeral breakpoint on the last tool caches the whole
        # tools array (every definition up to and including the breakpoint).
        if ant_tools and getattr(self, "cache_prompt", False):
            ant_tools[-1] = {**ant_tools[-1], "cache_control": {"type": "ephemeral"}}
        return ant_tools

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

    def _generate(
        self,
        prompt: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
        audio: Optional[list] = None,
        response_format: Optional[dict] = None,
    ) -> Union[str, Iterator[StreamChunk]]:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        if response_format is not None:
            content = self._generate_content(prompt, images, audio)
            return self._structured_call(
                anthropic.NOT_GIVEN, [{"role": "user", "content": content}], generate_kwargs, response_format
            )

        generate_kwargs = self._thinking_kwargs(generate_kwargs)

        if stream:
            return self._generate_streamed(prompt, generate_kwargs, images=images, audio=audio)

        response = self._client.messages.create(
            model=self.model.value,
            messages=[{"role": "user", "content": self._generate_content(prompt, images, audio)}],
            **generate_kwargs,
        )
        logger.debug("Anthropic raw response: %s", response)
        self.last_usage = usage_from_anthropic(response)

        self.last_thinking = ""
        content = ""
        for block in response.content:
            if block.type == "thinking":
                self.last_thinking = block.thinking
            elif block.type == "text":
                content = block.text
        return content

    @staticmethod
    def _generate_content(prompt: str, images: Optional[list], audio: Optional[list] = None):
        """Build the single-turn user content for stateless generate.

        Returns a plain string for text-only, or an Anthropic-format content block list
        when images or audio are provided. ``images`` and ``audio`` are mutually exclusive
        (validated upstream by ``BaseModelClient.generate()``).
        """
        if images:
            return _openai_blocks_to_anthropic(_build_user_content_blocks(prompt, images))
        if audio:
            from .._internal.audio_input import _build_audio_content_blocks

            return _openai_blocks_to_anthropic(_build_audio_content_blocks(prompt, audio))
        return prompt

    def _generate_streamed(
        self,
        prompt: str,
        generate_kwargs: dict[str, Any],
        images: Optional[list] = None,
        audio: Optional[list] = None,
    ) -> Iterator[StreamChunk]:
        self.last_thinking = ""
        self.last_usage = None

        with self._client.messages.stream(
            model=self.model.value,
            messages=[{"role": "user", "content": self._generate_content(prompt, images, audio)}],
            **generate_kwargs,
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "thinking_delta":
                        self.last_thinking += delta.thinking
                        yield StreamChunk(StreamingContentType.THINKING, delta.thinking)
                    elif delta.type == "text_delta":
                        yield StreamChunk(StreamingContentType.GENERATING, delta.text)
            self.last_usage = usage_from_anthropic(stream.get_final_message())

    def _chat(
        self,
        user_message: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        use_tools: bool = True,
        stream: bool = False,
        images: Optional[list] = None,
        audio: Optional[list] = None,
        response_format: Optional[dict] = None,
    ) -> Union[str, Iterator[StreamChunk]]:
        if response_format is not None and use_tools and self.tools:
            raise ValueError(
                "Anthropic structured output uses a forced tool, which is incompatible with active "
                "action tools. Drop tools (or use_tools=False), or use a provider whose response_format "
                "composes with tools (e.g. OpenAI)."
            )

        generate_kwargs, tools = self._chat_setup(user_message, generate_kwargs, use_tools, images=images, audio=audio)

        if response_format is not None:
            system_str, ant_messages = self._openai_messages_to_anthropic(self.messages)
            text = self._structured_call(
                system_str if system_str else anthropic.NOT_GIVEN, ant_messages, generate_kwargs, response_format
            )
            self.messages.append({"role": "assistant", "content": text})
            return text

        generate_kwargs = self._thinking_kwargs(generate_kwargs)

        if stream:
            return self._chat_streamed(generate_kwargs, tools)

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
            self._handle_tool_calls(tool_calls)
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

        self.last_usage = usage_from_anthropic(response)
        assistant_msg: dict = {"role": "assistant", "content": text_content}
        if self.last_thinking:
            assistant_msg["thinking"] = self.last_thinking
        self.messages.append(assistant_msg)
        return text_content

    def _chat_streamed(self, generate_kwargs: dict[str, Any], tools: list) -> Iterator[StreamChunk]:
        system_str, ant_messages = self._openai_messages_to_anthropic(self.messages)
        ant_tools = self._openai_tools_to_anthropic(tools) if tools else anthropic.NOT_GIVEN

        # Accumulated state from first stream pass
        tool_use_acc: list[dict] = []  # {"id": str, "name": str, "input_json": str}
        first_pass_chunks: list[StreamChunk] = []
        self.last_thinking = ""
        self.last_usage = None

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
            self.last_usage = usage_from_anthropic(stream.get_final_message())
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
        yield from self._handle_tool_calls_streamed(tool_calls)
        self._patch_tool_ids(msgs_before, parsed_blocks)

        # Store thinking from first stream pass in the tool-call assistant message
        if self.last_thinking:
            self.messages[msgs_before]["thinking"] = self.last_thinking

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
            self.last_usage = usage_from_anthropic(stream2.get_final_message())

        # Fall back to first-pass thinking if second call produced none
        if not self.last_thinking:
            self.last_thinking = first_pass_thinking

        assistant_msg: dict = {"role": "assistant", "content": full_content}
        if self.last_thinking:
            assistant_msg["thinking"] = self.last_thinking
        self.messages.append(assistant_msg)
