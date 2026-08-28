import json
import logging
import os
import re
from enum import Enum
from types import SimpleNamespace
from typing import Any, Iterator, Optional, Union

import anthropic
from dotenv import load_dotenv

from .._internal.generate_kwargs import CLOUD_MAX_TOKENS, Unsupported
from .._internal.sdk_config import sdk_client_kwargs
from ..base import (
    BaseModelClient,
    ContextOverflowError,
    ModelRefusalError,
    Model,
    ModelSpec,
    StreamingContentType,
    StreamChunk,
    classproperty,
)
from .._internal.image_input import _build_user_content_blocks, _openai_blocks_to_anthropic
from .._internal.chat_state import REFUSAL_STOP_REASON
from .._internal.thinking import THINKING_KWARG, pop_thinking
from .._internal.usage import usage_from_anthropic

logger = logging.getLogger(__name__)

# Default thinking budget in tokens (must be < max_tokens); used by the ENABLED style only.
_DEFAULT_THINKING_BUDGET = 8000
_THINKING_MAX_TOKENS_FLOOR = _DEFAULT_THINKING_BUDGET + 1024
# Adaptive thinking shares max_tokens with the answer, so give it room to avoid truncation.
_ADAPTIVE_THINKING_MAX_TOKENS_FLOOR = 4096

# Portable levels expressed as Anthropic token budgets. "medium" is the historical default,
# so thinking="medium" and thinking=None agree.
_THINKING_BUDGETS = {"low": 2048, "medium": _DEFAULT_THINKING_BUDGET, "high": 16000}

# The effort vocabularies, declared on the members that accept them so the mapping below can read
# what a given model actually takes rather than assuming one set. "xhigh" arrived with Opus 4.7.
_EFFORT_LEVELS_4_7 = ("low", "medium", "high", "xhigh", "max")
_EFFORT_LEVELS_4_6 = ("low", "medium", "high", "max")

# Opus 4.7 and later reject temperature/top_p/top_k outright, thinking or not; the 4.6 line and
# Haiku 4.5 still accept them. A model fact rather than a request shape, so it cannot be read off
# ThinkingStyle: the 4.6 line is adaptive *and* takes sampling parameters. Kept provider-local
# (like ThinkingStyle) rather than promoted to ModelSpec, since no other provider has the quirk.
_REJECTS_SAMPLING = frozenset(
    {"claude-fable-5", "claude-opus-5", "claude-opus-4-8", "claude-opus-4-7", "claude-sonnet-5"}
)

# Effort values AIMU will not pair with disabled thinking: Opus 5 rejects that combination with a
# 400, and validates the two independently on every request, so it cannot be settled per client.
_EFFORT_ABOVE_HIGH = ("xhigh", "max")


def _effort_for_level(level: str, effort_levels: tuple[str, ...]) -> str:
    """Translate a portable thinking level into one of ``effort_levels``.

    ``high`` reaches for ``xhigh`` where the model has it, following the same reasoning as
    QWEN_REASONING_EFFORT: Anthropic's effort defaults to "high" when the parameter is unset, so
    mapping to the literal "high" would make thinking="high" a silent no-op. It is also the
    vendor's own recommendation for demanding coding and agentic work, with "max" held back for
    correctness-over-cost -- reachable only by passing output_config through generate_kwargs.
    """
    if level == "high" and "xhigh" in effort_levels:
        return "xhigh"
    return level


class ThinkingStyle(Enum):
    """How a model's thinking parameter is expressed in the Anthropic Messages API.

    ENABLED  -> ``{"type": "enabled", "budget_tokens": N}``; the model always thinks up
                to the budget. Used by Opus <= 4.6, Sonnet 4.6, and Haiku 4.5. Thinking is
                disabled by omitting the parameter.
    ADAPTIVE -> ``{"type": "adaptive", "display": "summarized"}``; the model decides per
                request whether and how much to think (it may not think at all on simple
                prompts). Required by Opus 4.7+, Sonnet 5, and Fable 5 -- the ENABLED form
                returns a 400 on those models, which also reject temperature/top_p/top_k.
                ``display`` defaults to ``"omitted"`` (empty thinking text), so we request
                ``"summarized"`` to surface thinking as StreamChunks. Thinking is disabled
                with an explicit ``{"type": "disabled"}`` rather than by omission, because
                Opus 5 and Sonnet 5 reason by default when the parameter is absent (Fable 5
                cannot be disabled at all, and declares ``thinking_optional=False``).
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

    # thinking_levels=True on every member: a portable level is always translated into an
    # Anthropic-specific mechanism by _thinking_kwargs (a token budget for ENABLED, a warned
    # no-op for ADAPTIVE), so the generic resolver must never strip it beforehand.
    CLAUDE_FABLE_5 = (
        ModelSpec(
            "claude-fable-5",
            tools=True,
            thinking=True,
            vision=True,
            structured_output=True,
            thinking_levels=True,
            effort_levels=_EFFORT_LEVELS_4_7,
            # Fable 5 always reasons: omitting the parameter runs adaptive, and an explicit
            # {"type": "disabled"} is a 400. Declaring it here is what makes thinking=False warn
            # and continue instead of reaching _thinking_kwargs and being sent.
            thinking_optional=False,
        ),
        ThinkingStyle.ADAPTIVE,
    )
    CLAUDE_OPUS_5 = (
        ModelSpec(
            "claude-opus-5",
            tools=True,
            thinking=True,
            vision=True,
            structured_output=True,
            thinking_levels=True,
            effort_levels=_EFFORT_LEVELS_4_7,
        ),
        ThinkingStyle.ADAPTIVE,
    )
    CLAUDE_OPUS_4_8 = (
        ModelSpec(
            "claude-opus-4-8",
            tools=True,
            thinking=True,
            vision=True,
            structured_output=True,
            thinking_levels=True,
            effort_levels=_EFFORT_LEVELS_4_7,
        ),
        ThinkingStyle.ADAPTIVE,
    )
    CLAUDE_OPUS_4_7 = (
        ModelSpec(
            "claude-opus-4-7",
            tools=True,
            thinking=True,
            vision=True,
            structured_output=True,
            thinking_levels=True,
            effort_levels=_EFFORT_LEVELS_4_7,
        ),
        ThinkingStyle.ADAPTIVE,
    )
    CLAUDE_OPUS_4_6 = (
        ModelSpec(
            "claude-opus-4-6",
            tools=True,
            thinking=True,
            vision=True,
            structured_output=True,
            thinking_levels=True,
            effort_levels=_EFFORT_LEVELS_4_6,
        ),
        ThinkingStyle.ADAPTIVE,
    )
    CLAUDE_SONNET_5 = (
        ModelSpec(
            "claude-sonnet-5",
            tools=True,
            thinking=True,
            vision=True,
            structured_output=True,
            thinking_levels=True,
            effort_levels=_EFFORT_LEVELS_4_7,
        ),
        ThinkingStyle.ADAPTIVE,
    )
    CLAUDE_SONNET_4_6 = (
        ModelSpec(
            "claude-sonnet-4-6",
            tools=True,
            thinking=True,
            vision=True,
            structured_output=True,
            thinking_levels=True,
            effort_levels=_EFFORT_LEVELS_4_6,
        ),
        ThinkingStyle.ADAPTIVE,
    )
    CLAUDE_HAIKU_4_5 = ModelSpec(
        "claude-haiku-4-5", tools=True, thinking=True, vision=True, structured_output=True, thinking_levels=True
    )


# The Messages API takes temperature, top_p, and top_k. It has no min_p and no penalty parameters, and
# rejects an unknown one outright rather than ignoring it.
ANTHROPIC_GENERATE_KWARGS = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": Unsupported("The Anthropic Messages API has no min_p; use top_p or top_k."),
    "presence_penalty": Unsupported("The Anthropic Messages API has no penalty parameters."),
    "repetition_penalty": Unsupported("The Anthropic Messages API has no penalty parameters."),
    "max_tokens": "max_tokens",
    "context_length": Unsupported("This model's context window is fixed by the provider."),
}


def _raise_if_context_overflowed(exc: "anthropic.BadRequestError") -> None:
    """Translate Anthropic's prompt-too-long 400 into ``ContextOverflowError``, or return.

    On the 400 path specifically, Anthropic has no machine-readable error code for this case:
    every ``BadRequestError`` shares ``error.type == "invalid_request_error"`` (bad tool schema,
    non-alternating roles, an unsupported thinking/effort pairing, ...), so the code alone can't
    distinguish them there. (A 413 is a different exception class entirely -- see
    ``_raise_for_request_too_large`` below -- and *does* have a distinct signal, so this function
    only ever sees the 400s that genuinely have none.) The message text is the only signal on this
    path, and Anthropic's own wording for this specific failure is "prompt is too long: N tokens >
    M maximum" -- matched narrowly on that phrase rather than on "it was a 400", so an unrelated
    bad request still propagates as itself instead of a misleading overflow error a
    catch-compact-retry loop can't fix.
    """
    if "prompt is too long" not in str(exc).lower():
        return
    raise ContextOverflowError(
        "The request no longer fits the model's context window: Anthropic rejected the prompt as "
        "too long. Shorten the conversation, advertise fewer tools, or compact history first "
        "(aimu.context.trim_messages / summarize_messages)."
    ) from exc


def _raise_for_request_too_large(exc: "anthropic.RequestTooLargeError") -> None:
    """Translate a 413 (``RequestTooLargeError``) into ``ContextOverflowError``. Always raises.

    Unlike the 400 path above, this exception class *is* the machine-readable signal: Anthropic's
    SDK maps status 413 to this dedicated type (a sibling of ``BadRequestError``, not a subclass of
    it, so a bare ``except anthropic.BadRequestError`` never sees it), and Anthropic's own error
    reference lists "too many input tokens" among 413's causes. No message-text disambiguation is
    needed here the way it is for the 400 path: catching this specific type is itself the narrow
    match ("matching on the exception type rather than the status number" -- a generic ``except
    APIStatusError: if e.status_code == 413`` would be the "it was a 4xx" mistake this whole
    mapping is trying to avoid).
    """
    raise ContextOverflowError(
        "Anthropic rejected the request as too large (413). This is usually the context window, "
        "but a large image or document in the request can also trigger it. Shorten the "
        "conversation, advertise fewer tools, or compact history first "
        "(aimu.context.trim_messages / summarize_messages)."
    ) from exc


# Tier-1 defaults; the async twin imports this rather than restating it. See CLOUD_MAX_TOKENS.
ANTHROPIC_DEFAULT_GENERATE_KWARGS = {
    "max_tokens": CLOUD_MAX_TOKENS,
    "temperature": 0.1,
}


class AnthropicClient(BaseModelClient):
    """Client for Anthropic Claude models using the native anthropic SDK.

    Reads ANTHROPIC_API_KEY from the environment (or a .env file).
    self.messages is always stored in OpenAI format; conversion to the
    Anthropic API format happens at call time.
    """

    MODELS = AnthropicModel

    GENERATE_KWARG_SUPPORT = ANTHROPIC_GENERATE_KWARGS

    DEFAULT_GENERATE_KWARGS = ANTHROPIC_DEFAULT_GENERATE_KWARGS

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

    def _record_response(self, response) -> None:
        """Record usage and how the turn ended, and refuse to return a refusal silently.

        One call rather than three lines repeated at six request paths. A path that recorded
        usage but not the stop reason would leave ``last_output_truncated`` stale from the
        previous turn, which is worse than never setting it.
        """
        self.last_usage = usage_from_anthropic(response)
        self._record_stop_reason(getattr(response, "stop_reason", None))
        self._raise_if_refused(response)

    def _raise_if_refused(self, response) -> None:
        """Turn Anthropic's HTTP-200 refusal into a typed error, or return.

        A declined request is not an HTTP failure: the response is a 200 whose content carries no
        text block, so every read path here returns an empty string and the caller is told
        nothing. Opus 5 and Fable 5 ship the classifiers that produce it, and benign security and
        life-sciences work trips them, so this is reachable in ordinary use rather than only under
        abuse. Raising rather than returning "" is what lets a FallbackClient recover the request
        on another model, which is the vendor's own recommended handling.
        """
        if self.last_stop_reason != REFUSAL_STOP_REASON:
            return
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None)
        explanation = getattr(details, "explanation", None)
        described = f" ({category})" if category else ""
        because = f" {explanation}" if explanation else ""
        raise ModelRefusalError(
            f"{self.model.value} declined this request{described} rather than answering it, so the "
            f"response carries no content.{because} Rephrase, or route the call to another model "
            "(aimu.models.FallbackClient accepts this error class in retry_on).",
            category=category,
            explanation=explanation,
        )

    def _strip_thinking_for_structured(self, generate_kwargs: dict) -> dict:
        """Remove the reserved thinking key before a forced-tool structured-output request.

        A forced ``tool_choice`` cannot coexist with extended thinking, so a ``thinking=``
        request reaching this path is warned about once (not raised) and dropped, matching
        this feature's warn-and-continue rule for a request a model cannot honour.
        """
        kwargs = generate_kwargs.copy()
        resolved = pop_thinking(kwargs)
        if resolved is not None and resolved.enabled:
            thinking_value = resolved.level if resolved.level is not None else True
            self._warn_once(
                f"{self.model.value} structured output uses a forced tool, which is incompatible "
                f"with extended thinking; thinking={thinking_value!r} is ignored."
            )
        return kwargs

    def _structured_call(self, system_str, ant_messages: list, generate_kwargs: dict, response_format: dict) -> str:
        """Anthropic structured output via a forced single tool.

        Anthropic has no ``response_format`` param; the idiomatic enforcement is to expose
        one tool whose ``input_schema`` is the JSON Schema and force it with ``tool_choice``.
        Extended thinking is incompatible with a forced ``tool_choice``, so ``generate_kwargs``
        here must NOT carry the thinking param (callers route around ``_thinking_kwargs``).
        Returns the tool input as a JSON string so the base coerces it like every other provider.
        """
        generate_kwargs = self._strip_thinking_for_structured(generate_kwargs)
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", str(response_format.get("title", "Response")))[:64] or "Response"
        tool = {"name": name, "description": f"Emit the answer as a {name} object.", "input_schema": response_format}
        payload = {
            **generate_kwargs,
            "model": self.model.value,
            "system": system_str,
            "messages": ant_messages,
            "tools": [tool],
            "tool_choice": {"type": "tool", "name": name},
        }
        self._record_request(payload)
        try:
            response = self._client.messages.create(**payload)
        except anthropic.RequestTooLargeError as exc:
            _raise_for_request_too_large(exc)
        except anthropic.BadRequestError as exc:
            _raise_if_context_overflowed(exc)
            raise
        logger.debug("Anthropic raw response (structured): %s", response)
        self._record_response(response)
        for block in response.content:
            if block.type == "tool_use":
                return json.dumps(block.input)
        return "{}"

    def _structured_call_streamed(
        self,
        system_str,
        ant_messages: list,
        generate_kwargs: dict,
        response_format: dict,
        *,
        append_message: bool,
    ) -> Iterator[StreamChunk]:
        """Streamed Anthropic structured output via a forced single tool.

        Streams the tool-input JSON as it is built (``GENERATING`` chunks from
        ``input_json_delta``). Emits **no** ``THINKING``: a forced ``tool_choice`` is
        incompatible with extended thinking, so ``generate_kwargs`` carries no thinking param
        (same contract as :meth:`_structured_call`). The base accumulates the yielded JSON and
        parses it; ``append_message`` stores the assistant turn for the stateful chat path.
        """
        self.last_thinking = ""
        self.last_usage = None
        generate_kwargs = self._strip_thinking_for_structured(generate_kwargs)
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", str(response_format.get("title", "Response")))[:64] or "Response"
        tool = {"name": name, "description": f"Emit the answer as a {name} object.", "input_schema": response_format}
        payload = {
            **generate_kwargs,
            "model": self.model.value,
            "system": system_str,
            "messages": ant_messages,
            "tools": [tool],
            "tool_choice": {"type": "tool", "name": name},
        }
        self._record_request(payload)
        try:
            with self._client.messages.stream(**payload) as stream:
                for event in stream:
                    if event.type == "content_block_delta" and event.delta.type == "input_json_delta":
                        yield StreamChunk(StreamingContentType.GENERATING, event.delta.partial_json)
                final = stream.get_final_message()
        except anthropic.RequestTooLargeError as exc:
            _raise_for_request_too_large(exc)
        except anthropic.BadRequestError as exc:
            _raise_if_context_overflowed(exc)
            raise
        self._record_response(final)
        text = "{}"
        for block in final.content:
            if block.type == "tool_use":
                text = json.dumps(block.input)
                break
        if append_message:
            self._append_message({"role": "assistant", "content": text})

    # ------------------------------------------------------------------ #
    # generate_kwargs helpers                                              #
    # ------------------------------------------------------------------ #

    # Parameters not accepted by the Anthropic Messages API (e.g. HuggingFace-specific)
    _UNSUPPORTED_KWARGS = frozenset({"max_new_tokens", "do_sample", "num_return_sequences"})

    # anthropic 1.x removed temperature/top_p/top_k from the messages.create()/stream()
    # signatures; the ones that survive _route_sampling_kwargs travel in extra_body, which is
    # merged into the request JSON as-is.
    _SAMPLING_KWARGS = ("temperature", "top_p", "top_k")

    def _rewrite_generate_kwargs(self, kwargs: dict) -> dict:
        # Strip HuggingFace / other framework-specific keys the Anthropic API rejects
        for key in self._UNSUPPORTED_KWARGS:
            kwargs.pop(key, None)
        return self._route_sampling_kwargs(kwargs)

    def _route_sampling_kwargs(self, kwargs: dict) -> dict:
        """Drop the sampling parameters, or move them into ``extra_body``.

        Three separate facts meet here. Extended thinking rejects all three (the API fixes
        temperature at 1 while it is in effect). The ``ADAPTIVE``-style models reject them
        outright, thinking or not. And anthropic 1.x removed them from the ``messages.create()``
        signature, so passing one as a keyword argument is a ``TypeError`` before any request is
        made -- what survives has to go through ``extra_body`` instead.

        One method owns the decision because it has to hold on *every* request path, and the
        structured-output path routes around ``_thinking_kwargs`` entirely: leaving the stripping
        to the thinking helpers is what let a forced ``temperature=1`` reach every structured call.
        ``_resolve_generate_kwargs`` calls this hook on all of them.
        """
        # Peek at (rather than pop) the reserved key: _thinking_kwargs still needs it downstream
        # to build or omit the thinking parameter.
        resolved = kwargs.get(THINKING_KWARG)
        thinking_off_this_call = resolved is not None and not resolved.enabled
        thinking_in_effect = self.is_thinking_model and not thinking_off_this_call
        rejects_sampling = self.model.value in _REJECTS_SAMPLING

        sampling = {key: kwargs.pop(key) for key in self._SAMPLING_KWARGS if key in kwargs}
        if sampling and not (thinking_in_effect or rejects_sampling):
            kwargs["extra_body"] = {**kwargs.get("extra_body", {}), **sampling}
        return kwargs

    def _thinking_kwargs(self, generate_kwargs: dict) -> dict:
        """Inject the thinking parameter for thinking-capable models."""
        kwargs = generate_kwargs.copy()
        resolved = pop_thinking(kwargs)

        if not self.is_thinking_model:
            return kwargs

        style = getattr(self.model, "thinking_style", ThinkingStyle.ENABLED)
        if style is ThinkingStyle.ADAPTIVE:
            return self._adaptive_thinking_kwargs(kwargs, resolved)

        if resolved is not None and not resolved.enabled:
            # On the ENABLED-style models, omitting the parameter is how thinking is disabled.
            # Sampling parameters are only stripped to satisfy extended thinking, so they stay
            # put here.
            return kwargs

        budget = kwargs.pop("thinking_budget_tokens", None)
        if budget is None and resolved is not None and resolved.level is not None:
            budget = _THINKING_BUDGETS[resolved.level]
        if budget is None:
            budget = _DEFAULT_THINKING_BUDGET
        # max_tokens must exceed the thinking budget
        if kwargs.get("max_tokens", 0) <= budget:
            kwargs["max_tokens"] = budget + 1024
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
        # The sampling parameters are already gone: _route_sampling_kwargs drops them whenever
        # thinking is in effect, which is the only way this branch is reached.
        return kwargs

    def _adaptive_thinking_kwargs(self, kwargs: dict, resolved) -> dict:
        """Build the thinking parameter for an ADAPTIVE-style model (see :class:`ThinkingStyle`).

        These models reject ``budget_tokens`` whether or not the request asks them to think; the
        sampling parameters some of them also reject are dropped earlier, by
        ``_route_sampling_kwargs``.
        """
        if kwargs.pop("thinking_budget_tokens", None) is not None:
            # Dropping a parameter the caller explicitly set has to be said out loud. Since the
            # 4.6 line moved to this branch, Haiku 4.5 is the only model where the escape hatch
            # still does anything, and a caller reaching for it elsewhere would otherwise see
            # nothing happen.
            self._warn_once(
                f"{self.model.value} has no thinking budget parameter; thinking_budget_tokens is "
                "ignored. Use thinking='low'/'medium'/'high' to steer effort instead."
            )

        if resolved is not None and not resolved.enabled:
            # Omission is not "off" here the way it is for the ENABLED style: an absent parameter
            # runs adaptive on Opus 5 and Sonnet 5, so disabling has to be said out loud.
            kwargs["thinking"] = {"type": "disabled"}
            return self._cap_effort_for_disabled_thinking(kwargs)

        if resolved is not None and resolved.level is not None:
            kwargs = self._with_effort(kwargs, resolved.level)
        # The model decides whether to think. Give thinking room within the shared max_tokens.
        if kwargs.get("max_tokens", 0) < _ADAPTIVE_THINKING_MAX_TOKENS_FLOOR:
            kwargs["max_tokens"] = _ADAPTIVE_THINKING_MAX_TOKENS_FLOOR
        kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
        return kwargs

    def _with_effort(self, kwargs: dict, level: str) -> dict:
        """Express a portable thinking level as ``output_config.effort``.

        Merged, never assigned: ``output_config`` also carries ``format``, and a caller can
        already pass one straight through ``generate_kwargs`` today, since AIMU polices only the
        eight portable keys. That passthrough is tier 4 -- and the only route to ``"max"``, the
        effort level the three-value portable vocabulary cannot reach -- so a caller's own
        ``effort`` wins over one derived from ``thinking=``.
        """
        effort_levels = self.model.spec.effort_levels
        if not effort_levels:
            # An adaptive model with no declared vocabulary: warn and continue, as before. Every
            # shipped adaptive member declares one, so this covers the next one added without.
            self._warn_once(
                f"{self.model.value} uses adaptive thinking and chooses its own effort; thinking={level!r} ignored."
            )
            return kwargs

        output_config = dict(kwargs.get("output_config") or {})
        if "effort" in output_config:
            return kwargs
        output_config["effort"] = _effort_for_level(level, effort_levels)
        kwargs["output_config"] = output_config
        return kwargs

    def _cap_effort_for_disabled_thinking(self, kwargs: dict) -> dict:
        """Lower a caller's ``xhigh``/``max`` effort to ``high`` when thinking is disabled.

        Opus 5 rejects that pair with a 400, and validates the two independently on every
        request, so it cannot be settled once per client. AIMU never builds the pair itself --
        ``thinking=False`` resolves to ``level=None`` -- so this only ever fires on an effort the
        caller passed through ``generate_kwargs``.

        The effort is lowered rather than the disable reversed. ``thinking=`` is the supported
        surface and the caller said off; silently re-enabling reasoning they turned off is the
        worse failure, being invisible in the response and visible only on the bill.
        """
        output_config = kwargs.get("output_config")
        if not output_config or output_config.get("effort") not in _EFFORT_ABOVE_HIGH:
            return kwargs
        requested = output_config["effort"]
        self._warn_once(
            f"{self.model.value} rejects thinking=False combined with effort={requested!r}; "
            "lowering effort to 'high' and keeping thinking off."
        )
        kwargs["output_config"] = {**output_config, "effort": "high"}
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
          assistant tool_calls    → {"role": "assistant", "content": [{"type": "tool_use", ...}]},
                                    preceded by a text block when the turn also carried prose
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
                    # A turn can carry both prose and tool calls, and _append_assistant_tool_calls
                    # stores the prose deliberately; dropping it here would silently lose the
                    # model's stated reason for the call from every later request. Skipped when
                    # blank, since the API rejects an empty text block.
                    text = msg.get("content") or ""
                    blocks = [{"type": "text", "text": text}] if text.strip() else []
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
        generate_kwargs = self._resolve_generate_kwargs(generate_kwargs)

        if response_format is not None:
            content = self._generate_content(prompt, images, audio)
            messages = [{"role": "user", "content": content}]
            if stream:
                return self._structured_call_streamed(
                    anthropic.NOT_GIVEN, messages, generate_kwargs, response_format, append_message=False
                )
            return self._structured_call(anthropic.NOT_GIVEN, messages, generate_kwargs, response_format)

        generate_kwargs = self._thinking_kwargs(generate_kwargs)

        if stream:
            return self._generate_streamed(prompt, generate_kwargs, images=images, audio=audio)

        payload = {
            **generate_kwargs,
            "model": self.model.value,
            "messages": [{"role": "user", "content": self._generate_content(prompt, images, audio)}],
        }
        self._record_request(payload)
        try:
            response = self._client.messages.create(**payload)
        except anthropic.RequestTooLargeError as exc:
            _raise_for_request_too_large(exc)
        except anthropic.BadRequestError as exc:
            _raise_if_context_overflowed(exc)
            raise
        logger.debug("Anthropic raw response: %s", response)
        self._record_response(response)

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

        payload = {
            **generate_kwargs,
            "model": self.model.value,
            "messages": [{"role": "user", "content": self._generate_content(prompt, images, audio)}],
        }
        self._record_request(payload)
        try:
            with self._client.messages.stream(**payload) as stream:
                for event in stream:
                    if event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "thinking_delta":
                            self.last_thinking += delta.thinking
                            yield StreamChunk(StreamingContentType.THINKING, delta.thinking)
                        elif delta.type == "text_delta":
                            yield StreamChunk(StreamingContentType.GENERATING, delta.text)
                self._record_response(stream.get_final_message())
        except anthropic.RequestTooLargeError as exc:
            _raise_for_request_too_large(exc)
        except anthropic.BadRequestError as exc:
            _raise_if_context_overflowed(exc)
            raise

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
            system = system_str if system_str else anthropic.NOT_GIVEN
            if stream:
                return self._structured_call_streamed(
                    system, ant_messages, generate_kwargs, response_format, append_message=True
                )
            text = self._structured_call(system, ant_messages, generate_kwargs, response_format)
            self._append_message({"role": "assistant", "content": text})
            return text

        generate_kwargs = self._thinking_kwargs(generate_kwargs)

        if stream:
            return self._chat_streamed(generate_kwargs, tools)

        system_str, ant_messages = self._openai_messages_to_anthropic(self.messages)
        ant_tools = self._openai_tools_to_anthropic(tools) if tools else anthropic.NOT_GIVEN

        payload = {
            **generate_kwargs,
            "model": self.model.value,
            "system": system_str if system_str else anthropic.NOT_GIVEN,
            "messages": ant_messages,
            "tools": ant_tools,
        }
        self._record_request(payload)
        try:
            response = self._client.messages.create(**payload)
        except anthropic.RequestTooLargeError as exc:
            _raise_for_request_too_large(exc)
        except anthropic.BadRequestError as exc:
            _raise_if_context_overflowed(exc)
            raise
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

        self._record_response(response)

        # Single turn: record the requested tools (with Anthropic's real tool_use ids so tool
        # results match on the next request) and return. The Agent executes them.
        if tool_use_blocks:
            prepared = [({"name": b.name, "arguments": b.input}, b.id) for b in tool_use_blocks]
            self._append_assistant_tool_calls(prepared, content=text_content)
            if self.last_thinking:
                self.messages[-1]["thinking"] = self.last_thinking
            return text_content

        assistant_msg: dict = {"role": "assistant", "content": text_content}
        if self.last_thinking:
            assistant_msg["thinking"] = self.last_thinking
        self._append_message(assistant_msg)
        return text_content

    def _chat_streamed(self, generate_kwargs: dict[str, Any], tools: list) -> Iterator[StreamChunk]:
        system_str, ant_messages = self._openai_messages_to_anthropic(self.messages)
        ant_tools = self._openai_tools_to_anthropic(tools) if tools else anthropic.NOT_GIVEN

        # Accumulated state from first stream pass
        tool_use_acc: list[dict] = []  # {"id": str, "name": str, "input_json": str}
        first_pass_chunks: list[StreamChunk] = []
        self.last_thinking = ""
        self.last_usage = None

        payload = {
            **generate_kwargs,
            "model": self.model.value,
            "system": system_str if system_str else anthropic.NOT_GIVEN,
            "messages": ant_messages,
            "tools": ant_tools,
        }
        self._record_request(payload)
        try:
            with self._client.messages.stream(**payload) as stream:
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
        except anthropic.RequestTooLargeError as exc:
            _raise_for_request_too_large(exc)
        except anthropic.BadRequestError as exc:
            _raise_if_context_overflowed(exc)
            raise

        self._record_response(stream.get_final_message())

        if not tool_use_acc:
            # No tool calls; yield buffered chunks and store the assistant message.
            full_content = ""
            for sc in first_pass_chunks:
                if sc.phase == StreamingContentType.GENERATING:
                    full_content += sc.content
                yield sc
            assistant_msg: dict = {"role": "assistant", "content": full_content}
            if self.last_thinking:
                assistant_msg["thinking"] = self.last_thinking
            self._append_message(assistant_msg)
            return

        # Single turn: parse accumulated JSON, dispatch, yield TOOL_CALLING chunks, and return.
        # The model's response to the tool results comes on the next chat() call (loop in Agent).
        parsed_blocks = [
            SimpleNamespace(
                id=tub["id"],
                name=tub["name"],
                input=json.loads(tub["input_json"]) if tub["input_json"] else {},
            )
            for tub in tool_use_acc
        ]

        # Yield any prose/thinking the model emitted alongside the tool call, and record the
        # requested tools (with real tool_use ids) + the prose as the assistant message. The Agent
        # executes them; it will emit the TOOL_CALLING chunks.
        full_content = ""
        for sc in first_pass_chunks:
            if sc.phase == StreamingContentType.GENERATING:
                full_content += sc.content
            yield sc
        prepared = [({"name": b.name, "arguments": b.input}, b.id) for b in parsed_blocks]
        self._append_assistant_tool_calls(prepared, content=full_content)
        if self.last_thinking:
            self.messages[-1]["thinking"] = self.last_thinking
