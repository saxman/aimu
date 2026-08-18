"""Portable thinking control: the level vocabulary, argument resolution, and the reserved
key that carries a resolved request down to a provider.

One rule governs resolution: validate the argument, never the model. An invalid value
raises, so a typo cannot silently buy full-effort reasoning. An unsupported model warns and
continues, so swapping models never turns working code into an exception.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

THINKING_LEVELS = ("low", "medium", "high")

# Carries a ResolvedThinking down to the provider inside the generate_kwargs dict, so no
# _chat/_generate signature has to change. Underscore prefixed so a provider that forgets to
# consume it produces a loud SDK rejection rather than a silent wire change.
THINKING_KWARG = "_thinking"

# Qwen's own effort vocabulary tops out at "xhigh" rather than "high". Both the
# OpenAI-compatible wire field and the HuggingFace chat-template kwarg take these values, so
# the table lives here rather than being copied into each provider.
QWEN_REASONING_EFFORT = {"low": "low", "medium": "medium", "high": "xhigh"}


@dataclass(frozen=True)
class ResolvedThinking:
    """A thinking request a provider is expected to honour.

    ``level`` is None when the caller asked for plain on/off, or when the model has no
    effort control and the level was dropped.
    """

    enabled: bool
    level: Optional[str] = None


def resolve_thinking(
    model: Any,
    thinking: Optional[Union[bool, str]],
    *,
    warn: Callable[[str], None],
) -> Optional[ResolvedThinking]:
    """Normalise a ``thinking=`` argument against what ``model`` declares it can do.

    Returns None when a provider has nothing to do: either the caller passed None, or the
    request cannot be honoured and has been warned about. Raises ValueError when the value
    itself is invalid.
    """
    if thinking is None:
        return None

    if isinstance(thinking, str):
        if thinking not in THINKING_LEVELS:
            raise ValueError(
                f"Unknown thinking level {thinking!r}. Valid levels: {', '.join(THINKING_LEVELS)}, or True/False."
            )
        level, enabled = thinking, True
    elif isinstance(thinking, bool):
        level, enabled = None, thinking
    else:
        raise ValueError(f"thinking must be True, False, or one of {', '.join(THINKING_LEVELS)}, got {thinking!r}.")

    if not model.supports_thinking:
        if enabled:
            warn(f"AIMU does not expose reasoning control for {model.value}; thinking={thinking!r} ignored.")
        # Worded as a limit on AIMU rather than on the model: supports_thinking means
        # "reasoning is visible here", and o3 reasons even though this flag is False.
        # Disabling reasoning on a model that has none is a true statement, so it is silent:
        # it lets one call site serve a mixed fleet.
        return None

    if not enabled and not model.thinking_optional:
        warn(f"{model.value} always reasons and cannot be disabled; thinking=False ignored.")
        return None

    if level is not None and not model.thinking_levels:
        warn(f"{model.value} has no effort-level control; thinking={thinking!r} treated as thinking=True.")
        level = None

    return ResolvedThinking(enabled=enabled, level=level)


def pop_thinking(generate_kwargs: dict) -> Optional[ResolvedThinking]:
    """Remove the resolved thinking request from ``generate_kwargs`` and return it.

    Every provider consumes the reserved key through this helper, exactly once per request,
    at the point where the dict stops being AIMU's and becomes the provider's payload. Going
    through one function keeps the contract greppable for whoever adds the next provider, and
    it matters because a missed pop does not fail uniformly: the OpenAI, Anthropic and
    Transformers call paths reject an unknown keyword, but Ollama types its ``options`` field
    as an open mapping and would serialize the key into the request body silently.

    Providers that need the value *before* the payload is final (to pick a sampling profile,
    say) read it with ``generate_kwargs.get(THINKING_KWARG)`` first and still pop here.
    """
    return generate_kwargs.pop(THINKING_KWARG, None)
