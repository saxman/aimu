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
        raise ValueError(f"thinking must be True, False, or one of {THINKING_LEVELS}, got {thinking!r}.")

    if not model.supports_thinking:
        if enabled:
            warn(f"{model.value} is not a thinking model; thinking={thinking!r} ignored.")
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


def select_profile(model: Any, resolved: Optional[ResolvedThinking]) -> dict:
    """Return the sampling profile for the resolved mode.

    Model cards specify different sampling for thinking and instruct mode. Falls back to the
    single profile when the model declares no instruct-mode variant.
    """
    if resolved is not None and not resolved.enabled and model.nonthinking_generation_kwargs:
        return dict(model.nonthinking_generation_kwargs)
    return dict(model.generation_kwargs)
