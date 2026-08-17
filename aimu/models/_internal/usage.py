"""Normalize provider-specific per-response metadata: token usage, and whether output was cut off.

Every provider returns token counts on its response object under different field
names (OpenAI: ``prompt_tokens``/``completion_tokens``; Anthropic:
``input_tokens``/``output_tokens``; Ollama: ``prompt_eval_count``/``eval_count``).
These helpers flatten that into a single shape so ``client.last_usage`` reads the
same regardless of provider:

    {"input_tokens": int, "output_tokens": int, "total_tokens": int}

Returns ``None`` when the provider did not report usage (some local OpenAI-compat
servers omit it), so a missing value is distinguishable from a real zero.

Token *counts* are surfaced, not dollar cost: per-model pricing drifts and would
need a maintained price table, which the caller can layer on top of these counts.
"""

from __future__ import annotations

from typing import Any, Optional


def _usage_dict(input_tokens: Optional[int], output_tokens: Optional[int]) -> Optional[dict]:
    if input_tokens is None and output_tokens is None:
        return None
    input_tokens = input_tokens or 0
    output_tokens = output_tokens or 0
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def usage_from_openai(response: Any) -> Optional[dict]:
    """Extract usage from an OpenAI / OpenAI-compatible chat-completion response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return _usage_dict(getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None))


def usage_from_anthropic(response: Any) -> Optional[dict]:
    """Extract usage from an Anthropic Messages API response.

    Adds ``cache_creation_input_tokens`` / ``cache_read_input_tokens`` when the response
    reports them (prompt caching), so a cache hit/creation is observable via
    ``client.last_usage``; the base three keys are unchanged when caching is unused.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    result = _usage_dict(getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None))
    if result is None:
        return None
    for field in ("cache_creation_input_tokens", "cache_read_input_tokens"):
        value = getattr(usage, field, None)
        if value is not None:
            result[field] = value
    return result


def _ollama_field(response: Any, key: str) -> Any:
    """One field of an Ollama response, read defensively.

    Ollama responses behave like mappings *and* attribute objects depending on version.
    """
    value = getattr(response, key, None)
    if value is None and isinstance(response, dict):
        value = response.get(key)
    return value


def usage_from_ollama(response: Any) -> Optional[dict]:
    """Extract usage from an Ollama native ``generate`` / ``chat`` response."""
    return _usage_dict(_ollama_field(response, "prompt_eval_count"), _ollama_field(response, "eval_count"))


def truncated_from_ollama(response: Any) -> bool:
    """Whether Ollama stopped this response at a limit rather than at the model's own stopping point.

    ``done_reason == "length"`` covers both ways output runs out: a ``num_predict`` cap, and a prompt
    that leaves too little of ``num_ctx`` to generate in. The second is the one worth surfacing, because
    it looks exactly like a model that answered with nothing.
    """
    return _ollama_field(response, "done_reason") == "length"
