"""Shared resilience-config helper for SDK-backed providers (P0-C).

The ``anthropic`` and ``openai`` SDKs implement request timeout and bounded retries
natively. AIMU surfaces them as ``timeout`` / ``max_retries`` constructor kwargs and
forwards them verbatim; this helper builds the forward dict, omitting unset values so
each SDK's own defaults are preserved when the caller doesn't opt in.
"""

from typing import Optional


def sdk_client_kwargs(timeout: Optional[float] = None, max_retries: Optional[int] = None) -> dict:
    """Return the resilience kwargs to forward to an anthropic/openai SDK client.

    Only keys the caller set are included, so passing neither leaves the SDK defaults intact.
    """
    kwargs: dict = {}
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    return kwargs
