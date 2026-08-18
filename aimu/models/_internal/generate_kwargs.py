"""Generation-kwarg resolution: the four precedence tiers and the provider rewrite hook.

Every request, ``chat()`` and ``generate()`` alike, gets its sampling parameters from the same
place. Four tiers layer into one dict (see :func:`merge_generate_kwargs`), then the provider
reshapes the result for its own API. Both steps live here so no provider has to reimplement
either: spreading the tiers per provider is how three of them came to ignore the model card.
"""

from __future__ import annotations

from typing import Any, Optional

from .thinking import THINKING_KWARG, ResolvedThinking


def select_profile(model: Any, resolved: Optional[ResolvedThinking]) -> dict:
    """Return the sampling profile for the resolved mode.

    Model cards specify different sampling for thinking and instruct mode. Falls back to the
    single profile when the model declares no instruct-mode variant.
    """
    if resolved is not None and not resolved.enabled and model.nonthinking_generation_kwargs:
        return dict(model.nonthinking_generation_kwargs)
    return dict(model.generation_kwargs)


def merge_generate_kwargs(model: Any, fallbacks: dict, client_defaults: dict, generate_kwargs: Optional[dict]) -> dict:
    """Layer sampling parameters into one request dict, lowest precedence first.

    1. ``fallbacks``: the client's own library defaults, for parameters nobody else sets
       (``max_tokens``, HuggingFace's ``max_new_tokens``). Deliberately the *weakest* tier: a
       library-chosen ``temperature=0.1`` must not quietly beat every model card's tuned value.
    2. the model card's profile (``ModelSpec.generation_kwargs``, or the instruct-mode variant
       when thinking resolved off), as recommended sampling the caller may override.
    3. ``client_defaults``: ``client.default_generate_kwargs``, the caller's standing choice for
       every call on this client instance.
    4. ``generate_kwargs``: the caller's per-call dict, which wins over all of the above.

    Returns a fresh dict, so the caller's dict and the process-global profile on the enum
    member are both left untouched.

    The reserved thinking key is read but not popped: the profile depends on which mode was
    resolved, and every provider still needs the value afterwards to build its own request.
    """
    caller = dict(generate_kwargs or {})
    return {
        **fallbacks,
        **select_profile(model, caller.get(THINKING_KWARG)),
        **client_defaults,
        **caller,
    }


class _GenerateKwargsMixin:
    """Mixin resolving a request's generation kwargs, shared by both base clients.

    Subclasses must provide:
      - ``model``: a :class:`Model` enum member (carries the card's sampling profile)
      - ``default_generate_kwargs``: ``dict``, the caller's standing kwargs for this client
    """

    # Tier 1: the client's own library fallbacks, for parameters neither the model card nor the
    # caller sets. Overridden per provider (e.g. ``max_tokens``); deliberately the weakest tier,
    # so it never beats a model card.
    DEFAULT_GENERATE_KWARGS: dict = {}

    def _resolve_generate_kwargs(self, generate_kwargs: Optional[dict] = None) -> dict:
        """Resolve the generation kwargs for one request, in the provider's own vocabulary.

        Layers the four tiers, gathering them off ``self``, then hands the merged dict to the
        provider's :meth:`_rewrite_generate_kwargs` hook. Called once per request by
        ``_chat_setup`` and by each ``_generate`` implementation; returns a new dict, leaving
        ``generate_kwargs`` untouched.
        """
        merged = merge_generate_kwargs(
            self.model, self.DEFAULT_GENERATE_KWARGS, self.default_generate_kwargs, generate_kwargs
        )
        return self._rewrite_generate_kwargs(merged)

    def _rewrite_generate_kwargs(self, kwargs: dict) -> dict:
        """Rewrite already-merged kwargs into this provider's request shape.

        AIMU exposes one standard set of generation kwargs (``max_tokens``, ``temperature``,
        ``top_p``, ...) whatever the provider, so a client whose API spells them differently
        overrides this hook to:

        - **rename** a standard key (``max_tokens`` becomes ``num_predict`` on Ollama,
          ``max_new_tokens`` on HuggingFace, ``max_completion_tokens`` on OpenAI's o-series);
        - **drop** a key the provider rejects (Transformers' ``generate()`` raises on
          ``presence_penalty``; the Anthropic API rejects HuggingFace-specific keys);
        - **override** a value the API mandates (Anthropic forces ``temperature=1`` and drops
          ``top_p`` while extended thinking is in effect).

        The reserved thinking key (:data:`~aimu.models._internal.thinking.THINKING_KWARG`) rides
        along in ``kwargs``; a provider that translates a resolved thinking request into request
        fields does so here, otherwise a downstream helper consumes it before the API call.

        ``kwargs`` is a fresh dict owned by the caller, so rewrite it in place and return it. The
        default is no rewrite, for a provider that speaks the standard names as-is.
        """
        return kwargs
