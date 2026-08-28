"""Generation-kwarg resolution: the four precedence tiers and the provider rewrite hook.

Every request, ``chat()`` and ``generate()`` alike, gets its sampling parameters from the same
place. Four tiers layer into one dict (see :func:`merge_generate_kwargs`), then the provider
reshapes the result for its own API. Both steps live here so no provider has to reimplement
either: spreading the tiers per provider is how three of them came to ignore the model card.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from .thinking import THINKING_KWARG, ResolvedThinking

# The portable name for the size of the model's context window, in tokens. Backends spell it
# differently (Ollama's ``num_ctx``) or cannot be told at all, so it is translated or dropped
# during resolution rather than forwarded, which keeps a client-level default portable across a
# provider swap. Unprefixed, unlike THINKING_KWARG: this is a documented public kwarg a caller
# sets, not a resolved request the library threads through.
CONTEXT_LENGTH_KWARG = "context_length"


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


# Every generation parameter AIMU accepts under one portable name whatever the backend. Each client
# declares a verdict for every one of them (see ``_GenerateKwargsMixin.GENERATE_KWARG_SUPPORT``):
# backends spell them differently (Ollama's ``repeat_penalty``, ``num_predict``, ``num_ctx``) or cannot
# take them at all (Anthropic has no ``min_p``; only Ollama's native API sizes the context window per
# request), and an undeclared key either goes on the wire and fails the request or, on Ollama, is
# discarded during request validation with nothing said.
PORTABLE_GENERATE_KWARGS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "repetition_penalty",
    "max_tokens",
    CONTEXT_LENGTH_KWARG,
)


@dataclass(frozen=True)
class Unsupported:
    """A key this backend has no equivalent for: drop it, and name ``remedy`` in the warning.

    A verdict rather than a silent omission, because a caller who set a parameter and did not get one
    has no other way to find out. ``remedy`` says where to set it instead, or why there is nowhere.
    """

    remedy: str


# Tier-1 output caps, in one place so the provider families cannot drift apart on them. This is
# the weakest tier (a model card or either caller tier still wins), so what it has to get right is
# the value nobody sets -- and the old 1024 was low enough to truncate an ordinary answer, which is
# the worst shape of failure available here: silent, mid-sentence, and only visible as a retry.
#
# Two numbers rather than one, because the two deployments fail differently. A cloud endpoint bills
# per token and stops at EOS, so a high cap costs nothing on a well-behaved turn; 16000 is the
# current vendor guidance for a non-streaming request, large enough for a long answer and small
# enough to stay inside an SDK's default HTTP timeout. A local server spends wall-clock on every
# token it is allowed, and a quantized model that never emits EOS spends all of them, so local
# stays at 4096 -- the value HuggingFaceClient already chose for the same reason.
#
# Streaming could afford far more (vendors suggest ~64000), but AIMU has one tier for both modes,
# so these are the streaming-safe intersection. Raise it per client with
# ``client.default_generate_kwargs["max_tokens"]``.
CLOUD_MAX_TOKENS = 16000
LOCAL_MAX_TOKENS = 4096


def apply_kwarg_support(
    kwargs: dict,
    *,
    support: dict,
    caller_keys: set,
    model_id: str,
    warn: Callable[[str], None],
) -> dict:
    """Rename each portable key into the backend's own spelling, or drop it with a warning.

    Follows the thinking control's rule: validate the argument, never the model. A backend that cannot
    honour a key warns and continues, so moving a working client default to another provider never
    raises mid-run.

    Runs on the base, between the tier merge and the provider's :meth:`_rewrite_generate_kwargs` hook,
    because dropping a key is a rule that must hold on every provider and an opt-in hook cannot carry
    one. A hook still owns the reshapes that are not a rename: where a surviving key *goes* (the
    OpenAI-compatible ``extra_body``), and a rename that depends on the model rather than the client
    (the o-series ``max_completion_tokens``).

    ``caller_keys`` is what the caller's own two tiers set (``client.default_generate_kwargs`` and the
    per-call dict). Only those warn: a model card's profile carries ``min_p`` and ``repetition_penalty``
    on every Qwen member, so reporting on the merged dict would fire once per client for a value the
    user never chose.

    ``None`` means unset, so a per-call ``None`` cancels a client default without reporting an
    unsupported key.
    """
    for key in PORTABLE_GENERATE_KWARGS:
        if key not in kwargs:
            continue
        if kwargs[key] is None:
            del kwargs[key]
            continue
        verdict = support.get(key)
        if verdict is None:  # undeclared: pass through, which the audit test forbids in-tree
            continue
        if isinstance(verdict, Unsupported):
            del kwargs[key]
            if key in caller_keys:
                warn(f"{model_id}: {key} is not supported here. {verdict.remedy}")
        elif verdict != key:
            kwargs[verdict] = kwargs.pop(key)
    return kwargs


class _GenerateKwargsMixin:
    """Mixin resolving a request's generation kwargs, shared by both base clients.

    Subclasses must provide:
      - ``model``: a :class:`Model` enum member (carries the card's sampling profile)
      - ``default_generate_kwargs``: ``dict``, the caller's standing kwargs for this client
      - ``_warn_once``: from :class:`_ChatStateMixin`, alongside which this mixin is always used
    """

    # Tier 1: the client's own library fallbacks, for parameters neither the model card nor the
    # caller sets. Overridden per provider (e.g. ``max_tokens``); deliberately the weakest tier,
    # so it never beats a model card.
    DEFAULT_GENERATE_KWARGS: dict = {}

    # This backend's verdict for each portable key: its own spelling for the ones it accepts, an
    # ``Unsupported`` (carrying the remedy) for the ones it cannot honour. Empty here, which passes
    # every key through untouched; every shipped client declares all eight, and a test enforces it.
    GENERATE_KWARG_SUPPORT: dict = {}

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
        # Handled here rather than in each provider's hook because a provider that forgot the drop
        # would put an unknown parameter on the wire.
        merged = apply_kwarg_support(
            merged,
            support=self.GENERATE_KWARG_SUPPORT,
            # The caller's own two tiers, so a card-supplied key is dropped as quietly as it is
            # ignored today rather than warned about on every client.
            caller_keys=set(self.default_generate_kwargs) | set(generate_kwargs or {}),
            model_id=self.model.value,
            warn=self._warn_once,
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
