"""Mock-only tests for generation-kwarg precedence: one chain, identical on every provider.

Four tiers reach a request, lowest precedence first:

1. ``Client.DEFAULT_GENERATE_KWARGS`` -- AIMU's own fallbacks, for what nobody else sets.
2. the model card (``ModelSpec.generation_kwargs``, or ``nonthinking_generation_kwargs`` when
   ``thinking=False`` resolved off).
3. ``client.default_generate_kwargs`` -- the caller's standing kwargs for this client instance.
4. the per-call ``generate_kwargs``.

Every invariant is parametrized across **all** providers, sync and async, because each one owns
its own kwarg resolution and the tiers were historically lost one provider at a time:
v0.15.0 fixed tier 2 being *replaced* by tier 4 on Ollama and HuggingFace, and the release after
it fixed tier 2 being ignored outright on Anthropic, the OpenAI-compatible family, and llama.cpp.
A per-provider test would only ever pin the provider someone thought to check.
"""

from __future__ import annotations

import types

import anthropic
import llama_cpp
import ollama
import openai
import pytest
from helpers import client_stand_in

from aimu.models._internal.generate_kwargs import select_profile
from aimu.models._internal.thinking import THINKING_KWARG, ResolvedThinking
from aimu.models.base import Model, ModelSpec
from aimu.models.model_client import ModelClient
from aimu.models.providers.ollama import OllamaClient, OllamaModel


class _CardModel(Model):
    """Catalog members carrying a sampling profile, as the Ollama and HuggingFace members do.

    Synthetic rather than a real member, so the tiers can be asserted with values no client
    fallback shares and a catalog edit cannot quietly weaken these tests. One real-catalog test
    below pins the wiring to a shipped member.
    """

    PLAIN = ModelSpec(
        "test-card-plain",
        tools=True,
        generation_kwargs={"temperature": 0.9, "top_p": 0.3, "min_p": 0.05},
    )
    THINKER = ModelSpec(
        "test-card-thinker",
        tools=True,
        thinking=True,
        generation_kwargs={"temperature": 0.9, "top_p": 0.3},
        nonthinking_generation_kwargs={"temperature": 0.7, "top_p": 0.8},
    )


# --- one client per provider, built against a mocked SDK ---------------------------------------
#
# ``_resolve_generate_kwargs`` is a pure dict transform even on the async clients, so the async
# providers -- which carry their own copies of it -- are covered by the same tests.


def _build_ollama(monkeypatch, model):
    monkeypatch.setattr(ollama, "Client", lambda **kw: types.SimpleNamespace(pull=lambda *a, **k: None))
    return OllamaClient(model)


def _build_aio_ollama(monkeypatch, model):
    from aimu.aio.providers.ollama import AsyncOllamaClient

    monkeypatch.setattr(ollama, "AsyncClient", lambda **kw: types.SimpleNamespace(pull=lambda *a, **k: None))
    return AsyncOllamaClient(model)


def _build_openai_compat(monkeypatch, model):
    from aimu.models.providers.openai_compat import LMStudioOpenAIClient

    monkeypatch.setattr(openai, "OpenAI", lambda **kw: types.SimpleNamespace())
    return LMStudioOpenAIClient(model)


def _build_aio_openai_compat(monkeypatch, model):
    from aimu.aio.providers.openai_compat import AsyncLMStudioOpenAIClient

    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kw: types.SimpleNamespace())
    return AsyncLMStudioOpenAIClient(model)


def _build_anthropic(monkeypatch, model):
    from aimu.models.providers.anthropic import AnthropicClient

    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: types.SimpleNamespace())
    return AnthropicClient(model)


def _build_aio_anthropic(monkeypatch, model):
    from aimu.aio.providers.anthropic import AsyncAnthropicClient

    monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda **kw: types.SimpleNamespace())
    return AsyncAnthropicClient(model)


def _build_llamacpp(monkeypatch, model):
    from aimu.models.providers.llamacpp import LlamaCppClient

    monkeypatch.setattr(llama_cpp, "Llama", lambda **kw: types.SimpleNamespace())
    return LlamaCppClient(model, model_path=f"/nonexistent/{model.value}.gguf")


def _build_hf(monkeypatch, model):
    from aimu.models.providers.hf.text import HuggingFaceClient

    return client_stand_in(HuggingFaceClient, model)


_BUILDERS = {
    "aio_anthropic": _build_aio_anthropic,
    "aio_ollama": _build_aio_ollama,
    "aio_openai_compat": _build_aio_openai_compat,
    "anthropic": _build_anthropic,
    "hf": _build_hf,
    "llamacpp": _build_llamacpp,
    "ollama": _build_ollama,
    "openai_compat": _build_openai_compat,
}

# The three providers whose DEFAULT_GENERATE_KWARGS carries max_tokens under the API's own name.
# Ollama and HuggingFace rename it, and Ollama has no fallbacks at all (unset parameters fall
# through to the server's own defaults).
_MAX_TOKENS_FALLBACK = ["aio_anthropic", "aio_openai_compat", "anthropic", "llamacpp", "openai_compat"]


@pytest.fixture(params=sorted(_BUILDERS))
def card_client(request, monkeypatch):
    return _BUILDERS[request.param](monkeypatch, _CardModel.PLAIN)


@pytest.fixture(params=sorted(_BUILDERS))
def thinking_card_client(request, monkeypatch):
    return _BUILDERS[request.param](monkeypatch, _CardModel.THINKER)


# --- tier 2: the model card is a default the caller can override -------------------------------


def test_the_card_supplies_what_nobody_else_sets(card_client):
    merged = card_client._resolve_generate_kwargs()

    assert merged["temperature"] == 0.9  # the card's value, not any client's own fallback of 0.1
    assert merged["top_p"] == 0.3
    assert merged["min_p"] == 0.05


def test_a_per_call_kwarg_overrides_the_card_without_discarding_it(card_client):
    """The v0.15.0 regression, now pinned for every provider: one key must not replace a profile."""
    merged = card_client._resolve_generate_kwargs({"temperature": 0.2})

    assert merged["temperature"] == 0.2
    assert merged["top_p"] == 0.3


def test_the_instruct_profile_replaces_the_card_when_thinking_resolves_off(thinking_card_client):
    merged = thinking_card_client._resolve_generate_kwargs({THINKING_KWARG: ResolvedThinking(enabled=False)})

    assert merged["temperature"] == 0.7
    assert merged["top_p"] == 0.8


# --- tier 3: client.default_generate_kwargs, the caller's standing choice -----------------------


def test_client_defaults_start_empty(card_client):
    """An input, not a report of the card: "unset" has to stay distinguishable from "set"."""
    assert card_client.default_generate_kwargs == {}


def test_client_defaults_override_the_card(card_client):
    card_client.default_generate_kwargs["temperature"] = 0.5

    merged = card_client._resolve_generate_kwargs()

    assert merged["temperature"] == 0.5
    assert merged["top_p"] == 0.3  # the card still fills what the caller left alone


def test_client_defaults_apply_to_every_call(card_client):
    card_client.default_generate_kwargs["temperature"] = 0.5

    assert card_client._resolve_generate_kwargs()["temperature"] == 0.5
    assert card_client._resolve_generate_kwargs({"top_p": 0.9})["temperature"] == 0.5


def test_reassigning_client_defaults_wholesale_takes_effect(card_client):
    """Assignment, not only mutation: the wrappers used to copy this dict on construction."""
    card_client.default_generate_kwargs = {"temperature": 0.5}

    assert card_client._resolve_generate_kwargs()["temperature"] == 0.5


# --- tier 4: the per-call dict wins over everything --------------------------------------------


def test_per_call_kwargs_override_the_client_defaults(card_client):
    card_client.default_generate_kwargs.update({"temperature": 0.5, "top_p": 0.7})

    merged = card_client._resolve_generate_kwargs({"temperature": 0.2})

    assert merged["temperature"] == 0.2
    assert merged["top_p"] == 0.7  # a client default the call did not name still applies


# --- the merge must not corrupt the process-global profile on the enum member -------------------


def test_the_merge_never_mutates_the_catalog_profile(card_client):
    """``model.generation_kwargs`` is shared by every client of that model.

    ``max_tokens`` is in the call dict deliberately: the providers that rename it (Ollama's
    ``num_predict``, HuggingFace's ``max_new_tokens``) do so with a ``pop``, which would corrupt
    the card if the merge ever returned the card's own dict instead of a fresh one.
    """
    card_client.default_generate_kwargs["temperature"] = 0.5
    card_client._resolve_generate_kwargs({"max_tokens": 2000, "top_p": 0.9})
    card_client._resolve_generate_kwargs({"temperature": 0.3})

    assert _CardModel.PLAIN.generation_kwargs == {"temperature": 0.9, "top_p": 0.3, "min_p": 0.05}


# --- tier 1, and the per-provider request reshaping that runs after the merge -------------------


@pytest.mark.parametrize("provider", _MAX_TOKENS_FALLBACK)
def test_library_fallbacks_fill_what_neither_card_nor_caller_sets(provider, monkeypatch):
    client = _BUILDERS[provider](monkeypatch, _CardModel.PLAIN)

    assert client._resolve_generate_kwargs()["max_tokens"] == 1024


@pytest.mark.parametrize(
    "provider,renamed", [("ollama", "num_predict"), ("aio_ollama", "num_predict"), ("hf", "max_new_tokens")]
)
def test_max_tokens_is_translated_to_the_backends_own_name(provider, renamed, monkeypatch):
    client = _BUILDERS[provider](monkeypatch, _CardModel.PLAIN)

    merged = client._resolve_generate_kwargs({"max_tokens": 2000})

    assert merged[renamed] == 2000
    assert "max_tokens" not in merged


def test_a_real_catalog_profile_survives_a_partial_call_dict(monkeypatch):
    """The tests above use a synthetic card; this one pins the wiring to a shipped member."""
    client = _build_ollama(monkeypatch, OllamaModel.QWEN_3_5_9B)

    merged = client._resolve_generate_kwargs({"max_tokens": 2000})

    assert merged["num_predict"] == 2000
    assert merged["temperature"] == 1.0
    assert merged["top_p"] == 0.95
    assert merged["top_k"] == 20


# --- the wrappers: a layer set on a wrapper has to reach the client that actually runs ----------


def test_client_defaults_reach_the_wire(monkeypatch):
    """The end-to-end guard: what the provider actually sends, not what the merge returns."""
    sent = {}

    def _fake_chat(**kwargs):
        sent.update(kwargs)
        message = types.SimpleNamespace(tool_calls=None, content="hi", role="assistant", thinking=None)
        return {"message": message}

    monkeypatch.setattr(
        ollama, "Client", lambda **kw: types.SimpleNamespace(pull=lambda *a, **k: None, chat=_fake_chat)
    )
    client = ModelClient(OllamaModel.QWEN_3_5_9B)
    client.default_generate_kwargs = {"temperature": 0.42, "top_k": 7}

    client.chat("hello")

    assert sent["options"]["temperature"] == 0.42
    assert sent["options"]["top_k"] == 7
    assert sent["options"]["top_p"] == 0.95  # the card still fills the rest


def test_client_defaults_delegate_through_the_sync_factory(monkeypatch):
    """``aimu.client()`` returns a ModelClient wrapper, so the property must delegate both ways."""
    monkeypatch.setattr(ollama, "Client", lambda **kw: types.SimpleNamespace(pull=lambda *a, **k: None))
    client = ModelClient(OllamaModel.QWEN_3_5_9B)

    client.default_generate_kwargs["temperature"] = 0.5
    assert client._client.default_generate_kwargs["temperature"] == 0.5

    client.default_generate_kwargs = {"temperature": 0.4}
    assert client._client._resolve_generate_kwargs()["temperature"] == 0.4


def test_client_defaults_delegate_through_the_async_factory(monkeypatch):
    from aimu import aio

    monkeypatch.setattr(ollama, "AsyncClient", lambda **kw: types.SimpleNamespace(pull=lambda *a, **k: None))
    client = aio.client("ollama:qwen3.5:9b")

    client.default_generate_kwargs = {"temperature": 0.4}

    assert client._client._resolve_generate_kwargs({"top_p": 0.9})["temperature"] == 0.4


def test_client_defaults_delegate_through_as_model_client(monkeypatch):
    """``agent.as_model_client()`` is a view over the agent's client, so the layer must reach it."""
    from aimu.agents import Agent

    monkeypatch.setattr(ollama, "Client", lambda **kw: types.SimpleNamespace(pull=lambda *a, **k: None))
    inner = ModelClient(OllamaModel.QWEN_3_5_9B)
    view = Agent(inner).as_model_client()

    view.default_generate_kwargs = {"temperature": 0.42}

    assert inner.default_generate_kwargs == {"temperature": 0.42}


def test_client_defaults_reach_the_fallback_chain(monkeypatch):
    """FallbackClient owns the canonical state, so it must push this layer down like the rest."""
    from aimu.models.fallback import FallbackClient

    monkeypatch.setattr(ollama, "Client", lambda **kw: types.SimpleNamespace(pull=lambda *a, **k: None))
    primary = ModelClient(OllamaModel.QWEN_3_5_9B)
    fallback = FallbackClient([primary])
    fallback.default_generate_kwargs = {"temperature": 0.42}

    fallback._load_state(primary)

    assert primary.default_generate_kwargs == {"temperature": 0.42}


# Classes that legitimately own ``_resolve_generate_kwargs``: each delegates the whole resolution
# to an inner client, whose tiers are the ones that matter.
_DELEGATING_WRAPPERS = {
    "ModelClient",
    "AsyncModelClient",
    "_AgenticView",
    "_AsyncAgenticView",
    "_AsyncInProcessClient",
}


def test_no_provider_overrides_the_merge_entrypoint():
    """A provider declares rewrites in ``_rewrite_generate_kwargs``; merging is not its job.

    ``_resolve_generate_kwargs`` is concrete on ``_GenerateKwargsMixin`` and always merges before it
    calls the hook. Overriding it in a provider is the one way to skip the merge again, which is
    the regression this whole module exists to prevent, so every shipped client is checked rather
    than the handful that carry a rewrite today.
    """
    import aimu  # noqa: F401  -- imports every installed provider client

    from aimu.aio._base import AsyncBaseModelClient
    from aimu.models.base import BaseModelClient

    def descendants(cls):
        for subclass in cls.__subclasses__():
            yield subclass
            yield from descendants(subclass)

    offenders = sorted(
        f"{cls.__module__}.{cls.__qualname__}"
        for base in (BaseModelClient, AsyncBaseModelClient)
        for cls in descendants(base)
        if cls.__module__.startswith("aimu.")  # test doubles are free to override anything
        and "_resolve_generate_kwargs" in vars(cls)
        and cls.__name__ not in _DELEGATING_WRAPPERS
    )

    assert offenders == []


# --- tier 2 has two variants; select_profile chooses between them -------------------------------


def test_the_selected_card_profile_depends_on_the_resolved_thinking_mode():
    """Cards specify different sampling for thinking and instruct mode; the resolved mode picks."""
    assert select_profile(_CardModel.THINKER, ResolvedThinking(enabled=True)) == {"temperature": 0.9, "top_p": 0.3}
    assert select_profile(_CardModel.THINKER, ResolvedThinking(enabled=False)) == {"temperature": 0.7, "top_p": 0.8}


def test_profile_selection_falls_back_when_the_card_has_no_instruct_variant():
    """Most cards carry one profile, which then applies in both modes."""
    assert select_profile(_CardModel.PLAIN, ResolvedThinking(enabled=False)) == _CardModel.PLAIN.generation_kwargs
    assert select_profile(_CardModel.PLAIN, None) == _CardModel.PLAIN.generation_kwargs
