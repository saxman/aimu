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
    # The card's third key, min_p, is asserted per provider further down instead: what a request ends
    # up carrying depends on the backend's declared verdict for it (kept, dropped, or moved into
    # extra_body), so it cannot be one claim across every provider.


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


# --- the portable context_length key ------------------------------------------------------------
#
# The context window is sized per request on exactly one provider (Ollama's ``num_ctx``) and set
# out of band everywhere else -- at load time (llama.cpp's ``n_ctx``), at server launch
# (``--ctx-size`` / ``--max-model-len``), or not at all (a cloud model's window is fixed). So the
# portable key is renamed on the one provider that can honour it and dropped on the rest, which
# keeps a client default portable across a provider swap instead of putting an unknown parameter
# on the wire.

# Wrappers that hand a whole request to an inner client, so the inner client's declaration is
# the one that fires: the factories and agentic views (which delegate the resolution itself), the
# in-process async wrappers (whose _resolve_generate_kwargs calls the wrapped sync client's), and
# the fallback clients (which delegate the public chat/generate call).
_DELEGATES_A_WHOLE_REQUEST = _DELEGATING_WRAPPERS | {
    "AsyncHuggingFaceClient",
    "AsyncLlamaCppClient",
    "FallbackClient",
    "AsyncFallbackClient",
}

_CONTEXT_LENGTH_SUPPORTED = ["ollama", "aio_ollama"]
_CONTEXT_LENGTH_UNSUPPORTED = sorted(set(_BUILDERS) - set(_CONTEXT_LENGTH_SUPPORTED))


@pytest.mark.parametrize("provider", _CONTEXT_LENGTH_SUPPORTED)
def test_context_length_is_translated_to_the_backends_own_name(provider, monkeypatch):
    client = _BUILDERS[provider](monkeypatch, _CardModel.PLAIN)

    merged = client._resolve_generate_kwargs({"context_length": 8192})

    assert merged["num_ctx"] == 8192
    assert "context_length" not in merged


@pytest.mark.parametrize("provider", _CONTEXT_LENGTH_SUPPORTED)
def test_context_length_can_be_a_client_default(provider, monkeypatch):
    client = _BUILDERS[provider](monkeypatch, _CardModel.PLAIN)
    client.default_generate_kwargs = {"context_length": 32768}

    assert client._resolve_generate_kwargs()["num_ctx"] == 32768
    assert client._resolve_generate_kwargs({"temperature": 0.2})["num_ctx"] == 32768


@pytest.mark.parametrize("provider", _CONTEXT_LENGTH_SUPPORTED)
def test_a_per_call_context_length_overrides_the_client_default(provider, monkeypatch):
    client = _BUILDERS[provider](monkeypatch, _CardModel.PLAIN)
    client.default_generate_kwargs = {"context_length": 32768}

    assert client._resolve_generate_kwargs({"context_length": 4096})["num_ctx"] == 4096


@pytest.mark.parametrize("provider", _CONTEXT_LENGTH_SUPPORTED)
def test_a_per_call_none_cancels_a_default_context_length(provider, monkeypatch):
    """The only way back to the backend's own sizing once a client default is set."""
    client = _BUILDERS[provider](monkeypatch, _CardModel.PLAIN)
    client.default_generate_kwargs = {"context_length": 32768}

    merged = client._resolve_generate_kwargs({"context_length": None})

    assert "num_ctx" not in merged
    assert "context_length" not in merged


@pytest.mark.parametrize("provider", _CONTEXT_LENGTH_SUPPORTED)
def test_the_backends_own_key_still_passes_through(provider, monkeypatch):
    """``num_ctx`` was reaching Ollama's options verbatim before the portable key existed."""
    client = _BUILDERS[provider](monkeypatch, _CardModel.PLAIN)

    assert client._resolve_generate_kwargs({"num_ctx": 2048})["num_ctx"] == 2048


@pytest.mark.parametrize("provider", _CONTEXT_LENGTH_UNSUPPORTED)
def test_context_length_never_reaches_a_backend_that_cannot_honour_it(provider, monkeypatch):
    """Left in, it would go on the wire as an unknown parameter and fail the request."""
    client = _BUILDERS[provider](monkeypatch, _CardModel.PLAIN)

    merged = client._resolve_generate_kwargs({"context_length": 8192})

    assert "context_length" not in merged
    assert "num_ctx" not in merged


@pytest.mark.parametrize("provider", _CONTEXT_LENGTH_UNSUPPORTED)
def test_dropping_context_length_warns_with_the_backends_own_remedy(provider, monkeypatch, caplog):
    """Silent would be a bug: the caller asked for a window size and did not get one."""
    client = _BUILDERS[provider](monkeypatch, _CardModel.PLAIN)

    with caplog.at_level("WARNING"):
        client._resolve_generate_kwargs({"context_length": 8192})

    assert len(caplog.records) == 1
    message = caplog.records[0].message
    assert "context_length" in message
    assert client.GENERATE_KWARG_SUPPORT["context_length"].remedy in message


@pytest.mark.parametrize("provider", _CONTEXT_LENGTH_UNSUPPORTED)
def test_the_unsupported_context_length_warning_fires_once_per_client(provider, monkeypatch, caplog):
    """Every round of an agent loop resolves kwargs again."""
    client = _BUILDERS[provider](monkeypatch, _CardModel.PLAIN)

    with caplog.at_level("WARNING"):
        for _ in range(3):
            client._resolve_generate_kwargs({"context_length": 8192})

    assert len(caplog.records) == 1


@pytest.mark.parametrize("provider", _CONTEXT_LENGTH_UNSUPPORTED)
def test_cancelling_a_context_length_does_not_warn(provider, monkeypatch, caplog):
    """None means unset, so there is nothing the backend failed to honour."""
    client = _BUILDERS[provider](monkeypatch, _CardModel.PLAIN)

    with caplog.at_level("WARNING"):
        client._resolve_generate_kwargs({"context_length": None})

    assert caplog.records == []


def test_context_length_reaches_the_wire_as_num_ctx(monkeypatch):
    """The end-to-end guard: what Ollama is actually sent, not what the merge returns."""
    sent = {}

    def _fake_chat(**kwargs):
        sent.update(kwargs)
        message = types.SimpleNamespace(tool_calls=None, content="hi", role="assistant", thinking=None)
        return {"message": message}

    monkeypatch.setattr(
        ollama, "Client", lambda **kw: types.SimpleNamespace(pull=lambda *a, **k: None, chat=_fake_chat)
    )
    client = ModelClient(OllamaModel.QWEN_3_5_9B)
    client.default_generate_kwargs = {"context_length": 32768}

    client.chat("hello")

    assert sent["options"]["num_ctx"] == 32768
    assert "context_length" not in sent["options"]


# --- one declared verdict per portable key -------------------------------------------------------
#
# Eight keys, eight backends that spell them differently or lack them outright. Each client declares
# what it does with every one, so an unsupported key is renamed or dropped with a warning naming the
# remedy rather than going on the wire (or, on Ollama, being discarded by the SDK's own request
# validation). This absorbed two earlier mechanisms: the max_tokens rename each provider did in its
# rewrite hook, and the context-length pair of class attributes.

_MIN_P_UNSUPPORTED = ["aio_anthropic", "aio_ollama", "anthropic", "ollama"]
# Accepted *and* left at the top level. The OpenAI-compatible pair accepts min_p too, but its hook
# moves it into extra_body, which is asserted separately below.
_MIN_P_AT_THE_TOP_LEVEL = ["hf", "llamacpp"]


@pytest.mark.parametrize("provider", ["ollama", "aio_ollama", "llamacpp"])
def test_repetition_penalty_is_renamed_to_the_backends_own_spelling(provider, monkeypatch):
    """Ollama and llama.cpp both have the knob, under repeat_penalty; the portable spelling missed it."""
    client = _BUILDERS[provider](monkeypatch, _CardModel.PLAIN)

    merged = client._resolve_generate_kwargs({"repetition_penalty": 1.05})

    assert merged["repeat_penalty"] == 1.05
    assert "repetition_penalty" not in merged


@pytest.mark.parametrize("provider", _MIN_P_UNSUPPORTED)
def test_an_unsupported_key_never_reaches_the_backend(provider, monkeypatch):
    client = _BUILDERS[provider](monkeypatch, _CardModel.PLAIN)

    assert "min_p" not in client._resolve_generate_kwargs({"min_p": 0.05})


@pytest.mark.parametrize("provider", _MIN_P_UNSUPPORTED)
def test_dropping_a_key_warns_with_the_backends_own_remedy(provider, monkeypatch, caplog):
    client = _BUILDERS[provider](monkeypatch, _CardModel.PLAIN)

    with caplog.at_level("WARNING"):
        client._resolve_generate_kwargs({"min_p": 0.05})

    assert len(caplog.records) == 1
    message = caplog.records[0].message
    assert "min_p" in message
    assert client.GENERATE_KWARG_SUPPORT["min_p"].remedy in message


@pytest.mark.parametrize("provider", _MIN_P_UNSUPPORTED)
def test_the_unsupported_warning_fires_once_per_client(provider, monkeypatch, caplog):
    """Every round of an agent loop resolves kwargs again."""
    client = _BUILDERS[provider](monkeypatch, _CardModel.PLAIN)

    with caplog.at_level("WARNING"):
        for _ in range(3):
            client._resolve_generate_kwargs({"min_p": 0.05})

    assert len(caplog.records) == 1


@pytest.mark.parametrize("provider", _MIN_P_AT_THE_TOP_LEVEL)
def test_a_supported_key_passes_through_untouched(provider, monkeypatch):
    client = _BUILDERS[provider](monkeypatch, _CardModel.PLAIN)

    assert client._resolve_generate_kwargs({"min_p": 0.05})["min_p"] == 0.05


@pytest.mark.parametrize("provider", _MIN_P_AT_THE_TOP_LEVEL)
def test_a_cards_supported_key_still_reaches_the_request(provider, monkeypatch):
    """Tier 2's third key: the fixture-wide tier test above cannot claim it on every provider."""
    client = _BUILDERS[provider](monkeypatch, _CardModel.PLAIN)

    assert client._resolve_generate_kwargs()["min_p"] == 0.05


@pytest.mark.parametrize("provider", ["openai_compat", "aio_openai_compat"])
def test_the_non_openai_knobs_are_routed_into_extra_body(provider, monkeypatch):
    """vLLM and llama-server read them there; the OpenAI schema has no top-level place for them."""
    client = _BUILDERS[provider](monkeypatch, _CardModel.PLAIN)

    merged = client._resolve_generate_kwargs({"top_k": 40, "min_p": 0.05, "repetition_penalty": 1.05})

    assert merged["extra_body"] == {"top_k": 40, "min_p": 0.05, "repetition_penalty": 1.05}
    assert not {"top_k", "min_p", "repetition_penalty"} & set(merged)


@pytest.mark.parametrize("provider", ["openai_compat", "aio_openai_compat"])
def test_routing_into_extra_body_keeps_what_is_already_there(provider, monkeypatch):
    """The thinking translation writes chat_template_kwargs into the same dict."""
    client = _BUILDERS[provider](monkeypatch, _CardModel.PLAIN)

    merged = client._resolve_generate_kwargs({"top_k": 40, "extra_body": {"guided_json": {"a": 1}}})

    assert merged["extra_body"] == {"guided_json": {"a": 1}, "top_k": 40, "min_p": 0.05}


def test_a_cards_unsupported_key_is_dropped_without_warning(monkeypatch, caplog):
    """Every Qwen card carries min_p, so warning on the merged dict would fire for a value nobody set."""
    client = _build_ollama(monkeypatch, OllamaModel.QWEN_3_5_9B)

    with caplog.at_level("WARNING"):
        merged = client._resolve_generate_kwargs()

    assert "min_p" not in merged
    assert caplog.records == []


@pytest.mark.parametrize("provider", _MIN_P_UNSUPPORTED)
def test_a_none_value_is_unset_rather_than_unsupported(provider, monkeypatch, caplog):
    """None cancels a client default, the rule context_length already followed."""
    client = _BUILDERS[provider](monkeypatch, _CardModel.PLAIN)
    client.default_generate_kwargs = {"min_p": 0.05}

    with caplog.at_level("WARNING"):
        merged = client._resolve_generate_kwargs({"min_p": None})

    assert "min_p" not in merged
    assert caplog.records == []


def test_a_client_default_warns_like_a_per_call_value(monkeypatch, caplog):
    """Both caller tiers are the caller's own request, so both deserve the report."""
    client = _build_ollama(monkeypatch, _CardModel.PLAIN)
    client.default_generate_kwargs = {"min_p": 0.05}

    with caplog.at_level("WARNING"):
        client._resolve_generate_kwargs()

    assert len(caplog.records) == 1


def test_every_client_declares_a_verdict_for_every_portable_key():
    """A new provider must say, per key, whether it accepts it, renames it, or cannot honour it.

    An undeclared key passes through, which on a backend that rejects it fails the request and on
    Ollama is discarded during request validation with nothing said. The declaration is inherited, so
    a family (the OpenAI-compatible servers) declares it once.

    Presence alone is not enough, which is why the second half checks the shape of each verdict. The
    resolver reads any non-``Unsupported``, non-string value the same way it reads a missing key --
    ``support.get(key) is None`` means "undeclared, pass through" -- so a table written
    ``{"min_p": None}`` would satisfy a presence-only audit while still forwarding ``min_p`` to a
    backend that cannot take it, silently. A verdict is therefore the backend's own spelling as a
    ``str`` or an :class:`Unsupported` carrying the remedy, and nothing else.
    """
    import aimu  # noqa: F401  -- imports every installed provider client

    from aimu.aio._base import AsyncBaseModelClient
    from aimu.models._internal.generate_kwargs import PORTABLE_GENERATE_KWARGS, Unsupported
    from aimu.models.base import BaseModelClient

    def descendants(cls):
        for subclass in cls.__subclasses__():
            yield subclass
            yield from descendants(subclass)

    clients = [
        cls
        for base in (BaseModelClient, AsyncBaseModelClient)
        for cls in descendants(base)
        if cls.__module__.startswith("aimu.") and cls.__name__ not in _DELEGATES_A_WHOLE_REQUEST
    ]

    incomplete = sorted(
        f"{cls.__module__}.{cls.__qualname__}: missing {', '.join(sorted(missing))}"
        for cls in clients
        if (missing := set(PORTABLE_GENERATE_KWARGS) - set(cls.GENERATE_KWARG_SUPPORT))
    )

    assert incomplete == []

    malformed = sorted(
        f"{cls.__module__}.{cls.__qualname__}: {key} = {verdict!r}"
        for cls in clients
        for key in PORTABLE_GENERATE_KWARGS
        if not isinstance(verdict := cls.GENERATE_KWARG_SUPPORT.get(key), (str, Unsupported))
    )

    assert malformed == []


# --- two OpenAI-compatible servers that do not inherit the family verdict unchanged ---------------
#
# "OpenAI-compatible" describes the endpoint, not the sampling surface behind it. Ollama's shim maps a
# fixed OpenAI field set onto its native call and reads none of the three extra knobs, so declaring
# them supported would route them into extra_body to be discarded without a word. llama-server reads
# all three but spells the repetition one repeat_penalty, as llama.cpp's own /completion does.


def _build_ollama_openai(monkeypatch, model):
    from aimu.models.providers.openai_compat import OllamaOpenAIClient

    monkeypatch.setattr(openai, "OpenAI", lambda **kw: types.SimpleNamespace())
    return OllamaOpenAIClient(model)


def _build_aio_ollama_openai(monkeypatch, model):
    from aimu.aio.providers.openai_compat import AsyncOllamaOpenAIClient

    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kw: types.SimpleNamespace())
    return AsyncOllamaOpenAIClient(model)


def _build_llama_server(monkeypatch, model):
    from aimu.models.providers.openai_compat import LlamaServerOpenAIClient

    monkeypatch.setattr(openai, "OpenAI", lambda **kw: types.SimpleNamespace())
    return LlamaServerOpenAIClient(model)


def _build_aio_llama_server(monkeypatch, model):
    from aimu.aio.providers.openai_compat import AsyncLlamaServerOpenAIClient

    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kw: types.SimpleNamespace())
    return AsyncLlamaServerOpenAIClient(model)


_OLLAMA_SHIM_BUILDERS = [_build_ollama_openai, _build_aio_ollama_openai]
_LLAMA_SERVER_BUILDERS = [_build_llama_server, _build_aio_llama_server]


@pytest.mark.parametrize("build", _OLLAMA_SHIM_BUILDERS)
@pytest.mark.parametrize("key", ["top_k", "min_p", "repetition_penalty"])
def test_ollamas_openai_shim_drops_the_knobs_it_cannot_read(build, key, monkeypatch, caplog):
    """Routed into extra_body they would be dropped by the server instead, with nothing said."""
    client = build(monkeypatch, _CardModel.PLAIN)

    with caplog.at_level("WARNING"):
        merged = client._resolve_generate_kwargs({key: 0.5})

    assert key not in merged
    assert key not in merged.get("extra_body", {})
    assert len(caplog.records) == 1
    assert key in caplog.records[0].message
    assert client.GENERATE_KWARG_SUPPORT[key].remedy in caplog.records[0].message


@pytest.mark.parametrize("build", _OLLAMA_SHIM_BUILDERS)
def test_ollamas_openai_shim_keeps_the_five_the_family_declares(build, monkeypatch, caplog):
    """Only the three extra knobs differ from the family; the OpenAI set itself is untouched."""
    client = build(monkeypatch, _CardModel.PLAIN)

    with caplog.at_level("WARNING"):
        merged = client._resolve_generate_kwargs(
            {"temperature": 0.2, "top_p": 0.9, "presence_penalty": 0.5, "max_tokens": 2000}
        )

    assert merged["temperature"] == 0.2
    assert merged["top_p"] == 0.9
    assert merged["presence_penalty"] == 0.5
    assert merged["max_tokens"] == 2000
    assert caplog.records == []
    # context_length is the family's fifth key, dropped here with this client's own remedy.
    assert "OLLAMA_CONTEXT_LENGTH" in client.GENERATE_KWARG_SUPPORT["context_length"].remedy


@pytest.mark.parametrize("build", _LLAMA_SERVER_BUILDERS)
def test_llama_server_routes_the_repetition_knob_under_llama_cpps_own_name(build, monkeypatch, caplog):
    """vLLM and SGLang read repetition_penalty; llama.cpp spells the same knob repeat_penalty.

    The rename and the routing have to agree: the OpenAI SDK's ``create()`` takes no arbitrary
    keywords, so a renamed key left at the top level would raise ``TypeError`` instead of reaching the
    server.
    """
    client = build(monkeypatch, _CardModel.PLAIN)

    with caplog.at_level("WARNING"):
        merged = client._resolve_generate_kwargs({"top_k": 40, "repetition_penalty": 1.05})

    assert merged["extra_body"] == {"top_k": 40, "min_p": 0.05, "repeat_penalty": 1.05}
    assert not {"top_k", "min_p", "repetition_penalty", "repeat_penalty"} & set(merged)
    assert caplog.records == []
