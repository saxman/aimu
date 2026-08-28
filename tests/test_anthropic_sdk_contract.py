"""Bind the payloads ``AnthropicClient`` builds against the installed SDK's real signature.

Every other Anthropic test monkeypatches ``messages.create``, so a payload key the SDK no
longer accepts passes them all and fails only against the live API. That is how ``anthropic``
1.x's removal of ``temperature`` / ``top_p`` / ``top_k`` could have shipped unnoticed: the two
paths that still carried a sampling parameter (a thinking-capable model with ``thinking=False``,
and every structured-output call, which had ``temperature=1`` forced into it) raise ``TypeError``
before any request is made.

``inspect.signature(...).bind()`` is the whole test: it asks the installed SDK whether the call
AIMU is about to make is even expressible, without a key, a network, or a mock.
"""

from __future__ import annotations

import inspect
import types

import pytest

from aimu.models import HAS_ANTHROPIC

pytestmark = pytest.mark.skipif(not HAS_ANTHROPIC, reason="anthropic not installed")


def _client(model):
    """An AnthropicClient whose SDK client is a stub, so no API key is needed."""
    import anthropic

    from aimu.models.providers.anthropic import AnthropicClient

    original = anthropic.Anthropic
    anthropic.Anthropic = lambda **kwargs: types.SimpleNamespace()
    try:
        return AnthropicClient(model)
    finally:
        anthropic.Anthropic = original


def _models():
    from aimu.models.providers.anthropic import AnthropicModel

    return list(AnthropicModel)


def _payloads(client, thinking):
    """The final kwargs for both request paths: the chat/generate one and the structured one."""
    resolved = client._resolve_generate_kwargs(
        client._apply_thinking({"max_tokens": 1024, "temperature": 0.3, "top_p": 0.9, "top_k": 40}, thinking)
    )
    return {
        "chat": client._thinking_kwargs(resolved),
        "structured": client._strip_thinking_for_structured(resolved),
    }


@pytest.mark.parametrize("thinking", [None, True, False, "low"])
def test_every_request_path_binds_against_the_installed_sdk(thinking):
    from anthropic.resources.messages import Messages

    for model in _models():
        client = _client(model)
        for path, kwargs in _payloads(client, thinking).items():
            payload = {**kwargs, "model": model.value, "messages": [{"role": "user", "content": "hi"}]}
            for method in ("create", "stream"):
                signature = inspect.signature(getattr(Messages, method))
                try:
                    signature.bind(object(), **payload)
                except TypeError as exc:
                    pytest.fail(f"{model.name} thinking={thinking!r} {path} -> messages.{method}(): {exc}")


@pytest.mark.parametrize("thinking", [None, True, False, "low"])
def test_sampling_parameters_never_ride_as_keyword_arguments(thinking):
    """They are gone from the 1.x signature, so extra_body is the only route left."""
    from aimu.models.providers.anthropic import AnthropicClient

    for model in _models():
        client = _client(model)
        for path, kwargs in _payloads(client, thinking).items():
            leaked = set(AnthropicClient._SAMPLING_KWARGS) & set(kwargs)
            assert not leaked, f"{model.name} thinking={thinking!r} {path} passed {leaked} as keyword arguments"


def test_sampling_survives_in_extra_body_where_the_model_accepts_it():
    """Dropping them everywhere would be safe but wrong: the pre-4.7 models still honour them,
    and AIMU declares all three supported in ``ANTHROPIC_GENERATE_KWARGS``."""
    from aimu.models.providers.anthropic import AnthropicModel

    client = _client(AnthropicModel.CLAUDE_SONNET_4_6)
    kwargs = _payloads(client, False)["chat"]

    assert kwargs["extra_body"] == {"temperature": 0.3, "top_p": 0.9, "top_k": 40}


def test_sampling_is_dropped_on_the_adaptive_models():
    """Opus 4.7+ / Sonnet 5 / Fable 5 reject all three outright, thinking or not, so extra_body
    must not smuggle them back in."""
    from aimu.models.providers.anthropic import AnthropicModel

    for model in (AnthropicModel.CLAUDE_OPUS_5, AnthropicModel.CLAUDE_SONNET_5, AnthropicModel.CLAUDE_FABLE_5):
        client = _client(model)
        for path, kwargs in _payloads(client, False).items():
            assert "extra_body" not in kwargs, f"{model.name} {path}"


def test_removed_sdk_surface_is_actually_gone():
    """A guard on the pin, not on AIMU: these are the 1.x removals that would silently
    re-enable the old code paths if the environment resolved back to a 0.x release."""
    import anthropic
    from anthropic.resources.messages import Messages

    assert not hasattr(anthropic, "HUMAN_PROMPT")
    assert not hasattr(anthropic, "AI_PROMPT")
    assert not hasattr(anthropic.Anthropic, "completions")
    assert "temperature" not in inspect.signature(Messages.create).parameters
