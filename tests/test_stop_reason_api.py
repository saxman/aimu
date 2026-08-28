"""How a turn ended: truncation capture per provider, and Anthropic's HTTP-200 refusal.

The cross-provider guard that every request path records *something* lives in
``test_request_legibility.py`` next to its ``_record_request`` sibling, reusing that module's
driver harness. This file covers what the guard cannot: that the value each provider captures is
read correctly (each spells truncation its own way), and that the two consequences work -- the
agent loop's ``TruncatedTurnError`` and the typed ``ModelRefusalError``.

Mock-only: no key, no network, no weights.
"""

from __future__ import annotations

import types

import pytest

from aimu.models import HAS_ANTHROPIC, HAS_LLAMACPP, HAS_OPENAI_COMPAT

pytestmark_anthropic = pytest.mark.skipif(not HAS_ANTHROPIC, reason="anthropic not installed")
pytestmark_openai = pytest.mark.skipif(not HAS_OPENAI_COMPAT, reason="openai not installed")
pytestmark_llamacpp = pytest.mark.skipif(not HAS_LLAMACPP, reason="llama-cpp-python not installed")


# --------------------------------------------------------------------------------------- #
# The seam itself                                                                           #
# --------------------------------------------------------------------------------------- #


def _seam_holder():
    """A bare object carrying just what ``_record_stop_reason`` writes."""
    from aimu.models._internal.chat_state import _ChatStateMixin

    holder = types.SimpleNamespace(last_stop_reason="stale", last_output_truncated=True)
    holder._record_stop_reason = _ChatStateMixin._record_stop_reason.__get__(holder)
    return holder


@pytest.mark.parametrize("reason,truncated", [("length", True), ("max_tokens", True), ("stop", False), ("", False)])
def test_the_seam_maps_each_backends_word_for_running_out_of_room(reason, truncated):
    """One vocabulary rather than a per-provider comparison: OpenAI-compatible servers and Ollama
    say "length", Anthropic says "max_tokens", and the rule has to hold on every backend."""
    holder = _seam_holder()

    holder._record_stop_reason(reason)

    assert holder.last_stop_reason == reason
    assert holder.last_output_truncated is truncated


def test_the_seam_records_none_as_none_rather_than_as_finished():
    """``None`` means the provider said nothing, which is not the same claim as "ended normally".

    Recorded as-is so a caller reading ``last_stop_reason`` can tell the two apart; the derived
    ``last_output_truncated`` is False either way, because there is no evidence of truncation.
    """
    holder = _seam_holder()

    holder._record_stop_reason(None)

    assert holder.last_stop_reason is None
    assert holder.last_output_truncated is False


def test_reset_clears_it_so_a_stale_reason_cannot_outlive_a_conversation():
    from tests.helpers import MockModelClient  # noqa: PLC0415

    client = MockModelClient(["ok"])
    client.last_stop_reason = "length"
    client.last_output_truncated = True

    client.reset()

    assert client.last_stop_reason is None
    assert client.last_output_truncated is False


# --------------------------------------------------------------------------------------- #
# Per-provider capture: each backend spells it differently                                  #
# --------------------------------------------------------------------------------------- #


@pytestmark_openai
@pytest.mark.parametrize("finish,truncated", [("length", True), ("stop", False)])
def test_openai_compat_reads_finish_reason(finish, truncated):
    from aimu.models.providers.openai_compat import OpenAICompatClient

    message = types.SimpleNamespace(content="cut off mid-", tool_calls=None, reasoning_content=None)
    response = types.SimpleNamespace(choices=[types.SimpleNamespace(message=message, finish_reason=finish)], usage=None)
    holder = _seam_holder()
    holder.last_usage = None

    OpenAICompatClient._record_response(holder, response)

    assert holder.last_stop_reason == finish
    assert holder.last_output_truncated is truncated


@pytestmark_openai
def test_openai_compat_streaming_keeps_usage_and_finish_reason_from_separate_chunks():
    """Streaming splits them: finish_reason rides the last content chunk, usage rides a terminal
    chunk whose ``choices`` is empty. Recording each only when present keeps one from clobbering
    the other with None -- the bug a single unconditional assignment would introduce."""
    from aimu.models.providers.openai_compat import OpenAICompatClient

    holder = _seam_holder()
    holder.last_usage = None
    content_chunk = types.SimpleNamespace(
        usage=None, choices=[types.SimpleNamespace(delta=None, finish_reason="length")]
    )
    usage_chunk = types.SimpleNamespace(
        usage=types.SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10), choices=[]
    )

    OpenAICompatClient._record_stream_chunk(holder, content_chunk)
    OpenAICompatClient._record_stream_chunk(holder, usage_chunk)

    assert holder.last_stop_reason == "length"
    assert holder.last_output_truncated is True
    assert holder.last_usage["total_tokens"] == 10


@pytestmark_llamacpp
@pytest.mark.parametrize("finish,truncated", [("length", True), ("stop", False)])
def test_llamacpp_reads_finish_reason_from_a_plain_dict(finish, truncated):
    """llama-cpp-python returns dicts, not SDK objects, so the read indexes rather than getattr-s."""
    from aimu.models.providers.llamacpp import LlamaCppClient

    holder = _seam_holder()

    LlamaCppClient._record_response(holder, {"choices": [{"finish_reason": finish}]})

    assert holder.last_stop_reason == finish
    assert holder.last_output_truncated is truncated


def test_hf_infers_truncation_from_the_generated_length():
    """Transformers reports no reason, so hitting the cap is only visible as "generated the cap"."""
    from aimu.models.providers.hf.text import HuggingFaceClient

    holder = _seam_holder()

    HuggingFaceClient._record_generated_length(holder, 16, {"max_new_tokens": 16})
    assert holder.last_output_truncated is True

    HuggingFaceClient._record_generated_length(holder, 9, {"max_new_tokens": 16})
    assert holder.last_output_truncated is False
    assert holder.last_stop_reason == "stop"


def test_hf_without_a_declared_cap_claims_nothing():
    from aimu.models.providers.hf.text import HuggingFaceClient

    holder = _seam_holder()

    HuggingFaceClient._record_generated_length(holder, 999, {})

    assert holder.last_output_truncated is False


# --------------------------------------------------------------------------------------- #
# Anthropic's refusal: HTTP 200, no content                                                 #
# --------------------------------------------------------------------------------------- #


def _anthropic_client():
    import anthropic

    from aimu.models.providers.anthropic import AnthropicClient, AnthropicModel

    original = anthropic.Anthropic
    anthropic.Anthropic = lambda **kwargs: types.SimpleNamespace()
    try:
        return AnthropicClient(AnthropicModel.CLAUDE_OPUS_5)
    finally:
        anthropic.Anthropic = original


@pytestmark_anthropic
def test_anthropic_refusal_raises_instead_of_returning_an_empty_string():
    """A declined request is a 200 with no text block, so every read path here returned "" and
    told the caller nothing. Opus 5 and Fable 5 ship the classifiers that produce it."""
    from aimu.models import ModelRefusalError

    client = _anthropic_client()
    response = types.SimpleNamespace(
        content=[],
        usage=types.SimpleNamespace(input_tokens=5, output_tokens=0),
        stop_reason="refusal",
        stop_details=types.SimpleNamespace(category="cyber", explanation="declined for safety"),
    )

    with pytest.raises(ModelRefusalError) as info:
        client._record_response(response)

    assert info.value.category == "cyber"
    assert info.value.explanation == "declined for safety"
    assert "cyber" in str(info.value)
    # Usage is recorded before the raise, so a caller can still see what the attempt cost.
    assert client.last_usage["input_tokens"] == 5
    assert client.last_stop_reason == "refusal"


@pytestmark_anthropic
def test_anthropic_refusal_without_details_still_raises():
    """``stop_details`` is documented as populated only on refusal, and its category is an open
    set that may be None -- so the error must not depend on either being present."""
    from aimu.models import ModelRefusalError

    client = _anthropic_client()
    response = types.SimpleNamespace(content=[], usage=None, stop_reason="refusal", stop_details=None)

    with pytest.raises(ModelRefusalError) as info:
        client._record_response(response)

    assert info.value.category is None


@pytestmark_anthropic
@pytest.mark.parametrize("reason,truncated", [("max_tokens", True), ("end_turn", False), ("tool_use", False)])
def test_anthropic_ordinary_stop_reasons_do_not_raise(reason, truncated):
    client = _anthropic_client()
    response = types.SimpleNamespace(content=[], usage=None, stop_reason=reason)

    client._record_response(response)

    assert client.last_stop_reason == reason
    assert client.last_output_truncated is truncated


@pytestmark_anthropic
def test_refusal_is_a_distinct_class_so_fallback_can_retry_on_it():
    """The documented recovery for a refusal is another model, which is what FallbackClient does."""
    from aimu.models import FallbackClient, ModelRefusalError

    refusing = types.SimpleNamespace(
        chat=lambda *a, **k: (_ for _ in ()).throw(ModelRefusalError("declined", category="cyber")),
        messages=[],
        system_message=None,
        tools=[],
        default_generate_kwargs={},
    )
    answering = types.SimpleNamespace(
        chat=lambda *a, **k: "the other model answered",
        messages=[],
        system_message=None,
        tools=[],
        default_generate_kwargs={},
    )
    for stand_in in (refusing, answering):
        stand_in.reset = lambda *a, **k: None
        stand_in.last_usage = None
        stand_in.last_stop_reason = None
        stand_in.last_output_truncated = False
        stand_in.last_thinking = ""
        stand_in.last_structured = None
        stand_in.last_request = None
        stand_in.model = types.SimpleNamespace(value="fake")

    client = FallbackClient([refusing, answering], retry_on=(ModelRefusalError,))

    assert client.chat("hi") == "the other model answered"


# --------------------------------------------------------------------------------------- #
# The consequence the whole signal exists for                                               #
# --------------------------------------------------------------------------------------- #


def test_a_truncated_empty_turn_now_raises_through_an_agent_on_any_provider():
    """The defect this fixes end to end: before, only Ollama set the flag, so an output cut off
    before it produced anything surfaced as a bare empty string on every other backend and the
    agent loop reported the generic degenerate-turn story instead of the actionable one."""
    from aimu.agents import Agent, TruncatedTurnError
    from tests.helpers import MockModelClient  # noqa: PLC0415

    client = MockModelClient(["", "never reached"])
    client._record_stop_reason("length")  # what a provider now records for a capped response
    # No system message on purpose: an Agent that sets one resets the client on _prepare_run, and
    # reset() now clears the stop reason -- correctly, since it belongs to the turn, not the client.
    agent = Agent(client, name="squeezed")

    with pytest.raises(TruncatedTurnError):
        agent.run("do the thing")
