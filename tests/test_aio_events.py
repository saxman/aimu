"""Async mirror of tests/test_events.py: turn events from the async model client.

See tests/test_events.py for the sync surface; these tests exercise the same
ModelTurnStarted / ModelTurnFinished emission points on aimu.aio.
"""

from tests.helpers_aio import MockAsyncModelClient


async def test_async_client_emits_turn_events_with_no_agent():
    """A bare await chat() is observable with no agent, same as the sync surface."""
    from aimu.events import ModelTurnFinished, ModelTurnStarted

    seen = []
    client = MockAsyncModelClient(["hello"])
    client.events = seen.append
    await client.chat("hi")

    kinds = [type(e).__name__ for e in seen]
    assert "ModelTurnStarted" in kinds
    assert "ModelTurnFinished" in kinds
    started = next(e for e in seen if isinstance(e, ModelTurnStarted))
    finished = next(e for e in seen if isinstance(e, ModelTurnFinished))
    assert started.message_count >= 1
    assert finished.text == "hello"
    assert finished.duration_s >= 0.0


async def test_async_no_sink_means_no_behaviour_change():
    """The default path must be byte-identical to before."""
    client = MockAsyncModelClient(["hello"])
    assert client.events is None
    assert await client.chat("hi") == "hello"


async def test_async_generate_emits_turn_events_too():
    from aimu.events import ModelTurnFinished

    seen = []
    client = MockAsyncModelClient(["generated"])
    client.events = seen.append
    await client.generate("prompt")
    assert any(isinstance(e, ModelTurnFinished) for e in seen)


async def test_async_streamed_chat_emits_turn_finished_once_on_drain():
    """ModelTurnFinished must not fire until the async iterator is fully drained: usage
    only populates then, and emitting eagerly would report a turn that hasn't run yet."""
    from aimu.events import ModelTurnFinished, ModelTurnStarted

    seen = []
    client = MockAsyncModelClient(["streamed hello"])
    client.events = seen.append

    stream = await client.chat("hi", stream=True)
    # ModelTurnStarted fires as soon as chat() is called, before any chunk is produced.
    assert any(isinstance(e, ModelTurnStarted) for e in seen)
    assert not any(isinstance(e, ModelTurnFinished) for e in seen)

    chunks = [chunk async for chunk in stream]
    assert chunks  # sanity: the mock actually streamed something

    finished = [e for e in seen if isinstance(e, ModelTurnFinished)]
    assert len(finished) == 1
    assert finished[0].text == "streamed hello"


async def test_async_abandoned_stream_still_reports_partial_text():
    """A consumer that stops consuming part-way (triggering aclose()) still gets its turn
    reported, via the generator's finally block."""
    from aimu.events import ModelTurnFinished

    seen = []
    client = MockAsyncModelClient(["streamed hello"])
    client.events = seen.append

    stream = await client.chat("hi", stream=True)
    async for _chunk in stream:
        break  # abandon after the first chunk; triggers the async generator's aclose()
    await stream.aclose()

    finished = [e for e in seen if isinstance(e, ModelTurnFinished)]
    assert len(finished) == 1
