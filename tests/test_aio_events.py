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


# ---------------------------------------------------------------------------
# Wrapper double-emit regression tests (async mirror of tests/test_events.py).
# ---------------------------------------------------------------------------


async def test_async_agentic_view_single_turn_chat_emits_exactly_one_pair():
    from aimu.aio.agent import Agent
    from aimu.events import ModelTurnFinished, ModelTurnStarted

    client = MockAsyncModelClient(["final answer"])
    client.model.supports_tools = False
    view = Agent(client).as_model_client()
    seen = []
    view.events = seen.append

    assert await view.chat("question") == "final answer"

    assert sum(isinstance(e, ModelTurnStarted) for e in seen) == 1
    assert sum(isinstance(e, ModelTurnFinished) for e in seen) == 1


async def test_async_agentic_view_tool_loop_chat_emits_one_pair_per_real_turn():
    from aimu.aio.agent import Agent
    from aimu.events import ModelTurnFinished, ModelTurnStarted

    client = MockAsyncModelClient(["tool", "after tool"])
    view = Agent(client, max_iterations=5).as_model_client()
    seen = []
    view.events = seen.append

    assert await view.chat("do something with tools") == "after tool"
    assert client._call_count == 2  # sanity: two real requests did happen

    assert sum(isinstance(e, ModelTurnStarted) for e in seen) == 2
    assert sum(isinstance(e, ModelTurnFinished) for e in seen) == 2


async def test_async_agentic_view_generate_emits_exactly_one_pair():
    from aimu.aio.agent import Agent
    from aimu.events import ModelTurnFinished, ModelTurnStarted

    client = MockAsyncModelClient(["generated"])
    client.model.supports_tools = False
    view = Agent(client).as_model_client()
    seen = []
    view.events = seen.append

    assert await view.generate("prompt") == "generated"
    assert client._call_count == 1

    assert sum(isinstance(e, ModelTurnStarted) for e in seen) == 1
    assert sum(isinstance(e, ModelTurnFinished) for e in seen) == 1


async def test_async_agentic_view_streamed_chat_emits_exactly_one_pair():
    from aimu.aio.agent import Agent
    from aimu.events import ModelTurnFinished, ModelTurnStarted

    client = MockAsyncModelClient(["stream result"])
    client.model.supports_tools = False
    view = Agent(client).as_model_client()
    seen = []
    view.events = seen.append

    stream = await view.chat("task", stream=True)
    chunks = [c async for c in stream]
    assert chunks

    assert sum(isinstance(e, ModelTurnStarted) for e in seen) == 1
    assert sum(isinstance(e, ModelTurnFinished) for e in seen) == 1


async def test_async_fallback_client_chat_emits_exactly_one_pair():
    """AsyncFallbackClient shares the same shape as the sync FallbackClient: it fully
    overrides chat()/generate() and delegates to the winning inner client's own public
    chat(), so it never had the wrapper double-emit defect. Confirmed here so a future
    refactor can't silently reintroduce it."""
    from aimu.aio.fallback import AsyncFallbackClient
    from aimu.events import ModelTurnFinished, ModelTurnStarted

    primary = MockAsyncModelClient(["ok"])
    fc = AsyncFallbackClient([primary])
    seen = []
    fc.events = seen.append

    assert await fc.chat("hi") == "ok"

    assert sum(isinstance(e, ModelTurnStarted) for e in seen) == 1
    assert sum(isinstance(e, ModelTurnFinished) for e in seen) == 1
    assert primary.events is not None


async def test_async_in_process_client_chat_emits_exactly_one_pair():
    """_AsyncInProcessClient (AsyncHuggingFaceClient/AsyncLlamaCppClient's shared base)
    wraps a sync client but calls its private _chat()/_generate() directly (via
    asyncio.to_thread), never the sync client's own public chat()/generate() -- so only
    the async wrapper's inherited turn-tracking fires, exactly once per real request. No
    fix was needed here; this test documents and locks in that it was already correct."""
    from aimu.aio.providers._inprocess import _AsyncInProcessClient
    from aimu.events import ModelTurnFinished, ModelTurnStarted
    from tests.helpers import MockModelClient

    class _Wrapper(_AsyncInProcessClient):
        _SYNC_CLASS = MockModelClient

    sync_client = MockModelClient(["hi back"])
    sync_client.model.supports_tools = False
    wrapper = _Wrapper(sync_client)
    seen = []
    wrapper.events = seen.append

    assert await wrapper.chat("hello") == "hi back"

    assert sum(isinstance(e, ModelTurnStarted) for e in seen) == 1
    assert sum(isinstance(e, ModelTurnFinished) for e in seen) == 1
