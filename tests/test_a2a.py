"""Tests for the optional A2A interop layer (aimu.agents.a2a).

Skipped entirely unless the ``a2a`` extra is installed. Mixes mock-only unit tests (card
construction, text extraction, app routes via Starlette's TestClient) with a real HTTP
round-trip (uvicorn in a background thread) proving RemoteAgent composes like any Runner.
"""

from __future__ import annotations

import socket
import threading
import time
from contextlib import contextmanager

import pytest

pytest.importorskip("a2a", reason="requires the 'a2a' extra")

from aimu.agents import Agent, Chain, OrchestratorAgent, Runner  # noqa: E402
from aimu.agents.a2a import A2AConnectionError, RemoteAgent, build_a2a_app  # noqa: E402
from aimu.agents.a2a import _card  # noqa: E402


class _EchoRunner(Runner):
    """Deterministic runner used for round-trip tests (no model backend)."""

    def __init__(self, name="echo", system_message="Echo the input back."):
        self.name = name
        self.system_message = system_message

    def run(self, task, generate_kwargs=None, stream=False, images=None):
        return f"echo: {task}"

    @property
    def messages(self):
        return {self.name: []}


@contextmanager
def _serve(app):
    """Run an ASGI app under uvicorn on a free port in a background thread."""
    import uvicorn

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(200):
            if server.started:
                break
            time.sleep(0.02)
        assert server.started, "uvicorn server did not start in time"
        yield f"http://127.0.0.1:{port}/"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


# --- Pure helper unit tests (no network) ---------------------------------------------------


def test_card_helpers_name_and_description():
    runner = _EchoRunner(name="my agent", system_message="First line.\nSecond line.")
    assert _card.runner_name(runner) == "my_agent"
    assert _card.runner_description(runner, "my_agent") == "First line."


def test_card_helpers_generic_description_without_system_message():
    class _Bare(Runner):
        name = "bare"

        def run(self, *a, **k):
            return ""

        @property
        def messages(self):
            return {}

    assert _card.runner_description(_Bare(), "bare") == "An AIMU runner exposed over A2A: bare."


def test_build_agent_card_defaults():
    card = _card.build_agent_card(_EchoRunner(name="greeter"), url="http://x/")
    assert card.name == "greeter"
    assert card.url == "http://x/"
    assert [s.name for s in card.skills] == ["greeter"]
    assert card.default_input_modes == ["text/plain"]


def test_result_to_text_from_message():
    from a2a.types import Message, Part, Role, TextPart

    msg = Message(message_id="1", role=Role.agent, parts=[Part(root=TextPart(text="hello "))])
    msg.parts.append(Part(root=TextPart(text="world")))
    assert _card.result_to_text(msg) == "hello world"


def test_server_app_serves_agent_card():
    from starlette.testclient import TestClient

    agent = Agent_for_card()
    app = build_a2a_app(agent, url="http://testserver/")
    client = TestClient(app)
    resp = client.get("/.well-known/agent-card.json")
    assert resp.status_code == 200
    card = resp.json()
    assert card["name"] == "greeter"
    assert card["skills"][0]["name"] == "greeter"


def Agent_for_card():
    from helpers import MockModelClient

    return Agent(MockModelClient(["hi"]), system_message="Be helpful.", name="greeter")


# --- Real HTTP round-trip -----------------------------------------------------------------


def test_remote_agent_round_trip_and_composition():
    app = build_a2a_app(_EchoRunner(name="greeter", system_message="Greet the user."), url="http://placeholder/")
    with _serve(app) as url:
        remote = RemoteAgent.connect(url)
        try:
            assert remote.name == "greeter"
            assert remote.run("hi there") == "echo: hi there"
            assert remote.messages["greeter"][-1]["content"] == "echo: hi there"

            # Composes as a tool (description sourced from the remote card).
            fn = remote.as_tool()
            assert fn.__name__ == "greeter"
            assert fn.__tool_spec__["function"]["description"] == "Greet the user."
            assert fn("again") == "echo: again"

            # Composes as a workflow step (a Chain that ends with the remote agent).
            chain = Chain(agents=[remote])
            assert chain.run("chained") == "echo: chained"

            # Composes as an orchestrator worker.
            from helpers import MockModelClient

            orch = OrchestratorAgent.assemble(MockModelClient(["done"]), "Use the worker.", workers=[remote])
            assert "greeter" in {t.__name__ for t in orch._orchestrator.tools}
        finally:
            remote.close()


def test_remote_agent_streamed_yields_single_generating_chunk():
    from aimu.models import StreamingContentType

    app = build_a2a_app(_EchoRunner(name="greeter"), url="http://placeholder/")
    with _serve(app) as url:
        remote = RemoteAgent.connect(url)
        try:
            chunks = list(remote.run("stream me", stream=True))
            generating = [c for c in chunks if c.phase == StreamingContentType.GENERATING]
            assert [c.content for c in generating] == ["echo: stream me"]
            assert chunks[-1].phase == StreamingContentType.DONE
        finally:
            remote.close()


def test_connect_to_dead_url_raises_a2a_connection_error():
    with pytest.raises(A2AConnectionError):
        RemoteAgent.connect("http://127.0.0.1:1/", agent_card_path="/.well-known/agent-card.json")
