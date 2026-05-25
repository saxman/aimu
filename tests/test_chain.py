"""
Tests for aimu.agents.Chain: the sequential chaining workflow pattern.

All tests use MockModelClient from helpers (deterministic, no backend needed).
"""

from aimu.agents import Agent, Chain, Runner
from aimu.models import StreamChunk
from helpers import MockModelClient


# ---------------------------------------------------------------------------
# Chain tests
# ---------------------------------------------------------------------------


def test_chain_chains_output_to_next_input():
    """Output of step 0 becomes the task for step 1."""
    client_a = MockModelClient(["step A output"])
    client_b = MockModelClient(["step B output"])
    chain = Chain(agents=[Agent(client_a, name="a"), Agent(client_b, name="b")])
    result = chain.run("initial task")

    assert result == "step B output"
    user_msgs_b = [m["content"] for m in client_b.messages if m["role"] == "user"]
    assert user_msgs_b[0] == "step A output"


def test_chain_run_single_agent():
    client = MockModelClient(["only answer"])
    chain = Chain(agents=[Agent(client, name="solo")])
    assert chain.run("task") == "only answer"


def test_chain_streamed_yields_stream_chunks():
    """Chain.run(stream=True) yields StreamChunk objects; chain step lives in `iteration`."""
    client_a = MockModelClient(["part one"])
    client_b = MockModelClient(["part two"])
    chain = Chain(agents=[Agent(client_a, name="a"), Agent(client_b, name="b")])
    chunks = list(chain.run("go", stream=True))

    assert all(isinstance(c, StreamChunk) for c in chunks)
    steps = {c.iteration for c in chunks}
    assert steps == {0, 1}


def test_chain_streamed_step_tags():
    client_a = MockModelClient(["result a"])
    client_b = MockModelClient(["result b"])
    chain = Chain(agents=[Agent(client_a, name="alpha"), Agent(client_b, name="beta")])
    chunks = list(chain.run("start", stream=True))

    step0 = [c for c in chunks if c.iteration == 0]
    step1 = [c for c in chunks if c.iteration == 1]
    assert all(c.agent == "alpha" for c in step0)
    assert all(c.agent == "beta" for c in step1)


def test_chain_from_config():
    client = MockModelClient(["first output", "second output"])
    configs = [{"name": "first"}, {"name": "second"}]
    chain = Chain.from_config(configs, client)

    assert len(chain.agents) == 2
    assert chain.agents[0].name == "first"
    assert chain.agents[1].name == "second"
    assert chain.run("go") == "second output"


def test_chain_from_config_resets_messages_between_steps():
    """from_config shared-client mode clears messages before each step."""
    client = MockModelClient(["step A output", "step B output"])
    configs = [{"name": "a"}, {"name": "b"}]
    chain = Chain.from_config(configs, client)
    chain.run("start")

    user_msgs = [m["content"] for m in client.messages if m["role"] == "user"]
    assert user_msgs == ["step A output"]  # step B received step A's output as task


def test_chain_from_config_sets_system_message_per_step():
    """from_config applies each step's system_message at run time."""
    client = MockModelClient(["out a", "out b"])
    configs = [
        {"name": "a", "system_message": "You are A."},
        {"name": "b", "system_message": "You are B."},
    ]
    chain = Chain.from_config(configs, client)
    chain.run("go")

    assert client.system_message == "You are B."


def test_chain_is_runner_subclass():
    """Chain implements the single Runner interface."""
    client = MockModelClient(["hi"])
    chain = Chain(agents=[Agent(client)])
    assert isinstance(chain, Runner)


def test_chain_from_client_builds_runnable_chain():
    """Chain.from_client(client, prompts) builds and runs end-to-end."""
    client = MockModelClient(["first", "second"])
    chain = Chain.from_client(client, ["Step 1 prompt.", "Step 2 prompt."])
    assert len(chain.agents) == 2
    result = chain.run("task")
    assert result == "second"
