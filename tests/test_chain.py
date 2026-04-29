"""
Tests for aimu.agents.Chain: the sequential chaining workflow pattern.

All tests use MockModelClient from helpers (deterministic, no backend needed).
"""

from aimu.agents import Agent, Chain, ChainChunk, Runner, SimpleAgent, Workflow
from aimu.models.base import StreamingContentType
from helpers import MockModelClient


# ---------------------------------------------------------------------------
# Chain tests
# ---------------------------------------------------------------------------


def test_chain_chains_output_to_next_input():
    """Output of step 0 becomes the task for step 1."""
    client_a = MockModelClient(["step A output"])
    client_b = MockModelClient(["step B output"])
    chain = Chain(agents=[SimpleAgent(client_a, name="a"), SimpleAgent(client_b, name="b")])
    result = chain.run("initial task")

    assert result == "step B output"
    user_msgs_b = [m["content"] for m in client_b.messages if m["role"] == "user"]
    assert user_msgs_b[0] == "step A output"


def test_chain_run_single_agent():
    client = MockModelClient(["only answer"])
    chain = Chain(agents=[SimpleAgent(client, name="solo")])
    assert chain.run("task") == "only answer"


def test_chain_streamed_yields_chain_chunks():
    client_a = MockModelClient(["part one"])
    client_b = MockModelClient(["part two"])
    chain = Chain(agents=[SimpleAgent(client_a, name="a"), SimpleAgent(client_b, name="b")])
    chunks = list(chain.run_streamed("go"))

    assert all(isinstance(c, ChainChunk) for c in chunks)
    steps = {c.step for c in chunks}
    assert steps == {0, 1}


def test_chain_streamed_step_tags():
    client_a = MockModelClient(["result a"])
    client_b = MockModelClient(["result b"])
    chain = Chain(agents=[SimpleAgent(client_a, name="alpha"), SimpleAgent(client_b, name="beta")])
    chunks = list(chain.run_streamed("start"))

    step0 = [c for c in chunks if c.step == 0]
    step1 = [c for c in chunks if c.step == 1]
    assert all(c.agent_name == "alpha" for c in step0)
    assert all(c.agent_name == "beta" for c in step1)


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


def test_chain_is_workflow_subclass():
    """Chain must be a Workflow (not an Agent) and also a Runner."""
    client = MockModelClient(["hi"])
    chain = Chain(agents=[SimpleAgent(client)])
    assert isinstance(chain, Workflow)
    assert isinstance(chain, Runner)
    assert not isinstance(chain, Agent)
