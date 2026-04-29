"""
Tests for aimu.agents.EvaluatorOptimizer: the Evaluator-Optimizer workflow pattern.

All tests use MockModelClient from helpers (deterministic, no backend needed).
"""

from aimu.agents import AgentChunk, EvaluatorOptimizer, Runner, SimpleAgent, Workflow
from aimu.models.base import StreamingContentType
from helpers import MockModelClient


# ---------------------------------------------------------------------------
# EvaluatorOptimizer tests
# ---------------------------------------------------------------------------


def test_evaluator_pass_on_first_evaluation():
    """When the evaluator returns PASS immediately, the initial output is returned."""
    gen_client = MockModelClient(["initial draft"])
    eval_client = MockModelClient(["PASS"])

    eo = EvaluatorOptimizer(
        generator=SimpleAgent(gen_client, name="writer"),
        evaluator=SimpleAgent(eval_client, name="critic"),
        max_rounds=3,
    )
    result = eo.run("Explain gravity.")

    assert result == "initial draft"
    assert gen_client._call_count == 1  # only initial generation
    assert eval_client._call_count == 1


def test_evaluator_revises_once_then_passes():
    """After one revision the evaluator returns PASS; the revised output is returned."""
    gen_client = MockModelClient(["draft v1", "draft v2"])
    eval_client = MockModelClient(["REVISE: needs more detail", "PASS"])

    eo = EvaluatorOptimizer(
        generator=SimpleAgent(gen_client, name="writer"),
        evaluator=SimpleAgent(eval_client, name="critic"),
        max_rounds=5,
    )
    result = eo.run("Explain gravity.")

    assert result == "draft v2"
    assert gen_client._call_count == 2
    assert eval_client._call_count == 2


def test_evaluator_stops_at_max_rounds():
    """Loop stops at max_rounds even if the evaluator never returns PASS."""
    # max_rounds=3 → 1 initial + 2 revisions max; evaluator called at most 2 times
    gen_client = MockModelClient(["v1", "v2", "v3"])
    eval_client = MockModelClient(["REVISE: bad", "REVISE: still bad"])

    eo = EvaluatorOptimizer(
        generator=SimpleAgent(gen_client, name="writer"),
        evaluator=SimpleAgent(eval_client, name="critic"),
        max_rounds=3,
    )
    result = eo.run("Explain gravity.")

    # max_rounds=3 → 1 generate + (3-1)=2 evaluate/revise cycles max
    # After 2 revision cycles without PASS, returns last output
    assert result == "v3"
    assert gen_client._call_count == 3


def test_evaluator_custom_pass_keyword():
    """A custom pass_keyword triggers early exit."""
    gen_client = MockModelClient(["output"])
    eval_client = MockModelClient(["APPROVED"])

    eo = EvaluatorOptimizer(
        generator=SimpleAgent(gen_client, name="writer"),
        evaluator=SimpleAgent(eval_client, name="critic"),
        pass_keyword="APPROVED",
        max_rounds=5,
    )
    result = eo.run("task")
    assert result == "output"
    assert eval_client._call_count == 1


def test_evaluator_streamed_yields_single_generating_chunk():
    """run_streamed() yields exactly one GENERATING AgentChunk with the final output."""
    gen_client = MockModelClient(["good answer"])
    eval_client = MockModelClient(["PASS"])

    eo = EvaluatorOptimizer(
        generator=SimpleAgent(gen_client, name="writer"),
        evaluator=SimpleAgent(eval_client, name="critic"),
        max_rounds=3,
    )
    chunks = list(eo.run_streamed("task"))

    assert len(chunks) == 1
    assert isinstance(chunks[0], AgentChunk)
    assert chunks[0].phase == StreamingContentType.GENERATING
    assert chunks[0].content == "good answer"


def test_evaluator_streamed_does_not_yield_intermediate_drafts():
    """Intermediate revision rounds are not streamed; only the final result is."""
    gen_client = MockModelClient(["draft", "final"])
    eval_client = MockModelClient(["REVISE", "PASS"])

    eo = EvaluatorOptimizer(
        generator=SimpleAgent(gen_client, name="writer"),
        evaluator=SimpleAgent(eval_client, name="critic"),
        max_rounds=5,
    )
    chunks = list(eo.run_streamed("task"))

    generating = [c for c in chunks if c.phase == StreamingContentType.GENERATING]
    assert len(generating) == 1
    assert generating[0].content == "final"


def test_evaluator_is_workflow_subclass():
    """EvaluatorOptimizer must be a Workflow (not an Agent) and also a Runner."""
    gen_client = MockModelClient(["output"])
    eval_client = MockModelClient(["PASS"])

    eo = EvaluatorOptimizer(
        generator=SimpleAgent(gen_client, name="writer"),
        evaluator=SimpleAgent(eval_client, name="critic"),
    )
    from aimu.agents import Agent

    assert isinstance(eo, Workflow)
    assert isinstance(eo, Runner)
    assert not isinstance(eo, Agent)
