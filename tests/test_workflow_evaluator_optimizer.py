"""
Tests for aimu.agents.EvaluatorOptimizer: the Evaluator-Optimizer workflow pattern.

All tests use MockModelClient from helpers (deterministic, no backend needed).
"""

from aimu.agents import Agent, EvaluatorOptimizer, Runner
from aimu.models import StreamChunk, StreamingContentType
from helpers import MockModelClient


# ---------------------------------------------------------------------------
# EvaluatorOptimizer tests
# ---------------------------------------------------------------------------


def test_evaluator_pass_on_first_evaluation():
    """When the evaluator returns PASS immediately, the initial output is returned."""
    gen_client = MockModelClient(["initial draft"])
    eval_client = MockModelClient(["PASS"])

    eo = EvaluatorOptimizer(
        generator=Agent(gen_client, name="writer"),
        evaluator=Agent(eval_client, name="critic"),
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
        generator=Agent(gen_client, name="writer"),
        evaluator=Agent(eval_client, name="critic"),
        max_rounds=5,
    )
    result = eo.run("Explain gravity.")

    assert result == "draft v2"
    assert gen_client._call_count == 2
    assert eval_client._call_count == 2


def test_revision_prompt_includes_previous_output_and_feedback():
    """The revision prompt must carry the prior draft, the feedback, and the task.

    A generator Agent with a system prompt resets its conversation on every run, so it
    cannot see the response it is told to revise unless that response is in the prompt.
    Asserting on the prompt text catches the regression regardless of reset behavior.
    """
    gen_client = MockModelClient(["draft v1", "draft v2"])
    eval_client = MockModelClient(["REVISE: needs more detail", "PASS"])

    eo = EvaluatorOptimizer(
        generator=Agent(gen_client, name="writer"),
        evaluator=Agent(eval_client, name="critic"),
        max_rounds=5,
    )
    eo.run("Explain gravity.")

    user_turns = [m["content"] for m in gen_client.messages if m["role"] == "user"]
    assert len(user_turns) == 2  # initial task + one revision
    revision_prompt = user_turns[1]
    assert "draft v1" in revision_prompt  # the prior draft it is revising
    assert "needs more detail" in revision_prompt  # the evaluator's feedback
    assert "Explain gravity." in revision_prompt  # the original task


def test_evaluator_stops_at_max_rounds():
    """Loop stops at max_rounds even if the evaluator never returns PASS."""
    # max_rounds=3 → 1 initial + 2 revisions max; evaluator called at most 2 times
    gen_client = MockModelClient(["v1", "v2", "v3"])
    eval_client = MockModelClient(["REVISE: bad", "REVISE: still bad"])

    eo = EvaluatorOptimizer(
        generator=Agent(gen_client, name="writer"),
        evaluator=Agent(eval_client, name="critic"),
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
        generator=Agent(gen_client, name="writer"),
        evaluator=Agent(eval_client, name="critic"),
        pass_keyword="APPROVED",
        max_rounds=5,
    )
    result = eo.run("task")
    assert result == "output"
    assert eval_client._call_count == 1


def test_evaluator_streamed_yields_single_generating_chunk():
    """run(stream=True) yields exactly one GENERATING StreamChunk with the final output."""
    gen_client = MockModelClient(["good answer"])
    eval_client = MockModelClient(["PASS"])

    eo = EvaluatorOptimizer(
        generator=Agent(gen_client, name="writer"),
        evaluator=Agent(eval_client, name="critic"),
        max_rounds=3,
    )
    chunks = list(eo.run("task", stream=True))

    assert len(chunks) == 1
    assert isinstance(chunks[0], StreamChunk)
    assert chunks[0].phase == StreamingContentType.GENERATING
    assert chunks[0].content == "good answer"


def test_evaluator_streamed_does_not_yield_intermediate_drafts():
    """Intermediate revision rounds are not streamed; only the final result is."""
    gen_client = MockModelClient(["draft", "final"])
    eval_client = MockModelClient(["REVISE", "PASS"])

    eo = EvaluatorOptimizer(
        generator=Agent(gen_client, name="writer"),
        evaluator=Agent(eval_client, name="critic"),
        max_rounds=5,
    )
    chunks = list(eo.run("task", stream=True))

    generating = [c for c in chunks if c.phase == StreamingContentType.GENERATING]
    assert len(generating) == 1
    assert generating[0].content == "final"


def test_evaluator_stop_when_predicate_accepts():
    """A stop_when predicate over the evaluator text decides acceptance."""
    gen_client = MockModelClient(["draft"])
    eval_client = MockModelClient(['{"ok": true}'])

    eo = EvaluatorOptimizer(
        generator=Agent(gen_client, name="writer"),
        evaluator=Agent(eval_client, name="critic"),
        stop_when=lambda feedback: "true" in feedback,
        max_rounds=3,
    )
    result = eo.run("task")

    assert result == "draft"
    assert eval_client._call_count == 1


def test_evaluator_verdict_schema_passes_first_round():
    """A typed verdict whose `passed` is True accepts the initial output."""
    from dataclasses import dataclass

    @dataclass
    class Verdict:
        passed: bool
        feedback: str = ""

    gen_client = MockModelClient(["draft"])
    eval_client = MockModelClient(['{"passed": true, "feedback": ""}'])
    eval_client.model.supports_structured_output = False

    eo = EvaluatorOptimizer(
        generator=Agent(gen_client, name="writer"),
        evaluator=Agent(eval_client, name="critic"),
        verdict_schema=Verdict,
        max_rounds=3,
    )
    result = eo.run("task")

    assert result == "draft"
    assert eval_client._call_count == 1


def test_evaluator_verdict_schema_revises_then_passes():
    """A failing verdict drives one revision; the next verdict passes."""
    from dataclasses import dataclass

    @dataclass
    class Verdict:
        passed: bool
        feedback: str = ""

    gen_client = MockModelClient(["v1", "v2"])
    eval_client = MockModelClient(['{"passed": false, "feedback": "add detail"}', '{"passed": true}'])
    eval_client.model.supports_structured_output = False

    eo = EvaluatorOptimizer(
        generator=Agent(gen_client, name="writer"),
        evaluator=Agent(eval_client, name="critic"),
        verdict_schema=Verdict,
        max_rounds=5,
    )
    result = eo.run("task")

    assert result == "v2"
    assert gen_client._call_count == 2
    assert eval_client._call_count == 2


def test_evaluator_is_runner_subclass():
    """EvaluatorOptimizer implements the single Runner interface."""
    gen_client = MockModelClient(["output"])
    eval_client = MockModelClient(["PASS"])

    eo = EvaluatorOptimizer(
        generator=Agent(gen_client, name="writer"),
        evaluator=Agent(eval_client, name="critic"),
    )

    assert isinstance(eo, Runner)
