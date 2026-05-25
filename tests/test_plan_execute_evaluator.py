"""Tests for aimu.agents.PlanExecuteEvaluator.

All tests use ``MockModelClient`` from helpers — no backend required. The
``Scorer`` is mocked directly so we don't need a judge model for the loop
logic itself.
"""

from __future__ import annotations

from aimu.agents import (
    Agent,
    BaseAgent,
    PlanExecuteEvaluator,
    Runner,
    SkillAgent,
    Workflow,
)
from aimu.agents.workflows.plan_execute_evaluator import _parse_planner_output
from aimu.models import StreamingContentType
from aimu.prompts.tuners.scorers import Scorer
from aimu.skills.manager import SkillManager
from helpers import MockModelClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubScorer(Scorer):
    """Returns ``(score, feedback)`` from a fixed sequence; records every row seen."""

    def __init__(self, results: list[tuple[float, str]]):
        self._results = list(results)
        self._idx = 0
        self.rows_seen: list = []

    def score(self, row) -> tuple[float, str]:
        if self._idx >= len(self._results):
            raise AssertionError(f"_StubScorer exhausted after {len(self._results)} calls")
        result = self._results[self._idx]
        self._idx += 1
        self.rows_seen.append(row)
        return result


def _make_workflow(
    *,
    planner_responses: list[str],
    executor_responses: list[str],
    scorer_results: list[tuple[float, str]],
    criteria: str | None = None,
    max_rounds: int = 3,
    pass_threshold: float = 0.7,
    pass_keyword: str | None = None,
) -> tuple[PlanExecuteEvaluator, MockModelClient, MockModelClient, _StubScorer]:
    planner_client = MockModelClient(planner_responses)
    executor_client = MockModelClient(executor_responses)
    scorer = _StubScorer(scorer_results)

    planner = SkillAgent(
        planner_client,
        "Plan tasks carefully.",
        name="planner",
        skill_manager=SkillManager(skill_dirs=["/nonexistent-so-no-skills"]),
    )
    executor = Agent(executor_client, "Execute the plan.", name="executor")

    wf = PlanExecuteEvaluator(
        planner=planner,
        executor=executor,
        scorer=scorer,
        criteria=criteria,
        max_rounds=max_rounds,
        pass_threshold=pass_threshold,
        pass_keyword=pass_keyword,
    )
    return wf, planner_client, executor_client, scorer


# ---------------------------------------------------------------------------
# Hierarchy and shape
# ---------------------------------------------------------------------------


def test_is_workflow_subclass_not_baseagent():
    wf, _, _, _ = _make_workflow(
        planner_responses=["plan"],
        executor_responses=["out"],
        scorer_results=[(0.9, "pass")],
    )
    assert isinstance(wf, Workflow)
    assert isinstance(wf, Runner)
    assert not isinstance(wf, BaseAgent)


def test_messages_merges_planner_and_executor():
    wf, _, _, _ = _make_workflow(
        planner_responses=["plan"],
        executor_responses=["out"],
        scorer_results=[(0.9, "pass")],
    )
    wf.run("hi")
    msgs = wf.messages
    assert wf.planner.name in msgs
    assert wf.executor.name in msgs


# ---------------------------------------------------------------------------
# Pass / fail loop behaviour
# ---------------------------------------------------------------------------


def test_first_round_pass_returns_executor_output():
    wf, planner_c, executor_c, scorer = _make_workflow(
        planner_responses=["step 1; step 2"],
        executor_responses=["final answer"],
        scorer_results=[(0.9, "looks good")],
        criteria="answer the question",
    )
    result = wf.run("what is 2+2?")
    assert result == "final answer"
    assert planner_c._call_count == 1
    assert executor_c._call_count == 1
    assert len(scorer.rows_seen) == 1


def test_pass_on_round_two_after_replan():
    """Round 1 fails; replan in round 2 succeeds. Second planner prompt contains round 1 feedback."""
    wf, planner_c, executor_c, scorer = _make_workflow(
        planner_responses=["plan v1", "plan v2"],
        executor_responses=["bad output", "good output"],
        scorer_results=[(0.3, "needs more detail"), (0.85, "looks good")],
        criteria="be detailed",
    )
    result = wf.run("write something")
    assert result == "good output"
    assert planner_c._call_count == 2
    assert executor_c._call_count == 2

    # SkillAgent resets its client between runs, so only the latest user message
    # survives in planner_c.messages — and that's round 2's, which must contain
    # round 1's feedback.
    planner_user_messages = [m["content"] for m in planner_c.messages if m["role"] == "user"]
    assert len(planner_user_messages) == 1
    round_two_prompt = planner_user_messages[0]
    assert "needs more detail" in round_two_prompt
    assert "plan v1" in round_two_prompt
    assert "bad output" in round_two_prompt


def test_max_rounds_exhaustion_returns_best_attempt():
    """Three rounds, scores [0.4, 0.6, 0.5] — round 2 wins on score."""
    wf, _, _, _ = _make_workflow(
        planner_responses=["p1", "p2", "p3"],
        executor_responses=["o1", "o2", "o3"],
        scorer_results=[(0.4, "fb1"), (0.6, "fb2"), (0.5, "fb3")],
        max_rounds=3,
        pass_threshold=0.7,
        criteria="anything",
    )
    result = wf.run("task")
    assert result == "o2"  # highest-scoring of the three


def test_pass_keyword_marks_pass_even_below_threshold():
    """A score below pass_threshold but feedback containing pass_keyword passes."""
    wf, _, _, scorer = _make_workflow(
        planner_responses=["plan"],
        executor_responses=["out"],
        scorer_results=[(0.5, "OK: APPROVED")],
        pass_threshold=0.8,
        pass_keyword="APPROVED",
        criteria="anything",
    )
    result = wf.run("task")
    assert result == "out"
    # Should have stopped after the first score even though 0.5 < 0.8
    assert len(scorer.rows_seen) == 1


# ---------------------------------------------------------------------------
# Criteria modes
# ---------------------------------------------------------------------------


def test_user_supplied_criteria_passed_to_planner_and_scorer():
    """`criteria` at construction is in every planner prompt and is `reference` on every row."""
    criteria = "the answer must be one sentence"
    wf, planner_c, _, scorer = _make_workflow(
        planner_responses=["plan v1", "plan v2"],
        executor_responses=["out v1", "out v2"],
        scorer_results=[(0.2, "too long"), (0.9, "good")],
        criteria=criteria,
    )
    wf.run("task")

    # The latest round's planner prompt contains the criteria (SkillAgent resets
    # between runs, so the round-1 prompt isn't preserved in client.messages —
    # but we know it was passed because the workflow built the same way per round).
    planner_user_messages = [m["content"] for m in planner_c.messages if m["role"] == "user"]
    assert any(criteria in msg for msg in planner_user_messages)

    # Scorer received the criteria as `reference` on every call (both rounds).
    assert len(scorer.rows_seen) == 2
    assert all(row.reference == criteria for row in scorer.rows_seen)


def test_planner_invented_criteria_extracted_and_passed_to_scorer():
    """`criteria=None` ⇒ planner invents; workflow parses and uses parsed criteria as `reference`."""
    planner_response = (
        "## Evaluation criteria\n"
        "Output must be exactly one sentence.\n\n"
        "## Plan\n"
        "1. Read the task.\n"
        "2. Produce a one-sentence answer."
    )
    wf, _, executor_c, scorer = _make_workflow(
        planner_responses=[planner_response],
        executor_responses=["A short answer."],
        scorer_results=[(0.9, "good")],
        criteria=None,
    )
    wf.run("explain")

    # Scorer's `reference` is the parsed criteria, not the literal planner response.
    assert scorer.rows_seen[0].reference == "Output must be exactly one sentence."

    # Executor receives the plan portion only (not the criteria).
    executor_user_messages = [m["content"] for m in executor_c.messages if m["role"] == "user"]
    assert "Read the task" in executor_user_messages[0]
    assert "## Plan" not in executor_user_messages[0]
    assert "Evaluation criteria" not in executor_user_messages[0]


def test_malformed_planner_output_falls_back_to_default_criteria():
    """When the planner ignores the format, the workflow treats the whole response as the plan."""
    plain_text = "just do something, I don't know"
    wf, _, _, scorer = _make_workflow(
        planner_responses=[plain_text],
        executor_responses=["did something"],
        scorer_results=[(0.9, "ok")],
        criteria=None,
    )
    result = wf.run("task")
    assert result == "did something"
    # Falls back to the default criteria sentence.
    assert "Successfully complete the task" in scorer.rows_seen[0].reference


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_stream_yields_single_generating_chunk_with_final_output():
    wf, _, _, _ = _make_workflow(
        planner_responses=["plan"],
        executor_responses=["the answer"],
        scorer_results=[(0.9, "pass")],
        criteria="answer the question",
    )
    chunks = list(wf.run_streamed("what?"))
    assert len(chunks) == 1
    assert chunks[0].phase == StreamingContentType.GENERATING
    assert chunks[0].content == "the answer"
    assert chunks[0].agent == wf.name


# ---------------------------------------------------------------------------
# Attempt introspection
# ---------------------------------------------------------------------------


def test_last_attempts_records_every_round():
    wf, _, _, _ = _make_workflow(
        planner_responses=["p1", "p2"],
        executor_responses=["o1", "o2"],
        scorer_results=[(0.3, "fb1"), (0.9, "fb2")],
        criteria="anything",
    )
    wf.run("task")
    attempts = wf.last_attempts
    assert len(attempts) == 2
    assert [a["score"] for a in attempts] == [0.3, 0.9]
    assert [a["output"] for a in attempts] == ["o1", "o2"]
    assert all("plan" in a and "criteria" in a and "feedback" in a for a in attempts)


# ---------------------------------------------------------------------------
# .of() factory
# ---------------------------------------------------------------------------


def test_of_factory_builds_runnable_workflow():
    """`PlanExecuteEvaluator.of(client)` builds a workflow with defaults that runs end-to-end."""
    from aimu.prompts.tuners.scorers import LLMJudgeScorer

    # Single client used for planner, executor, AND (default) judge.
    # Sequence accounts for: planner round 1, executor round 1, judge round 1 (returns "8"
    # which the LLMJudgeScorer parses to 0.8 → pass).
    client = MockModelClient(["plan it", "did it", "8"])

    wf = PlanExecuteEvaluator.of(client, criteria="answer the task")
    assert isinstance(wf, PlanExecuteEvaluator)
    assert isinstance(wf.scorer, LLMJudgeScorer)
    assert isinstance(wf.planner, SkillAgent)
    assert isinstance(wf.executor, Agent)

    result = wf.run("hi")
    assert result == "did it"


def test_of_factory_propagates_executor_tools():
    """`executor_tools=` arg lands on the executor agent."""
    from aimu.tools import tool

    @tool
    def my_tool(x: str) -> str:
        """A tool."""
        return x

    client = MockModelClient(["plan", "out", "9"])
    wf = PlanExecuteEvaluator.of(client, executor_tools=[my_tool], criteria="X")
    assert wf.executor.tools == [my_tool]


def test_of_factory_uses_separate_judge_client_when_provided():
    """`judge_client=` builds the LLMJudgeScorer over the judge client, not the main one."""
    main = MockModelClient(["plan", "out"])
    judge = MockModelClient(["9"])
    wf = PlanExecuteEvaluator.of(main, judge_client=judge, criteria="X")
    # The scorer's underlying client should be the judge, not main.
    assert wf.scorer.judge_client is judge

    result = wf.run("task")
    assert result == "out"
    assert main._call_count == 2  # planner + executor
    assert judge._call_count == 1  # scorer


# ---------------------------------------------------------------------------
# Parser unit test
# ---------------------------------------------------------------------------


def test_parser_splits_on_plan_header():
    text = "## Evaluation criteria\nClear answer.\n\n## Plan\n1. step\n2. step\n"
    criteria, plan = _parse_planner_output(text)
    assert criteria == "Clear answer."
    assert plan.startswith("1. step")


def test_parser_falls_back_when_no_plan_header():
    criteria, plan = _parse_planner_output("just some text")
    assert "Successfully complete" in criteria
    assert plan == "just some text"


def test_parser_handles_missing_criteria_header():
    """Plan header present, criteria header missing — still split correctly."""
    text = "some preamble\n\n## Plan\n1. do it"
    criteria, plan = _parse_planner_output(text)
    assert plan == "1. do it"
    # Criteria falls back to "some preamble" since there's no `## Evaluation criteria` header.
    assert criteria == "some preamble"
