from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Optional, Union

from aimu.agents.agent import Agent
from aimu.agents.base import MessageHistory, Runner
from aimu.agents.skill_agent import SkillAgent
from aimu.models.base import BaseModelClient, StreamChunk, StreamingContentType
from aimu.prompts.tuners.scorers import LLMJudgeScorer, Scorer
from aimu.skills.manager import SkillManager

logger = logging.getLogger(__name__)


_DEFAULT_FALLBACK_CRITERIA = "Successfully complete the task as stated."

_DEFAULT_PLANNER_SYSTEM = (
    "You are a planner. Given a task, produce a numbered list of concrete "
    "steps that an executor with tool access will follow. Be specific. Avoid "
    "vague steps. The plan is your only handoff to the executor."
)

_DEFAULT_EXECUTOR_SYSTEM = (
    "You execute the plan you receive. Use available tools. Return only the "
    "final artifact or answer, not a summary of what you did."
)

_PLAN_HEADER_RE = re.compile(r"^\s*##\s*Plan\s*$", re.MULTILINE | re.IGNORECASE)
_CRITERIA_HEADER_RE = re.compile(r"^\s*##\s*Evaluation criteria\s*$", re.MULTILINE | re.IGNORECASE)


def _parse_planner_output(text: str) -> tuple[str, str]:
    """Split planner output into (criteria, plan).

    Expects the planner to emit:

        ## Evaluation criteria
        <criteria text>

        ## Plan
        <plan text>

    Falls back to ``(_DEFAULT_FALLBACK_CRITERIA, text)`` if no ``## Plan``
    header is found.
    """
    plan_match = _PLAN_HEADER_RE.search(text)
    if not plan_match:
        logger.debug("Planner output had no '## Plan' header; using full text as plan.")
        return _DEFAULT_FALLBACK_CRITERIA, text.strip()

    before, after = text[: plan_match.start()], text[plan_match.end() :]
    plan_text = after.strip()

    criteria_match = _CRITERIA_HEADER_RE.search(before)
    if criteria_match:
        criteria_text = before[criteria_match.end() :].strip()
    else:
        criteria_text = before.strip() or _DEFAULT_FALLBACK_CRITERIA

    return criteria_text or _DEFAULT_FALLBACK_CRITERIA, plan_text


def _truncate(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


@dataclass
class PlanExecuteEvaluator(Runner):
    """Plan → execute → evaluate → replan-on-fail loop.

    A planner agent produces a plan for the task; an executor agent runs the
    plan with its tools; a pluggable :class:`Scorer` judges the output. On
    failure, the planner is invoked again with the prior round's feedback to
    produce a *new* plan (not a revision). The loop bounds at ``max_rounds``
    and on exhaustion returns the highest-scoring attempt.

    Two criteria modes:

    - **User-supplied** (``criteria="..."`` at construction): the criteria is
      passed to the planner's prompt every round, and to the scorer as the
      row's ``reference`` field.
    - **Planner-invented** (``criteria=None``): the planner is asked to emit
      ``## Evaluation criteria`` + ``## Plan`` sections; the workflow parses
      both. The parsed criteria is passed to the scorer as ``reference``.

    Usage::

        wf = PlanExecuteEvaluator.from_client(
            client=aimu.client("anthropic:claude-sonnet-4-6"),
            judge_client=aimu.client("openai:gpt-4o"),
            executor_tools=builtin.web + builtin.fs,
            criteria="The summary cites 3+ sources and is under 200 words.",
            max_rounds=3,
        )
        result = wf.run("Summarise the latest news on quantum computing.")
    """

    planner: SkillAgent
    executor: Agent
    scorer: Scorer
    criteria: Optional[str] = None
    name: str = "plan_execute_evaluator"
    max_rounds: int = 3
    pass_threshold: float = 0.7
    pass_keyword: Optional[str] = None
    _last_attempts: list = field(default_factory=list, init=False, repr=False)

    @property
    def messages(self) -> MessageHistory:
        result: MessageHistory = {}
        result.update(self.planner.messages)
        result.update(self.executor.messages)
        return result

    def run(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
    ) -> Union[str, Iterator[StreamChunk]]:
        """Run the plan-execute-evaluate loop. ``images`` are forwarded to the executor each round."""
        if stream:
            return self._run_streamed(task, generate_kwargs, images=images)

        attempts: list[dict] = []
        prior_attempt_block = ""

        for round_num in range(self.max_rounds):
            planner_prompt = self._build_planner_prompt(task, prior_attempt_block)
            planner_response = self.planner.run(planner_prompt, generate_kwargs=generate_kwargs)

            round_criteria, plan_text = self._extract_criteria_and_plan(planner_response)
            output = self.executor.run(plan_text, generate_kwargs=generate_kwargs, images=images)

            score, feedback = self._score(task, output, round_criteria)
            attempt = {
                "round": round_num,
                "plan": plan_text,
                "criteria": round_criteria,
                "output": output,
                "score": score,
                "feedback": feedback,
            }
            attempts.append(attempt)
            logger.debug("PlanExecuteEvaluator '%s' round %d: score=%.2f", self.name, round_num, score)

            if self._is_pass(score, feedback):
                logger.debug("PlanExecuteEvaluator '%s' passed on round %d.", self.name, round_num)
                self._last_attempts = attempts
                return output

            prior_attempt_block = self._build_prior_attempt_block(attempt)

        self._last_attempts = attempts
        best = max(attempts, key=lambda a: a["score"])
        logger.debug(
            "PlanExecuteEvaluator '%s' exhausted %d rounds; returning best score=%.2f from round %d.",
            self.name,
            self.max_rounds,
            best["score"],
            best["round"],
        )
        return best["output"]

    def _run_streamed(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        images: Optional[list] = None,
    ) -> Iterator[StreamChunk]:
        output = self.run(task, generate_kwargs=generate_kwargs, images=images)
        yield StreamChunk(StreamingContentType.GENERATING, output, agent=self.name, iteration=0)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _build_planner_prompt(self, task: str, prior_attempt_block: str) -> str:
        parts = [f"Task: {task}"]
        if self.criteria is not None:
            parts.append(f"\nEvaluation criteria (your plan must satisfy these):\n{self.criteria}")
        if prior_attempt_block:
            parts.append("\n" + prior_attempt_block)
        if self.criteria is not None:
            parts.append("\nProduce a plan as a numbered list of concrete steps that the executor will follow.")
        else:
            parts.append(
                "\nReturn your response in two sections:\n\n"
                "## Evaluation criteria\n"
                "One paragraph describing what a successful completion looks like.\n\n"
                "## Plan\n"
                "A numbered list of concrete steps the executor will follow."
            )
        return "\n".join(parts)

    def _build_prior_attempt_block(self, attempt: dict) -> str:
        return (
            f"Your previous plan was executed and the evaluator scored it {attempt['score']:.2f}/1.0:\n\n"
            f"Plan:\n{attempt['plan']}\n\n"
            f"Output:\n{_truncate(attempt['output'])}\n\n"
            f"Evaluator feedback:\n{attempt['feedback']}\n\n"
            f"Produce a *new* plan that addresses the feedback."
        )

    def _extract_criteria_and_plan(self, planner_response: str) -> tuple[str, str]:
        if self.criteria is not None:
            return self.criteria, planner_response.strip()
        return _parse_planner_output(planner_response)

    def _score(self, task: str, output: str, criteria: str) -> tuple[float, str]:
        row = SimpleNamespace(content=task, output=output, reference=criteria)
        return self.scorer.score(row)

    def _is_pass(self, score: float, feedback: str) -> bool:
        if score >= self.pass_threshold:
            return True
        if self.pass_keyword and self.pass_keyword in feedback:
            return True
        return False

    @property
    def last_attempts(self) -> list[dict]:
        """Per-round attempts from the most recent ``run()``.

        Each entry has ``round``, ``plan``, ``criteria``, ``output``, ``score``,
        ``feedback``. Useful for introspection or post-hoc analysis.
        """
        return self._last_attempts

    # ------------------------------------------------------------------ #
    # Convenience factory                                                  #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_client(
        cls,
        client: BaseModelClient,
        *,
        judge_client: Optional[BaseModelClient] = None,
        criteria: Optional[str] = None,
        executor_tools: Optional[list[Callable]] = None,
        skill_manager: Optional[SkillManager] = None,
        planner_system_message: Optional[str] = None,
        executor_system_message: Optional[str] = None,
        max_rounds: int = 3,
        pass_threshold: float = 0.7,
        pass_keyword: Optional[str] = None,
        name: str = "plan_execute_evaluator",
    ) -> PlanExecuteEvaluator:
        """Build a PlanExecuteEvaluator from a single client and the common knobs.

        - ``planner`` is a :class:`SkillAgent` over ``client``; pass
          ``skill_manager=`` to use task-type planning skills from disk.
        - ``executor`` is an :class:`Agent` over ``client`` with the given
          ``executor_tools``.
        - ``scorer`` is an :class:`LLMJudgeScorer` over ``judge_client``
          (defaults to ``client`` if not provided; a stronger separate model is
          recommended). Its ``criteria`` is the workflow's ``criteria`` if
          supplied, otherwise a generic fallback.

        For full control (e.g. a custom :class:`Scorer`), construct the
        dataclass directly.
        """
        planner = SkillAgent(
            client,
            planner_system_message or _DEFAULT_PLANNER_SYSTEM,
            name="planner",
            skill_manager=skill_manager or SkillManager(),
            reset_messages_on_run=True,
        )
        executor = Agent(
            client,
            executor_system_message or _DEFAULT_EXECUTOR_SYSTEM,
            name="executor",
            tools=list(executor_tools) if executor_tools else [],
            reset_messages_on_run=True,
        )
        scorer = LLMJudgeScorer(
            judge_client or client,
            criteria=criteria or _DEFAULT_FALLBACK_CRITERIA,
        )
        return cls(
            planner=planner,
            executor=executor,
            scorer=scorer,
            criteria=criteria,
            name=name,
            max_rounds=max_rounds,
            pass_threshold=pass_threshold,
            pass_keyword=pass_keyword,
        )
