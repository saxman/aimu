"""Async PlanExecuteEvaluator workflow (plan → execute → score → replan)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, AsyncIterator, Callable, Optional, Union

from aimu.agents.base import MessageHistory
from aimu.agents.workflows.plan_execute_evaluator import (
    _DEFAULT_EXECUTOR_SYSTEM,
    _DEFAULT_FALLBACK_CRITERIA,
    _DEFAULT_PLANNER_SYSTEM,
    _parse_planner_output,
    _truncate,
)
from aimu.models.base import StreamChunk, StreamingContentType
from aimu.prompts.tuners.scorers import LLMJudgeScorer, Scorer
from aimu.skills.manager import SkillManager

from .._base import AsyncBaseModelClient
from ..agent import Agent, AsyncRunner
from ..skill_agent import SkillAgent

logger = logging.getLogger(__name__)


@dataclass
class PlanExecuteEvaluator(AsyncRunner):
    """Async plan → execute → evaluate → replan-on-fail loop.

    Same semantics as :class:`aimu.agents.PlanExecuteEvaluator` but with
    ``async def run()`` and ``await``ed delegation to planner/executor.

    The scorer's ``score()`` is sync (it's a CPU/judge-call concern, not an
    AIMU-async concern). If you wire an LLM judge, that judge call blocks the
    event loop unless wrapped; use ``asyncio.to_thread`` from your scorer's
    ``score()`` if needed.
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

    async def run(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
    ) -> Union[str, AsyncIterator[StreamChunk]]:
        if stream:
            return self._run_streamed(task, generate_kwargs, images=images)

        attempts: list[dict] = []
        prior_attempt_block = ""

        for round_num in range(self.max_rounds):
            planner_prompt = self._build_planner_prompt(task, prior_attempt_block)
            planner_response = await self.planner.run(planner_prompt, generate_kwargs=generate_kwargs)

            round_criteria, plan_text = self._extract_criteria_and_plan(planner_response)
            output = await self.executor.run(plan_text, generate_kwargs=generate_kwargs, images=images)

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
                self._last_attempts = attempts
                return output

            prior_attempt_block = self._build_prior_attempt_block(attempt)

        self._last_attempts = attempts
        best = max(attempts, key=lambda a: a["score"])
        return best["output"]

    async def _run_streamed(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        images: Optional[list] = None,
    ) -> AsyncIterator[StreamChunk]:
        output = await self.run(task, generate_kwargs=generate_kwargs, images=images)
        yield StreamChunk(StreamingContentType.GENERATING, output, agent=self.name, iteration=0)

    # ---- helpers (mostly copied from sync version; pure logic) ----

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
        return self._last_attempts

    @classmethod
    def from_client(
        cls,
        client: AsyncBaseModelClient,
        *,
        judge_client: Optional[Any] = None,
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
        """Build a PlanExecuteEvaluator from a single async client.

        The ``judge_client`` for the scorer may be sync or async (the scorer is
        sync; if it's an :class:`LLMJudgeScorer` it expects a sync client).
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
