from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional, Union

from aimu.agents.agent import Agent
from aimu.agents.base import MessageHistory, Runner
from aimu.models.base import StreamChunk, StreamingContentType

logger = logging.getLogger(__name__)


@dataclass
class EvaluatorOptimizer(Runner):
    """**Evaluator-Optimizer** pattern: iteratively improve via critic feedback.

    The generator produces an initial response. The evaluator reviews it against the
    original task and decides whether it is accepted; if not, its feedback drives a revision.
    The loop stops when the evaluator accepts or ``max_rounds`` is reached.

    Acceptance is decided by one of three mechanisms, in priority order:

    * ``stop_when`` — a predicate over the evaluator's output (the raw text, or the typed
      verdict when ``verdict_schema`` is set). The most flexible and robust option.
    * ``verdict_schema`` — a dataclass / Pydantic model the evaluator must return (via
      structured output). Acceptance reads its ``passed`` (bool) attribute and revision uses
      its ``feedback`` (str) attribute. Requires a structured-output-capable model; it raises
      on a malformed verdict rather than silently continuing.
    * ``pass_keyword`` (default) — accept when this substring appears in the evaluator's text.

    Usage::

        eo = EvaluatorOptimizer(
            generator=Agent(client, "Write a clear, accurate explanation.", name="writer"),
            evaluator=Agent(client, "Reply PASS or REVISE: <feedback>.", name="critic"),
            max_rounds=4,
        )
        result = eo.run("Explain gradient descent.")

        # Typed verdict instead of substring matching:
        class Verdict(BaseModel):
            passed: bool
            feedback: str = ""
        eo = EvaluatorOptimizer(generator=writer, evaluator=critic, verdict_schema=Verdict)
    """

    generator: Agent
    evaluator: Agent
    name: str = "evaluator_optimizer"
    max_rounds: int = 3
    pass_keyword: str = "PASS"
    stop_when: Optional[Callable[[Any], bool]] = None
    verdict_schema: Optional[type] = None
    passed_attr: str = "passed"
    feedback_attr: str = "feedback"

    def _evaluate(self, eval_input: str, generate_kwargs: Optional[dict[str, Any]]) -> tuple[bool, str]:
        """Run the evaluator and return ``(accepted, feedback_text)``."""
        if self.verdict_schema is not None:
            verdict = self.evaluator.run(eval_input, generate_kwargs=generate_kwargs, schema=self.verdict_schema)
            accepted = self.stop_when(verdict) if self.stop_when else bool(getattr(verdict, self.passed_attr))
            feedback = str(getattr(verdict, self.feedback_attr, "") or verdict)
            return accepted, feedback
        feedback = self.evaluator.run(eval_input, generate_kwargs=generate_kwargs)
        accepted = self.stop_when(feedback) if self.stop_when else (self.pass_keyword in feedback)
        return accepted, feedback

    @property
    def messages(self) -> MessageHistory:
        result: MessageHistory = {}
        result.update(self.generator.messages)
        result.update(self.evaluator.messages)
        return result

    def restore(self, messages: list[dict]) -> None:
        """Restore the generator's state from a saved message list.

        The evaluator starts fresh on the next round. See :meth:`Agent.restore`
        for the full save/restore pattern and system-message handling details.
        """
        self.generator.restore(messages)

    def run(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
    ) -> Union[str, Iterator[StreamChunk]]:
        """Run the generate-evaluate loop. Streaming yields only the final output."""
        if stream:
            return self._run_streamed(task, generate_kwargs, images=images)
        output = self.generator.run(task, generate_kwargs=generate_kwargs, images=images)
        for round_num in range(self.max_rounds - 1):
            eval_input = f"Task: {task}\n\nResponse:\n{output}"
            accepted, feedback = self._evaluate(eval_input, generate_kwargs)
            if accepted:
                logger.debug("EvaluatorOptimizer '%s' accepted output on round %d.", self.name, round_num + 1)
                break
            logger.debug("EvaluatorOptimizer '%s' revising on round %d.", self.name, round_num + 1)
            # Re-supply the prior output: a generator Agent with a system prompt resets its
            # conversation on every run, so it cannot see the response it is asked to revise
            # unless that response is in the prompt.
            revision_prompt = (
                "Revise your previous response based on the feedback below.\n\n"
                f"Original task:\n{task}\n\n"
                f"Your previous response:\n{output}\n\n"
                f"Feedback:\n{feedback}"
            )
            output = self.generator.run(revision_prompt, generate_kwargs=generate_kwargs)
        return output

    def _run_streamed(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        images: Optional[list] = None,
    ) -> Iterator[StreamChunk]:
        output = self.run(task, generate_kwargs=generate_kwargs, images=images)
        yield StreamChunk(StreamingContentType.GENERATING, output, agent=self.name, iteration=0)
