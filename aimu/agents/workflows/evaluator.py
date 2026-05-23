from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Union

from aimu.agents.agent import Agent
from aimu.agents.base import MessageHistory, Workflow
from aimu.models.base import StreamChunk, StreamingContentType

logger = logging.getLogger(__name__)


@dataclass
class EvaluatorOptimizer(Workflow):
    """Anthropic's **Evaluator-Optimizer** pattern: iteratively improve via critic feedback.

    The generator produces an initial response. The evaluator reviews it against the
    original task and either returns ``pass_keyword`` (accepted) or revision feedback.
    The loop stops when the evaluator accepts or ``max_rounds`` is reached.

    Prompt the evaluator to respond with the exact ``pass_keyword`` for acceptance and
    specific revision feedback otherwise.

    Usage::

        eo = EvaluatorOptimizer(
            generator=Agent(client, "Write a clear, accurate explanation.", name="writer"),
            evaluator=Agent(
                client,
                "Review for accuracy and clarity. If acceptable, reply PASS. "
                "Otherwise reply REVISE: <specific feedback>.",
                name="critic",
            ),
            max_rounds=4,
        )
        result = eo.run("Explain gradient descent.")
    """

    generator: Agent
    evaluator: Agent
    name: str = "evaluator_optimizer"
    max_rounds: int = 3
    pass_keyword: str = "PASS"

    @property
    def messages(self) -> MessageHistory:
        result: MessageHistory = {}
        result.update(self.generator.messages)
        result.update(self.evaluator.messages)
        return result

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
            feedback = self.evaluator.run(eval_input, generate_kwargs=generate_kwargs)
            if self.pass_keyword in feedback:
                logger.debug("EvaluatorOptimizer '%s' accepted output on round %d.", self.name, round_num + 1)
                break
            logger.debug("EvaluatorOptimizer '%s' revising on round %d.", self.name, round_num + 1)
            revision_prompt = f"Revise your response based on this feedback:\n{feedback}\n\nOriginal task: {task}"
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
