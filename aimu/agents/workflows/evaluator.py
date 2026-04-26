from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Union

from aimu.agents.base import Workflow, AgentChunk, MessageHistory
from aimu.agents.simple_agent import SimpleAgent
from aimu.models.base import StreamingContentType

logger = logging.getLogger(__name__)


@dataclass
class EvaluatorOptimizer(Workflow):
    """
    Workflow pattern: iteratively improve output using a generator-evaluator loop.

    The generator produces an initial response to the task. The evaluator receives
    the task and the generated output and must return either ``pass_keyword``
    (accepted) or a revision request. On a revision request the generator receives
    the evaluator's feedback and tries again. The loop stops when the evaluator
    returns the pass keyword or ``max_rounds`` is reached.

    Prompt the evaluator to respond with the exact ``pass_keyword`` string when
    the output is acceptable, and with specific revision feedback otherwise.

    Usage::

        eo = EvaluatorOptimizer(
            generator=SimpleAgent(client, name="writer",
                system_message="Write a clear, accurate explanation of the concept."),
            evaluator=SimpleAgent(client, name="critic",
                system_message=(
                    "Review the response for accuracy and clarity. "
                    "If it meets the bar, reply PASS. "
                    "Otherwise reply REVISE: <specific feedback>."
                )),
            max_rounds=4,
        )
        result = eo.run("Explain gradient descent.")
    """

    generator: SimpleAgent
    evaluator: SimpleAgent
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
    ) -> Union[str, Iterator[AgentChunk]]:
        """Run the generate-evaluate loop. stream=True yields the final output as a single GENERATING chunk."""
        if stream:
            return self._run_streamed(task, generate_kwargs)
        output = self.generator.run(task, generate_kwargs=generate_kwargs)
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

    def _run_streamed(self, task: str, generate_kwargs: Optional[dict[str, Any]] = None) -> Iterator[AgentChunk]:
        output = self.run(task, generate_kwargs=generate_kwargs)
        yield AgentChunk(self.name, 0, StreamingContentType.GENERATING, output)
