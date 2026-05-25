"""Async EvaluatorOptimizer workflow (generator + critic loop)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional, Union

from aimu.agents.base import MessageHistory
from aimu.models.base import StreamChunk, StreamingContentType

from ..agent import Agent, AsyncRunner

logger = logging.getLogger(__name__)


@dataclass
class EvaluatorOptimizer(AsyncRunner):
    """Async generate-evaluate-revise loop, identical semantics to sync version."""

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

    async def run(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        images: Optional[list] = None,
    ) -> Union[str, AsyncIterator[StreamChunk]]:
        if stream:
            return self._run_streamed(task, generate_kwargs, images=images)
        output = await self.generator.run(task, generate_kwargs=generate_kwargs, images=images)
        for round_num in range(self.max_rounds - 1):
            eval_input = f"Task: {task}\n\nResponse:\n{output}"
            feedback = await self.evaluator.run(eval_input, generate_kwargs=generate_kwargs)
            if self.pass_keyword in feedback:
                logger.debug("EvaluatorOptimizer '%s' accepted on round %d.", self.name, round_num + 1)
                break
            logger.debug("EvaluatorOptimizer '%s' revising on round %d.", self.name, round_num + 1)
            revision_prompt = (
                f"Revise your response based on this feedback:\n{feedback}\n\nOriginal task: {task}"
            )
            output = await self.generator.run(revision_prompt, generate_kwargs=generate_kwargs)
        return output

    async def _run_streamed(
        self,
        task: str,
        generate_kwargs: Optional[dict[str, Any]] = None,
        images: Optional[list] = None,
    ) -> AsyncIterator[StreamChunk]:
        output = await self.run(task, generate_kwargs=generate_kwargs, images=images)
        yield StreamChunk(StreamingContentType.GENERATING, output, agent=self.name, iteration=0)
