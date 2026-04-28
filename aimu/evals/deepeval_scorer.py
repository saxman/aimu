"""
aimu.evals.deepeval_scorer: DeepEval-backed Scorer for JudgedPromptTuner.

Wraps a list of DeepEval ``BaseMetric`` instances. For each row, an
``LLMTestCase`` is built from the row's ``content`` (input), ``output``
(actual_output), and optional ``reference`` (expected_output). Every
metric's ``measure()`` is called; the returned score is the mean of the
per-metric scores, and the feedback string concatenates each metric's
reason for use in the mutation prompt.
"""

from __future__ import annotations

import logging

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from aimu.prompts.tuners.scorers import Scorer

logger = logging.getLogger(__name__)


class DeepEvalScorer(Scorer):
    """
    Score each row using one or more DeepEval metrics, averaged.

    Args:
        metrics:      One or more DeepEval ``BaseMetric`` instances. Each
                      metric should already be configured with its own
                      judge model (typically wrapped via
                      :class:`aimu.evals.DeepEvalModel`).
        input_field:  Row attribute used as ``LLMTestCase.input``. Default
                      ``"content"`` matches the column convention used by
                      :class:`aimu.prompts.JudgedPromptTuner`.
        output_field: Row attribute used as ``LLMTestCase.actual_output``.
                      Default ``"output"`` matches the column produced by
                      :meth:`JudgedPromptTuner.generate_responses`.
    """

    def __init__(
        self,
        metrics: list[BaseMetric],
        input_field: str = "content",
        output_field: str = "output",
    ):
        if not metrics:
            raise ValueError("DeepEvalScorer requires at least one metric")
        self.metrics = metrics
        self.input_field = input_field
        self.output_field = output_field

    def score(self, row) -> tuple[float, str]:
        test_case = LLMTestCase(
            input=getattr(row, self.input_field),
            actual_output=getattr(row, self.output_field),
            expected_output=getattr(row, "reference", None),
        )

        scores: list[float] = []
        reasons: list[str] = []
        for metric in self.metrics:
            metric.measure(test_case)
            scores.append(float(metric.score))
            reasons.append(f"{type(metric).__name__}: {metric.reason}")

        mean_score = sum(scores) / len(scores)
        feedback = "\n".join(reasons)
        logger.info("deepeval scores=%s mean=%.3f", scores, mean_score)
        return mean_score, feedback
