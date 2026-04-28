"""
aimu.prompts.tuners.scorers: Pluggable per-row scoring for JudgedPromptTuner.

A Scorer takes a single row (with a ``content`` and ``output`` column) and
returns a normalised score in the [0, 1] range plus a free-form feedback
string used by the mutation prompt.

LLMJudgeScorer is the built-in implementation: it asks a separate judge
model to rate the output on a 1-10 scale and parses the first integer in
that range from the reply. DeepEval-backed scoring lives in
:mod:`aimu.evals.deepeval_scorer`.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

_DEFAULT_JUDGE_TEMPLATE = """\
Task criteria: {criteria}

Input: {content}
{reference_line}
Response: {output}

Rate how well this response satisfies the criteria. \
Reply with only a number from 1 to 10.\
"""

_SCORE_RE = re.compile(r"\b(10|[1-9])\b")


def _parse_judge_score(text: str) -> float:
    """
    Parse a 1-10 integer score from judge output and normalise to 0-1.

    Finds the first standalone integer in 1-10. Returns 0.5 as a neutral
    fallback when no number is found, and logs a warning.
    """
    m = _SCORE_RE.search(text)
    if m:
        return int(m.group(1)) / 10.0
    logger.warning("Could not parse a 1-10 score from judge output, defaulting to 0.5: %r", text)
    return 0.5


class Scorer(ABC):
    """
    Base class for per-row scorers used by :class:`JudgedPromptTuner`.

    Implementations receive a row that has at least ``content`` and
    ``output`` attributes (pandas Series). They return ``(score, feedback)``
    where *score* is a float in [0, 1] and *feedback* is a string passed
    through to the mutation prompt to help the model improve.
    """

    @abstractmethod
    def score(self, row) -> tuple[float, str]: ...


class LLMJudgeScorer(Scorer):
    """
    Score each row by asking a judge model for a 1-10 rating.

    Args:
        judge_client:    ModelClient used to evaluate outputs.
        criteria:        Plain-text description of what a good response looks like.
        prompt_template: Override the default judge prompt. Must contain
                         ``{criteria}``, ``{content}``, ``{output}``, and
                         ``{reference_line}`` placeholders.
        generate_kwargs: Optional kwargs passed through to ``judge_client.generate``.
    """

    def __init__(
        self,
        judge_client,
        criteria: str,
        prompt_template: str | None = None,
        generate_kwargs: dict | None = None,
    ):
        self.judge_client = judge_client
        self.criteria = criteria
        self.prompt_template = prompt_template or _DEFAULT_JUDGE_TEMPLATE
        self.generate_kwargs = generate_kwargs

    def score(self, row) -> tuple[float, str]:
        reference_line = ""
        reference = getattr(row, "reference", None)
        if reference:
            reference_line = f"Reference: {reference}"

        judge_prompt = self.prompt_template.format(
            criteria=self.criteria,
            content=row.content,
            output=row.output,
            reference_line=reference_line,
        )
        feedback = self.judge_client.generate(judge_prompt, self.generate_kwargs)
        logger.info(f"judge feedback: {feedback}")
        return _parse_judge_score(feedback), feedback
