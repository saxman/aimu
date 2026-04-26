"""
aimu.prompts.tuners.judged: Prompt tuner for open-ended generation judged by an LLM.

JudgedPromptTuner handles tasks where there is no single correct answer (summarisation,
rewriting, question answering, creative generation) by delegating evaluation to a
separate judge model that scores each output on a 1–10 scale.

Unlike other tuners, the primary optimisation target is the mean judge *score* (0–1),
not binary accuracy. JudgedPromptTuner overrides ``score()`` to demonstrate how the
base-class design supports non-accuracy primary metrics.

- Data must have a ``content`` column (input text). An optional ``reference`` column
  (ideal/expected output) is included in the judge prompt when present.
- The main prompt template must contain a ``{content}`` placeholder.
- The judge prompt template uses ``{criteria}``, ``{content}``, ``{output}``, and
  ``{reference_line}`` (empty string when no reference is available).

Example::

    from aimu.prompts import JudgedPromptTuner

    tuner = JudgedPromptTuner(
        model_client=main_client,
        judge_client=judge_client,
        criteria="Responses should be concise, accurate, and in plain English.",
    )
    prompt = tuner.tune(df, initial_prompt="Summarise this article: {content}")
"""

from __future__ import annotations

import logging
import re

from tqdm import tqdm

from aimu.prompts.tuner import DEFAULT_MUTATION_KWARGS, PromptTuner

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
    Parse a 1–10 integer score from judge output and normalise to 0–1.

    Finds the first standalone integer in 1–10. Returns 0.5 as a neutral
    fallback when no number is found, and logs a warning.
    """
    m = _SCORE_RE.search(text)
    if m:
        return int(m.group(1)) / 10.0
    logger.warning("Could not parse a 1-10 score from judge output, defaulting to 0.5: %r", text)
    return 0.5


class JudgedPromptTuner(PromptTuner):
    """
    Prompt tuner for open-ended generation tasks evaluated by a judge model.

    Overrides ``score()`` to optimise mean judge score rather than accuracy,
    demonstrating the PromptTuner base-class extension point.

    Args:
        model_client:          ModelClient that generates responses.
        judge_client:          Separate ModelClient used only for evaluation.
        criteria:              Plain-text description of what a good response looks like.
        pass_threshold:        Minimum normalised score (0–1) for a row to be ``_correct``.
                               Default ``0.7`` (i.e. judge score ≥ 7/10).
        judge_prompt_template: Override the default judge prompt. Must contain
                               ``{criteria}``, ``{content}``, ``{output}``, and
                               ``{reference_line}`` placeholders.
    """

    def __init__(
        self,
        model_client,
        judge_client,
        criteria: str,
        pass_threshold: float = 0.7,
        judge_prompt_template: str | None = None,
    ):
        super().__init__(model_client)
        self.judge_client = judge_client
        self.criteria = criteria
        self.pass_threshold = pass_threshold
        self.judge_prompt_template = judge_prompt_template or _DEFAULT_JUDGE_TEMPLATE
        # Generating full responses needs more tokens than short classification outputs.
        self.response_kwargs = DEFAULT_MUTATION_KWARGS.copy()
        if model_client.is_thinking_model:
            from aimu.prompts.tuner import _THINKING_GENERATE_KWARGS

            self.response_kwargs.update(_THINKING_GENERATE_KWARGS)

    # --- PromptTuner overrides ---

    def score(self, metrics: dict) -> float:
        """Optimise mean judge score rather than binary accuracy."""
        return metrics["score"]

    # --- PromptTuner abstract method implementations ---

    def apply_prompt(self, prompt: str, data):
        data = self.generate_responses(prompt, data)
        data = self.judge_responses(data)
        data["_correct"] = data["judge_score"] >= self.pass_threshold
        return data

    def evaluate(self, data) -> dict:
        return self.evaluate_results(data)

    def mutation_prompt(self, current_prompt: str, items: list) -> str:
        parts = [
            f"Given this task prompt:\n{current_prompt}\n",
            f"Task criteria: {self.criteria}\n",
            "The following responses scored below the quality threshold:\n",
        ]

        for item in items:
            parts.append(f"Input: {item.content}")
            parts.append(f"Response: {item.output}")
            parts.append(f"Score: {item.judge_score:.1f}/1.0")
            feedback = getattr(item, "judge_feedback", "")
            if feedback:
                parts.append(f"Judge feedback: {feedback}")
            parts.append("")

        parts.append(
            "Please improve the task prompt so it produces higher-scoring responses."
            " Return the improved prompt wrapped in <prompt></prompt> tags."
        )
        return "\n".join(parts)

    # --- Public helpers (also usable standalone) ---

    def generate_responses(self, prompt: str, data):
        """
        Apply *prompt* to every row's ``content`` and store results in an ``output`` column.

        Args:
            prompt: Prompt template with a ``{content}`` placeholder.
            data:   DataFrame with a ``content`` column.

        Returns:
            DataFrame with added ``output`` column.
        """
        outputs = []
        df = data.copy()

        for i in tqdm(range(len(data)), desc="generating"):
            row = data.iloc[i]
            formatted = prompt.format(content=row.content)
            output = self.model_client.generate(formatted, self.response_kwargs)
            logger.info(f"generated output: {output}")
            outputs.append(output)

        df["output"] = outputs
        return df

    def judge_responses(self, data):
        """
        Judge every row's ``output`` and store normalised scores in ``judge_score``
        and the full judge response in ``judge_feedback``.

        Args:
            data: DataFrame with ``content`` and ``output`` columns.

        Returns:
            DataFrame with added ``judge_score`` (float 0–1) and ``judge_feedback`` columns.
        """
        scores = []
        feedbacks = []
        df = data.copy()

        for i in tqdm(range(len(data)), desc="judging"):
            row = data.iloc[i]
            reference_line = ""
            if "reference" in data.columns and row.reference:
                reference_line = f"Reference: {row.reference}"

            judge_prompt = self.judge_prompt_template.format(
                criteria=self.criteria,
                content=row.content,
                output=row.output,
                reference_line=reference_line,
            )
            feedback = self.judge_client.generate(judge_prompt, self.generate_kwargs)
            logger.info(f"judge feedback: {feedback}")

            score = _parse_judge_score(feedback)
            scores.append(score)
            feedbacks.append(feedback)

        df["judge_score"] = scores
        df["judge_feedback"] = feedbacks
        return df

    def evaluate_results(self, data) -> dict:
        """
        Compute mean judge score and pass rate.

        Args:
            data: DataFrame with ``judge_score`` (float) and ``_correct`` (bool) columns.

        Returns:
            Dict with ``score`` (mean judge score, 0–1) and ``pass_rate`` (fraction passing).
        """
        return {
            "score": data["judge_score"].mean(),
            "pass_rate": data["_correct"].mean(),
        }
