"""
aimu.prompts.tuners.judged: Prompt tuner for open-ended generation evaluated by a Scorer.

JudgedPromptTuner handles tasks where there is no single correct answer (summarisation,
rewriting, question answering, creative generation) by delegating per-row evaluation
to a pluggable :class:`Scorer`.

Unlike other tuners, the primary optimisation target is the mean score (0-1), not
binary accuracy. JudgedPromptTuner overrides ``score()`` to demonstrate how the
base-class design supports non-accuracy primary metrics.

- Data must have a ``content`` column (input text). An optional ``reference`` column
  (ideal/expected output) is forwarded to scorers that consume it.
- The main prompt template must contain a ``{content}`` placeholder.

Example with the built-in LLM-as-judge scorer::

    from aimu.prompts import JudgedPromptTuner
    from aimu.prompts.tuners.scorers import LLMJudgeScorer

    tuner = JudgedPromptTuner(
        model_client=main_client,
        scorer=LLMJudgeScorer(judge_client, criteria="Concise, accurate, plain English."),
    )
    prompt = tuner.tune(df, initial_prompt="Summarise this article: {content}")

Example with DeepEval metrics::

    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCaseParams
    from aimu.evals import DeepEvalModel, DeepEvalScorer

    geval = GEval(
        name="quality",
        criteria="Concise, accurate, plain English.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=DeepEvalModel(judge_client),
    )
    tuner = JudgedPromptTuner(model_client=main_client, scorer=DeepEvalScorer([geval]))
"""

from __future__ import annotations

import logging

from tqdm import tqdm

from aimu.prompts.tuner import DEFAULT_MUTATION_KWARGS, PromptTuner
from aimu.prompts.tuners.scorers import Scorer

logger = logging.getLogger(__name__)


class JudgedPromptTuner(PromptTuner):
    """
    Prompt tuner for open-ended generation tasks evaluated by a :class:`Scorer`.

    Overrides ``score()`` to optimise mean judge score rather than accuracy,
    demonstrating the PromptTuner base-class extension point.

    Args:
        model_client:   ModelClient that generates responses.
        scorer:         Per-row :class:`Scorer` used for evaluation. Use
                        :class:`aimu.prompts.tuners.scorers.LLMJudgeScorer` for
                        LLM-as-judge or :class:`aimu.evals.DeepEvalScorer` for
                        DeepEval-backed metrics.
        pass_threshold: Minimum normalised score (0-1) for a row to be ``_correct``.
                        Default ``0.7``.
    """

    def __init__(
        self,
        model_client,
        scorer: Scorer,
        pass_threshold: float = 0.7,
    ):
        super().__init__(model_client)
        self.scorer = scorer
        self.pass_threshold = pass_threshold
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
            "The following responses scored below the quality threshold:\n",
        ]

        for item in items:
            parts.append(f"Input: {item.content}")
            parts.append(f"Response: {item.output}")
            parts.append(f"Score: {item.judge_score:.2f}/1.0")
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
        Run the configured :class:`Scorer` on every row and store results.

        Args:
            data: DataFrame with ``content`` and ``output`` columns.

        Returns:
            DataFrame with added ``judge_score`` (float 0-1) and ``judge_feedback`` columns.
        """
        scores = []
        feedbacks = []
        df = data.copy()

        for i in tqdm(range(len(data)), desc="judging"):
            row = data.iloc[i]
            score, feedback = self.scorer.score(row)
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
            Dict with ``score`` (mean judge score, 0-1) and ``pass_rate`` (fraction passing).
        """
        return {
            "score": data["judge_score"].mean(),
            "pass_rate": data["_correct"].mean(),
        }
