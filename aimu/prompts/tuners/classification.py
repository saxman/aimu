"""
aimu.prompts.tuners.classification: Prompt tuner for binary classification tasks.

ClassificationPromptTuner implements the three abstract methods of PromptTuner
for the common case of yes/no classification:

- Data must have a ``content`` column (text to classify) and an ``actual_class``
  boolean column (ground truth).
- The prompt template must contain a ``{content}`` placeholder.
- The model's output must contain ``[YES]`` or ``[NO]``.

Example::

    from aimu.prompts import ClassificationPromptTuner

    tuner = ClassificationPromptTuner(model_client)
    prompt = tuner.tune(df, initial_prompt="Does this text mention AI? {content}")
"""

from __future__ import annotations

import pandas as pd

from aimu.prompts.tuner import PromptTuner


class ClassificationPromptTuner(PromptTuner):
    """
    Prompt tuner for binary (YES/NO) classification.

    Inherits the hill-climbing loop from PromptTuner.tune() and implements
    apply_prompt, evaluate, and mutation_prompt for classification tasks.

    Args:
        model_client: Any ModelClient instance.
    """

    # --- PromptTuner abstract method implementations ---

    def apply_prompt(self, prompt: str, data: pd.DataFrame) -> pd.DataFrame:
        data = self.classify_data(prompt, data)
        data["_correct"] = data["actual_class"] == data["predicted_class"]
        return data

    def evaluate(self, data: pd.DataFrame) -> dict:
        return self.evaluate_results(data)

    def mutation_prompt(self, current_prompt: str, items: list) -> str:
        positives = [item.content for item in items if item.actual_class]
        negatives = [item.content for item in items if not item.actual_class]

        parts = [f"Given this task prompt:\n{current_prompt}\n"]

        if positives:
            parts.append("The following examples should be classified as YES but were classified as NO:")
            parts.extend(f"- {c}" for c in positives)

        if negatives:
            parts.append("\nThe following examples should be classified as NO but were classified as YES:")
            parts.extend(f"- {c}" for c in negatives)

        parts.append(
            "\nPlease improve the task prompt to correctly handle all of these examples. Return the improved prompt wrapped in <prompt></prompt> tags."
        )
        return "\n".join(parts)

    # --- Public helpers (also usable standalone) ---

    def classify_data(self, classification_prompt: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run the prompt on every row of *data* and return it with a
        ``predicted_class`` boolean column added.

        Args:
            classification_prompt: Prompt template with a ``{content}`` placeholder.
            data:                  DataFrame with a ``content`` column.

        Returns:
            DataFrame with added ``predicted_class`` column.
        """

        def _parse(result: str, _row: object) -> bool:
            lowered = result.lower()
            if "[yes]" in lowered:
                return True
            if "[no]" in lowered:
                return False
            raise ValueError(f"unexpected classification result: {result}")

        return self._apply_to_rows(classification_prompt, data, _parse, column="predicted_class", desc="classifying")

    def evaluate_results(self, data: pd.DataFrame) -> dict:
        """
        Compute accuracy, precision, and recall from ``actual_class`` /
        ``predicted_class`` columns.

        Args:
            data: DataFrame with boolean ``actual_class`` and ``predicted_class`` columns.

        Returns:
            Dict with ``accuracy``, ``precision``, and ``recall`` keys.
        """
        true_pos = len(data.query("actual_class == True and predicted_class == True"))
        true_neg = len(data.query("actual_class == False and predicted_class == False"))

        accuracy = (true_pos + true_neg) / len(data)
        pos = len(data.query("predicted_class == True"))
        precision = true_pos / pos if pos > 0 else 0
        pos = len(data.query("actual_class == True"))
        recall = true_pos / pos if pos > 0 else 0

        return {"accuracy": accuracy, "precision": precision, "recall": recall}
