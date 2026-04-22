"""
aimu.prompts.tuners.multiclass: Prompt tuner for multi-class classification tasks.

MultiClassPromptTuner extends PromptTuner for tasks where the model must assign
one of N named categories to each input, rather than a binary YES/NO decision.

- Data must have a ``content`` column (text to classify) and an ``actual_class``
  column (ground-truth category string, must be one of *classes*).
- The prompt template must contain a ``{content}`` placeholder.
- The model's output must contain ``[ClassName]`` for one of the registered classes
  (case-insensitive).

Example::

    from aimu.prompts import MultiClassPromptTuner

    tuner = MultiClassPromptTuner(
        model_client,
        classes=["positive", "negative", "neutral"],
    )
    prompt = tuner.tune(df, initial_prompt="Classify the sentiment: {content}")
"""

from __future__ import annotations

import logging
from collections import defaultdict

from tqdm import tqdm

from aimu.prompts.tuner import PromptTuner

logger = logging.getLogger(__name__)


class MultiClassPromptTuner(PromptTuner):
    """
    Prompt tuner for multi-class (N-way) classification.

    Inherits the hill-climbing loop from PromptTuner.tune() and implements
    apply_prompt, evaluate, and mutation_prompt for multi-class tasks.

    Args:
        model_client: Any ModelClient instance.
        classes:      Ordered list of valid class name strings. The model is
                      expected to output ``[ClassName]`` for one of these.
    """

    def __init__(self, model_client, classes: list[str]):
        super().__init__(model_client)
        self.classes = classes

    # --- PromptTuner abstract method implementations ---

    def apply_prompt(self, prompt: str, data):
        data = self.classify_data(prompt, data)
        data["_correct"] = data["actual_class"] == data["predicted_class"]
        return data

    def evaluate(self, data) -> dict:
        return self.evaluate_results(data)

    def mutation_prompt(self, current_prompt: str, items: list) -> str:
        # Group misclassified items by (predicted, actual) confusion pair
        confusion: dict[tuple[str, str], list] = defaultdict(list)
        for item in items:
            key = (item.predicted_class, item.actual_class)
            confusion[key].append(item.content)

        parts = [f"Given this task prompt:\n{current_prompt}\n"]
        for (predicted, actual), examples in confusion.items():
            parts.append(f"The following were classified as [{predicted}] but should be [{actual}]:")
            parts.extend(f"- {ex}" for ex in examples)
            parts.append("")

        parts.append(
            "Please improve the task prompt to correctly handle all of these examples."
            " Return the improved prompt wrapped in <prompt></prompt> tags."
        )
        return "\n".join(parts)

    # --- Public helpers (also usable standalone) ---

    def classify_data(self, classification_prompt: str, data):
        """
        Run the prompt on every row of *data* and return it with a
        ``predicted_class`` column added.

        Args:
            classification_prompt: Prompt template with a ``{content}`` placeholder.
            data:                  DataFrame with a ``content`` column.

        Returns:
            DataFrame with added ``predicted_class`` column.

        Raises:
            ValueError: If the model output does not contain any known ``[ClassName]``.
        """
        predicted_class = []
        df = data.copy()
        lower_classes = [c.lower() for c in self.classes]

        for i in tqdm(range(len(data)), desc="classifying"):
            row = data.iloc[i]
            prompt = classification_prompt.format(content=row.content)
            result = self.model_client.generate(prompt, self.generate_kwargs)
            logger.info(f"raw classification result: {result}")

            result_lower = result.lower()
            matched = None
            for cls, cls_lower in zip(self.classes, lower_classes):
                if f"[{cls_lower}]" in result_lower:
                    matched = cls
                    break

            if matched is None:
                raise ValueError(
                    f"Model output did not contain any known class tag {[f'[{c}]' for c in self.classes]}. "
                    f"Got: {result!r}"
                )

            predicted_class.append(matched)
            logger.info(f"classification: {matched}")

        df["predicted_class"] = predicted_class
        return df

    def evaluate_results(self, data) -> dict:
        """
        Compute accuracy and per-class precision, recall, and F1.

        Args:
            data: DataFrame with ``actual_class``, ``predicted_class`` columns.

        Returns:
            Dict with ``accuracy``, ``macro_f1``, and per-class
            ``per_class_{name}_precision``, ``per_class_{name}_recall``,
            ``per_class_{name}_f1`` keys.
        """
        n = len(data)
        accuracy = (data["actual_class"] == data["predicted_class"]).sum() / n

        per_class_f1 = []
        metrics: dict[str, float] = {"accuracy": accuracy}

        for cls in self.classes:
            tp = len(data[(data["actual_class"] == cls) & (data["predicted_class"] == cls)])
            fp = len(data[(data["actual_class"] != cls) & (data["predicted_class"] == cls)])
            fn = len(data[(data["actual_class"] == cls) & (data["predicted_class"] != cls)])

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics[f"per_class_{cls}_precision"] = precision
            metrics[f"per_class_{cls}_recall"] = recall
            metrics[f"per_class_{cls}_f1"] = f1
            per_class_f1.append(f1)

        metrics["macro_f1"] = sum(per_class_f1) / len(per_class_f1) if per_class_f1 else 0.0
        return metrics
