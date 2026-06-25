"""
aimu.prompts.tuners.extraction: Prompt tuner for structured field extraction tasks.

ExtractionPromptTuner tunes prompts that pull structured data from text, including named
entities, key-value pairs, or any fixed set of fields that can be expressed as JSON.

- Data must have a ``content`` column (source text) and an ``expected`` column
  (dict mapping field name → expected value string).
- The prompt template must contain a ``{content}`` placeholder.
- The model's output must be parseable as JSON (raw or within a ```json``` block).
- A row is ``_correct`` only if every field in *fields* matches exactly (after stripping).

Example::

    from aimu.prompts import ExtractionPromptTuner

    tuner = ExtractionPromptTuner(model_client, fields=["name", "company"])
    prompt = tuner.tune(
        df,
        initial_prompt="Extract name and company as JSON. Text: {content}",
    )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from aimu.models._internal.json import parse_json_response
from aimu.prompts.tuner import PromptTuner

if TYPE_CHECKING:
    from aimu.models.base import BaseModelClient

logger = logging.getLogger(__name__)


class ExtractionPromptTuner(PromptTuner):
    """
    Prompt tuner for structured field extraction.

    Constructor: ``ExtractionPromptTuner(model_client, fields)`` — ``fields`` (the field
    names to extract and compare) is **required**, unlike the bare-``model_client`` tuners.

    Inherits the hill-climbing loop from PromptTuner.tune() and implements
    apply_prompt, evaluate, and mutation_prompt for extraction tasks.

    Args:
        model_client: Any ModelClient instance.
        fields:       List of field names to extract and compare. Only these
                      fields are considered when computing ``_correct`` and metrics.
    """

    def __init__(self, model_client: "BaseModelClient", fields: list[str]):
        super().__init__(model_client)
        self.fields = fields

    # --- PromptTuner abstract method implementations ---

    def apply_prompt(self, prompt: str, data: pd.DataFrame) -> pd.DataFrame:
        data = self.extract_data(prompt, data)
        data["_correct"] = data.apply(self._row_correct, axis=1)
        return data

    def evaluate(self, data: pd.DataFrame) -> dict:
        return self.evaluate_results(data)

    def mutation_prompt(self, current_prompt: str, items: list) -> str:
        parts = [f"Given this task prompt:\n{current_prompt}\n"]
        parts.append("The following examples were not extracted correctly:\n")

        for item in items:
            parts.append(f"Text: {item.content}")
            extracted = item.extracted if isinstance(item.extracted, dict) else {}
            expected = item.expected if isinstance(item.expected, dict) else {}
            wrong_fields = [
                f for f in self.fields if str(extracted.get(f, "")).strip() != str(expected.get(f, "")).strip()
            ]
            for field in wrong_fields:
                parts.append(f"  Field '{field}': expected {expected.get(field)!r}, got {extracted.get(field)!r}")
            parts.append("")

        parts.append(
            "Please improve the task prompt to correctly extract all fields for these examples."
            " Return the improved prompt wrapped in <prompt></prompt> tags."
        )
        return "\n".join(parts)

    # --- Public helpers (also usable standalone) ---

    def extract_data(self, extraction_prompt: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run the prompt on every row of *data* and return it with an
        ``extracted`` column (dict) added.

        Args:
            extraction_prompt: Prompt template with a ``{content}`` placeholder.
            data:              DataFrame with a ``content`` column.

        Returns:
            DataFrame with added ``extracted`` column.
        """

        def _parse(result: str, _row: object) -> dict:
            try:
                return parse_json_response(result)
            except ValueError:
                logger.warning(f"Failed to parse JSON from model output, treating as empty: {result!r}")
                return {}

        return self._apply_to_rows(extraction_prompt, data, _parse, column="extracted", desc="extracting")

    def evaluate_results(self, data: pd.DataFrame) -> dict:
        """
        Compute row-level accuracy and per-field match rates.

        Args:
            data: DataFrame with ``expected`` (dict) and ``extracted`` (dict) columns,
                  plus a ``_correct`` boolean column.

        Returns:
            Dict with ``accuracy`` (fraction of fully-correct rows) and
            ``field_{name}_accuracy`` for each field in *self.fields*.
        """
        n = len(data)
        metrics: dict[str, float] = {"accuracy": data["_correct"].sum() / n}

        for field in self.fields:
            correct_count = data.apply(
                lambda row, f=field: (
                    str((row.extracted or {}).get(f, "")).strip() == str((row.expected or {}).get(f, "")).strip()
                ),
                axis=1,
            ).sum()
            metrics[f"field_{field}_accuracy"] = correct_count / n

        return metrics

    # --- Internal ---

    def _row_correct(self, row) -> bool:
        extracted = row.extracted if isinstance(row.extracted, dict) else {}
        expected = row.expected if isinstance(row.expected, dict) else {}
        return all(str(extracted.get(f, "")).strip() == str(expected.get(f, "")).strip() for f in self.fields)
