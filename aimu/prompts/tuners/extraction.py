"""
aimu.prompts.tuners.extraction — Prompt tuner for structured field extraction tasks.

ExtractionPromptTuner tunes prompts that pull structured data from text — named
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

import json
import logging
import re

from tqdm import tqdm

from aimu.prompts.tuner import PromptTuner

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _parse_json(text: str) -> dict:
    """
    Extract a JSON object from *text*.

    Tries, in order:
    1. Direct ``json.loads`` on the full string.
    2. Content of the first ```json``` fenced block.
    3. Substring from the first ``{`` to the last ``}``.

    Raises:
        ValueError: If no valid JSON object is found.
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse a JSON object from model output: {text!r}")


class ExtractionPromptTuner(PromptTuner):
    """
    Prompt tuner for structured field extraction.

    Inherits the hill-climbing loop from PromptTuner.tune() and implements
    apply_prompt, evaluate, and mutation_prompt for extraction tasks.

    Args:
        model_client: Any ModelClient instance.
        fields:       List of field names to extract and compare. Only these
                      fields are considered when computing ``_correct`` and metrics.
    """

    def __init__(self, model_client, fields: list[str]):
        super().__init__(model_client)
        self.fields = fields

    # --- PromptTuner abstract method implementations ---

    def apply_prompt(self, prompt: str, data):
        data = self.extract_data(prompt, data)
        data["_correct"] = data.apply(self._row_correct, axis=1)
        return data

    def evaluate(self, data) -> dict:
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

    def extract_data(self, extraction_prompt: str, data):
        """
        Run the prompt on every row of *data* and return it with an
        ``extracted`` column (dict) added.

        Args:
            extraction_prompt: Prompt template with a ``{content}`` placeholder.
            data:              DataFrame with a ``content`` column.

        Returns:
            DataFrame with added ``extracted`` column.
        """
        extracted_list = []
        df = data.copy()

        for i in tqdm(range(len(data)), desc="extracting"):
            row = data.iloc[i]
            prompt = extraction_prompt.format(content=row.content)
            result = self.model_client.generate(prompt, self.generate_kwargs)
            logger.info(f"raw extraction result: {result}")

            try:
                parsed = _parse_json(result)
            except ValueError:
                logger.warning(f"Failed to parse JSON from model output, treating as empty: {result!r}")
                parsed = {}

            extracted_list.append(parsed)
            logger.info(f"extracted fields: {parsed}")

        df["extracted"] = extracted_list
        return df

    def evaluate_results(self, data) -> dict:
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
                lambda row, f=field: str((row.extracted or {}).get(f, "")).strip()
                == str((row.expected or {}).get(f, "")).strip(),
                axis=1,
            ).sum()
            metrics[f"field_{field}_accuracy"] = correct_count / n

        return metrics

    # --- Internal ---

    def _row_correct(self, row) -> bool:
        extracted = row.extracted if isinstance(row.extracted, dict) else {}
        expected = row.expected if isinstance(row.expected, dict) else {}
        return all(
            str(extracted.get(f, "")).strip() == str(expected.get(f, "")).strip() for f in self.fields
        )
