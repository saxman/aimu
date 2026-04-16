"""
aimu.prompts.tuner — Abstract base class for hill-climbing prompt optimisation.

Subclass PromptTuner and implement three methods:

- apply_prompt(prompt, data)  — run the prompt on the dataset, return annotated
  data that includes a boolean '_correct' column.
- evaluate(data) -> dict       — score the annotated data; the dict must contain
  an 'accuracy' key so the loop knows whether to keep or revert a mutation.
- mutation_prompt(prompt, items) -> str — return a prompt that asks the LLM to
  improve *prompt* given the list of incorrectly handled *items*.

The shared hill-climbing loop lives in tune().
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aimu.prompts.catalog import PromptCatalog

logger = logging.getLogger(__name__)

# kwargs used when classifying / applying the prompt (short outputs like [YES]/[NO])
DEFAULT_GENERATE_KWARGS = {"max_new_tokens": 3, "temperature": 0.01, "do_sample": True}

# kwargs used when generating a mutated prompt (needs full-length text output)
DEFAULT_MUTATION_KWARGS = {"max_new_tokens": 512, "temperature": 0.3, "do_sample": True}

_THINKING_GENERATE_KWARGS = {"max_new_tokens": 1024, "temperature": 0.6}


class PromptTuner(ABC):
    def __init__(self, model_client):
        self.model_client = model_client
        self.generate_kwargs = DEFAULT_GENERATE_KWARGS.copy()
        self.mutation_kwargs = DEFAULT_MUTATION_KWARGS.copy()
        if model_client.is_thinking_model:
            self.generate_kwargs.update(_THINKING_GENERATE_KWARGS)
            self.mutation_kwargs.update(_THINKING_GENERATE_KWARGS)

    @abstractmethod
    def apply_prompt(self, prompt: str, data):
        """
        Apply *prompt* to *data* and return the annotated dataset.

        The returned object must contain a boolean column named ``_correct``
        (True when the model's output matched ground truth, False otherwise).
        Implementations may mutate *data* in-place or return a copy.
        """
        ...

    @abstractmethod
    def evaluate(self, data) -> dict:
        """
        Score the annotated dataset.

        Args:
            data: Annotated dataset as returned by apply_prompt.

        Returns:
            Metrics dict. Must include an ``'accuracy'`` key (float 0–1) so the
            loop can decide whether to keep or revert a mutation.
        """
        ...

    @abstractmethod
    def mutation_prompt(self, current_prompt: str, items: list) -> str:
        """
        Build a prompt that asks the LLM to improve *current_prompt*.

        Args:
            current_prompt: The prompt version that failed on *items*.
            items:          List of incorrectly handled rows (pandas-like row
                            objects with attribute access).

        Returns:
            A prompt string. The LLM's response is expected to contain the
            improved prompt wrapped in ``<prompt>...</prompt>`` tags.
        """
        ...

    def score(self, metrics: dict) -> float:
        """
        Scalar to maximise during hill-climbing. Default: ``metrics['accuracy']``.

        Override to optimise a different metric (e.g. ``metrics['score']`` for
        a judge-based tuner, or ``metrics['macro_f1']`` for multi-class tasks).
        """
        return metrics["accuracy"]

    def extract_mutated_prompt(self, result: str) -> str:
        """
        Extract the improved prompt text from a mutation LLM response.

        Default: parses content between ``<prompt>`` and ``</prompt>`` tags.
        Override for a different extraction format (JSON, bare text, etc.).
        """
        return result.split("<prompt>")[-1].split("</prompt>")[0].strip()

    def tune(
        self,
        training_data,
        initial_prompt: str,
        max_iterations: int = 20,
        max_examples: int = 5,
        catalog: PromptCatalog | None = None,
        prompt_name: str | None = None,
    ) -> str:
        """
        Hill-climbing loop: mutate the prompt until 100% accuracy is reached
        or *max_iterations* is exhausted.

        Each iteration applies the current prompt, evaluates results, and — if
        accuracy improved — picks up to *max_examples* incorrect items and asks
        the LLM to produce a better prompt. If a mutation makes accuracy worse
        the previous best prompt is restored immediately without re-evaluating.

        Args:
            training_data:  Dataset passed to apply_prompt and evaluate.
            initial_prompt: Starting prompt text.
            max_iterations: Hard stop on the number of evaluation rounds.
                            Prevents infinite loops when 100% is unreachable.
            max_examples:   Maximum number of incorrect examples passed to
                            mutation_prompt per round (sampled randomly).
            catalog:        Optional PromptCatalog. When provided (together with
                            *prompt_name*), each improvement is persisted.
            prompt_name:    Catalog key for saving prompt versions.

        Returns:
            The best prompt found.
        """
        data = training_data.copy()
        current_prompt = initial_prompt
        best_prompt = initial_prompt
        best_metrics: dict | None = None
        best_data = None
        iteration = mutation = 0

        while True:
            data = self.apply_prompt(current_prompt, data)
            metrics = self.evaluate(data)

            metric_str = " ".join(f"{k}={v:.3f}" for k, v in metrics.items() if isinstance(v, float))
            out = f"iteration={iteration} mutation={mutation} {metric_str}"
            print(out)
            logger.info(out)

            iteration += 1

            if best_metrics is not None and self.score(metrics) < self.score(best_metrics):
                # Mutation made things worse — revert to cached best state without
                # re-evaluating (avoids a full classification pass).
                print("reverting to last best prompt")
                logger.info("reverting to last best prompt")
                current_prompt = best_prompt
                data = best_data
                metrics = best_metrics
                mutation -= 1
            else:
                # Equal or better — promote to best and optionally persist.
                best_prompt = current_prompt
                best_metrics = metrics
                best_data = data.copy()

                if catalog is not None and prompt_name is not None:
                    self._save_to_catalog(catalog, prompt_name, current_prompt, metrics)

            bad = data[~data["_correct"]]

            if bad.empty:
                break

            if iteration >= max_iterations:
                logger.warning(
                    "Reached max_iterations=%d without 100%% accuracy. Best accuracy=%.3f",
                    max_iterations,
                    best_metrics["accuracy"],
                )
                break

            bad_sample = bad.sample(min(len(bad), max_examples))
            items = [bad_sample.iloc[i] for i in range(len(bad_sample))]
            mut_prompt = self.mutation_prompt(current_prompt, items)
            print("mutating prompt")
            logger.info(f"mutation prompt:\n{mut_prompt}")

            result = self.model_client.generate(mut_prompt, self.mutation_kwargs)
            logger.info(f"raw mutation result:\n{result}")

            current_prompt = self.extract_mutated_prompt(result)
            logger.info(f"mutated prompt:\n{current_prompt}")

            mutation += 1

        return best_prompt

    def _save_to_catalog(self, catalog: PromptCatalog, prompt_name: str, prompt: str, metrics: dict) -> None:
        from aimu.prompts.catalog import Prompt

        model = self.model_client.model
        try:
            model_id = model.value[0]
        except (AttributeError, TypeError, IndexError):
            model_id = str(model)

        catalog.store_prompt(Prompt(name=prompt_name, model_id=model_id, prompt=prompt, metrics=metrics))
