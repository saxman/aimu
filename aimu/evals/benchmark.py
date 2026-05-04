"""
aimu.evals.benchmark: Multi-model benchmark harness.

Benchmark runs the same prompt and dataset across several BaseModelClient
instances (including AgenticModelClient wrapping a SimpleAgent) and returns
a DataFrame of aggregate metrics for direct comparison.

Rows are driven through chat() with messages reset between rows so that an
AgenticModelClient invokes its agent loop. The system_message attribute
survives the reset (it lives outside the messages list) and is re-injected
on the next chat() call. Scoring is delegated to any Scorer implementation
(LLMJudgeScorer, DeepEvalScorer, or custom).

Example::

    from aimu.evals import Benchmark
    from aimu.prompts.tuners.scorers import LLMJudgeScorer

    bench = Benchmark(
        prompt="Summarise this article: {content}",
        data=df,                                  # column 'content' required
        scorer=LLMJudgeScorer(judge, criteria="Concise, accurate, plain English."),
    )
    results = bench.run({
        "claude":  ModelClient(AnthropicModel.CLAUDE_OPUS_4_7),
        "gpt":     ModelClient(OpenAIModel.GPT_5),
        "agent":   AgenticModelClient(SimpleAgent(ModelClient(OllamaModel.QWEN_3_8B))),
    })
    results.to_csv("bench.csv")
    results.to_catalog(catalog, prompt_name="summary-bench")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import pandas as pd
from tqdm import tqdm

from aimu.models.base import BaseModelClient
from aimu.prompts.tuners.scorers import Scorer

if TYPE_CHECKING:
    from aimu.prompts.catalog import PromptCatalog

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResults:
    """
    Aggregate results from a :class:`Benchmark` run.

    Attributes:
        metrics: DataFrame indexed by client name with one column per metric
                 (currently ``score`` and ``pass_rate``).
        prompt:  The prompt template used for the run, kept so that
                 :meth:`to_catalog` can persist it alongside the metrics.
    """

    metrics: pd.DataFrame
    prompt: str

    def to_csv(self, path: str) -> None:
        self.metrics.to_csv(path, index_label="model")

    def to_json(self, path: str) -> None:
        self.metrics.to_json(path, orient="index", indent=2)

    def to_catalog(self, catalog: PromptCatalog, prompt_name: str) -> None:
        """
        Persist one row per client to *catalog* keyed by ``(prompt_name, client_name)``.

        Catalog auto-versions on each store, so re-running a benchmark with the
        same *prompt_name* appends new versions rather than overwriting.
        """
        from aimu.prompts.catalog import Prompt

        for client_name, row in self.metrics.iterrows():
            catalog.store_prompt(
                Prompt(
                    name=prompt_name,
                    model_id=str(client_name),
                    prompt=self.prompt,
                    metrics={k: float(v) for k, v in row.items()},
                )
            )


class Benchmark:
    """
    Run the same prompt and dataset across multiple model clients.

    Args:
        prompt:          Prompt template containing a ``{content}`` placeholder.
        data:            DataFrame with at least a ``content`` column. An optional
                         ``reference`` column is forwarded to scorers that consume
                         it (e.g. :class:`aimu.evals.DeepEvalScorer`).
        scorer:          Per-row :class:`Scorer` used to evaluate model outputs.
        pass_threshold:  Minimum normalised score (0-1) for a row to count
                         toward ``pass_rate``. Default ``0.7``.
        generate_kwargs: Optional kwargs forwarded to ``client.chat(...)``.
    """

    def __init__(
        self,
        prompt: str,
        data: pd.DataFrame,
        scorer: Scorer,
        pass_threshold: float = 0.7,
        generate_kwargs: Optional[dict] = None,
    ):
        if "content" not in data.columns:
            raise ValueError("Benchmark data must include a 'content' column")
        if not data.index.is_unique:
            raise ValueError("Benchmark data must have a unique index")
        self.prompt = prompt
        self.data = data
        self.scorer = scorer
        self.pass_threshold = pass_threshold
        self.generate_kwargs = generate_kwargs

    def run(self, clients: dict[str, BaseModelClient]) -> BenchmarkResults:
        """
        Evaluate every client on the dataset and return aggregate metrics.

        Each row is run as a fresh chat (``client.messages = []``) so an
        :class:`aimu.agents.AgenticModelClient` engages its agent loop. The
        client's original messages are restored after its run completes.

        Args:
            clients: Mapping of display name to client. The display name is
                     used as the row label in the result DataFrame and as the
                     ``model_id`` when persisting to a catalog.

        Returns:
            :class:`BenchmarkResults` with one row per client.
        """
        if not clients:
            raise ValueError("Benchmark requires at least one client")

        rows: dict[str, dict[str, float]] = {}
        for name, client in clients.items():
            logger.info("benchmarking client=%s rows=%d", name, len(self.data))
            rows[name] = self._run_one(name, client)

        # Preserve insertion order so the result rows match the clients dict order.
        metrics_df = pd.DataFrame.from_dict(rows, orient="index")
        return BenchmarkResults(metrics=metrics_df, prompt=self.prompt)

    def _run_one(self, name: str, client: BaseModelClient) -> dict[str, float]:
        original_messages = list(client.messages)
        try:
            scored = self._score_rows(name, client)
        finally:
            client.messages = original_messages

        scores = scored["score"]
        return {
            "score": float(scores.mean()),
            "pass_rate": float((scores >= self.pass_threshold).mean()),
        }

    def _score_rows(self, name: str, client: BaseModelClient) -> pd.DataFrame:
        df = self.data.copy()
        outputs: list[str] = []
        scores: list[float] = []
        feedbacks: list[str] = []

        for i in tqdm(range(len(df)), desc=f"benchmarking {name}"):
            row = df.iloc[i]
            # Reset messages each row so AgenticModelClient runs a fresh agent
            # loop. system_message is stored separately and re-injected on the
            # next chat() call, so it survives this reset.
            client.messages = []
            user_message = self.prompt.format(content=row.content)
            output = client.chat(user_message, self.generate_kwargs)
            outputs.append(output)

            scored_row = row.copy()
            scored_row["output"] = output
            score, feedback = self.scorer.score(scored_row)
            scores.append(score)
            feedbacks.append(feedback)

        df["output"] = outputs
        df["score"] = scores
        df["feedback"] = feedbacks
        return df
