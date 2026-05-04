"""
Benchmark harness tests.

Usage:
- pytest tests/test_benchmark.py                    # mock-only, no backend needed
- pytest tests/test_benchmark.py -k catalog          # exercise PromptCatalog persistence
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from aimu.evals import Benchmark, BenchmarkResults
from aimu.prompts.tuners.scorers import Scorer


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class _StubChatClient:
    """
    Minimal BaseModelClient-shaped stub.

    chat() returns from a configurable function so tests can vary outputs per
    row; generate() raises so any accidental use shows up immediately.
    """

    is_thinking_model = False

    def __init__(self, responder, system_message: str | None = None, initial_messages=None):
        self._responder = responder
        self._system_message = system_message
        self.messages = list(initial_messages or [])
        self.chat_calls: list[dict] = []

    def chat(self, user_message, generate_kwargs=None, use_tools=True, stream=False):
        # Snapshot messages at the moment of the call so tests can verify reset.
        self.chat_calls.append(
            {
                "user_message": user_message,
                "messages_before": list(self.messages),
                "generate_kwargs": generate_kwargs,
            }
        )
        return self._responder(user_message)

    def generate(self, prompt, generate_kwargs=None, stream=False, include_thinking=True):
        raise AssertionError("Benchmark must drive rows through chat(), not generate()")

    @property
    def system_message(self):
        return self._system_message

    @system_message.setter
    def system_message(self, value):
        self._system_message = value


class _SeqScorer(Scorer):
    """Scorer that returns scores from a fixed sequence."""

    def __init__(self, scores):
        self._scores = list(scores)
        self._idx = 0
        self.rows_seen: list[pd.Series] = []

    def score(self, row):
        if self._idx >= len(self._scores):
            raise AssertionError(f"_SeqScorer exhausted after {len(self._scores)} calls")
        score = self._scores[self._idx]
        self._idx += 1
        self.rows_seen.append(row)
        return score, f"score={score}"


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_benchmark_requires_content_column():
    df = pd.DataFrame({"text": ["hello"]})
    with pytest.raises(ValueError, match="content"):
        Benchmark(prompt="Q: {content}", data=df, scorer=_SeqScorer([0.5]))


def test_benchmark_requires_unique_index():
    df = pd.DataFrame({"content": ["a", "b"]}, index=[0, 0])
    with pytest.raises(ValueError, match="unique"):
        Benchmark(prompt="Q: {content}", data=df, scorer=_SeqScorer([0.5, 0.5]))


def test_benchmark_run_requires_clients():
    df = pd.DataFrame({"content": ["a"]})
    with pytest.raises(ValueError, match="at least one"):
        Benchmark(prompt="Q: {content}", data=df, scorer=_SeqScorer([0.5])).run({})


# ---------------------------------------------------------------------------
# Core run() behaviour
# ---------------------------------------------------------------------------


def test_run_aggregates_score_and_pass_rate():
    df = pd.DataFrame({"content": ["a", "b", "c"]})
    client = _StubChatClient(responder=lambda msg: f"reply:{msg}")
    scorer = _SeqScorer([0.9, 0.5, 0.8])  # threshold 0.7 → 2/3 pass

    bench = Benchmark(prompt="Q: {content}", data=df, scorer=scorer, pass_threshold=0.7)
    results = bench.run({"stub": client})

    assert isinstance(results, BenchmarkResults)
    assert list(results.metrics.index) == ["stub"]
    row = results.metrics.loc["stub"]
    assert row["score"] == pytest.approx((0.9 + 0.5 + 0.8) / 3)
    assert row["pass_rate"] == pytest.approx(2 / 3)


def test_run_uses_chat_not_generate():
    """If Benchmark called generate(), the stub raises — this test would fail."""
    df = pd.DataFrame({"content": ["a", "b"]})
    client = _StubChatClient(responder=lambda msg: "ok")
    bench = Benchmark(prompt="{content}", data=df, scorer=_SeqScorer([0.7, 0.7]))

    bench.run({"stub": client})

    assert len(client.chat_calls) == 2


def test_run_resets_messages_between_rows():
    df = pd.DataFrame({"content": ["a", "b"]})
    client = _StubChatClient(
        responder=lambda msg: "ok",
        initial_messages=[{"role": "user", "content": "old turn"}],
    )
    bench = Benchmark(prompt="{content}", data=df, scorer=_SeqScorer([0.7, 0.7]))

    bench.run({"stub": client})

    # Each chat call should have started from an empty messages list.
    for call in client.chat_calls:
        assert call["messages_before"] == []


def test_run_restores_original_messages_after_run():
    df = pd.DataFrame({"content": ["a", "b"]})
    initial = [{"role": "user", "content": "old turn"}]
    client = _StubChatClient(responder=lambda msg: "ok", initial_messages=initial)
    bench = Benchmark(prompt="{content}", data=df, scorer=_SeqScorer([0.7, 0.7]))

    bench.run({"stub": client})

    assert client.messages == initial


def test_run_restores_original_messages_when_chat_raises():
    df = pd.DataFrame({"content": ["a", "b"]})
    initial = [{"role": "user", "content": "old turn"}]

    def boom(_msg):
        raise RuntimeError("simulated chat failure")

    client = _StubChatClient(responder=boom, initial_messages=initial)
    bench = Benchmark(prompt="{content}", data=df, scorer=_SeqScorer([0.7]))

    with pytest.raises(RuntimeError, match="simulated"):
        bench.run({"stub": client})

    assert client.messages == initial


def test_run_formats_prompt_with_content():
    df = pd.DataFrame({"content": ["hello", "world"]})
    seen: list[str] = []

    def capture(msg):
        seen.append(msg)
        return "ok"

    client = _StubChatClient(responder=capture)
    bench = Benchmark(prompt="Greet: {content}", data=df, scorer=_SeqScorer([0.7, 0.7]))

    bench.run({"stub": client})

    assert seen == ["Greet: hello", "Greet: world"]


def test_run_passes_reference_to_scorer():
    df = pd.DataFrame({"content": ["a"], "reference": ["expected answer"]})
    client = _StubChatClient(responder=lambda msg: "actual answer")
    scorer = _SeqScorer([0.8])

    bench = Benchmark(prompt="{content}", data=df, scorer=scorer)
    bench.run({"stub": client})

    row = scorer.rows_seen[0]
    assert row["content"] == "a"
    assert row["reference"] == "expected answer"
    assert row["output"] == "actual answer"


def test_run_passes_generate_kwargs():
    df = pd.DataFrame({"content": ["a"]})
    client = _StubChatClient(responder=lambda msg: "ok")
    bench = Benchmark(
        prompt="{content}",
        data=df,
        scorer=_SeqScorer([0.7]),
        generate_kwargs={"temperature": 0.2},
    )

    bench.run({"stub": client})

    assert client.chat_calls[0]["generate_kwargs"] == {"temperature": 0.2}


def test_run_compares_multiple_clients():
    df = pd.DataFrame({"content": ["a", "b"]})
    high = _StubChatClient(responder=lambda msg: "good")
    low = _StubChatClient(responder=lambda msg: "bad")
    # Scorer is called per (client, row) in client order:
    # high → 0.9, 0.9; low → 0.4, 0.4
    scorer = _SeqScorer([0.9, 0.9, 0.4, 0.4])

    bench = Benchmark(prompt="{content}", data=df, scorer=scorer, pass_threshold=0.7)
    results = bench.run({"high": high, "low": low})

    assert list(results.metrics.index) == ["high", "low"]
    assert results.metrics.loc["high", "score"] == pytest.approx(0.9)
    assert results.metrics.loc["high", "pass_rate"] == pytest.approx(1.0)
    assert results.metrics.loc["low", "score"] == pytest.approx(0.4)
    assert results.metrics.loc["low", "pass_rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# BenchmarkResults persistence
# ---------------------------------------------------------------------------


def _build_simple_results() -> BenchmarkResults:
    df = pd.DataFrame({"content": ["a", "b"]})
    client = _StubChatClient(responder=lambda msg: "ok")
    bench = Benchmark(prompt="P: {content}", data=df, scorer=_SeqScorer([0.8, 0.6]), pass_threshold=0.7)
    return bench.run({"stub": client})


def test_results_to_csv(tmp_path):
    results = _build_simple_results()
    path = tmp_path / "bench.csv"
    results.to_csv(str(path))

    reloaded = pd.read_csv(path, index_col="model")
    assert list(reloaded.index) == ["stub"]
    assert reloaded.loc["stub", "score"] == pytest.approx(0.7)
    assert reloaded.loc["stub", "pass_rate"] == pytest.approx(0.5)


def test_results_to_json(tmp_path):
    results = _build_simple_results()
    path = tmp_path / "bench.json"
    results.to_json(str(path))

    payload = json.loads(path.read_text())
    assert "stub" in payload
    assert payload["stub"]["score"] == pytest.approx(0.7)
    assert payload["stub"]["pass_rate"] == pytest.approx(0.5)


def test_results_to_catalog(tmp_path):
    from aimu.prompts.catalog import PromptCatalog

    df = pd.DataFrame({"content": ["a", "b"]})
    high = _StubChatClient(responder=lambda msg: "good")
    low = _StubChatClient(responder=lambda msg: "bad")
    scorer = _SeqScorer([0.9, 0.9, 0.4, 0.4])

    bench = Benchmark(prompt="P: {content}", data=df, scorer=scorer, pass_threshold=0.7)
    results = bench.run({"high": high, "low": low})

    catalog = PromptCatalog(db_path=str(tmp_path / "catalog.db"))
    results.to_catalog(catalog, prompt_name="bench-test")

    high_rows = catalog.retrieve_all("bench-test", "high")
    low_rows = catalog.retrieve_all("bench-test", "low")
    assert len(high_rows) == 1
    assert len(low_rows) == 1
    assert high_rows[0].prompt == "P: {content}"
    assert high_rows[0].metrics["score"] == pytest.approx(0.9)
    assert high_rows[0].metrics["pass_rate"] == pytest.approx(1.0)
    assert low_rows[0].metrics["score"] == pytest.approx(0.4)
    assert low_rows[0].metrics["pass_rate"] == pytest.approx(0.0)


def test_results_to_catalog_appends_versions(tmp_path):
    from aimu.prompts.catalog import PromptCatalog

    catalog = PromptCatalog(db_path=str(tmp_path / "catalog.db"))

    # Two runs of the same benchmark should produce two versions per client.
    for _ in range(2):
        df = pd.DataFrame({"content": ["a"]})
        client = _StubChatClient(responder=lambda msg: "ok")
        bench = Benchmark(prompt="P: {content}", data=df, scorer=_SeqScorer([0.8]))
        bench.run({"stub": client}).to_catalog(catalog, prompt_name="repeat-test")

    rows = catalog.retrieve_all("repeat-test", "stub")
    assert len(rows) == 2
    assert {r.version for r in rows} == {1, 2}
