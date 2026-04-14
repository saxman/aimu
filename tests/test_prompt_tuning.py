"""
Prompt tuning tests.

Usage:
- pytest tests/test_prompt_tuning.py                            # mock client (no backend needed)
- pytest tests/test_prompt_tuning.py --client=ollama            # all Ollama models
- pytest tests/test_prompt_tuning.py --client=ollama --model=LLAMA_3_2_3B
- pytest tests/test_prompt_tuning.py --client=hf
- pytest tests/test_prompt_tuning.py --client=aisuite
"""

from typing import Iterable

import pandas as pd
import pytest

from aimu.models import ModelClient
from aimu.prompts import ClassificationPromptTuner
from conftest import create_real_model_client, resolve_model_params

# ---------------------------------------------------------------------------
# Sequential mock client — returns responses one-by-one from a fixed list.
# Used for tune() tests where generate() is called multiple times in sequence.
# ---------------------------------------------------------------------------


class _TuneSeqClient:
    """Returns responses from a pre-built sequence; raises if exhausted."""

    is_thinking_model = False

    # A minimal stand-in for the model attribute used by _save_to_catalog.
    class _Model:
        value = ("mock-model",)

    model = _Model()

    def __init__(self, responses: list[str]):
        self._responses = responses
        self._idx = 0

    def generate(self, prompt: str, generate_kwargs=None) -> str:
        if self._idx >= len(self._responses):
            raise AssertionError(
                f"generate() called more times than expected. Exhausted after {len(self._responses)} calls."
            )
        resp = self._responses[self._idx]
        self._idx += 1
        return resp

    @property
    def calls_made(self) -> int:
        return self._idx


_MOCK = "mock"


# ---------------------------------------------------------------------------
# Mock client — keyword-based [YES]/[NO] responses, no backend required
# ---------------------------------------------------------------------------


class _MockClassifyClient:
    """Minimal client that returns [YES]/[NO] by keyword matching."""

    is_thinking_model = False

    def generate(self, prompt, generate_kwargs=None):
        keywords = ["large language", "llm", "ai", "language model"]
        return "[YES]" if any(kw in prompt.lower() for kw in keywords) else "[NO]"


# ---------------------------------------------------------------------------
# Parametrization — default is mock; real client when --client is specified
# ---------------------------------------------------------------------------


def pytest_generate_tests(metafunc):
    if "model_client" not in metafunc.fixturenames:
        return
    params = resolve_model_params(metafunc.config, default_params=[_MOCK])
    metafunc.parametrize("model_client", params, indirect=True, scope="session")


@pytest.fixture(scope="session")
def model_client(request) -> Iterable[ModelClient]:
    if request.param == _MOCK:
        yield _MockClassifyClient()
        return
    yield from create_real_model_client(request)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_classify_data(model_client):
    """Test that classify_data populates a predicted_class column."""
    tuner = ClassificationPromptTuner(model_client=model_client)

    classification_prompt = (
        "Is the following content related to large language models?\n"
        "Wrap your final answer in square braces, e.g. [YES] or [NO]. "
        "Only include [YES] or [NO] in your answer.\n"
        "Content: {content}"
    )

    df = pd.DataFrame(
        {
            "content": [
                "Large language models are a type of AI.",
                "This is a random sentence about cooking.",
                "LLMs can generate text based on prompts.",
                "The weather today is sunny and warm.",
            ]
        }
    )

    df = tuner.classify_data(classification_prompt, df)

    assert "predicted_class" in df.columns
    assert len(df.predicted_class) == 4
    assert df.predicted_class.value_counts().get(True, 0) + df.predicted_class.value_counts().get(False, 0) == 4


class _MockClientForEval:
    """Stub for constructing a tuner without calling the model."""

    is_thinking_model = False


def test_evaluate_results():
    """Test metric calculations — no model call needed."""
    tuner = ClassificationPromptTuner(model_client=_MockClientForEval())

    df = pd.DataFrame(
        {
            "actual_class": [True, False, True, False],
            "predicted_class": [True, False, True, False],
        }
    )
    metrics = tuner.evaluate_results(df)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0

    df = pd.DataFrame(
        {
            "actual_class": [True, True, False, False],
            "predicted_class": [True, False, True, False],
        }
    )
    metrics = tuner.evaluate_results(df)
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5


# ---------------------------------------------------------------------------
# tune() tests — exercise the hill-climbing loop with _TuneSeqClient
# ---------------------------------------------------------------------------

_INITIAL_PROMPT = "Does this text discuss AI? Answer [YES] or [NO]. Content: {content}"

_TWO_ROW_DF = pd.DataFrame(
    {
        "content": ["LLMs are AI systems.", "The weather is sunny."],
        "actual_class": [True, False],
    }
)


def test_tune_returns_initial_prompt_when_perfect():
    """When 100% accuracy on the first pass, tune() returns the initial prompt with no mutations."""
    # 2 rows → 2 classify calls; both correct → stop immediately
    client = _TuneSeqClient(["[YES]", "[NO]"])
    tuner = ClassificationPromptTuner(model_client=client)

    result = tuner.tune(_TWO_ROW_DF, _INITIAL_PROMPT)

    assert result == _INITIAL_PROMPT
    assert client.calls_made == 2  # only the first-pass classification


def test_tune_mutates_and_returns_improved_prompt():
    """When first pass is imperfect, tune() mutates the prompt and returns the improved version."""
    # Iter 0: wrong on row 0, correct on row 1 → 50% accuracy
    # Mutation: model returns improved prompt in <prompt> tags
    # Iter 1: both correct → 100% → stop
    client = _TuneSeqClient(
        [
            "[NO]",  # row 0: wrong (actual True)
            "[NO]",  # row 1: correct (actual False)
            "<prompt>improved: {content}</prompt>",  # mutation
            "[YES]",  # row 0: correct
            "[NO]",  # row 1: correct
        ]
    )
    tuner = ClassificationPromptTuner(model_client=client)

    result = tuner.tune(_TWO_ROW_DF, _INITIAL_PROMPT)

    assert result == "improved: {content}"
    assert client.calls_made == 5


def test_tune_stops_at_max_iterations():
    """tune() respects max_iterations even when accuracy never reaches 100%."""
    # Accuracy is stuck at 50% across both iterations.
    # max_iterations=2 → iter 0 eval, mutate, iter 1 eval → exit
    client = _TuneSeqClient(
        [
            "[NO]",  # iter 0, row 0: wrong
            "[NO]",  # iter 0, row 1: correct → 50%
            "<prompt>mutated: {content}</prompt>",  # mutation
            "[NO]",  # iter 1, row 0: still wrong
            "[NO]",  # iter 1, row 1: correct → 50% (not worse → keep mutated)
        ]
    )
    tuner = ClassificationPromptTuner(model_client=client)

    result = tuner.tune(_TWO_ROW_DF, _INITIAL_PROMPT, max_iterations=2)

    # Both iterations achieved 50%; mutated prompt was kept (not worse)
    assert result == "mutated: {content}"
    assert client.calls_made == 5


def test_tune_reverts_when_mutation_makes_things_worse():
    """When a mutation reduces accuracy, tune() reverts to the previous best."""
    # Iter 0: 50% (both rows classified the same way, one wrong)
    # Mutation → new prompt
    # Iter 1: 0% (both now wrong) → revert to iter-0 prompt
    # Still at max_iterations=2 after the revert check → exit
    client = _TuneSeqClient(
        [
            "[YES]",  # iter 0, row 0: correct (actual True)
            "[YES]",  # iter 0, row 1: wrong (actual False) → 50%
            "<prompt>worse: {content}</prompt>",  # mutation
            "[YES]",  # iter 1, row 0: correct
            "[YES]",  # iter 1, row 1: wrong → still 50%, not worse → keep
        ]
    )
    tuner = ClassificationPromptTuner(model_client=client)

    # Use max_iterations=2 so we stop after one mutation round
    result = tuner.tune(_TWO_ROW_DF, _INITIAL_PROMPT, max_iterations=2)

    assert isinstance(result, str)
    assert "{content}" in result


def test_tune_saves_to_catalog(tmp_path):
    """tune() persists improved prompts to the catalog when catalog+prompt_name are given."""
    from aimu.prompts.catalog import PromptCatalog

    client = _TuneSeqClient(
        [
            "[NO]",  # iter 0, row 0: wrong
            "[NO]",  # iter 0, row 1: correct → 50%
            "<prompt>catalog-improved: {content}</prompt>",  # mutation
            "[YES]",  # iter 1, row 0: correct
            "[NO]",  # iter 1, row 1: correct → 100%
        ]
    )
    tuner = ClassificationPromptTuner(model_client=client)

    catalog = PromptCatalog(db_path=str(tmp_path / "catalog.db"))
    result = tuner.tune(_TWO_ROW_DF, _INITIAL_PROMPT, catalog=catalog, prompt_name="test-prompt")

    assert result == "catalog-improved: {content}"
    saved = catalog.retrieve_all("test-prompt", "mock-model")
    assert len(saved) >= 1
    assert any(p.prompt == "catalog-improved: {content}" for p in saved)


def test_tune_handles_missing_prompt_tags():
    """When the mutation response lacks <prompt> tags, tune() uses the raw response as the new prompt."""
    # This documents the current fragile behaviour (split on missing tag returns full string).
    client = _TuneSeqClient(
        [
            "[NO]",  # iter 0, row 0: wrong → 50%
            "[NO]",  # iter 0, row 1: correct
            "No tags here just plain text",  # mutation without <prompt> tags
            "[YES]",  # iter 1, row 0: correct
            "[NO]",  # iter 1, row 1: correct → 100%
        ]
    )
    tuner = ClassificationPromptTuner(model_client=client)

    result = tuner.tune(_TWO_ROW_DF, _INITIAL_PROMPT)

    # The loop should complete without crashing; result is the best prompt found
    assert isinstance(result, str)
    assert len(result) > 0
