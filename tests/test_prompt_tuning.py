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
