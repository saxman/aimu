from aimu.models.ollama.ollama_client import OllamaClient
from aimu.prompts import ClassificationPromptTuner

import pytest
import pandas as pd


@pytest.fixture(scope="session")
def model_client():
    client = OllamaClient(OllamaClient.MODEL_LLAMA_3_2_3B)
    yield client
    del client


@pytest.fixture(scope="session")
def thinking_model_client():
    client = OllamaClient(OllamaClient.MODEL_QWEN_3_8B)
    yield client
    del client


def test_classify_data(model_client):
    """Test classification with a standard model."""

    prompt_tuner = ClassificationPromptTuner(model_client=model_client)

    classification_prompt = "Is the following content related to large language models?\nWrap your final answer in square braces, e.g. [YES] or [NO]. Only include [YES] or [NO] in your answer.\nContent: {content}"

    df = pd.DataFrame(
        {
            "content": [
                "Large language models are a type of AI.",
                "This is a random sentence.",
                "LLMs can generate text based on prompts.",
                "This sentence is not related to AI.",
            ]
        }
    )

    df = prompt_tuner.classify_data(classification_prompt, df)

    assert "predicted_class" in df.columns
    assert len(df.predicted_class) == 4
    assert df.predicted_class.value_counts().get(True) + df.predicted_class.value_counts().get(False) == 4


def test_classify_data_thinking_model(thinking_model_client):
    """Test classification with a thinking model."""

    test_classify_data(thinking_model_client)


def test_evaluate_results(model_client):
    """Test the evaluation of classification results."""

    prompt_tuner = ClassificationPromptTuner(model_client=model_client)

    df = pd.DataFrame(
        {
            "actual_class": [True, False, True, False],
            "predicted_class": [True, False, True, False],
        }
    )

    metrics = prompt_tuner.evaluate_results(df)

    assert metrics["accuracy"] == 1.0, "Accuracy should be 100%."
    assert metrics["precision"] == 1.0, "Precision should be 100%."
    assert metrics["recall"] == 1.0, "Recall should be 100%."

    df = pd.DataFrame(
        {
            "actual_class": [True, True, False, False],
            "predicted_class": [True, False, True, False],
        }
    )

    metrics = prompt_tuner.evaluate_results(df)

    assert metrics["accuracy"] == 0.5, "Accuracy should be 50%."
    assert metrics["precision"] == 0.5, "Precision should be 50%."
    assert metrics["recall"] == 0.5, "Recall should be 50%."
