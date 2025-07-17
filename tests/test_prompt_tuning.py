from aimu.models.ollama.ollama_client import OllamaClient
from aimu.prompts import ClassificationPromptTuner

import pytest
import pandas as pd


@pytest.fixture(scope="session")
def model_client():
    client = OllamaClient(OllamaClient.MODEL_LLAMA_3_2_3B)
    yield client
    del client

def test_classify_data(model_client):
    prompt_tuner = ClassificationPromptTuner(model_client=model_client)

    classification_prompt = "Is the following content related to large language models?\nWrap your final answer in square braces, e.g. [YES] or [NO]. Only include [YES] or [NO] in your answer.\nContent: {content}"

    df = pd.DataFrame({
        "content": [
            "Large language models are a type of AI.",
            "This is a random sentence.",
            "LLMs can generate text based on prompts.",
            "This sentence is not related to AI."
        ]
    })

    df = prompt_tuner.classify_data(classification_prompt, df)

    assert df["predicted_class"].tolist() == ["[YES]", "[NO]", "[YES]", "[NO]"], "Classification results do not match expected values."
