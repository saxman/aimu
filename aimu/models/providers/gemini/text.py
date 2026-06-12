import os
from typing import Optional

from dotenv import load_dotenv

from ...base import Model, ModelSpec
from ..openai_compat import OpenAICompatClient

# Google's OpenAI-compatible endpoint; same openai SDK, different base_url + key
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class GeminiModel(Model):
    # Gemini 2.x supports audio input; 1.5 does not
    GEMINI_2_0_FLASH = ModelSpec("gemini-2.0-flash", tools=True, vision=True, audio=True, structured_output=True)
    GEMINI_2_0_FLASH_LITE = ModelSpec("gemini-2.0-flash-lite", tools=True, vision=True, audio=True, structured_output=True)
    GEMINI_1_5_PRO = ModelSpec("gemini-1.5-pro", tools=True, vision=True, structured_output=True)
    GEMINI_1_5_FLASH = ModelSpec("gemini-1.5-flash", tools=True, vision=True, structured_output=True)
    # Thinking models expose reasoning via <think>...</think> tags on this endpoint
    GEMINI_2_5_PRO = ModelSpec("gemini-2.5-pro", tools=True, thinking=True, vision=True, audio=True, structured_output=True)
    GEMINI_2_5_FLASH = ModelSpec("gemini-2.5-flash", tools=True, thinking=True, vision=True, audio=True, structured_output=True)


class GeminiClient(OpenAICompatClient):
    """Client for Google Gemini models via Google's OpenAI-compatible REST API.

    Reads GOOGLE_API_KEY from the environment (or a .env file).
    """

    MODELS = GeminiModel

    def __init__(
        self,
        model: GeminiModel,
        system_message: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
    ):
        load_dotenv()
        api_key = os.environ.get("GOOGLE_API_KEY", "not-set")
        super().__init__(
            model, base_url=GEMINI_BASE_URL, api_key=api_key, system_message=system_message, model_kwargs=model_kwargs
        )
