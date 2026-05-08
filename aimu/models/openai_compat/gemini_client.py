import os
from typing import Optional

from dotenv import load_dotenv

from ..base import Model
from .openai_compat_client import OpenAICompatClient

# Google's OpenAI-compatible endpoint; same openai SDK, different base_url + key
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class GeminiModel(Model):
    def __init__(self, value, supports_tools=False, supports_thinking=False, supports_vision=False):
        super().__init__(value, supports_tools, supports_thinking, supports_vision)

    GEMINI_2_0_FLASH = ("gemini-2.0-flash", True, False, True)
    GEMINI_2_0_FLASH_LITE = ("gemini-2.0-flash-lite", True, False, True)
    GEMINI_1_5_PRO = ("gemini-1.5-pro", True, False, True)
    GEMINI_1_5_FLASH = ("gemini-1.5-flash", True, False, True)
    # Thinking models expose reasoning via <think>...</think> tags on this endpoint
    GEMINI_2_5_PRO = ("gemini-2.5-pro", True, True, True)
    GEMINI_2_5_FLASH = ("gemini-2.5-flash", True, True, True)


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
