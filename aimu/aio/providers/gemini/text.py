"""Async Google Gemini text client (via the OpenAI-compatible endpoint).

Mirrors ``aimu.models.providers.gemini.text.GeminiClient`` using ``openai.AsyncOpenAI``
(via the shared :class:`AsyncOpenAICompatClient` base).
"""

from __future__ import annotations

import os
from typing import Optional

from aimu.models.providers.gemini.text import GeminiModel

from ..openai_compat import AsyncOpenAICompatClient


class AsyncGeminiClient(AsyncOpenAICompatClient):
    """Google Gemini via OpenAI-compatible endpoint."""

    MODELS = GeminiModel

    def __init__(
        self,
        model: GeminiModel,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key: Optional[str] = None,
        system_message: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
    ):
        from dotenv import load_dotenv

        load_dotenv()
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key or os.environ.get("GOOGLE_API_KEY", "not-needed"),
            system_message=system_message,
            model_kwargs=model_kwargs,
        )
