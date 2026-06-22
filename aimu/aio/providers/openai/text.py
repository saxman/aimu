"""Async OpenAI cloud text client (GPT / o-series).

Mirrors ``aimu.models.providers.openai.text.OpenAIClient`` using ``openai.AsyncOpenAI``
(via the shared :class:`AsyncOpenAICompatClient` base).
"""

from __future__ import annotations

import os
from typing import Any, Optional

from aimu.models.providers.openai.text import OpenAIModel

from ..openai_compat import AsyncOpenAICompatClient


class AsyncOpenAIClient(AsyncOpenAICompatClient):
    """OpenAI cloud API."""

    MODELS = OpenAIModel

    def __init__(
        self,
        model: OpenAIModel,
        base_url: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
        system_message: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        from dotenv import load_dotenv

        load_dotenv()
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "not-needed"),
            system_message=system_message,
            model_kwargs=model_kwargs,
            timeout=timeout,
            max_retries=max_retries,
        )

    def _update_generate_kwargs(self, generate_kwargs: Optional[dict[str, Any]] = None) -> dict:
        kwargs = super()._update_generate_kwargs(generate_kwargs)
        # o-series models: rename max_tokens, force temperature=1.
        model_id = self.model.value
        if any(model_id.startswith(prefix) for prefix in ("o1", "o3", "o4")):
            if "max_tokens" in kwargs:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
            kwargs["temperature"] = 1
        return kwargs
