import os
from typing import Optional

from dotenv import load_dotenv

from ...base import Model, ModelSpec
from ..openai_compat import CLOUD_OPENAI_GENERATE_KWARGS, OpenAICompatClient

OPENAI_BASE_URL = "https://api.openai.com/v1"

# o-series models reject max_tokens and require max_completion_tokens + temperature=1
_O_SERIES_PREFIXES = ("o1", "o3", "o4")


class OpenAIModel(Model):
    # Standard GPT models: GPT-4o and GPT-4.1 accept audio via input_audio blocks
    GPT_4O_MINI = ModelSpec("gpt-4o-mini", tools=True, vision=True, audio=True, structured_output=True)
    GPT_4O = ModelSpec("gpt-4o", tools=True, vision=True, audio=True, structured_output=True)
    GPT_4_1 = ModelSpec("gpt-4.1", tools=True, vision=True, audio=True, structured_output=True)
    GPT_4_1_MINI = ModelSpec("gpt-4.1-mini", tools=True, vision=True, audio=True, structured_output=True)
    GPT_4_1_NANO = ModelSpec("gpt-4.1-nano", tools=True, vision=True, audio=True, structured_output=True)
    # o-series reasoning models: reasoning tokens not accessible as text chunks,
    # so thinking=False; pass reasoning_effort via generate_kwargs if needed
    O4_MINI = ModelSpec("o4-mini", tools=True, vision=True, structured_output=True)
    O3 = ModelSpec("o3", tools=True, vision=True, structured_output=True)
    O3_MINI = ModelSpec("o3-mini", tools=True, structured_output=True)


class OpenAIClient(OpenAICompatClient):
    """Client for the OpenAI API (GPT and o-series models).

    Reads OPENAI_API_KEY from the environment (or a .env file).
    """

    MODELS = OpenAIModel

    GENERATE_KWARG_SUPPORT = CLOUD_OPENAI_GENERATE_KWARGS

    # OpenAI's endpoint has no chat-template convention; ``enable_thinking`` is a Qwen/vLLM
    # concept and would be rejected or silently ignored here.
    _SUPPORTS_CHAT_TEMPLATE_KWARGS = False

    def __init__(
        self,
        model: OpenAIModel,
        system_message: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY", "not-set")
        super().__init__(
            model,
            base_url=OPENAI_BASE_URL,
            api_key=api_key,
            system_message=system_message,
            model_kwargs=model_kwargs,
            timeout=timeout,
            max_retries=max_retries,
        )

    def _rewrite_generate_kwargs(self, kwargs: dict) -> dict:
        kwargs = super()._rewrite_generate_kwargs(kwargs)
        if any(self.model.value.startswith(p) for p in _O_SERIES_PREFIXES):
            # o-series requires max_completion_tokens instead of max_tokens,
            # and temperature must be 1 (no sampling control)
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens", 1024)
            kwargs["temperature"] = 1
            kwargs.pop("top_p", None)
        return kwargs
