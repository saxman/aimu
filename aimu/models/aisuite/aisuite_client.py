from typing import Any, Iterator, Optional
from ..base_client import Model, ModelClient

import aisuite
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class AisuiteModel(Model):
    GPT_4O_MINI = "openai:gpt-4o-mini"
    GPT_4O = "openai:gpt-4o"

class AisuiteClient(ModelClient):
    MODELS = AisuiteModel

    TOOL_MODELS = [
        MODELS.GPT_4O_MINI,
        MODELS.GPT_4O
    ]

    THINKING_MODELS = [
        MODELS.GPT_4O_MINI,
        MODELS.GPT_4O
    ]

    DEFAULT_GENERATE_KWARGS = {
        "max_new_tokens": 1024,
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 50,
    }

    def __init__(
        self,
        model: AisuiteModel,
        model_kwargs: Optional[dict] = None,
        system_message: Optional[str] = None,
    ):
        super().__init__(model, model_kwargs, system_message)
        self.default_generate_kwargs = self.DEFAULT_GENERATE_KWARGS.copy()

        load_dotenv()

        self.ai_client = aisuite.Client()

    def _update_generate_kwargs(self, generate_kwargs: Optional[dict[str, Any]] = None) -> dict[str, None]:
        if not generate_kwargs:
            generate_kwargs = self.default_generate_kwargs
        
        return generate_kwargs

    def generate(self, prompt: str, generate_kwargs: Optional[dict[str, Any]] = None) -> str:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)

        messages = [{"role": "user", "content": prompt}]

        response = self.ai_client.chat.completions.create(
            self.model.value, messages, temperature=generate_kwargs["temperature"], max_tokens=generate_kwargs["max_new_tokens"]
        )

        return response.choices[0].message.content

    def generate_streamed(self, prompt: str, generate_kwargs: Optional[dict[str, Any]] = None) -> Iterator[str]:
        generate_kwargs = self._update_generate_kwargs(generate_kwargs)
        
        messages = [{"role": "user", "content": prompt}]

        response = self.ai_client.chat.completions.create(
            self.model.value, messages, stream=True, temperature=generate_kwargs["temperature"], max_tokens=generate_kwargs["max_new_tokens"]
        )

        return response

    def chat(
        self, user_message: str, generate_kwargs: Optional[dict[str, Any]] = None, use_tools: Optional[bool] = True
    ) -> str:
        pass

    def chat_streamed(
        self, user_message: str, generate_kwargs: Optional[dict[str, Any]] = None, use_tools: Optional[bool] = True
    ) -> Iterator[str]:
        pass
