import asyncio
from typing import Optional, Type, Union

from deepeval.models.base_model import DeepEvalBaseLLM
from pydantic import BaseModel

from aimu.models._internal.json import parse_json_response
from aimu.models.base import BaseModelClient


class DeepEvalModel(DeepEvalBaseLLM):
    """Wraps any AIMU BaseModelClient for use as a DeepEval judge model."""

    def __init__(self, model_client: BaseModelClient):
        self._client = model_client

    def get_model_name(self) -> str:
        return str(self._client.model.name)

    def load_model(self):
        return self._client

    def generate(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
    ) -> Union[str, BaseModel]:
        result = self._client.generate(prompt)
        if schema is not None:
            return parse_json_response(result, schema)
        return result

    async def a_generate(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
    ) -> Union[str, BaseModel]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt, schema)
