import asyncio
import json
import re
from typing import Optional, Type, Union

from deepeval.models.base_model import DeepEvalBaseLLM
from pydantic import BaseModel

from aimu.models.base import BaseModelClient

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _parse_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from model output: {text!r}")


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
            return schema.model_validate(_parse_json(result))
        return result

    async def a_generate(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
    ) -> Union[str, BaseModel]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt, schema)
