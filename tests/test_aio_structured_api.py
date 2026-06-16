"""Mock-only async structured-output tests (mirrors the sync base-logic tests)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from aimu.aio._base import AsyncBaseModelClient
from aimu.models.base import Model, ModelSpec


@dataclass
class Person:
    name: str
    age: int


class _SchemaModel(Model):
    NATIVE = ModelSpec("a-native", structured_output=True)
    PARSE = ModelSpec("a-parse", structured_output=False)


class FakeAsyncClient(AsyncBaseModelClient):
    MODELS = _SchemaModel

    def __init__(self, model):
        super().__init__(model)
        self.seen_response_format = "UNSET"
        self.seen_prompt = None

    def _update_generate_kwargs(self, generate_kwargs=None):
        return generate_kwargs or {}

    async def _generate(
        self, prompt, generate_kwargs=None, stream=False, images=None, audio=None, response_format=None
    ):
        self.seen_response_format = response_format
        self.seen_prompt = prompt
        return '{"name": "Ada", "age": 36}'

    async def _chat(
        self,
        user_message,
        generate_kwargs=None,
        use_tools=True,
        stream=False,
        images=None,
        audio=None,
        response_format=None,
    ):
        self.seen_response_format = response_format
        self.seen_prompt = user_message
        self.messages.append({"role": "assistant", "content": '{"name": "Ada", "age": 36}'})
        return '{"name": "Ada", "age": 36}'


async def test_async_native_passes_response_format():
    client = FakeAsyncClient(_SchemaModel.NATIVE)
    out = await client.generate("Extract: Ada, 36", schema=Person)
    assert isinstance(out, Person) and out.age == 36
    assert client.seen_response_format is not None
    assert client.seen_prompt == "Extract: Ada, 36"


async def test_async_parse_path_injects_instruction():
    client = FakeAsyncClient(_SchemaModel.PARSE)
    out = await client.chat("Extract: Ada, 36", schema=Person)
    assert isinstance(out, Person)
    assert client.seen_response_format is None
    assert "JSON Schema" in client.seen_prompt
    assert all(isinstance(m["content"], str) for m in client.messages)


async def test_async_schema_with_stream_raises():
    client = FakeAsyncClient(_SchemaModel.NATIVE)
    with pytest.raises(ValueError, match="mutually exclusive"):
        await client.chat("x", schema=Person, stream=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        await client.generate("x", schema=Person, stream=True)
