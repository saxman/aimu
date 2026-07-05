"""Mock-only async structured-output tests (mirrors the sync base-logic tests)."""

from __future__ import annotations

from dataclasses import dataclass


from aimu.aio._base import AsyncBaseModelClient
from aimu.models.base import Model, ModelSpec, StreamChunk, StreamingContentType


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

    @staticmethod
    async def _fake_stream():
        yield StreamChunk(StreamingContentType.THINKING, "reasoning...")
        yield StreamChunk(StreamingContentType.GENERATING, '{"name": "Ada",')
        yield StreamChunk(StreamingContentType.GENERATING, ' "age": 36}')

    async def _generate(
        self, prompt, generate_kwargs=None, stream=False, images=None, audio=None, response_format=None
    ):
        self.seen_response_format = response_format
        self.seen_prompt = prompt
        if stream:
            return self._fake_stream()
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
        if stream:
            return self._fake_stream()
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


async def test_async_schema_stream_yields_chunks_and_terminal_result():
    client = FakeAsyncClient(_SchemaModel.NATIVE)
    chunks = [c async for c in await client.chat("x", schema=Person, stream=True)]
    phases = [c.phase for c in chunks]
    assert StreamingContentType.THINKING in phases
    assert chunks[-1].is_done()
    result = chunks[-1].content["result"]
    assert isinstance(result, Person) and result.age == 36
    assert client.last_structured == result


async def test_async_generate_schema_stream_result():
    client = FakeAsyncClient(_SchemaModel.PARSE)
    chunks = [c async for c in await client.generate("x", schema=Person, stream=True)]
    assert chunks[-1].is_done()
    assert isinstance(chunks[-1].content["result"], Person)
    assert client.last_structured == chunks[-1].content["result"]


async def test_async_schema_stream_include_thinking_still_delivers_result():
    client = FakeAsyncClient(_SchemaModel.NATIVE)
    chunks = [c async for c in await client.chat("x", schema=Person, stream=True, include=["thinking"])]
    phases = [c.phase for c in chunks]
    assert StreamingContentType.GENERATING not in phases
    assert StreamingContentType.THINKING in phases
    assert chunks[-1].is_done()


async def test_async_reset_clears_last_structured():
    client = FakeAsyncClient(_SchemaModel.NATIVE)
    [c async for c in await client.chat("x", schema=Person, stream=True)]
    assert client.last_structured is not None
    client.reset()
    assert client.last_structured is None
