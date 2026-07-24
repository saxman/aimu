"""Mock-only tests for native/parse structured output (``chat``/``generate`` ``schema=``).

Covers the schema→JSON-Schema helper, the base auto-escalate logic (native vs parse) via
fake clients, the per-provider request envelopes (OpenAI ``response_format``, Ollama
``format=``, Anthropic forced-tool), coercion, the plain-messages invariant, and the guards
(``schema`` + ``stream``; Anthropic ``schema`` + tools).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from aimu.models._internal.structured import schema_to_json_schema
from aimu.models.base import BaseModelClient, Model, ModelSpec, StreamChunk, StreamingContentType


@dataclass
class Person:
    name: str
    age: int
    nickname: str = ""


# ---------------------------------------------------------------------------
# schema_to_json_schema
# ---------------------------------------------------------------------------


def test_schema_to_json_schema_dataclass():
    js = schema_to_json_schema(Person)
    assert js["type"] == "object"
    assert set(js["properties"]) == {"name", "age", "nickname"}
    assert js["properties"]["age"]["type"] == "integer"
    assert js["required"] == ["name", "age"]  # nickname has a default → optional


def test_schema_to_json_schema_pydantic():
    pydantic = pytest.importorskip("pydantic")

    class PM(pydantic.BaseModel):
        name: str
        age: int

    js = schema_to_json_schema(PM)
    assert set(js["properties"]) == {"name", "age"}


def test_schema_to_json_schema_rejects_other():
    with pytest.raises(TypeError):
        schema_to_json_schema(dict)


# ---------------------------------------------------------------------------
# Base auto-escalate via fake clients
# ---------------------------------------------------------------------------


class _SchemaModel(Model):
    NATIVE = ModelSpec("native", structured_output=True)
    PARSE = ModelSpec("parse", structured_output=False)


class FakeClient(BaseModelClient):
    MODELS = _SchemaModel

    def __init__(self, model):
        super().__init__(model)
        self.seen_response_format = "UNSET"
        self.seen_prompt = None

    def _update_generate_kwargs(self, generate_kwargs=None):
        return generate_kwargs or {}

    @staticmethod
    def _fake_stream():
        """Emit a THINKING chunk then the JSON answer split across GENERATING chunks."""
        yield StreamChunk(StreamingContentType.THINKING, "reasoning...")
        yield StreamChunk(StreamingContentType.GENERATING, '{"name": "Ada",')
        yield StreamChunk(StreamingContentType.GENERATING, ' "age": 36}')

    def _generate(self, prompt, generate_kwargs=None, stream=False, images=None, audio=None, response_format=None):
        self.seen_response_format = response_format
        self.seen_prompt = prompt
        if stream:
            return self._fake_stream()
        return '{"name": "Ada", "age": 36}'

    def _chat(
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
        if stream:
            self.messages.append({"role": "assistant", "content": '{"name": "Ada", "age": 36}'})
            return self._fake_stream()
        self.messages.append({"role": "assistant", "content": '{"name": "Ada", "age": 36}'})
        return '{"name": "Ada", "age": 36}'


def test_native_passes_response_format_and_keeps_prompt_clean():
    client = FakeClient(_SchemaModel.NATIVE)
    out = client.generate("Extract: Ada, 36", schema=Person)
    assert isinstance(out, Person) and out.name == "Ada" and out.age == 36
    assert client.seen_response_format is not None  # JSON schema threaded to provider
    assert client.seen_prompt == "Extract: Ada, 36"  # no prompt injection on native path


def test_parse_path_injects_instruction_and_no_response_format():
    client = FakeClient(_SchemaModel.PARSE)
    out = client.chat("Extract: Ada, 36", schema=Person)
    assert isinstance(out, Person)
    assert client.seen_response_format is None
    assert "JSON Schema" in client.seen_prompt


def test_messages_stay_plain_strings_after_schema_chat():
    client = FakeClient(_SchemaModel.PARSE)
    client.chat("Extract: Ada, 36", schema=Person)
    assert all(isinstance(m["content"], str) for m in client.messages)


def test_schema_stream_yields_chunks_and_terminal_result():
    client = FakeClient(_SchemaModel.NATIVE)
    chunks = list(client.chat("x", schema=Person, stream=True))
    phases = [c.phase for c in chunks]
    assert StreamingContentType.THINKING in phases
    assert phases[-1] == StreamingContentType.DONE
    result = chunks[-1].content["result"]
    assert isinstance(result, Person) and result.name == "Ada" and result.age == 36
    # Also exposed on the attribute (populated once the stream is consumed).
    assert client.last_structured == result
    # Native path threaded the JSON schema to the provider.
    assert client.seen_response_format is not None


def test_generate_schema_stream_yields_terminal_result():
    client = FakeClient(_SchemaModel.PARSE)
    chunks = list(client.generate("x", schema=Person, stream=True))
    assert chunks[-1].is_done()
    assert isinstance(chunks[-1].content["result"], Person)
    assert client.last_structured == chunks[-1].content["result"]
    # Parse path: no response_format, schema appended to prompt.
    assert client.seen_response_format is None
    assert "JSON Schema" in client.seen_prompt


def test_schema_stream_include_thinking_still_delivers_result():
    client = FakeClient(_SchemaModel.NATIVE)
    chunks = list(client.chat("x", schema=Person, stream=True, include=["thinking"]))
    phases = [c.phase for c in chunks]
    assert StreamingContentType.GENERATING not in phases  # filtered out
    assert StreamingContentType.THINKING in phases
    assert chunks[-1].is_done()  # terminal result never filtered
    assert isinstance(chunks[-1].content["result"], Person)


def test_reset_clears_last_structured():
    client = FakeClient(_SchemaModel.NATIVE)
    list(client.chat("x", schema=Person, stream=True))
    assert client.last_structured is not None
    client.reset()
    assert client.last_structured is None


# ---------------------------------------------------------------------------
# OpenAI-compat envelope
# ---------------------------------------------------------------------------


def test_openai_response_format_envelope():
    from aimu.models.providers.openai_compat import OpenAICompatClient

    js = schema_to_json_schema(Person)
    gk = OpenAICompatClient._with_response_format({"max_tokens": 10}, js)
    rf = gk["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "Person"
    assert rf["json_schema"]["schema"] == js
    assert rf["json_schema"]["strict"] is False
    # None response_format is a no-op
    assert OpenAICompatClient._with_response_format({"max_tokens": 10}, None) == {"max_tokens": 10}


# ---------------------------------------------------------------------------
# Ollama format= threading
# ---------------------------------------------------------------------------


def test_ollama_threads_format(monkeypatch):
    pytest.importorskip("ollama")
    from aimu.models.providers import ollama as ollama_mod
    from aimu.models.providers.ollama import OllamaClient, OllamaModel

    captured = {}

    class FakeClient:
        def __init__(self, **kw):
            pass

        def pull(self, *a, **k):
            return None

        def generate(self, **kwargs):
            captured.update(kwargs)
            return {"response": '{"name": "Ada", "age": 36}'}

    monkeypatch.setattr(ollama_mod.ollama, "Client", FakeClient)

    client = OllamaClient(OllamaModel.LLAMA_3_2_3B)  # non-thinking → dict response is enough
    out = client.generate("Extract: Ada, 36", schema=Person)
    assert isinstance(out, Person)
    assert captured["format"] == schema_to_json_schema(Person)  # native format= envelope


def test_ollama_threads_format_when_streaming(monkeypatch):
    pytest.importorskip("ollama")
    from aimu.models.providers import ollama as ollama_mod
    from aimu.models.providers.ollama import OllamaClient, OllamaModel

    captured = {}

    class FakeStreamPart(dict):
        thinking = None

    class FakeClient:
        def __init__(self, **kw):
            pass

        def pull(self, *a, **k):
            return None

        def generate(self, **kwargs):
            captured.update(kwargs)
            assert kwargs.get("stream") is True
            part = FakeStreamPart(response='{"name": "Ada", "age": 36}', done=True)
            return iter([part])

    monkeypatch.setattr(ollama_mod.ollama, "Client", FakeClient)

    client = OllamaClient(OllamaModel.LLAMA_3_2_3B)
    chunks = list(client.generate("Extract: Ada, 36", schema=Person, stream=True))
    assert captured["format"] == schema_to_json_schema(Person)  # format threaded into the streamed call
    assert chunks[-1].is_done()
    assert isinstance(chunks[-1].content["result"], Person)


# ---------------------------------------------------------------------------
# Anthropic forced-tool + tools conflict
# ---------------------------------------------------------------------------


def _bare_anthropic(model, tools=None):
    """Build an AnthropicClient without running __init__ (no SDK/key needed)."""
    from aimu.models.providers.anthropic import AnthropicClient

    client = object.__new__(AnthropicClient)
    client.model = model
    client.model_kwargs = None
    client._system_message = None
    client.default_generate_kwargs = {}
    client.messages = []
    client.tools = tools or []
    client.last_thinking = ""
    client.last_usage = None
    client.last_structured = None
    client.concurrent_tool_calls = False
    return client


def test_anthropic_structured_forces_tool():
    from aimu.models import HAS_ANTHROPIC

    if not HAS_ANTHROPIC:
        pytest.skip("anthropic not installed")
    from aimu.models import AnthropicModel

    client = _bare_anthropic(AnthropicModel.CLAUDE_SONNET_4_6)
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="tool_use", input={"name": "Ada", "age": 36})],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )

    client._client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))
    text = client._structured_call("sys", [{"role": "user", "content": "hi"}], {}, schema_to_json_schema(Person))

    assert captured["tool_choice"] == {"type": "tool", "name": "Person"}
    assert captured["tools"][0]["input_schema"] == schema_to_json_schema(Person)
    assert text == '{"name": "Ada", "age": 36}'


def test_anthropic_structured_streamed_yields_json_no_thinking():
    from aimu.models import HAS_ANTHROPIC

    if not HAS_ANTHROPIC:
        pytest.skip("anthropic not installed")
    from aimu.models import AnthropicModel

    client = _bare_anthropic(AnthropicModel.CLAUDE_SONNET_4_6)

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def __iter__(self):
            # input_json_delta events carry the tool-input JSON as it is built.
            yield SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="input_json_delta", partial_json='{"name": "Ada",'),
            )
            yield SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="input_json_delta", partial_json=' "age": 36}'),
            )

        def get_final_message(self):
            return SimpleNamespace(
                content=[SimpleNamespace(type="tool_use", input={"name": "Ada", "age": 36})],
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            )

    client._client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kw: FakeStream()))
    chunks = list(
        client._structured_call_streamed(
            "sys", [{"role": "user", "content": "hi"}], {}, schema_to_json_schema(Person), append_message=True
        )
    )
    phases = [c.phase for c in chunks]
    assert StreamingContentType.THINKING not in phases  # forced tool ⊥ thinking
    assert all(c.phase == StreamingContentType.GENERATING for c in chunks)
    assert "".join(c.content for c in chunks) == '{"name": "Ada", "age": 36}'
    # Assistant turn stored for the stateful chat path (minus the inert append-time timestamp).
    stored = {key: value for key, value in client.messages[-1].items() if key != "timestamp"}
    assert stored == {"role": "assistant", "content": '{"name": "Ada", "age": 36}'}


def test_anthropic_schema_plus_tools_raises():
    from aimu.models import HAS_ANTHROPIC

    if not HAS_ANTHROPIC:
        pytest.skip("anthropic not installed")
    from aimu.models import AnthropicModel

    def some_tool():
        """A tool."""

    client = _bare_anthropic(AnthropicModel.CLAUDE_SONNET_4_6, tools=[some_tool])
    with pytest.raises(ValueError, match="incompatible with active"):
        client._chat("hi", response_format=schema_to_json_schema(Person), use_tools=True)
