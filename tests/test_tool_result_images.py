"""ToolResult: a tool can hand the model an image, not just a sentence about one (mock-only).

Covers the type itself, both tool loops on every dispatch path that builds a result message (a plain
tool, an async tool, a sync and an async streaming tool, concurrent dispatch), the text-only fallback
for a model without vision, and each provider adapter that has to reshape a tool message carrying
images: Anthropic, Ollama, and the OpenAI-format servers (which also serve Gemini, llama.cpp, and HF).
"""

from __future__ import annotations

import base64

import pytest

from aimu.agents._tool_loop import _ToolLoop
from aimu.aio._tool_loop import _AsyncToolLoop
from aimu.models import StreamChunk, StreamingContentType
from aimu.models._internal.image_input import _adapt_messages_for_ollama, _hoist_tool_images
from aimu.tools import ToolResult, tool
from aimu.tools.result import tool_message_text
from helpers import MockModelClient
from helpers_aio import MockAsyncModelClient

PNG = b"\x89PNG\r\n\x1a\nfake"
PNG_URL = "data:image/png;base64," + base64.b64encode(PNG).decode("ascii")


@pytest.fixture
def image_file(tmp_path):
    path = tmp_path / "giraffe.png"
    path.write_bytes(PNG)
    return path


# --------------------------------------------------------------------------- #
# The type                                                                    #
# --------------------------------------------------------------------------- #


def test_str_is_the_text_so_stringifying_code_keeps_working():
    assert str(ToolResult("made it", images=[PNG])) == "made it"


def test_images_normalize_at_construction_to_data_urls(image_file):
    result = ToolResult("made it", images=[image_file, str(image_file), PNG, PNG_URL])
    assert result.images == (PNG_URL,) * 4


def test_a_missing_image_file_raises_inside_the_tool(tmp_path):
    with pytest.raises(FileNotFoundError):
        ToolResult("made it", images=[tmp_path / "nope.png"])


def test_content_blocks_are_the_canonical_user_image_shape():
    assert ToolResult("made it", images=[PNG]).content_blocks() == [
        {"type": "text", "text": "made it"},
        {"type": "image_url", "image_url": {"url": PNG_URL}},
    ]


def test_tool_message_text_reads_both_content_shapes():
    assert tool_message_text("plain") == "plain"
    assert tool_message_text(ToolResult("made it", images=[PNG]).content_blocks()) == "made it"


# --------------------------------------------------------------------------- #
# The loops                                                                   #
# --------------------------------------------------------------------------- #


@tool
def draw(subject: str):
    """A plain tool returning an image."""
    return ToolResult(f"drew {subject}", images=[PNG])


@tool
async def draw_async(subject: str):
    """An async tool returning an image."""
    return ToolResult(f"drew {subject}", images=[PNG])


@tool
def draw_streamed(subject: str):
    """A sync streaming tool whose final chunk carries the result, as an image client's does."""
    yield StreamChunk(
        StreamingContentType.IMAGE_GENERATING, {"final": True, "result": ToolResult(f"drew {subject}", images=[PNG])}
    )


@tool
async def draw_streamed_async(subject: str):
    """An async streaming tool whose final chunk carries the result."""
    yield StreamChunk(
        StreamingContentType.IMAGE_GENERATING, {"final": True, "result": ToolResult(f"drew {subject}", images=[PNG])}
    )


EXPECTED_BLOCKS = [
    {"type": "text", "text": "drew a giraffe"},
    {"type": "image_url", "image_url": {"url": PNG_URL}},
]


def _client(cls, vision: bool):
    client = cls([])
    client.model.supports_vision = vision
    return client


def _stage(client, names):
    client.messages.append(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": name, "arguments": {"subject": "a giraffe"}},
                    "id": f"call-{i}",
                }
                for i, name in enumerate(names)
            ],
        }
    )


def _tool_messages(client):
    return [m for m in client.messages if m["role"] == "tool"]


@pytest.mark.parametrize("tools", [[draw], [draw_streamed]])
def test_sync_streamed_dispatch_stores_blocks_and_streams_text(tools):
    client = _client(MockModelClient, vision=True)
    _stage(client, [tools[0].__name__])
    chunks = list(_ToolLoop(client, tools)._dispatch_streamed(1))
    assert _tool_messages(client)[0]["content"] == EXPECTED_BLOCKS
    called = [c for c in chunks if c.phase == StreamingContentType.TOOL_CALLING]
    assert called[0].content["response"] == "drew a giraffe"


def test_sync_plain_dispatch_stores_blocks():
    client = _client(MockModelClient, vision=True)
    _stage(client, ["draw"])
    _ToolLoop(client, [draw])._dispatch(1)
    assert _tool_messages(client)[0]["content"] == EXPECTED_BLOCKS


def test_sync_concurrent_dispatch_streams_text_for_each_call():
    client = _client(MockModelClient, vision=True)
    _stage(client, ["draw", "draw"])
    chunks = list(_ToolLoop(client, [draw], concurrent_tool_calls=True)._dispatch_streamed(1))
    assert [m["content"] for m in _tool_messages(client)] == [EXPECTED_BLOCKS, EXPECTED_BLOCKS]
    assert [c.content["response"] for c in chunks if c.phase == StreamingContentType.TOOL_CALLING] == [
        "drew a giraffe",
        "drew a giraffe",
    ]


@pytest.mark.parametrize("tools", [[draw], [draw_async], [draw_streamed], [draw_streamed_async]])
async def test_async_streamed_dispatch_stores_blocks_and_streams_text(tools):
    client = _client(MockAsyncModelClient, vision=True)
    _stage(client, [tools[0].__name__])
    chunks = [c async for c in _AsyncToolLoop(client, tools)._dispatch_streamed(1)]
    assert _tool_messages(client)[0]["content"] == EXPECTED_BLOCKS
    called = [c for c in chunks if c.phase == StreamingContentType.TOOL_CALLING]
    assert called[0].content["response"] == "drew a giraffe"


@pytest.mark.parametrize("tools", [[draw], [draw_async]])
async def test_async_plain_dispatch_stores_blocks(tools):
    client = _client(MockAsyncModelClient, vision=True)
    _stage(client, [tools[0].__name__])
    await _AsyncToolLoop(client, tools)._dispatch(1)
    assert _tool_messages(client)[0]["content"] == EXPECTED_BLOCKS


def test_a_model_without_vision_is_sent_the_text_alone():
    client = _client(MockModelClient, vision=False)
    _stage(client, ["draw"])
    _ToolLoop(client, [draw])._dispatch(1)
    assert _tool_messages(client)[0]["content"] == "drew a giraffe"


async def test_an_async_model_without_vision_is_sent_the_text_alone():
    client = _client(MockAsyncModelClient, vision=False)
    _stage(client, ["draw_async"])
    await _AsyncToolLoop(client, [draw_async])._dispatch(1)
    assert _tool_messages(client)[0]["content"] == "drew a giraffe"


def test_tool_called_event_carries_the_text():
    events = []
    client = _client(MockModelClient, vision=True)
    _stage(client, ["draw"])
    _ToolLoop(client, [draw], events=events.append)._dispatch(1)
    called = [e for e in events if type(e).__name__ == "ToolCalled"]
    assert called[0].result == "drew a giraffe"


def test_a_tool_that_fails_building_its_result_reports_the_error(tmp_path):
    @tool
    def broken(subject: str):
        """Returns an image that is not there."""
        return ToolResult("drew it", images=[tmp_path / "missing.png"])

    client = _client(MockModelClient, vision=True)
    _stage(client, ["broken"])
    _ToolLoop(client, [broken])._dispatch(1)
    assert "raised an error" in _tool_messages(client)[0]["content"]


# --------------------------------------------------------------------------- #
# The provider adapters                                                       #
# --------------------------------------------------------------------------- #


def _transcript(*tool_contents):
    """An assistant tool-call turn followed by one tool message per content."""
    calls = [
        {"type": "function", "function": {"name": f"tool{i}", "arguments": "{}"}, "id": f"call-{i}"}
        for i in range(len(tool_contents))
    ]
    return [
        {"role": "user", "content": "draw a giraffe"},
        {"role": "assistant", "content": "", "tool_calls": calls},
        *(
            {"role": "tool", "name": f"tool{i}", "content": content, "tool_call_id": f"call-{i}"}
            for i, content in enumerate(tool_contents)
        ),
    ]


def test_hoist_leaves_a_list_without_tool_images_untouched():
    messages = _transcript("plain")
    assert _hoist_tool_images(messages) is messages


def test_hoist_moves_images_into_one_user_message_after_the_run():
    messages = _transcript(EXPECTED_BLOCKS, "plain", EXPECTED_BLOCKS) + [{"role": "assistant", "content": "done"}]
    hoisted = _hoist_tool_images(messages)
    assert [m["role"] for m in hoisted] == ["user", "assistant", "tool", "tool", "tool", "user", "assistant"]
    assert [m["content"] for m in hoisted[2:5]] == ["drew a giraffe", "plain", "drew a giraffe"]
    assert hoisted[5]["content"] == [
        {"type": "text", "text": "Returned by the tool0 call above:"},
        {"type": "image_url", "image_url": {"url": PNG_URL}},
        {"type": "text", "text": "Returned by the tool2 call above:"},
        {"type": "image_url", "image_url": {"url": PNG_URL}},
    ]
    assert messages[2]["content"] == EXPECTED_BLOCKS  # the stored history is not mutated


def test_hoist_appends_the_user_message_when_the_run_ends_the_list():
    hoisted = _hoist_tool_images(_transcript(EXPECTED_BLOCKS))
    assert hoisted[-1]["role"] == "user"
    assert hoisted[-2] == {"role": "tool", "name": "tool0", "content": "drew a giraffe", "tool_call_id": "call-0"}


def test_ollama_puts_tool_images_on_the_tool_message():
    adapted = _adapt_messages_for_ollama(_transcript(EXPECTED_BLOCKS))
    assert adapted[-1]["content"] == "drew a giraffe"
    assert adapted[-1]["images"] == [base64.b64encode(PNG).decode("ascii")]


def test_anthropic_puts_tool_images_inside_the_tool_result():
    pytest.importorskip("anthropic")
    from aimu.models.providers.anthropic import AnthropicClient

    blank_text = [{"type": "text", "text": ""}, EXPECTED_BLOCKS[1]]
    _, ant = AnthropicClient._openai_messages_to_anthropic(
        object.__new__(AnthropicClient), _transcript(EXPECTED_BLOCKS, blank_text, "plain")
    )
    results = ant[-1]["content"]
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_URL.split(",", 1)[1]}}
    assert results[0]["content"] == [{"type": "text", "text": "drew a giraffe"}, image]
    assert results[1]["content"] == [image]  # an empty text block is one Anthropic rejects
    assert results[2]["content"] == "plain"


def test_openai_compat_request_hoists_tool_images():
    pytest.importorskip("openai")
    from unittest.mock import MagicMock

    from aimu.models.providers.openai_compat import LMStudioOpenAIClient, LMStudioOpenAIModel

    client = LMStudioOpenAIClient(list(LMStudioOpenAIModel)[0])
    client.messages = _transcript(EXPECTED_BLOCKS)
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        message = MagicMock(tool_calls=None, content="a giraffe, as asked")
        return MagicMock(choices=[MagicMock(message=message)])

    client._client = MagicMock()
    client._client.chat.completions.create = fake_create
    client._chat("how did it come out?", use_tools=False)

    roles = [m["role"] for m in captured["messages"]]
    tool_index = roles.index("tool")
    assert captured["messages"][tool_index]["content"] == "drew a giraffe"
    assert captured["messages"][tool_index + 1]["content"][1] == EXPECTED_BLOCKS[1]
    assert client.messages[tool_index]["content"] == EXPECTED_BLOCKS  # history keeps the canonical shape
