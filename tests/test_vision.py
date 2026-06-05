"""
Vision input tests.

Mock-only by default (``pytest tests/test_vision.py``); pass ``--client=<provider>``
to additionally run real-provider smoke tests against vision-capable models.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Iterable

import pytest

from aimu.agents.agent import Agent
from aimu.models import BaseModelClient
from aimu.models._internal.image_input import (
    _adapt_messages_for_ollama,
    _build_user_content_blocks,
    _decode_image_url_to_pil,
    _extract_pil_images,
    _normalize_image,
    _openai_blocks_to_anthropic,
    _parse_data_url,
    _replace_image_url_with_image_placeholder,
)

from helpers import MockModelClient, create_real_model_client, resolve_model_params


# 1×1 transparent PNG (smallest valid PNG)
_TINY_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)
_TINY_PNG_DATA_URL = "data:image/png;base64," + base64.b64encode(_TINY_PNG_BYTES).decode("ascii")


# --------------------------------------------------------------------------- #
# Helper: a vision-capable mock model client                                   #
# --------------------------------------------------------------------------- #


class _VisionMockClient(MockModelClient):
    """MockModelClient with supports_vision=True so _chat_setup accepts images."""

    def __init__(self, responses):
        super().__init__(responses)
        self.model.supports_vision = True

    def chat(self, user_message, generate_kwargs=None, use_tools=True, stream=False, images=None, tools=None):
        # Run the real _chat_setup so we exercise the image-block normalization,
        # then fall back to MockModelClient's canned-response behavior.
        if images:
            # _chat_setup appends a content-block user message; replicate the rest of mock chat.
            self._chat_setup(user_message, generate_kwargs, use_tools, images=images)
            response = self._responses[self._call_count]
            self._call_count += 1
            self.messages.append({"role": "assistant", "content": response})
            return response
        return super().chat(user_message, generate_kwargs, use_tools, stream)


# --------------------------------------------------------------------------- #
# _normalize_image                                                              #
# --------------------------------------------------------------------------- #


def test_normalize_image_passes_through_http_url():
    url = "https://example.com/cat.jpg"
    assert _normalize_image(url) == url


def test_normalize_image_passes_through_data_url():
    assert _normalize_image(_TINY_PNG_DATA_URL) == _TINY_PNG_DATA_URL


def test_normalize_image_encodes_bytes_as_png_data_url():
    out = _normalize_image(_TINY_PNG_BYTES)
    assert out.startswith("data:image/png;base64,")
    mime, data = _parse_data_url(out)
    assert mime == "image/png"
    assert base64.b64decode(data) == _TINY_PNG_BYTES


def test_normalize_image_reads_path_string(tmp_path: Path):
    fp = tmp_path / "tiny.png"
    fp.write_bytes(_TINY_PNG_BYTES)
    out = _normalize_image(str(fp))
    mime, data = _parse_data_url(out)
    assert mime == "image/png"
    assert base64.b64decode(data) == _TINY_PNG_BYTES


def test_normalize_image_reads_pathlib_path(tmp_path: Path):
    fp = tmp_path / "tiny.png"
    fp.write_bytes(_TINY_PNG_BYTES)
    out = _normalize_image(fp)
    assert out.startswith("data:image/png;base64,")


def test_normalize_image_rejects_unknown_type():
    with pytest.raises(TypeError):
        _normalize_image(12345)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# _build_user_content_blocks + _chat_setup                                      #
# --------------------------------------------------------------------------- #


def test_build_user_content_blocks_basic():
    blocks = _build_user_content_blocks("describe", [_TINY_PNG_DATA_URL])
    assert blocks[0] == {"type": "text", "text": "describe"}
    assert blocks[1] == {"type": "image_url", "image_url": {"url": _TINY_PNG_DATA_URL}}


def test_chat_setup_with_images_builds_content_block_list():
    client = _VisionMockClient(["yes"])
    client._chat_setup("describe", images=[_TINY_PNG_DATA_URL])
    user_msg = client.messages[-1]
    assert user_msg["role"] == "user"
    assert isinstance(user_msg["content"], list)
    assert user_msg["content"][0]["type"] == "text"
    assert user_msg["content"][1]["type"] == "image_url"


def test_chat_with_images_runs_through_mock_client():
    client = _VisionMockClient(["it's a cat"])
    response = client.chat("what is this?", images=[_TINY_PNG_DATA_URL])
    assert response == "it's a cat"
    user_msg = client.messages[-2]
    assert isinstance(user_msg["content"], list)


def test_chat_setup_rejects_images_on_non_vision_model():
    client = MockModelClient(["unused"])
    client.model.supports_vision = False
    with pytest.raises(ValueError, match="does not support vision"):
        client._chat_setup("describe", images=[_TINY_PNG_DATA_URL])


def test_chat_setup_without_images_keeps_string_content():
    client = _VisionMockClient(["unused"])
    client._chat_setup("hello")
    assert client.messages[-1]["content"] == "hello"


# --------------------------------------------------------------------------- #
# Anthropic adapter                                                             #
# --------------------------------------------------------------------------- #


def test_openai_blocks_to_anthropic_data_url():
    blocks = [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": _TINY_PNG_DATA_URL}},
    ]
    out = _openai_blocks_to_anthropic(blocks)
    assert out[0] == {"type": "text", "text": "describe"}
    assert out[1]["type"] == "image"
    assert out[1]["source"]["type"] == "base64"
    assert out[1]["source"]["media_type"] == "image/png"
    assert base64.b64decode(out[1]["source"]["data"]) == _TINY_PNG_BYTES


def test_openai_blocks_to_anthropic_http_url():
    url = "https://example.com/cat.jpg"
    blocks = [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": url}},
    ]
    out = _openai_blocks_to_anthropic(blocks)
    assert out[1] == {"type": "image", "source": {"type": "url", "url": url}}


def test_anthropic_client_message_conversion_with_images():
    pytest.importorskip("anthropic")
    from aimu.models.providers.anthropic import AnthropicClient

    client = AnthropicClient.__new__(AnthropicClient)
    messages = [
        {"role": "system", "content": "be helpful"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": _TINY_PNG_DATA_URL}},
            ],
        },
    ]
    system_str, ant = client._openai_messages_to_anthropic(messages)
    assert system_str == "be helpful"
    assert ant[0]["role"] == "user"
    assert ant[0]["content"][0]["type"] == "text"
    assert ant[0]["content"][1]["type"] == "image"
    assert ant[0]["content"][1]["source"]["type"] == "base64"


# --------------------------------------------------------------------------- #
# Ollama adapter                                                                #
# --------------------------------------------------------------------------- #


def test_adapt_messages_for_ollama_extracts_images():
    messages = [
        {"role": "system", "content": "be helpful"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": _TINY_PNG_DATA_URL}},
            ],
        },
    ]
    out = _adapt_messages_for_ollama(messages)
    assert out[0]["content"] == "be helpful"
    assert out[1]["role"] == "user"
    assert out[1]["content"] == "describe"
    assert "images" in out[1]
    assert base64.b64decode(out[1]["images"][0]) == _TINY_PNG_BYTES


def test_adapt_messages_for_ollama_does_not_mutate_input():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": _TINY_PNG_DATA_URL}},
            ],
        }
    ]
    snapshot = messages[0]["content"][:]
    _adapt_messages_for_ollama(messages)
    assert messages[0]["content"] == snapshot
    assert "images" not in messages[0]


def test_adapt_messages_for_ollama_rejects_http_url():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}},
            ],
        }
    ]
    with pytest.raises(ValueError, match="inline base64"):
        _adapt_messages_for_ollama(messages)


# --------------------------------------------------------------------------- #
# HF adapter                                                                    #
# --------------------------------------------------------------------------- #


def test_extract_pil_images_decodes_data_url():
    pytest.importorskip("PIL")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": _TINY_PNG_DATA_URL}},
            ],
        }
    ]
    images = _extract_pil_images(messages)
    assert len(images) == 1
    assert images[0].size == (1, 1)


def test_replace_image_url_with_image_placeholder():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": _TINY_PNG_DATA_URL}},
            ],
        }
    ]
    out = _replace_image_url_with_image_placeholder(messages)
    assert out[0]["content"] == [
        {"type": "text", "text": "describe"},
        {"type": "image"},
    ]
    # Original is not mutated
    assert messages[0]["content"][1]["type"] == "image_url"


def test_decode_image_url_to_pil_data_url():
    pytest.importorskip("PIL")
    img = _decode_image_url_to_pil(_TINY_PNG_DATA_URL)
    assert img.size == (1, 1)


# --------------------------------------------------------------------------- #
# Agentic vision integration                                                    #
# --------------------------------------------------------------------------- #


def test_agent_as_model_client_forwards_images():
    inner = _VisionMockClient(["it's a tiny pixel"])
    agent = Agent(inner, name="vision-agent", max_iterations=2)
    client = agent.as_model_client()

    response = client.chat("what is this?", images=[_TINY_PNG_DATA_URL])
    assert response == "it's a tiny pixel"

    # The inner client's user message should be a content-block list
    user_msg = next(m for m in inner.messages if m["role"] == "user")
    assert isinstance(user_msg["content"], list)
    assert user_msg["content"][1]["type"] == "image_url"


def test_simple_agent_only_attaches_images_to_first_turn():
    """Continuation turns must not re-send the user's images."""
    inner = _VisionMockClient(["intermediate", "done"])
    inner.model.supports_tools = False  # simplify; we drive iterations manually

    captured: list[dict] = []
    real_chat = inner.chat

    def tracking_chat(user_message, generate_kwargs=None, use_tools=True, stream=False, images=None, tools=None):
        captured.append({"user_message": user_message, "images": images})
        return real_chat(user_message, generate_kwargs, use_tools, stream, images)

    inner.chat = tracking_chat  # type: ignore[method-assign]

    # Force two iterations by lying about tool usage in the loop check.
    call_count = {"n": 0}

    def fake_last_turn():
        call_count["n"] += 1
        return call_count["n"] == 1  # True after first turn → triggers continuation

    agent = Agent(inner, name="vision-agent", max_iterations=3)
    agent._last_turn_called_tools = fake_last_turn  # type: ignore[method-assign]

    agent.run("describe", images=[_TINY_PNG_DATA_URL])

    assert len(captured) == 2
    assert captured[0]["images"] == [_TINY_PNG_DATA_URL]
    assert captured[1]["images"] is None


# --------------------------------------------------------------------------- #
# Real-provider smoke tests                                                     #
# --------------------------------------------------------------------------- #


def pytest_generate_tests(metafunc):
    if "vision_model_client" in metafunc.fixturenames:
        params = resolve_model_params(metafunc.config, default_params=[])
        metafunc.parametrize("vision_model_client", params, indirect=True, scope="session")


@pytest.fixture(scope="session")
def vision_model_client(request) -> Iterable[BaseModelClient]:
    model = request.param
    if not getattr(model, "supports_vision", False):
        pytest.skip(f"Model {model.name} does not support vision")
    yield from create_real_model_client(request)


def test_chat_with_image(vision_model_client):
    """Real model: send a 1x1 PNG and confirm we get back a non-empty response."""
    response = vision_model_client.chat(
        "Describe the image in one short sentence.",
        images=[_TINY_PNG_BYTES],
    )
    assert isinstance(response, str)
    assert len(response.strip()) > 0
