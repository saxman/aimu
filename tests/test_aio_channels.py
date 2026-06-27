"""Tests for the async Channel ABC and CLIChannel adapter."""

from __future__ import annotations

import pytest

from aimu.aio import CLIChannel, Channel, ChannelMessage
from aimu.models import StreamChunk, StreamingContentType


def test_channel_is_abstract():
    with pytest.raises(TypeError):
        Channel()


def test_channel_message_is_plain_data():
    msg = ChannelMessage(text="hi")
    assert msg.text == "hi"
    assert msg.sender is None and msg.channel is None and msg.images is None
    assert msg.metadata == {}
    # metadata is per-instance, not shared.
    other = ChannelMessage(text="yo")
    msg.metadata["a"] = 1
    assert other.metadata == {}


async def test_cli_receive_yields_until_eof(monkeypatch):
    lines = iter(["hello\n", "  \n", "world\n", ""])  # blank line skipped, "" is EOF
    monkeypatch.setattr("sys.stdin.readline", lambda: next(lines))

    channel = CLIChannel()
    received = [m async for m in channel.receive()]

    assert [m.text for m in received] == ["hello", "world"]
    assert all(m.sender == "cli" and m.channel == "cli" for m in received)


async def test_cli_send_string(capsys):
    await CLIChannel().send("done")
    assert "done" in capsys.readouterr().out


async def test_cli_send_stream(capsys):
    async def chunks():
        yield StreamChunk(StreamingContentType.GENERATING, "ab")
        yield StreamChunk(StreamingContentType.THINKING, "ignored")
        yield StreamChunk(StreamingContentType.GENERATING, "cd")

    await CLIChannel().send(chunks())
    out = capsys.readouterr().out
    assert "abcd" in out
    assert "ignored" not in out


async def test_cli_send_ignores_reply_to(capsys):
    msg = ChannelMessage(text="ping", sender="someone")
    await CLIChannel().send("pong", reply_to=msg)  # does not raise
    assert "pong" in capsys.readouterr().out
