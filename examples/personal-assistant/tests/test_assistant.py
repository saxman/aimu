"""Mock-only tests for the reference personal assistant."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import AsyncIterator

# tests/helpers_aio.py lives under the repo's tests dir; add it to the path so we can reuse
# the shared async mock client.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))

from _assistant_common import Assistant, AssistantConfig  # noqa: E402
from assistant import build_arg_parser, config_from_args  # noqa: E402
from helpers_aio import MockAsyncModelClient  # noqa: E402

from aimu.aio.channels.base import Channel, ChannelMessage  # noqa: E402
from aimu.models import StreamingContentType  # noqa: E402


class FakeChannel(Channel):
    name = "fake"

    def __init__(self, inbound: list[str] | None = None):
        self._inbound = inbound or []
        self.sent: list[str] = []

    async def receive(self) -> AsyncIterator[ChannelMessage]:
        for text in self._inbound:
            yield ChannelMessage(text=text, sender="fake", channel="fake")

    async def send(self, content, *, reply_to=None) -> None:
        if isinstance(content, str):
            self.sent.append(content)
            return
        parts = []
        async for chunk in content:
            if chunk.phase == StreamingContentType.GENERATING:
                parts.append(chunk.content)
        self.sent.append("".join(parts))


def _config(tmp_path: Path, **overrides) -> AssistantConfig:
    base = {
        "skills_dir": tmp_path / "skills",
        "history_path": str(tmp_path / "history.json"),
    }
    base.update(overrides)
    return AssistantConfig(**base)


def test_arg_parser_defaults():
    args = build_arg_parser().parse_args([])
    assert args.model is None
    assert args.skills_dir is None  # falls back to the output-dir default in config_from_args
    assert args.history is None
    assert args.reminder_seconds is None


def test_default_config_lives_under_output_dir():
    from aimu import paths

    cfg = config_from_args(build_arg_parser().parse_args([]))
    out = paths.output / "personal-assistant"
    assert cfg.skills_dir == out / "skills"
    assert cfg.history_path == str(out / "history.json")


def test_arg_parser_overrides():
    args = build_arg_parser().parse_args(
        [
            "--model",
            "anthropic:claude-sonnet-4-6",
            "--reminder-seconds",
            "5",
            "--skills-dir",
            "/tmp/s",
            "--history",
            "/tmp/h.json",
        ]
    )
    cfg = config_from_args(args)
    assert cfg.model == "anthropic:claude-sonnet-4-6"
    assert cfg.reminder_seconds == 5.0
    assert cfg.skills_dir == Path("/tmp/s")
    assert cfg.history_path == "/tmp/h.json"


async def test_assistant_wires_fixed_builtin_tools(tmp_path):
    assistant = await Assistant.create(_config(tmp_path), FakeChannel(), client=MockAsyncModelClient([]))
    names = {fn.__name__ for fn in assistant._agent.tools}
    # The fixed set is builtin.web + builtin.misc...
    assert {"get_weather", "get_current_date_and_time"} <= names
    # ...not the other groups (fs / compute) or the generative tools.
    assert "read_file" not in names and "calculate" not in names
    assert "generate_image" not in names
    # The assistant's own skill tools are always present.
    assert {"author_skill", "add_skill_script"} <= names


async def test_assistant_handles_message(tmp_path):
    channel = FakeChannel()
    client = MockAsyncModelClient(["Sure, done."])
    assistant = await Assistant.create(_config(tmp_path), channel, client=client)

    await assistant._handle(ChannelMessage(text="do a thing", channel="fake"))

    assert channel.sent == ["Sure, done."]
    assert assistant._conversation.messages  # persisted at least the turn


async def test_assistant_proactive_message(tmp_path):
    channel = FakeChannel()
    client = MockAsyncModelClient(["Don't forget lunch."])
    assistant = await Assistant.create(_config(tmp_path, reminder_text="remind"), channel, client=client)

    await assistant._proactive()

    assert channel.sent == ["Don't forget lunch."]


async def test_assistant_persists_and_restores(tmp_path):
    cfg = _config(tmp_path)

    channel1 = FakeChannel()
    client1 = MockAsyncModelClient(["first reply"])
    assistant1 = await Assistant.create(cfg, channel1, client=client1)
    await assistant1._handle(ChannelMessage(text="remember this"))
    assistant1._conversation.close()  # flush TinyDB

    channel2 = FakeChannel()
    client2 = MockAsyncModelClient([])  # no turn; just restore
    assistant2 = await Assistant.create(cfg, channel2, client=client2)

    restored = [m.get("content") for m in assistant2._agent.model_client.messages]
    assert "remember this" in restored
    assert "first reply" in restored


async def test_assistant_wires_author_skill_tool(tmp_path):
    cfg = _config(tmp_path)
    assistant = await Assistant.create(cfg, FakeChannel(), client=MockAsyncModelClient([]))

    tools = assistant._agent.tools
    author = next((t for t in tools if t.__name__ == "author_skill"), None)
    assert author is not None and author.__tool_is_async__ is True

    await author(name="format-standup", description="Format a standup update.", body="# Standup\n\nDo X.")
    assert (cfg.skills_dir / "format-standup" / "SKILL.md").exists()
    assert "format-standup" in assistant._agent.skill_manager.skills


async def test_assistant_authors_and_registers_runnable_script(tmp_path):
    cfg = _config(tmp_path)
    assistant = await Assistant.create(cfg, FakeChannel(), client=MockAsyncModelClient([]))

    tools = assistant._agent.tools
    author = next(t for t in tools if t.__name__ == "author_skill")
    add_script = next(t for t in tools if t.__name__ == "add_skill_script")
    assert add_script.__tool_is_async__ is True

    await author(name="disk", description="Disk helpers.", body="# Disk")
    msg = await add_script(skill_name="disk", filename="usage.py", content="print('disk ok')\n")

    assert "disk__usage" in msg
    assert (cfg.skills_dir / "disk" / "scripts" / "usage.py").exists()
    # reload_skills() ran, so the new script tool is callable on the live client.
    assert "disk__usage" in [fn.__name__ for fn in assistant._agent.model_client.tools]
