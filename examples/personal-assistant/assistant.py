"""A reference personal-assistant daemon built on AIMU primitives.

Demonstrates the channel + scheduler + skill-authoring primitives wired into a single-user,
always-on assistant (the shape of OpenClaw / Hermes Agent), using the terminal as its
channel. Run it::

    python examples/personal-assistant/assistant.py \\
        --model anthropic:claude-sonnet-4-6 --reminder-seconds 30

Then chat at the prompt. After ~30s a proactive reminder appears. Ask the assistant to
remember a procedure and it will author a reusable skill under ``--skills-dir``.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from _assistant_common import Assistant, AssistantConfig
from aimu.aio import CLIChannel


def build_arg_parser(prog: str = "assistant.py") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="A reference personal-assistant daemon.")
    parser.add_argument(
        "--model",
        default=None,
        help="Model string (e.g. 'anthropic:claude-sonnet-4-6'). Defaults to AIMU_LANGUAGE_MODEL "
        "/ a locally available model.",
    )
    parser.add_argument("--system", default=None, help="Override the assistant's system message.")
    parser.add_argument(
        "--skills-dir",
        default=None,
        help="Directory where authored skills are written and discovered. Default: <output>/personal-assistant/skills.",
    )
    parser.add_argument(
        "--history",
        default=None,
        help="Conversation history database path. Default: <output>/personal-assistant/history.json.",
    )
    parser.add_argument(
        "--reminder-seconds",
        type=float,
        default=None,
        help="If set, send a proactive reminder this many seconds after startup.",
    )
    parser.add_argument(
        "--reminder-text",
        default=None,
        help="Override the prompt used to generate the proactive reminder.",
    )
    parser.add_argument(
        "--show-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the model's reasoning as it streams. Default: on (use --no-show-thinking to hide).",
    )
    parser.add_argument(
        "--show-tools",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show tool calls as they happen. Default: on (use --no-show-tools to hide).",
    )
    parser.add_argument(
        "--tools",
        default="web,fs,compute,misc",
        help="Comma-separated AIMU built-in tool groups to expose: web, fs, compute, misc, image, "
        "audio, speech, transcription (or 'all' / 'none'). Default: web,fs,compute,misc. The "
        "generative groups (image/audio/speech/transcription) require their AIMU_*_MODEL env var.",
    )
    parser.add_argument(
        "--mcp",
        action="append",
        default=None,
        metavar="URL",
        help="Remote MCP server URL whose tools the assistant should use (repeatable). The "
        "assistant can also connect more servers mid-session via the add_mcp_server tool.",
    )
    parser.add_argument(
        "--mcp-bearer",
        default=None,
        help="Bearer token applied to all --mcp servers that require authentication.",
    )
    parser.add_argument(
        "--memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persistent memory across conversations: facts about the user (semantic) plus "
        "user-provided documents. Default: on (use --no-memory to disable).",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> AssistantConfig:
    # Omitted path flags fall back to the AssistantConfig defaults (under the output dir).
    kwargs = {
        "model": args.model,
        "reminder_seconds": args.reminder_seconds,
        "show_thinking": args.show_thinking,
        "show_tools": args.show_tools,
        "tools": [group.strip() for group in args.tools.split(",") if group.strip()],
        "mcp_servers": args.mcp or [],
        "mcp_bearer": args.mcp_bearer,
        "memory": args.memory,
    }
    if args.skills_dir is not None:
        kwargs["skills_dir"] = Path(args.skills_dir)
    if args.history is not None:
        kwargs["history_path"] = args.history
    if args.system is not None:
        kwargs["system_message"] = args.system
    if args.reminder_text is not None:
        kwargs["reminder_text"] = args.reminder_text
    return AssistantConfig(**kwargs)


async def _amain(config: AssistantConfig) -> None:
    print(
        "[notice] This assistant can author and run Python/shell scripts with full access to this "
        "machine (no sandbox), and can connect to remote MCP servers and run whatever tools they "
        "expose. Only use it with a model, inputs, and MCP servers you trust.",
        file=sys.stderr,
    )
    channel = CLIChannel(show_thinking=config.show_thinking, show_tools=config.show_tools)
    assistant = await Assistant.create(config, channel)
    await assistant.run()


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        asyncio.run(_amain(config_from_args(args)))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
