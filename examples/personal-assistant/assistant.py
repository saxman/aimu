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
    return parser


def config_from_args(args: argparse.Namespace) -> AssistantConfig:
    # Omitted path flags fall back to the AssistantConfig defaults (under the output dir).
    kwargs = {"model": args.model, "reminder_seconds": args.reminder_seconds}
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
    channel = CLIChannel()
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
