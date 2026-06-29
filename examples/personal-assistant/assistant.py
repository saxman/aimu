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

# Tools that require a y/n confirmation in the terminal before they run. add_skill_script writes and
# runs code with full machine access, so it is gated by default. To change this, edit this set (e.g.
# add the script-run "{skill}__{stem}" tools, or empty it), or pass tool_approval=None in _amain.
CONFIRM_BEFORE = {"add_skill_script"}


async def _confirm_in_terminal(name: str, arguments: dict) -> bool:
    """Approval policy: prompt the user in the terminal before a gated tool runs.

    Safe alongside ``CLIChannel.receive()``: during a tool call the serve loop is inside
    ``agent.run()``, so the channel is not reading stdin and this prompt has it to itself.
    """
    if name not in CONFIRM_BEFORE:
        return True
    answer = await asyncio.to_thread(input, f"\n[approve] Run tool '{name}' with {arguments}? [y/N] ")
    return answer.strip().lower() in ("y", "yes")


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
    kwargs = {
        "model": args.model,
        "reminder_seconds": args.reminder_seconds,
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
        "machine (no sandbox). Only use it with a model and inputs you trust. Authoring a script "
        "(add_skill_script) asks for y/n confirmation in the terminal first (see CONFIRM_BEFORE).",
        file=sys.stderr,
    )
    channel = CLIChannel(show_thinking=config.show_thinking, show_tools=config.show_tools)
    assistant = await Assistant.create(config, channel, tool_approval=_confirm_in_terminal)
    await assistant.run()


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        asyncio.run(_amain(config_from_args(args)))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
