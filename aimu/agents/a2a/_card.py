"""Pure A2A <-> AIMU adapters shared by the sync and async surfaces.

These helpers build an :class:`a2a.types.AgentCard` from any runner, construct a user
``Message`` for an outbound task, and flatten an A2A response/result back to plain text.
They contain no sync/async coupling so both ``aimu.agents.a2a`` and ``aimu.aio.a2a``
reuse them — the agent-level analog of :mod:`aimu.tools.mcp_format`.
"""

from __future__ import annotations

import re
from typing import Any, Optional
from uuid import uuid4

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Message,
    MessageSendParams,
    Part,
    Role,
    SendMessageRequest,
    SendStreamingMessageRequest,
    TextPart,
)

DEFAULT_INPUT_MODES = ["text/plain"]
DEFAULT_OUTPUT_MODES = ["text/plain"]


def runner_name(runner: Any, fallback: str = "aimu-agent") -> str:
    """A safe identifier derived from ``runner.name`` (or ``fallback``)."""
    raw = getattr(runner, "name", None) or fallback
    return re.sub(r"\W+", "_", raw).strip("_") or fallback


def runner_description(runner: Any, name: str) -> str:
    """First line of the runner's ``system_message`` if present, else a generic string."""
    system_message = getattr(runner, "system_message", None)
    if system_message:
        return system_message.splitlines()[0]
    return f"An AIMU runner exposed over A2A: {name}."


def build_agent_card(
    runner: Any,
    *,
    url: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    skills: Optional[list[AgentSkill]] = None,
    streaming: bool = False,
    version: str = "1.0.0",
) -> AgentCard:
    """Construct an :class:`AgentCard` advertising ``runner`` at ``url``.

    ``name`` / ``description`` default to the runner's ``name`` / ``system_message``;
    ``skills`` defaults to a single catch-all skill so the card is valid without callers
    having to enumerate capabilities.
    """
    resolved_name = name or runner_name(runner)
    resolved_description = description or runner_description(runner, resolved_name)
    resolved_skills = skills or [
        AgentSkill(
            id=resolved_name,
            name=resolved_name,
            description=resolved_description,
            tags=["aimu"],
        )
    ]
    return AgentCard(
        name=resolved_name,
        description=resolved_description,
        url=url,
        version=version,
        capabilities=AgentCapabilities(streaming=streaming),
        default_input_modes=DEFAULT_INPUT_MODES,
        default_output_modes=DEFAULT_OUTPUT_MODES,
        skills=resolved_skills,
    )


def _user_message(task: str) -> Message:
    return Message(message_id=uuid4().hex, role=Role.user, parts=[Part(root=TextPart(text=task))])


def make_send_request(task: str) -> SendMessageRequest:
    """Build a ``message/send`` request carrying ``task`` as a single text part."""
    return SendMessageRequest(id=uuid4().hex, params=MessageSendParams(message=_user_message(task)))


def make_stream_request(task: str) -> SendStreamingMessageRequest:
    """Build a ``message/stream`` request carrying ``task`` as a single text part."""
    return SendStreamingMessageRequest(id=uuid4().hex, params=MessageSendParams(message=_user_message(task)))


def parts_to_text(parts: Optional[list]) -> str:
    """Concatenate the text of every ``TextPart`` in a parts list; skip non-text parts."""
    if not parts:
        return ""
    chunks = []
    for part in parts:
        root = getattr(part, "root", part)
        if getattr(root, "kind", None) == "text":
            chunks.append(root.text)
    return "".join(chunks)


def result_to_text(result: Any) -> str:
    """Flatten a ``Message`` or ``Task`` result into plain text.

    A ``Message`` result returns its joined text parts. A ``Task`` result prefers its
    latest status message, then any artifact parts.
    """
    # Message result: has .parts directly.
    if getattr(result, "parts", None) is not None and getattr(result, "artifacts", None) is None:
        return parts_to_text(result.parts)

    # Task result: status message first, then artifacts.
    status = getattr(result, "status", None)
    status_message = getattr(status, "message", None) if status is not None else None
    if status_message is not None:
        text = parts_to_text(getattr(status_message, "parts", None))
        if text:
            return text
    artifacts = getattr(result, "artifacts", None) or []
    return "\n".join(parts_to_text(getattr(a, "parts", None)) for a in artifacts).strip()
