"""Message-dict hygiene: inert metadata, and the one field OpenAI types differently.

``self.messages`` is plain OpenAI-format data. A few non-standard keys are attached to
message dicts as inert metadata for UIs and persistence, never for the model:

- ``"thinking"``: per-turn reasoning attached to assistant messages by thinking providers.
- ``"provenance"``: marks a turn the framework injected rather than the user authoring it,
  so a display or persistence layer can hide or visually distinguish it. Real user turns and
  ordinary assistant turns are left untagged; absence means "ordinary turn".
- ``"timestamp"``: added by ``ConversationManager`` when persisting.

These keys must never reach a provider. Anthropic and HuggingFace rebuild their request
payloads from ``role``/``content``/``tool_calls`` and drop them automatically, but
OpenAI-compat and Ollama forward unknown message-dict keys verbatim, so those two request
paths call :func:`strip_inert_keys` first.

Separately, one *standard* key needs adapting rather than dropping: a tool call's
``arguments``. The store holds it parsed, and OpenAI's schema types it as a string, so the
OpenAI-format request paths also call :func:`encode_tool_call_arguments` and Ollama's
deliberately do not.
"""

from __future__ import annotations

import json

PROVENANCE_KEY = "provenance"

# Provenance values (the framework-injected turns worth distinguishing from user input).
# Between successful tool rounds the agent continues by calling chat() with no user message, so
# nothing is injected there. This tag marks the recovery nudge the loop injects when a turn comes
# back degenerate (empty: no content and no tool calls), so replay/display can hide or distinguish it.
PROVENANCE_CONTINUATION = "continuation"
PROVENANCE_FINAL_ANSWER = "final_answer"
PROVENANCE_PROACTIVE = "proactive"

# Non-standard message-dict keys that are UI/persistence metadata and must never be sent to a provider.
INERT_MESSAGE_KEYS = frozenset({"thinking", PROVENANCE_KEY, "timestamp"})


def strip_inert_keys(messages: list[dict]) -> list[dict]:
    """Return ``messages`` with :data:`INERT_MESSAGE_KEYS` removed.

    Only dicts that actually carry an inert key are copied; the rest pass through by identity
    to avoid churning the request hot path. Standard OpenAI keys
    (``role``/``content``/``tool_calls``/``tool_call_id``/``name``) are preserved.
    """
    cleaned = []
    for message in messages:
        if INERT_MESSAGE_KEYS.isdisjoint(message):
            cleaned.append(message)
        else:
            cleaned.append({key: value for key, value in message.items() if key not in INERT_MESSAGE_KEYS})
    return cleaned


def encode_tool_call_arguments(messages: list[dict]) -> list[dict]:
    """Return ``messages`` with every tool call's ``arguments`` as a JSON string.

    ``self.messages`` stores a tool call's arguments parsed: that is what Ollama's and
    Anthropic's request paths want on the wire, and what a UI or a transcript reads.
    OpenAI's schema types ``tool_calls[].function.arguments`` as a *string*, so this single
    field is where the store's shape and the OpenAI wire's shape disagree, and adapting it
    at request time is what keeps the provider format out of the store (the plain-data
    principle).

    Sending the dict is not merely off-schema, it raises server-side: a server rendering its
    chat template calls ``json.loads`` on this field, and mlx-lm answers
    ``404 {'error': 'the JSON object must be str, bytes or bytearray, not dict'}``. That
    failed every turn which had called a tool, on the request *after* the tool result.

    Deliberately not folded into :func:`strip_inert_keys`, which Ollama's request path calls
    too: a parsed dict is correct there, so encoding for every caller would move this bug
    rather than fix it. A value that is already a string passes through, so history a caller
    hand-built in OpenAI's own shape is not double-encoded, and a tool call carrying no
    ``arguments`` key does not acquire one.
    """
    adapted = []
    for message in messages:
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            adapted.append(message)
        else:
            adapted.append({**message, "tool_calls": [_with_encoded_arguments(tc) for tc in tool_calls]})
    return adapted


def _with_encoded_arguments(tool_call: dict) -> dict:
    function = tool_call.get("function")
    if not isinstance(function, dict) or "arguments" not in function or isinstance(function["arguments"], str):
        return tool_call
    return {**tool_call, "function": {**function, "arguments": json.dumps(function["arguments"])}}
