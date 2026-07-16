"""Inert message-dict metadata: provenance markers and request-time hygiene.

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
"""

from __future__ import annotations

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
