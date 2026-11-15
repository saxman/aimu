"""Compaction: the strategies that decide what a request to a model leaves out, bundled for use.

Like :mod:`aimu.tools.builtin` for tools, this module is where AIMU's ready-made compactions live,
in two forms for two jobs:

- **Ready-made compactions** for ``Agent(compaction=...)``: :func:`mask` and :func:`trim` return a
  :data:`~aimu.context.Compaction` with its settings bound, and :func:`chain` runs several in
  order. ``Agent(compaction=compaction.chain(compaction.mask(6000), compaction.trim(6000)))``
  masks old content first and drops whole messages only if masking was not enough.
- **The strategy functions** behind them, :func:`mask_messages`, :func:`trim_messages`, and
  :func:`summarize_messages`, for a one-off edit you make yourself
  (``client.messages = trim_messages(client.messages, 8000)``), which, unlike a compaction,
  rewrites the history you assign it to.

**A compaction decides what a request sends, never what the conversation holds.** The agent loop
runs it on a copy of the full history for every request, sends the result, and announces what it
leaves out when that changes; ``client.messages`` is never rewritten, so a later request under a
different compaction can send what an earlier one left out.

**Masking replaces content in place; trimming drops messages.** :func:`mask_messages` keeps every
message where it is and replaces old content with a short note naming the message, so the model
still sees who said what and which tools ran, and the note names the way back:
``get_message(<index>)``, a tool every agent with a compaction is given (see
:func:`aimu.tools.builtin.make_history_tools`). It masks the cheapest content to lose first: tool
results, then images, then assistant text, then user text, oldest first within each, and never the
task. A note still costs tokens, so masking slows growth without bounding it; :func:`trim_messages`
is the backstop that drops whole messages, and comes after masking in a :func:`chain`.

**Order matters in a chain.** A mask's notes name each message by its position in the history the
compaction was given, so a mask has to run on the full history, ahead of anything that drops
messages. ``chain(mask(...), trim(...))`` is right; ``chain(trim(...), mask(...))`` would number
the notes against a list that is already missing messages.

**Summarizing has no ready-made compaction.** :func:`summarize_messages` calls a model, and as a
compaction it would call one on every request, over a longer history each time. It stays a
function for a deliberate edit (issue #19 tracks a version that can run per request).
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

from . import Compaction, count_tokens
from ._groups import _flatten, _group_messages, _protect_head, _protect_tail

DEFAULT_SUMMARIZE_PROMPT = (
    "Summarize the conversation below in a few sentences. Preserve concrete facts, "
    "decisions, and any open questions a continuation of this conversation would still "
    "need to know."
)

# The name of the tool a mask's notes point to. A literal rather than an import from aimu.tools,
# because this package stays stdlib-only (it joins the `import aimu` chain); a test pins it to
# aimu.tools.builtin.HISTORY_TOOL_NAME.
_HISTORY_TOOL = "get_message"

#: The order :func:`mask_messages` masks in: the content cheapest to lose first. A tool result is
#: usually the bulk and the model's own record of having made the call stays beside it; an image
#: is large and comes back exactly; the dialogue itself is what the model reasons from.
MASK_ORDER = ("tool", "image", "assistant", "user")

# A tool-call argument longer than this is masked along with the assistant text it belongs to: a
# call has to stay (its result needs it), but its arguments can be a whole file's content.
_LONG_ARGUMENT_CHARS = 200

NoteFn = Callable[[int, dict, str], str]


def default_note(index: int, message: dict, kind: str) -> str:
    """The note :func:`mask_messages` puts in place of masked content, naming the way back.

    ``kind`` is one of :data:`MASK_ORDER`. The text says what is missing and where, that it is a
    context saving rather than a gap in the conversation, and which call returns it.
    """
    back = f"{_HISTORY_TOOL}({index})"
    if kind == "image":
        return f"[An image in message {index} is not shown here, to save context. {back} shows it.]"
    what = {
        "tool": f"the result of {message.get('name') or 'a tool call'}",
        "assistant": "an earlier reply of yours",
        "user": "an earlier user message",
    }[kind]
    return f"[Message {index}, {what}, is not shown here, to save context. {back} returns it in full.]"


def mask_messages(
    messages: list[dict],
    max_tokens: int,
    *,
    keep_first: int = 1,
    keep_last: int = 2,
    counter: Optional[Callable[[str], int]] = None,
    note: NoteFn = default_note,
) -> list[dict]:
    """Return *messages* with old content replaced by notes until they fit within *max_tokens*.

    Every message keeps its place and role, so the list is the same length, its shape is the
    history's shape (roles alternate wherever they did), and a tool call never loses its result:
    only content changes. Content is masked in :data:`MASK_ORDER`, oldest first within each kind,
    and masking stops the moment the list fits:

    - ``tool``: a tool result's content becomes the note. A ``get_message`` result holds another
      message's content, so its note names that message (following reads of reads), never the read:
      a note naming the read sent a model to read its own read.
    - ``image``: a message's image blocks become one text block carrying the note, beside any text.
    - ``assistant``: a reply's text becomes the note, and any tool-call argument longer than 200
      characters becomes a short placeholder (the call itself stays).
    - ``user``: a user message's content becomes the note.

    System messages are never masked, nor the first ``keep_first`` and last ``keep_last`` messages
    after them (extended outward to whole tool-call groups, like :func:`trim_messages`): in an agent
    run the first is the task. A message is masked only if its note is shorter than what it
    replaces. *messages* is never mutated. If every candidate is masked and the list still does not
    fit, it is returned as masked; follow with :func:`trim_messages` (see :func:`chain`) to bound it.

    Args:
        messages: The conversation, OpenAI message-dict format.
        max_tokens: The budget to fit, per ``counter`` (or the default estimate).
        keep_first: Leading non-system messages never masked.
        keep_last: Trailing messages never masked.
        counter: Optional real tokenizer; see :func:`~aimu.context.count_tokens`.
        note: ``note(index, message, kind) -> str``, the text put in place of masked content.
            ``index`` is the message's position in *messages*, which is its position in the history
            when this runs as a compaction. The default, :func:`default_note`, names
            ``get_message(index)``.
    """
    masked = list(messages)
    total = count_tokens(masked, counter=counter)
    if total <= max_tokens:
        return masked

    # The running total moves by each replacement's own saving, so the whole list is recounted only
    # when it looks done; a counter that is not additive (a real tokenizer) gets the final say.
    for kind in MASK_ORDER:
        for index in _maskable_indices(masked, keep_first, keep_last, kinds=(kind,)):
            replacement = _masked(masked, index, kind, note)
            if replacement is None:
                continue
            saving = count_tokens([masked[index]], counter=counter) - count_tokens([replacement], counter=counter)
            if saving <= 0:
                continue
            masked[index] = replacement
            total -= saving
            if total <= max_tokens and count_tokens(masked, counter=counter) <= max_tokens:
                return masked
    return masked


def _maskable_indices(messages: list[dict], keep_first: int, keep_last: int, *, kinds: tuple[str, ...]):
    """Positions of the messages a mask may change, oldest first, that hold content of *kinds*."""
    rest = [index for index, message in enumerate(messages) if message.get("role") != "system"]
    groups, start = [], 0
    for group in _group_messages([messages[index] for index in rest]):
        groups.append(rest[start : start + len(group)])
        start += len(group)
    _, remainder = _protect_head(groups, keep_first)
    middle, _ = _protect_tail(remainder, keep_last)
    for index in _flatten(middle):
        if any(_holds(messages[index], kind) for kind in kinds):
            yield index


def _holds(message: dict, kind: str) -> bool:
    role, content = message.get("role"), message.get("content")
    if kind == "image":
        return isinstance(content, list) and any(block.get("type") == "image_url" for block in content)
    if kind == "tool":
        return role == "tool"
    return role == kind


def _masked(messages: list[dict], index: int, kind: str, note: NoteFn) -> Optional[dict]:
    """``messages[index]`` with its *kind* content replaced by a note, or None when there is nothing to replace."""
    message = messages[index]
    recalled = _recalled_index(messages, index) if kind == "tool" else None
    if recalled is not None:
        # A get_message result holds another message's content, so its note names that message. A note
        # naming the read itself would send the model to read its own read, then that read's read.
        original = messages[recalled]
        text = note(recalled, original, "tool" if original.get("role") == "tool" else original.get("role", "user"))
    else:
        text = note(index, message, kind)
    content = message.get("content")
    if kind == "image":
        blocks = [block for block in content if block.get("type") != "image_url"]
        return {**message, "content": [*blocks, {"type": "text", "text": text}]}
    if kind == "assistant":
        calls = message.get("tool_calls")
        shortened = [_with_short_arguments(call, index) for call in calls] if calls else None
        replaced = {**message, "content": text} if content else dict(message)
        if shortened is not None:
            replaced["tool_calls"] = shortened
        return replaced if replaced != message else None
    return {**message, "content": text} if content else None


def _recalled_index(messages: list[dict], index: int) -> Optional[int]:
    """The message a ``get_message`` result at *index* read, following reads of reads, or None.

    Found through the call the result answers (the ``assistant`` message carrying a ``tool_calls`` entry
    with this result's ``tool_call_id``), whose ``index`` argument names the message read.
    """
    seen = set()
    while index not in seen:
        seen.add(index)
        message = messages[index]
        if message.get("role") != "tool" or message.get("name") != _HISTORY_TOOL:
            return index if seen != {index} else None
        target = _call_argument(messages, index, message.get("tool_call_id"), "index")
        if not isinstance(target, int) or not 0 <= target < len(messages):
            return None
        index = target
    return None


def _call_argument(messages: list[dict], before: int, call_id: Optional[str], name: str) -> Any:
    for message in reversed(messages[:before]):
        for call in message.get("tool_calls") or ():
            if call.get("id") == call_id:
                arguments = (call.get("function") or {}).get("arguments")
                return arguments.get(name) if isinstance(arguments, dict) else None
    return None


def _with_short_arguments(call: dict, index: int) -> dict:
    function = call.get("function") or {}
    arguments = function.get("arguments")
    if not isinstance(arguments, dict):
        return call
    placeholder = f"[not shown; {_HISTORY_TOOL}({index})]"
    shortened = {
        key: placeholder if isinstance(value, str) and len(value) > _LONG_ARGUMENT_CHARS else value
        for key, value in arguments.items()
    }
    return call if shortened == arguments else {**call, "function": {**function, "arguments": shortened}}


def trim_messages(
    messages: list[dict],
    max_tokens: int,
    *,
    keep_system: bool = True,
    keep_first: int = 1,
    keep_last: int = 2,
    counter: Optional[Callable[[str], int]] = None,
) -> list[dict]:
    """Return a new, trimmed message list that fits within *max_tokens* (estimated).

    Always returns a new list; *messages* (and its dicts) are never mutated. A conversation
    already at or under *max_tokens* is returned unchanged (a copy) -- this is a no-op, not
    a forced rewrite.

    Dropping proceeds oldest-first over the non-system messages, one indivisible group at a
    time: an ``assistant`` message carrying ``tool_calls`` is always dropped together with every
    ``tool`` message answering it, never separately. This is the invariant that justifies this
    function over a hand-written ``messages[-n:]`` slice, which produces exactly that illegal shape.

    Dropping leaves no trace, so as a compaction it belongs behind :func:`mask_messages` (see
    :func:`chain`), where it only removes messages masking has already reduced to notes.

    The first non-system message is kept by default (``keep_first=1``). In an agent run it is
    the task, the user message ``agent.run(task)`` appended, and oldest-first dropping would
    otherwise remove it before anything else, leaving the agent working without knowing what
    it was asked. That failure raises nothing: the run still returns a plausible answer.

    Args:
        messages: The conversation, OpenAI message-dict format.
        max_tokens: The token budget to trim down to, per ``counter`` (or the default
            estimate; see :func:`~aimu.context.count_tokens`).
        keep_system: When ``True`` (default), every ``{"role": "system"}`` message is kept
            regardless of budget, and is **moved to the front** of the returned list. That is
            invisible for the usual single leading system prompt, but a system message that
            sat mid-conversation (an earlier summary inserted by :func:`summarize_messages`,
            say) comes back ahead of the turns it originally followed. When ``False``, system
            messages are ordinary droppable messages like any other and order is preserved.
        keep_first: The number of leading **messages** to always protect from dropping,
            counted after the system messages when ``keep_system`` is ``True`` and from the
            very first message otherwise. Extended outward to whole tool-call groups, like
            ``keep_last``. Pass ``0`` for plain oldest-first dropping.
        keep_last: The number of trailing **messages** (not exchanges) to always protect from
            dropping, extended outward to whole tool-call groups when the raw count would
            otherwise split one.
        counter: Optional real tokenizer; see :func:`~aimu.context.count_tokens`.
    """
    messages = list(messages)
    if count_tokens(messages, counter=counter) <= max_tokens:
        return messages

    if keep_system:
        kept_system = [m for m in messages if m.get("role") == "system"]
        rest = [m for m in messages if m.get("role") != "system"]
    else:
        kept_system = []
        rest = list(messages)

    head, remainder = _protect_head(_group_messages(rest), keep_first)
    droppable, tail = _protect_tail(remainder, keep_last)
    kept_head = kept_system + _flatten(head)

    while droppable and count_tokens(kept_head + _flatten(droppable) + _flatten(tail), counter=counter) > max_tokens:
        droppable.pop(0)

    return kept_head + _flatten(droppable) + _flatten(tail)


def mask(max_tokens: int, **options: Any) -> Compaction:
    """A compaction that runs :func:`mask_messages` with these settings on every request.

    ``options`` are :func:`mask_messages`' keyword arguments, checked now rather than on the first
    request, so a misspelled one raises here.
    """
    _check_options(mask_messages, options)

    def compaction(messages: list[dict]) -> list[dict]:
        return mask_messages(messages, max_tokens, **options)

    compaction.__name__ = compaction.__qualname__ = f"mask({max_tokens})"
    return compaction


def trim(max_tokens: int, **options: Any) -> Compaction:
    """A compaction that runs :func:`trim_messages` with these settings on every request.

    ``options`` are :func:`trim_messages`' keyword arguments, checked now rather than on the first
    request, so a misspelled one raises here.
    """
    _check_options(trim_messages, options)

    def compaction(messages: list[dict]) -> list[dict]:
        return trim_messages(messages, max_tokens, **options)

    compaction.__name__ = compaction.__qualname__ = f"trim({max_tokens})"
    return compaction


def chain(*compactions: Compaction) -> Compaction:
    """A compaction that runs each of *compactions* in turn, each on the previous one's result.

    The way to combine strategies, as ``builtin.web + [my_tool]`` combines tools: masking first,
    trimming as the backstop, ``chain(mask(n), trim(n))``. Put a mask ahead of anything that drops
    messages, since its notes name positions in the list it is given (see the module docstring).
    """
    if not compactions:
        raise ValueError("chain() needs at least one compaction.")
    for index, step in enumerate(compactions):
        if not callable(step):
            raise ValueError(f"chain() argument {index} is not a compaction (a callable): {step!r}")

    def compaction(messages: list[dict]) -> list[dict]:
        for step in compactions:
            messages = step(messages)
        return messages

    names = ", ".join(getattr(step, "__name__", "compaction") for step in compactions)
    compaction.__name__ = compaction.__qualname__ = f"chain({names})"
    return compaction


def _check_options(function: Callable, options: dict) -> None:
    allowed = {
        name
        for name, parameter in inspect.signature(function).parameters.items()
        if parameter.kind is parameter.KEYWORD_ONLY
    }
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise TypeError(f"{function.__name__}() has no option(s) {unknown}; it takes {sorted(allowed)}.")


def _message_text(message: dict) -> str:
    """Extract human-readable text from one message, for rendering a summarization transcript.

    Only ``content`` (string, or the text blocks of a content-block list; image/audio
    blocks contribute nothing meaningful as text) and ``tool_calls`` (function name +
    arguments) are considered -- this is for a human/LLM reader, not token counting.
    """
    parts: list[str] = []
    content = message.get("content")
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)

    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        name = function.get("name", "")
        arguments = function.get("arguments", "")
        text = f"{name} {arguments}".strip()
        if text:
            parts.append(text)

    return " ".join(p for p in parts if p)


def _partition_for_summary(
    messages: list[dict], keep_first: int, keep_last: int
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Split *messages* into ``(system, head, prefix, tail)`` for :func:`summarize_messages`.

    Shared with the async twin so the two cannot disagree about what gets summarized.
    """
    messages = list(messages)
    kept_system = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    head, remainder = _protect_head(_group_messages(rest), keep_first)
    prefix, tail = _protect_tail(remainder, keep_last)
    return kept_system, _flatten(head), _flatten(prefix), _flatten(tail)


def _summary_prompt(instruction: str, head: list[dict], prefix: list[dict]) -> str:
    """The summarization request: *instruction*, the kept *head* as background, then *prefix*.

    The head is shown because the instruction asks for what a continuation will need, and in
    an agent run the head is the task that need is relative to. It is labeled as kept, so the
    summarizer leaves it out rather than repeating it.
    """
    if not head:
        return f"{instruction}\n\n{_render_transcript(prefix)}"
    return (
        f"{instruction}\n\n"
        "For context only, the conversation began with the message(s) below. "
        "They are kept as written; do not summarize them:\n"
        f"{_render_transcript(head)}\n\n"
        f"Summarize this part:\n{_render_transcript(prefix)}"
    )


def _summary_message(summary_text: str) -> dict:
    return {"role": "system", "content": f"Summary of earlier conversation:\n{summary_text}"}


def _render_transcript(messages: list[dict]) -> str:
    """Render *messages* as a simple ``role: text`` transcript for a summarization prompt."""
    return "\n".join(f"{m.get('role', '?')}: {_message_text(m)}" for m in messages)


def summarize_messages(
    client: Any,
    messages: list[dict],
    *,
    keep_first: int = 1,
    keep_last: int = 2,
    prompt: Optional[str] = None,
) -> list[dict]:
    """Replace the middle of a conversation with a one-message LLM summary.

    Any ``{"role": "system"}`` messages are always preserved unchanged, and are returned at
    the front (a mid-conversation system message is hoisted, the same reordering
    :func:`trim_messages` does under ``keep_system=True``). Of the rest, the first
    ``keep_first`` and the last ``keep_last`` **messages** (each extended outward to whole
    tool-call groups, the same invariant :func:`trim_messages` enforces) are kept verbatim.
    Everything between them (the "prefix") is rendered to a plain transcript, summarized by one
    ``client.generate()`` call, and replaced with a single ``{"role": "system", ...}`` message
    carrying the summary text -- system, rather than user or assistant, because it is
    out-of-band context neither party actually said, not a real conversational turn.

    ``keep_first=1`` is the default for the reason it is on :func:`trim_messages`: in an agent
    run the first non-system message is the task, and a summarizer may paraphrase it or leave
    it out. The kept first messages are still shown to the summarizer, labeled as background
    it should not summarize, because the instruction asks for what a continuation will need
    and the task is what that need is relative to. Models tend to restate the task in the
    summary anyway, a small duplication. The returned order is system messages, then the
    summary, then the kept first messages, then the tail. The summary goes ahead of the kept
    first messages, though it describes what came after them, because several local chat
    templates reject a system message anywhere but the start of the conversation.

    ``client`` is any object exposing ``generate(prompt: str) -> str`` (a plain
    :class:`~aimu.models.BaseModelClient`, or a :class:`~aimu.agents.Agent` via
    ``agent.as_model_client()``); this function takes it as an argument rather than
    constructing one itself, so it stays free of any provider dependency. There is no
    ready-made compaction for it; see the module docstring.

    When nothing lies between the kept first and last messages, no summarization call is
    made and the conversation is returned unchanged (a new list; *messages* is never
    mutated).

    Args:
        client: Anything with a ``generate(prompt: str) -> str`` method.
        messages: The conversation, OpenAI message-dict format.
        keep_first: Leading non-system messages to keep verbatim; ``0`` summarizes from the
            first one.
        keep_last: Trailing messages (not exchanges) to keep verbatim; see above.
        prompt: Optional instruction prepended to the rendered transcript, replacing
            :data:`DEFAULT_SUMMARIZE_PROMPT`.
    """
    kept_system, head, prefix, tail = _partition_for_summary(messages, keep_first, keep_last)
    if not prefix:
        return kept_system + head + tail

    summary_text = client.generate(_summary_prompt(prompt or DEFAULT_SUMMARIZE_PROMPT, head, prefix))
    return kept_system + [_summary_message(summary_text)] + head + tail


__all__ = [
    "DEFAULT_SUMMARIZE_PROMPT",
    "MASK_ORDER",
    "chain",
    "default_note",
    "mask",
    "mask_messages",
    "summarize_messages",
    "trim",
    "trim_messages",
]
