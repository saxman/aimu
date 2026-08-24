"""Context management: plain functions over the conversation.

The invariant that justifies a library function rather than a user-written slice: trimming
must never orphan a `tool` message from the `assistant` message carrying its `tool_calls`.
Every provider rejects that shape, and it is exactly what a naive slice produces.
"""

from aimu.context import count_tokens, summarize_messages, trim_messages


def test_trim_never_orphans_a_tool_result():
    """The reason this function exists."""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    trimmed = trim_messages(messages, max_tokens=12, keep_last=1)
    ids = {m["tool_call_id"] for m in trimmed if m.get("role") == "tool"}
    advertised = {call["id"] for m in trimmed if m.get("role") == "assistant" for call in (m.get("tool_calls") or [])}
    assert ids <= advertised, f"orphaned tool results: {ids - advertised}"


def test_trim_never_orphans_a_tool_result_even_at_the_keep_last_boundary():
    """The tool-call group must not be split even when keep_last would cut through it."""
    messages = [
        {"role": "user", "content": "q1"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
    ]
    # keep_last=1 would naively cut off just the tool result, orphaning it.
    trimmed = trim_messages(messages, max_tokens=1, keep_last=1)
    ids = {m["tool_call_id"] for m in trimmed if m.get("role") == "tool"}
    advertised = {call["id"] for m in trimmed if m.get("role") == "assistant" for call in (m.get("tool_calls") or [])}
    assert ids <= advertised, f"orphaned tool results: {ids - advertised}"


def test_trim_keeps_the_system_message_by_default():
    messages = [{"role": "system", "content": "you are a helpful assistant, be concise and thorough"}]
    messages += [{"role": "user", "content": f"question number {i} " * 10} for i in range(20)]
    trimmed = trim_messages(messages, max_tokens=20, keep_last=2)
    system_messages = [m for m in trimmed if m.get("role") == "system"]
    assert len(system_messages) == 1
    assert system_messages[0]["content"] == messages[0]["content"]


def test_trim_can_drop_the_system_message_when_asked():
    messages = [{"role": "system", "content": "sys " * 50}, {"role": "user", "content": "q1"}]
    # max_tokens=-1 is unsatisfiable by any non-empty remainder, forcing every droppable
    # group -- including the system message, since keep_system=False -- to be dropped.
    trimmed = trim_messages(messages, max_tokens=-1, keep_system=False, keep_last=0)
    assert trimmed == []


def test_trim_keeps_the_last_n_turns():
    messages = [{"role": "system", "content": "sys"}]
    for i in range(10):
        messages.append({"role": "user", "content": f"question {i} " * 20})
        messages.append({"role": "assistant", "content": f"answer {i} " * 20})
    trimmed = trim_messages(messages, max_tokens=1, keep_last=4)
    # keep_last counts messages, not exchanges: exactly the last 4 messages survive
    # (plus the always-kept system message).
    non_system = [m for m in trimmed if m.get("role") != "system"]
    assert non_system == messages[-4:]


def test_trim_returns_a_new_list_and_does_not_mutate():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello " * 50},
        {"role": "assistant", "content": "world " * 50},
    ]
    original = [dict(m) for m in messages]
    trimmed = trim_messages(messages, max_tokens=1, keep_last=1)
    assert trimmed is not messages
    assert messages == original


def test_count_tokens_default_is_documented_as_an_estimate():
    assert "estimate" in count_tokens.__doc__.lower()
    short = count_tokens([{"role": "user", "content": "a" * 40}])
    long = count_tokens([{"role": "user", "content": "a" * 4000}])
    # An estimate, not a measurement: it scales with content length and never undercounts
    # below the raw content-length/4 floor (the JSON envelope only ever adds overhead).
    assert long > short
    assert long >= 4000 // 4


def test_count_tokens_excludes_inert_keys():
    with_thinking = [{"role": "assistant", "content": "hi", "thinking": "x" * 1000, "timestamp": "now"}]
    without_thinking = [{"role": "assistant", "content": "hi"}]
    assert count_tokens(with_thinking) == count_tokens(without_thinking)


def test_count_tokens_accepts_a_real_tokenizer():
    # A fixed-output stand-in for "a real tokenizer": proves count_tokens actually
    # dispatches to the supplied counter rather than always using the default estimate.
    messages = [{"role": "user", "content": "some perfectly ordinary text"}]

    def fake_tokenizer(text: str) -> int:
        return 4242

    assert count_tokens(messages, counter=fake_tokenizer) == 4242
    assert count_tokens(messages) != 4242


def test_summarize_replaces_the_prefix_and_keeps_the_tail():
    class FakeClient:
        def __init__(self):
            self.prompts = []

        def generate(self, prompt: str) -> str:
            self.prompts.append(prompt)
            return "a short summary"

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    client = FakeClient()
    result = summarize_messages(client, messages, keep_last=2)

    # System preserved, tail (last 2 messages) kept verbatim, prefix collapsed to one message.
    assert result[0] == messages[0]
    assert result[-2:] == messages[-2:]
    assert len(result) == 1 + 1 + 2  # system + summary + tail
    summary_message = result[1]
    assert summary_message["role"] == "system"
    assert "a short summary" in summary_message["content"]
    assert len(client.prompts) == 1
    # The summarized prefix (q1/a1) must have reached the client, the kept tail must not.
    assert "q1" in client.prompts[0]
    assert "q2" not in client.prompts[0]


def test_summarize_is_a_noop_when_everything_fits_in_keep_last():
    class FailingClient:
        def generate(self, prompt: str) -> str:
            raise AssertionError("should not be called when there is no prefix to summarize")

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]
    result = summarize_messages(FailingClient(), messages, keep_last=10)
    assert result == messages
    assert result is not messages


def test_trim_on_an_already_small_conversation_is_a_noop():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    trimmed = trim_messages(messages, max_tokens=10_000)
    assert trimmed == messages
    assert trimmed is not messages


def test_summarize_never_orphans_a_tool_result_at_the_keep_last_boundary():
    """Mirrors test_trim_never_orphans_a_tool_result_even_at_the_keep_last_boundary.

    A naive index slice (last keep_last raw messages as the tail) would cut this
    conversation between the tool-call group's two messages, summarizing away the
    assistant message carrying tool_calls while leaving its tool result in the kept tail.
    """

    class FakeClient:
        def generate(self, prompt: str) -> str:
            return "summary"

    messages = [
        {"role": "user", "content": "q1"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
        {"role": "assistant", "content": "a1"},
    ]
    # A naive messages[-2:] tail would be [tool, a1], orphaning the tool result from the
    # assistant+tool_calls message that would land in the summarized prefix.
    result = summarize_messages(FakeClient(), messages, keep_last=2)
    ids = {m["tool_call_id"] for m in result if m.get("role") == "tool"}
    advertised = {call["id"] for m in result if m.get("role") == "assistant" for call in (m.get("tool_calls") or [])}
    assert ids <= advertised, f"orphaned tool results: {ids - advertised}"


def test_trim_drops_oldest_first_not_newest_first():
    """A partial drop -- some but not all droppable messages removed -- must remove the
    oldest ones, per the documented "oldest-first" order. A budget that fits exactly one
    droppable message keeps the newest and drops the rest.
    """
    oldest = {"role": "user", "content": "OLDEST"}
    middle = {"role": "user", "content": "MIDDLE"}
    newest = {"role": "user", "content": "NEWEST"}
    messages = [oldest, middle, newest]

    budget = count_tokens([newest])  # exactly enough for the single newest message
    trimmed = trim_messages(messages, max_tokens=budget, keep_last=0)

    assert trimmed == [newest]
