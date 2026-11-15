"""aimu.context.compaction: masking in place, the ready-made compactions, and chaining them (mock-only).

Masking keeps every message where it is and replaces old content with a note naming the message,
cheapest content first; the factories bind a strategy's settings so ``Agent(compaction=...)`` takes it
directly; ``chain`` runs several in order. Trimming and summarizing keep their own tests in
tests/test_context.py.
"""

from __future__ import annotations

import pytest

from aimu.context import count_tokens
from aimu.context.compaction import MASK_ORDER, chain, default_note, mask, mask_messages, trim, trim_messages

PNG_URL = "data:image/png;base64," + "A" * 400


def _research_run() -> list[dict]:
    """A worker's history: system, task, two tool rounds (one with a long argument), an image, replies."""
    big = "word " * 400
    return [
        {"role": "system", "content": "You are a researcher."},
        {"role": "user", "content": "find the release date"},  # 1: the task
        {
            "role": "assistant",
            "content": "Searching first.",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "search", "arguments": {"q": "x"}}}],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "search", "content": big},  # 3
        {"role": "assistant", "content": "Here is a chart. " + big},  # 4
        {
            "role": "user",
            "content": [{"type": "text", "text": "look at this"}, {"type": "image_url", "image_url": {"url": PNG_URL}}],
        },  # 5
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "write_file", "arguments": {"path": "a", "content": big}},
                }
            ],
        },  # 6
        {"role": "tool", "tool_call_id": "c2", "name": "write_file", "content": "wrote 400 words"},  # 7
        {"role": "user", "content": "and?"},  # 8
        {"role": "assistant", "content": "done"},  # 9
    ]


def test_masking_keeps_every_message_in_place():
    messages = _research_run()
    masked = mask_messages(messages, max_tokens=200)
    assert len(masked) == len(messages)
    assert [m["role"] for m in masked] == [m["role"] for m in messages]
    assert [m.get("tool_call_id") for m in masked] == [m.get("tool_call_id") for m in messages]
    assert messages == _research_run()  # the input is never mutated


def test_tool_results_go_first_and_stop_once_it_fits():
    messages = _research_run()
    budget = (
        count_tokens(messages) - count_tokens([messages[3]]) + count_tokens([{"role": "tool", "content": "x" * 160}])
    )
    masked = mask_messages(messages, max_tokens=budget)
    assert masked[3]["content"] == default_note(3, messages[3], "tool")
    assert masked[4] == messages[4]  # the assistant reply was not needed
    assert masked[5] == messages[5]  # nor the image


def test_the_order_is_tool_image_assistant_user():
    assert MASK_ORDER == ("tool", "image", "assistant", "user")
    masked = mask_messages(_research_run(), max_tokens=1)  # nothing fits: every candidate masked
    assert masked[3]["content"].startswith("[Message 3, the result of search")
    assert masked[4]["content"].startswith("[Message 4, an earlier reply of yours")
    # Its image went at the image step, and then the user's words at the last step.
    assert (
        masked[5]["content"]
        == "[Message 5, an earlier user message, is not shown here, to save context. get_message(5) returns it in full.]"
    )


def test_an_image_becomes_a_note_beside_the_text_until_the_user_message_itself_is_masked():
    messages = _research_run()
    # One token less than what is left once the tool result is masked, so the image has to go too,
    # and masking an image saves enough that the user's own words are not reached.
    after_tool = list(messages)
    after_tool[3] = {**messages[3], "content": default_note(3, messages[3], "tool")}
    masked = mask_messages(messages, max_tokens=count_tokens(after_tool) - 1)
    assert masked[5]["content"] == [
        {"type": "text", "text": "look at this"},
        {"type": "text", "text": default_note(5, messages[5], "image")},
    ]


def test_a_long_tool_argument_is_shortened_and_the_call_kept():
    masked = mask_messages(_research_run(), max_tokens=1)
    call = masked[6]["tool_calls"][0]
    assert call["id"] == "c2" and call["function"]["name"] == "write_file"
    assert call["function"]["arguments"] == {"path": "a", "content": "[not shown; get_message(6)]"}


def test_the_task_system_message_and_tail_are_never_masked():
    messages = _research_run()
    masked = mask_messages(messages, max_tokens=1)
    assert masked[0] == messages[0]
    assert masked[1] == messages[1]
    assert masked[-2:] == messages[-2:]


def test_a_message_shorter_than_its_note_is_left_alone():
    masked = mask_messages(_research_run(), max_tokens=1)
    assert masked[7] == _research_run()[7]  # "wrote 400 words" is shorter than any note


def test_a_list_that_fits_is_returned_unchanged():
    messages = _research_run()
    assert mask_messages(messages, max_tokens=10**6) == messages


def test_a_custom_note_is_given_the_index_the_message_and_the_kind():
    seen = []

    def note(index, message, kind):
        seen.append((index, message["role"], kind))
        return f"<{kind} {index}>"

    masked = mask_messages(_research_run(), max_tokens=1, note=note)
    assert (3, "tool", "tool") in seen
    assert masked[3]["content"] == "<tool 3>"


def test_the_default_note_names_the_message_and_the_way_back():
    note = default_note(12, {"role": "tool", "name": "search"}, "tool")
    assert (
        note
        == "[Message 12, the result of search, is not shown here, to save context. get_message(12) returns it in full.]"
    )


def test_the_history_tool_name_matches_builtin():
    from aimu.context import compaction
    from aimu.tools.builtin import HISTORY_TOOL_NAME

    assert compaction._HISTORY_TOOL == HISTORY_TOOL_NAME


# --------------------------------------------------------------------------- #
# The ready-made compactions                                                  #
# --------------------------------------------------------------------------- #


def test_mask_and_trim_bind_their_settings():
    messages = _research_run()
    assert mask(200)(messages) == mask_messages(messages, 200)
    assert trim(200, keep_last=1)(messages) == trim_messages(messages, 200, keep_last=1)


def test_a_misspelled_option_raises_when_the_compaction_is_made():
    with pytest.raises(TypeError, match="keep_lsat"):
        trim(200, keep_lsat=1)
    with pytest.raises(TypeError, match="marker"):
        mask(200, marker="x")


def test_a_compaction_says_what_it_is():
    assert trim(200).__name__ == "trim(200)"
    assert chain(mask(300), trim(200)).__name__ == "chain(mask(300), trim(200))"


def test_chain_runs_each_in_turn():
    messages = _research_run()
    chained = chain(mask(200), trim(200))(messages)
    assert chained == trim_messages(mask_messages(messages, 200), 200)


def test_masking_first_lets_the_trim_drop_less():
    messages = _research_run()
    budget = count_tokens(messages) // 3
    trimmed_only = trim(budget)(messages)
    masked_then_trimmed = chain(mask(budget), trim(budget))(messages)
    assert len(masked_then_trimmed) > len(trimmed_only)


def test_chain_refuses_nothing_or_a_non_callable():
    with pytest.raises(ValueError):
        chain()
    with pytest.raises(ValueError, match="argument 1"):
        chain(mask(10), "trim")


def _with_reads() -> list[dict]:
    """A history where the model read message 3 back with get_message, then read that read."""
    big = "word " * 400

    def read(call_id, index):
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": "get_message", "arguments": {"index": index}}}
            ],
        }

    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "fetch", "arguments": {}}}],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "fetch", "content": big},  # 3
        read("c2", 3),
        {
            "role": "tool",
            "tool_call_id": "c2",
            "name": "get_message",
            "content": "Message 3 (tool, fetch)\n" + big,
        },  # 5
        read("c3", 5),
        {
            "role": "tool",
            "tool_call_id": "c3",
            "name": "get_message",
            "content": "Message 5 (tool, get_message)\n" + big,
        },  # 7
        {"role": "user", "content": "and?"},
        {"role": "assistant", "content": "ok"},
    ]


def test_a_masked_read_names_the_message_it_read_not_itself():
    """Observed live: a note naming the read sent the model to read its own read. A read's note names
    the original message, so every way back leads to the content, not to another read."""
    masked = mask_messages(_with_reads(), max_tokens=1)
    original_note = default_note(3, _with_reads()[3], "tool")
    assert masked[3]["content"] == original_note
    assert masked[5]["content"] == original_note
    assert masked[7]["content"] == original_note  # a read of a read resolves all the way back
