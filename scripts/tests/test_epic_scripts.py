import json
from unittest.mock import MagicMock

from _epic_common import parse_judge_response, write_summary


# ---------------------------------------------------------------------------
# Prompt contracts — rewording must not break the parser or drop the gate
# ---------------------------------------------------------------------------


def test_judge_prompt_keeps_parser_contract():
    """Rewording must preserve the SCORE/DONE/CONTINUE structure parse_judge_response needs."""
    from _epic_common import JUDGE_PROMPT

    assert "SCORE: N/10" in JUDGE_PROMPT
    assert "DONE:" in JUDGE_PROMPT
    assert "CONTINUE:" in JUDGE_PROMPT


def test_judge_prompt_has_fidelity_gate():
    """The judge gates on a single grammatical sentence about the same errand: drifters score 1/10."""
    from _epic_common import JUDGE_PROMPT

    low = JUDGE_PROMPT.lower()
    assert "one single grammatical sentence" in low
    assert "score: 1/10" in low  # the explicit failure score the gate emits


def test_generator_prompt_anchors_seed_and_single_sentence():
    """The generator instruction must restate the seed and the one-sentence rule every call."""
    from _epic_common import GENERATOR_PROMPT

    assert "{seed}" in GENERATOR_PROMPT
    assert "{direction}" in GENERATOR_PROMPT
    low = GENERATOR_PROMPT.lower()
    assert "one single grammatical sentence" in low
    assert "only the rewritten sentence" in low


# ---------------------------------------------------------------------------
# parse_judge_response — same contract as hotdog's parse_evaluator_response
# ---------------------------------------------------------------------------


def test_parse_done_response():
    text = "Epicness: 9/10\nDONE: This turns a milk run into a cosmic saga; it cannot get grander."
    result = parse_judge_response(text)
    assert result["action"] == "DONE"
    assert result["score"] == 9
    assert "cosmic" in result["reasoning"]
    assert result["next_direction"] is None


def test_parse_continue_response():
    text = "8/10 epic.\nCONTINUE: frame the errand as a prophecy foretold across a thousand ages"
    result = parse_judge_response(text)
    assert result["action"] == "CONTINUE"
    assert result["score"] == 8
    assert result["next_direction"] == "frame the errand as a prophecy foretold across a thousand ages"
    assert result["reasoning"] is None


def test_parse_unknown_response():
    text = "I cannot determine how epic this sentence is."
    result = parse_judge_response(text)
    assert result["action"] == "unknown"
    assert result["score"] is None


def test_parse_done_takes_priority_over_continue():
    text = "CONTINUE: grander\nDONE: actually it is done"
    result = parse_judge_response(text)
    assert result["action"] == "DONE"


def test_parse_score_out_of_range_ignored():
    text = "11/10 epic.\nDONE: off the charts"
    result = parse_judge_response(text)
    assert result["score"] is None


def test_parse_continue_captures_multiline_direction():
    # The CONTINUE directive is matched with re.DOTALL so a multi-line directive is captured.
    text = "8/10 epic.\nCONTINUE: render it as a celestial decree\nwitnessed by the gods themselves."
    result = parse_judge_response(text)
    assert result["action"] == "CONTINUE"
    assert result["next_direction"] == "render it as a celestial decree\nwitnessed by the gods themselves."


def test_parse_does_not_match_done_mid_sentence():
    text = "Not done: it needs more grandeur. Score: 5/10\nCONTINUE: invoke ancient prophecy"
    result = parse_judge_response(text)
    assert result["action"] == "CONTINUE"
    assert result["next_direction"] == "invoke ancient prophecy"


# ---------------------------------------------------------------------------
# generate_sentence / evaluate_sentence / refine_sentence — stateless, text-only
# ---------------------------------------------------------------------------


def test_generate_sentence_is_stateless_and_strips_quotes():
    from _epic_common import GENERATOR_PROMPT, generate_sentence

    client = MagicMock()
    client.generate.return_value = '  "The hero strode forth for milk."  '

    out = generate_sentence(client, "A man buys milk.", "make it grand")

    client.reset.assert_not_called()
    # Wrapping quotes and whitespace are stripped.
    assert out == "The hero strode forth for milk."
    # The prompt embeds the seed and the directive; text-only (no images forwarded).
    args, kwargs = client.generate.call_args
    assert args[0] == GENERATOR_PROMPT.format(seed="A man buys milk.", direction="make it grand")
    assert "images" not in kwargs


def test_evaluate_sentence_uses_stateless_generate():
    from _epic_common import JUDGE_PROMPT, evaluate_sentence

    client = MagicMock()
    client.generate.return_value = "7/10\nCONTINUE: more mythic weight"

    response, parsed = evaluate_sentence(client, "The hero walks for milk.")

    client.reset.assert_not_called()
    # The sentence rides in the prompt; no images.
    args, kwargs = client.generate.call_args
    assert args[0].startswith(JUDGE_PROMPT)
    assert "The hero walks for milk." in args[0]
    assert "images" not in kwargs
    assert parsed["score"] == 7
    assert parsed["action"] == "CONTINUE"


def test_evaluate_sentence_retries_when_no_score():
    """A scoreless first reply triggers a retry with SCORE_REMINDER appended."""
    from _epic_common import SCORE_REMINDER, evaluate_sentence

    client = MagicMock()
    client.generate.side_effect = ["no score here", "9/10\nDONE: peak grandeur"]

    response, parsed = evaluate_sentence(client, "A sentence.", max_retries=2)

    assert client.generate.call_count == 2
    assert client.generate.call_args_list[1].args[0].endswith(SCORE_REMINDER)
    assert parsed["score"] == 9
    assert parsed["action"] == "DONE"


def test_refine_sentence_stateless_and_avoids_rejected():
    from _epic_common import refine_sentence

    client = MagicMock()
    client.generate.return_value = "  frame it as a prophecy  "

    out = refine_sentence(client, "The hero walks for milk.", rejected=["add more adjectives"])

    client.reset.assert_not_called()
    args, kwargs = client.generate.call_args
    assert "add more adjectives" in args[0]  # the rejected direction is listed in the avoid clause
    assert "images" not in kwargs
    assert out == "frame it as a prophecy"


def test_refine_sentence_threads_temperature():
    """refine_sentence forwards the (annealed) proposer temperature as generate_kwargs."""
    from _epic_common import refine_sentence

    client = MagicMock()
    client.generate.return_value = "  grander framing  "

    out = refine_sentence(client, "A sentence.", [], temperature=0.9)

    args, _ = client.generate.call_args
    assert args[1] == {"temperature": 0.9}  # generate_kwargs (positional)
    assert out == "grander framing"


def test_refine_sentence_without_temperature_uses_model_default():
    from _epic_common import refine_sentence

    client = MagicMock()
    client.generate.return_value = "grander"

    refine_sentence(client, "A sentence.", [])

    args, _ = client.generate.call_args
    assert args[1] is None  # no generate_kwargs → model default temperature


# ---------------------------------------------------------------------------
# write_summary
# ---------------------------------------------------------------------------


def test_write_summary(tmp_path):
    trace = [
        {
            "iteration": 1,
            "direction": "make it grand",
            "sentence": "The hero strode forth for milk.",
            "judge_response": "6/10\nCONTINUE: grander",
            "score": 6,
            "action": "CONTINUE",
            "next_direction": "grander",
        },
        {
            "iteration": 2,
            "direction": "grander",
            "sentence": "Beneath a dying sun the hero claimed the sacred milk.",
            "judge_response": "9/10\nDONE: cosmic",
            "score": 9,
            "action": "DONE",
            "next_direction": None,
            "status": "accepted (new best)",
        },
    ]
    summary_path = write_summary(tmp_path, trace, seed_sentence="A man buys milk.", gen_model="g", judge_model="j")
    assert summary_path.exists()
    text = summary_path.read_text()
    assert "Iteration 1" in text
    assert "make it grand" in text
    assert "The hero strode forth for milk." in text
    assert "Iteration 2" in text
    assert "DONE: cosmic" in text
    assert "Status: accepted (new best)" in text
    assert "A man buys milk." in text


def test_write_best(tmp_path):
    from _epic_common import write_best

    path = write_best(tmp_path, {"iteration": 3, "score": 8, "sentence": "Beneath a dying sun he claimed the milk."})
    assert path == tmp_path / "best.txt"
    text = path.read_text()
    assert "iteration 3" in text
    assert "8/10" in text
    assert "Beneath a dying sun he claimed the milk." in text


# ---------------------------------------------------------------------------
# Arg parsers
# ---------------------------------------------------------------------------


def test_loop_arg_parser_defaults():
    from epic_loop import build_arg_parser

    args = build_arg_parser("test").parse_args([])
    assert args.gen_model == "ollama:gemma4:e4b"
    assert args.judge_model == "ollama:gemma4:e4b"
    assert args.output_dir is None
    assert args.max_iterations == 10
    assert "milk" in args.seed_sentence


def test_loop_arg_parser_overrides():
    from epic_loop import build_arg_parser

    args = build_arg_parser("test").parse_args(
        [
            "--gen-model",
            "ollama:qwen3:8b",
            "--judge-model",
            "anthropic:claude-sonnet-4-6",
            "--seed-sentence",
            "She parks the car.",
            "--max-iterations",
            "3",
        ]
    )
    assert args.gen_model == "ollama:qwen3:8b"
    assert args.judge_model == "anthropic:claude-sonnet-4-6"
    assert args.seed_sentence == "She parks the car."
    assert args.max_iterations == 3


# ---------------------------------------------------------------------------
# Agent trace reconstruction (epic_agent.py)
# ---------------------------------------------------------------------------


def test_agent_parse_trace_single_iteration():
    from epic_agent import parse_agent_trace

    messages = [
        {"role": "user", "content": "Begin."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "write_epic_sentence",
                        "arguments": json.dumps({"directive": "make it grand"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "The hero strode forth for milk."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c2",
                    "type": "function",
                    "function": {
                        "name": "judge_epicness",
                        "arguments": json.dumps({"sentence": "The hero strode forth for milk."}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c2", "content": "9/10\nDONE: Maximum grandeur."},
        {"role": "assistant", "content": "Complete."},
    ]
    trace = parse_agent_trace(messages)
    assert len(trace) == 1
    assert trace[0]["iteration"] == 1
    assert trace[0]["direction"] == "make it grand"
    assert trace[0]["sentence"] == "The hero strode forth for milk."
    assert trace[0]["score"] == 9
    assert trace[0]["action"] == "DONE"


def test_agent_parse_trace_two_iterations():
    from epic_agent import parse_agent_trace

    messages = [
        {"role": "user", "content": "Begin."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "write_epic_sentence",
                        "arguments": json.dumps({"directive": "make it grand"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "The hero strode forth for milk."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "judge_epicness", "arguments": json.dumps({"sentence": "..."})},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c2", "content": "6/10\nCONTINUE: invoke a prophecy"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c3",
                    "type": "function",
                    "function": {
                        "name": "write_epic_sentence",
                        "arguments": json.dumps({"directive": "invoke a prophecy"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c3", "content": "As prophecy foretold, he claimed the milk."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c4",
                    "type": "function",
                    "function": {"name": "judge_epicness", "arguments": json.dumps({"sentence": "..."})},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c4", "content": "9/10\nDONE: Peak grandeur."},
        {"role": "assistant", "content": "Done."},
    ]
    trace = parse_agent_trace(messages)
    assert len(trace) == 2
    assert trace[0]["direction"] == "make it grand"
    assert trace[0]["action"] == "CONTINUE"
    assert trace[1]["direction"] == "invoke a prophecy"
    assert trace[1]["action"] == "DONE"


def test_agent_make_tools_increments_and_judges(tmp_path):
    from epic_agent import make_tools

    gen_client = MagicMock()
    gen_client.generate.return_value = "The hero strode forth for milk."
    judge_client = MagicMock()
    judge_client.generate.return_value = "8/10\nCONTINUE: invoke a prophecy"

    write_fn, judge_fn = make_tools(gen_client, judge_client, "A man buys milk.")

    sentence = write_fn("make it grand")
    assert sentence == "The hero strode forth for milk."

    result = judge_fn(sentence)
    # evaluate_sentence is stateless — no reset() dance.
    judge_client.reset.assert_not_called()
    assert "CONTINUE" in result


def test_agent_judge_falls_back_to_latest_when_relay_empty():
    """An empty relayed sentence falls back to the latest generation, not judging nothing."""
    from epic_agent import make_tools

    gen_client = MagicMock()
    gen_client.generate.return_value = "The hero strode forth for milk."
    judge_client = MagicMock()
    judge_client.generate.return_value = "9/10\nDONE: grandeur"

    write_fn, judge_fn = make_tools(gen_client, judge_client, "A man buys milk.")
    write_fn("make it grand")
    judge_fn("   ")  # empty relay

    # The judge call embedded the real latest sentence, not the blank.
    args, _ = judge_client.generate.call_args
    assert "The hero strode forth for milk." in args[0]


# ---------------------------------------------------------------------------
# EvaluatorOptimizer variant (epic_evaluator.py)
# ---------------------------------------------------------------------------


def test_evaluator_make_tools_records():
    from epic_evaluator import make_tools

    gen_client = MagicMock()
    gen_client.generate.return_value = "The hero strode forth for milk."
    judge_client = MagicMock()
    judge_client.generate.return_value = "8/10\nCONTINUE: invoke a prophecy"

    records: list[dict] = []
    write_fn, judge_fn = make_tools(gen_client, judge_client, "A man buys milk.", records)

    sentence = write_fn("make it grand")
    assert records[0]["iteration"] == 1
    assert records[0]["judge_response"] is None  # not yet judged

    result = judge_fn(sentence)
    judge_client.reset.assert_not_called()
    assert "CONTINUE" in result
    assert records[0]["action"] == "CONTINUE"
    assert records[0]["score"] == 8


def test_evaluator_judge_falls_back_to_latest():
    """An empty relayed sentence falls back to the most recent generation, not a crash."""
    from epic_evaluator import make_tools

    gen_client = MagicMock()
    gen_client.generate.return_value = "The hero strode forth for milk."
    judge_client = MagicMock()
    judge_client.generate.return_value = "9/10\nDONE: grandeur"

    records: list[dict] = []
    write_fn, judge_fn = make_tools(gen_client, judge_client, "A man buys milk.", records)
    write_fn("make it grand")
    judge_fn("")  # empty relay

    args, _ = judge_client.generate.call_args
    assert "The hero strode forth for milk." in args[0]
    assert records[0]["action"] == "DONE"


# ---------------------------------------------------------------------------
# Simulated annealing (epic_anneal.py)
# ---------------------------------------------------------------------------


def test_anneal_accept_uphill_and_ties_always():
    import random

    from epic_anneal import _accept

    rng = random.Random(0)
    assert _accept(2, 0.01, rng) is True  # uphill, even at near-zero T
    assert _accept(0, 0.01, rng) is True  # tie — a free sideways move


def test_anneal_accept_downhill_greedy_at_zero_temperature():
    import random

    from epic_anneal import _accept

    assert _accept(-1, 0.0, random.Random(0)) is False


def test_anneal_accept_downhill_high_temperature_mostly_accepts():
    import random

    from epic_anneal import _accept

    rng = random.Random(0)
    accepts = sum(_accept(-1, 100.0, rng) for _ in range(100))
    assert accepts > 90


def test_anneal_arg_parser_has_sa_knobs():
    from epic_anneal import build_parser

    args = build_parser().parse_args([])
    assert args.initial_temp == 2.0
    assert args.cooling_rate == 0.85
    assert args.seed is None

    args = build_parser().parse_args(["--initial-temp", "3", "--cooling-rate", "0.9", "--seed", "7"])
    assert (args.initial_temp, args.cooling_rate, args.seed) == (3.0, 0.9, 7)


def test_anneal_proposer_temperature_hot_to_cold():
    from epic_anneal import PROPOSER_TEMP_COLD, PROPOSER_TEMP_HOT, _proposer_temperature

    assert _proposer_temperature(2.0, 2.0) == PROPOSER_TEMP_HOT  # f = 1.0 → fully hot
    assert _proposer_temperature(0.0, 2.0) == PROPOSER_TEMP_COLD  # f = 0.0 → fully cold
    mid = _proposer_temperature(1.0, 2.0)  # f = 0.5 → halfway
    assert PROPOSER_TEMP_COLD < mid < PROPOSER_TEMP_HOT
    assert _proposer_temperature(4.0, 4.0) == PROPOSER_TEMP_HOT  # schedule depends on fraction, not T0
