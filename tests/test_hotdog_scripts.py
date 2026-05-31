import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from _hotdog_common import parse_evaluator_response, write_summary


# ---------------------------------------------------------------------------
# build_image_prompt — the subject anchor must lead the prompt
# ---------------------------------------------------------------------------


def test_build_image_prompt_prepends_for_plain_prompt():
    from _hotdog_common import SUBJECT_ANCHOR, build_image_prompt

    assert build_image_prompt("a hot hotdog") == f"{SUBJECT_ANCHOR}, a hot hotdog"


def test_build_image_prompt_front_loads_when_subject_buried():
    """A 'single hotdog' mention buried mid-prompt still gets the anchor front-loaded.

    Regression for non-hotdog output: the summarizer puts hot details first, so the subject
    lands late; the old 'contains anywhere' guard skipped the anchor and the subject was
    under-rendered.
    """
    from _hotdog_common import SUBJECT_ANCHOR, build_image_prompt

    buried = "raging white-hot flames, a single hotdog at the center, molten lava glow"
    out = build_image_prompt(buried)
    assert out.startswith(f"{SUBJECT_ANCHOR},")
    assert out.lower().startswith("a single hotdog")


def test_build_image_prompt_no_double_anchor_when_already_leading():
    from _hotdog_common import SUBJECT_ANCHOR, build_image_prompt

    already = f"{SUBJECT_ANCHOR}, engulfed in flames"
    assert build_image_prompt(already) == already


def test_build_summarizer_prompt_targets_heat_not_subject():
    """The summarizer spends the budget on heat and defers the subject to the anchor."""
    from _hotdog_common import build_summarizer_prompt, prompt_word_budget

    out = build_summarizer_prompt(256)
    assert str(prompt_word_budget(256)) in out          # {max_words} budget is filled in
    assert "do not restate the subject" in out.lower()  # subject is the anchor's job, not the summary's


def test_parse_done_response():
    text = "Hotness: 9/10\nDONE: The hotdog is fully engulfed in flames and cannot get hotter."
    result = parse_evaluator_response(text)
    assert result["action"] == "DONE"
    assert result["score"] == 9
    assert "flames" in result["reasoning"]
    assert result["next_prompt"] is None


def test_parse_continue_response():
    text = "8/10 hotness.\nCONTINUE: a flaming hotdog with scorched bun, lava background"
    result = parse_evaluator_response(text)
    assert result["action"] == "CONTINUE"
    assert result["score"] == 8
    assert result["next_prompt"] == "a flaming hotdog with scorched bun, lava background"
    assert result["reasoning"] is None


def test_parse_unknown_response():
    text = "I cannot determine the hotness level of this image."
    result = parse_evaluator_response(text)
    assert result["action"] == "unknown"
    assert result["score"] is None


def test_parse_done_takes_priority_over_continue():
    text = "CONTINUE: hotter prompt\nDONE: actually it is done"
    result = parse_evaluator_response(text)
    assert result["action"] == "DONE"


def test_parse_score_out_of_range_ignored():
    text = "11/10 hotness.\nDONE: off the charts"
    result = parse_evaluator_response(text)
    assert result["score"] is None


def test_write_summary(tmp_path):
    trace = [
        {
            "iteration": 1,
            "prompt": "a hot hotdog",
            "image_path": str(tmp_path / "01.png"),
            "evaluator_response": "6/10\nCONTINUE: hotter",
            "score": 6,
            "action": "CONTINUE",
            "next_prompt": "hotter",
        },
        {
            "iteration": 2,
            "prompt": "hotter",
            "image_path": str(tmp_path / "02.png"),
            "evaluator_response": "9/10\nDONE: max heat",
            "score": 9,
            "action": "DONE",
            "next_prompt": None,
        },
    ]
    summary_path = write_summary(tmp_path, trace)
    assert summary_path.exists()
    text = summary_path.read_text()
    assert "Iteration 1" in text
    assert "a hot hotdog" in text
    assert "CONTINUE: hotter" in text
    assert "Iteration 2" in text
    assert "DONE: max heat" in text



def test_parse_continue_captures_multiline_description():
    # The CONTINUE description is matched with re.DOTALL so a multi-line, free-form
    # description (the describe → summarize chain's first stage) is captured in full.
    text = "8/10 hotness.\nCONTINUE: a flaming hotdog on lava\nwith dramatic backlighting."
    result = parse_evaluator_response(text)
    assert result["action"] == "CONTINUE"
    assert result["next_prompt"] == "a flaming hotdog on lava\nwith dramatic backlighting."


def test_parse_does_not_match_done_mid_sentence():
    text = "Not done: the hotdog needs more heat. Score: 5/10\nCONTINUE: hotter hotdog on fire"
    result = parse_evaluator_response(text)
    assert result["action"] == "CONTINUE"
    assert result["next_prompt"] == "hotter hotdog on fire"


def test_loop_arg_parser_defaults():
    from aimu.models import HuggingFaceImageModel
    from hotdog_loop import build_arg_parser
    args = build_arg_parser("test").parse_args([])
    assert args.image_model == HuggingFaceImageModel.SD_3_5_MEDIUM
    assert args.eval_model == "ollama:gemma4:e4b"
    assert args.output_dir is None
    assert args.max_iterations == 10


def test_loop_arg_parser_overrides():
    from hotdog_loop import build_arg_parser
    args = build_arg_parser("test").parse_args([
        "--image-model", "hf:stabilityai/stable-diffusion-xl-base-1.0",
        "--eval-model", "ollama:gemma4:26b",
        "--output-dir", "/tmp/hotdog",
        "--max-iterations", "3",
    ])
    assert args.image_model == "hf:stabilityai/stable-diffusion-xl-base-1.0"
    assert args.eval_model == "ollama:gemma4:26b"
    assert args.output_dir == "/tmp/hotdog"
    assert args.max_iterations == 3


def test_agent_arg_parser_defaults():
    from aimu.models import HuggingFaceImageModel
    from hotdog_agent import build_arg_parser
    args = build_arg_parser("test").parse_args([])
    assert args.image_model == HuggingFaceImageModel.SD_3_5_MEDIUM
    assert args.eval_model == "ollama:gemma4:e4b"
    assert args.output_dir is None
    assert args.max_iterations == 10


def test_agent_parse_trace_single_iteration():
    from hotdog_agent import parse_agent_trace
    messages = [
        {"role": "user", "content": "Start the experiment."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "generate_hotdog_image", "arguments": json.dumps({"prompt": "a hot hotdog"})},
            }],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "/out/01.png"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "c2", "type": "function",
                "function": {"name": "evaluate_hotness", "arguments": json.dumps({"image_path": "/out/01.png"})},
            }],
        },
        {"role": "tool", "tool_call_id": "c2", "content": "9/10\nDONE: Maximum achieved."},
        {"role": "assistant", "content": "Experiment complete."},
    ]
    trace = parse_agent_trace(messages)
    assert len(trace) == 1
    assert trace[0]["iteration"] == 1
    assert trace[0]["prompt"] == "a hot hotdog"
    assert trace[0]["image_path"] == "/out/01.png"
    assert trace[0]["score"] == 9
    assert trace[0]["action"] == "DONE"


def test_agent_parse_trace_two_iterations():
    from hotdog_agent import parse_agent_trace
    messages = [
        {"role": "user", "content": "Start."},
        {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "generate_hotdog_image", "arguments": json.dumps({"prompt": "a hot hotdog"})}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "/out/01.png"},
        {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "evaluate_hotness", "arguments": json.dumps({"image_path": "/out/01.png"})}}],
        },
        {"role": "tool", "tool_call_id": "c2", "content": "6/10\nCONTINUE: flaming hotdog on lava"},
        {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "c3", "type": "function", "function": {"name": "generate_hotdog_image", "arguments": json.dumps({"prompt": "flaming hotdog on lava"})}}],
        },
        {"role": "tool", "tool_call_id": "c3", "content": "/out/02.png"},
        {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "c4", "type": "function", "function": {"name": "evaluate_hotness", "arguments": json.dumps({"image_path": "/out/02.png"})}}],
        },
        {"role": "tool", "tool_call_id": "c4", "content": "9/10\nDONE: Peak hotness."},
        {"role": "assistant", "content": "Done."},
    ]
    trace = parse_agent_trace(messages)
    assert len(trace) == 2
    assert trace[0]["prompt"] == "a hot hotdog"
    assert trace[0]["action"] == "CONTINUE"
    assert trace[1]["prompt"] == "flaming hotdog on lava"
    assert trace[1]["action"] == "DONE"


def test_make_tools_counter_increments(tmp_path):
    from hotdog_agent import make_tools

    image_client = MagicMock()
    image_client.generate.return_value = str(tmp_path / "generated.png")
    (tmp_path / "generated.png").touch()

    eval_client = MagicMock()
    eval_client.chat.return_value = "8/10\nCONTINUE: hotter hotdog"

    generate_fn, _, _ = make_tools(image_client, eval_client, tmp_path, 256)

    path1 = generate_fn("first prompt")
    assert path1 == str(tmp_path / "01.png")
    assert (tmp_path / "01.png").exists()

    (tmp_path / "generated.png").touch()
    path2 = generate_fn("second prompt")
    assert path2 == str(tmp_path / "02.png")
    assert (tmp_path / "02.png").exists()


def test_make_tools_evaluate_uses_stateless_generate(tmp_path):
    """The agent's evaluate tool routes through evaluate_image → stateless generate(images=)."""
    from _hotdog_common import EVALUATOR_PROMPT
    from hotdog_agent import make_tools

    image_client = MagicMock()
    eval_client = MagicMock()
    eval_client.generate.return_value = "8/10\nCONTINUE: hotter"

    _, evaluate_fn, _ = make_tools(image_client, eval_client, tmp_path, 256)
    result = evaluate_fn("/some/01.png")

    # generate() is stateless: no reset() dance, no history pollution.
    eval_client.reset.assert_not_called()
    eval_client.generate.assert_called_once_with(EVALUATOR_PROMPT, images=["/some/01.png"])
    assert "CONTINUE" in result


def test_evaluate_image_uses_stateless_generate(tmp_path):
    from _hotdog_common import EVALUATOR_PROMPT, evaluate_image

    eval_client = MagicMock()
    eval_client.generate.return_value = "7/10\nCONTINUE: more char"

    response, parsed = evaluate_image(eval_client, tmp_path / "01.png")

    eval_client.reset.assert_not_called()
    eval_client.generate.assert_called_once_with(EVALUATOR_PROMPT, images=[str(tmp_path / "01.png")])
    assert parsed["score"] == 7
    assert parsed["action"] == "CONTINUE"


def test_evaluate_image_retries_when_no_score(tmp_path):
    """A scoreless first reply triggers a retry with SCORE_REMINDER appended."""
    from _hotdog_common import EVALUATOR_PROMPT, SCORE_REMINDER, evaluate_image

    eval_client = MagicMock()
    eval_client.generate.side_effect = ["no score here", "9/10\nDONE: peak heat"]

    response, parsed = evaluate_image(eval_client, tmp_path / "01.png", max_retries=2)

    assert eval_client.generate.call_count == 2
    assert eval_client.generate.call_args_list[1].args[0] == EVALUATOR_PROMPT + SCORE_REMINDER
    assert parsed["score"] == 9
    assert parsed["action"] == "DONE"


def test_evaluator_arg_parser_defaults():
    from aimu.models import HuggingFaceImageModel
    from hotdog_evaluator import build_arg_parser

    args = build_arg_parser("test").parse_args([])
    assert args.image_model == HuggingFaceImageModel.SD_3_5_MEDIUM
    assert args.eval_model == "ollama:gemma4:e4b"
    assert args.max_iterations == 10


def test_evaluator_make_tools_records(tmp_path):
    from hotdog_evaluator import make_tools

    image_client = MagicMock()
    image_client.generate.return_value = str(tmp_path / "generated.png")
    (tmp_path / "generated.png").touch()

    vision_client = MagicMock()
    vision_client.generate.return_value = "8/10\nCONTINUE: hotter hotdog"

    records = []
    generate_fn, evaluate_fn = make_tools(image_client, vision_client, tmp_path, records)

    path1 = generate_fn("first prompt")
    assert path1 == str(tmp_path / "01.png")
    assert records[0]["iteration"] == 1
    assert records[0]["evaluator_response"] is None  # not yet evaluated

    result = evaluate_fn(path1)
    # evaluate_image uses stateless generate(images=) — no reset() dance.
    vision_client.reset.assert_not_called()
    assert "CONTINUE" in result
    # The matching record is filled in with the parsed verdict.
    assert records[0]["action"] == "CONTINUE"
    assert records[0]["score"] == 8


def test_evaluator_evaluate_falls_back_to_latest_image(tmp_path):
    """A mangled relayed path falls back to the most recent generation, not a crash."""
    from _hotdog_common import EVALUATOR_PROMPT
    from hotdog_evaluator import make_tools

    image_client = MagicMock()
    image_client.generate.return_value = str(tmp_path / "generated.png")
    (tmp_path / "generated.png").touch()

    vision_client = MagicMock()
    vision_client.generate.return_value = "9/10\nDONE: maximal char"

    records = []
    generate_fn, evaluate_fn = make_tools(image_client, vision_client, tmp_path, records)
    real_path = generate_fn("a hot hotdog")

    evaluate_fn("/nonexistent/path/that/llm/hallucinated.png")
    # The vision call used the real latest image, not the bogus path.
    vision_client.generate.assert_called_once_with(EVALUATOR_PROMPT, images=[real_path])
    assert records[0]["action"] == "DONE"


def test_summarize_for_image_uses_stateless_generate():
    """summarize_for_image is text-only and stateless (generate, no reset)."""
    from _hotdog_common import summarize_for_image

    client = MagicMock()
    client.generate.return_value = "  flaming hotdog, charred, lava lighting  "

    out = summarize_for_image(client, "a long flaming description", 256)

    client.reset.assert_not_called()
    assert out == "flaming hotdog, charred, lava lighting"  # stripped
    # Text-only: no images forwarded.
    _, kwargs = client.generate.call_args
    assert "images" not in kwargs


def test_refine_from_best_uses_stateless_generate(tmp_path):
    """refine_from_best is a one-shot vision call via stateless generate(images=)."""
    from hotdog_loop_climbing import refine_from_best

    client = MagicMock()
    client.generate.return_value = "  brighter flames and more char  "

    out = refine_from_best(client, str(tmp_path / "best.png"), rejected=["more steam"])

    client.reset.assert_not_called()
    client.generate.assert_called_once()
    args, kwargs = client.generate.call_args
    assert kwargs["images"] == [str(tmp_path / "best.png")]
    assert "more steam" in args[0]  # the rejected idea is listed in the avoid clause
    assert out == "brighter flames and more char"


# ---------------------------------------------------------------------------
# Simulated annealing (hotdog_anneal.py)
# ---------------------------------------------------------------------------


def test_anneal_accept_uphill_and_ties_always():
    """Non-worsening moves (Δ >= 0) are always accepted, at any temperature."""
    import random

    from hotdog_anneal import _accept

    rng = random.Random(0)
    assert _accept(2, 0.01, rng) is True   # uphill, even at near-zero T
    assert _accept(0, 0.01, rng) is True   # tie — a free sideways move


def test_anneal_accept_downhill_greedy_at_zero_temperature():
    """At T = 0 a worsening move is never accepted — the pure-greedy (climber) limit."""
    import random

    from hotdog_anneal import _accept

    assert _accept(-1, 0.0, random.Random(0)) is False


def test_anneal_accept_downhill_high_temperature_mostly_accepts():
    """High T accepts worsening moves nearly always (exp(-1/100) ≈ 0.99)."""
    import random

    from hotdog_anneal import _accept

    rng = random.Random(0)
    accepts = sum(_accept(-1, 100.0, rng) for _ in range(100))
    assert accepts > 90


def test_anneal_arg_parser_has_sa_knobs():
    from hotdog_anneal import build_parser

    args = build_parser().parse_args([])
    assert args.initial_temp == 2.0
    assert args.cooling_rate == 0.85
    assert args.seed is None

    args = build_parser().parse_args(["--initial-temp", "3", "--cooling-rate", "0.9", "--seed", "7"])
    assert (args.initial_temp, args.cooling_rate, args.seed) == (3.0, 0.9, 7)
