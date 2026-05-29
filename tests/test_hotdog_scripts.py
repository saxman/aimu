import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from _hotdog_common import parse_evaluator_response, write_summary, get_ollama_model


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


def test_get_ollama_model_valid():
    from aimu.models.ollama import OllamaModel
    model = get_ollama_model("gemma4:e4b")
    assert model == OllamaModel.GEMMA_4_E4B


def test_get_ollama_model_unknown():
    with pytest.raises(ValueError, match="Unknown Ollama model"):
        get_ollama_model("nonexistent:model")


def test_get_ollama_model_no_vision():
    with pytest.raises(ValueError, match="does not support vision"):
        get_ollama_model("llama3.1:8b")


def test_parse_does_not_match_done_mid_sentence():
    text = "Not done: the hotdog needs more heat. Score: 5/10\nCONTINUE: hotter hotdog on fire"
    result = parse_evaluator_response(text)
    assert result["action"] == "CONTINUE"
    assert result["next_prompt"] == "hotter hotdog on fire"


def test_loop_arg_parser_defaults():
    from hotdog_loop import build_arg_parser
    args = build_arg_parser().parse_args([])
    assert args.image_model == "FLUX_SCHNELL"
    assert args.eval_model == "gemma4:e4b"
    assert args.output_dir is None
    assert args.max_iterations == 10


def test_loop_arg_parser_overrides():
    from hotdog_loop import build_arg_parser
    args = build_arg_parser().parse_args([
        "--image-model", "SDXL_BASE",
        "--eval-model", "gemma4:26b",
        "--output-dir", "/tmp/hotdog",
        "--max-iterations", "3",
    ])
    assert args.image_model == "SDXL_BASE"
    assert args.eval_model == "gemma4:26b"
    assert args.output_dir == "/tmp/hotdog"
    assert args.max_iterations == 3
