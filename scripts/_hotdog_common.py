"""Shared helpers for the hotdog heating loop scripts."""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

EVALUATOR_PROMPT = """\
You are evaluating how visually "hot" this hotdog image is.
Rate its hotness from 1 to 10 (10 = blazing inferno hotdog, 1 = cold).
Then decide: can this hotdog get any hotter? If not, output exactly:
DONE: <your reasoning>
If it can get hotter, output exactly:
CONTINUE: <a refined image generation prompt that will make it hotter>
The prompt must depict exactly ONE single hotdog — never multiple hotdogs, a
pile, or a platter. Keep the CONTINUE prompt concise — under 40 words, as
comma-separated visual descriptors (no full sentences). Image encoders truncate
long prompts, so put the most important "hot" details first.
"""


# Pushes the diffusion model away from rendering more than one hotdog. Applied as
# the image client's negative_prompt — used by HuggingFace; providers without
# negative-prompt support (e.g. Gemini Nano Banana) silently ignore it.
NEGATIVE_PROMPT = "multiple hotdogs, two hotdogs, several hotdogs, pile of hotdogs, platter, group of hotdogs"


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """Build an argument parser with the options common to both hotdog scripts."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--image-model",
        default="hf:stabilityai/stable-diffusion-xl-base-1.0",
        help="Image model string in 'provider:model_id' form (default: hf:stabilityai/stable-diffusion-xl-base-1.0)",
    )
    p.add_argument(
        "--eval-model",
        default="ollama:gemma4:e4b",
        help="Vision eval model string in 'provider:model_id' form (default: ollama:gemma4:e4b)",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for images and summary (default: output/hotdog/<timestamp>/)",
    )
    p.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Hard cap on iteration count (default: 10; 0 = run until the evaluator says DONE)",
    )
    return p


def resolve_output_dir(output_dir: str | None) -> Path:
    """Resolve the output directory, defaulting to output/hotdog/<timestamp>/."""
    if output_dir:
        return Path(output_dir)
    from aimu import paths as aimu_paths

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return aimu_paths.output / "hotdog" / timestamp


def parse_evaluator_response(text: str) -> dict:
    """Parse DONE/CONTINUE signal and hotness score from an evaluator response.

    Returns a dict with keys:
      action      -- "DONE", "CONTINUE", or "unknown"
      score       -- int 1-10 or None
      reasoning   -- text after DONE: (or None)
      next_prompt -- text after CONTINUE: (or None)
    """
    score = None
    score_match = re.search(r'\b(\d+)/10\b', text)
    if score_match:
        val = int(score_match.group(1))
        if 1 <= val <= 10:
            score = val

    done_match = re.search(r'^DONE\s*:\s*(.+)', text, re.DOTALL | re.IGNORECASE | re.MULTILINE)
    continue_match = re.search(r'^CONTINUE\s*:\s*(.+)$', text, re.IGNORECASE | re.MULTILINE)

    # DONE takes priority if both are present
    if done_match:
        return {
            "action": "DONE",
            "score": score,
            "reasoning": done_match.group(1).strip(),
            "next_prompt": None,
        }
    if continue_match:
        return {
            "action": "CONTINUE",
            "score": score,
            "reasoning": None,
            "next_prompt": continue_match.group(1).strip(),
        }
    return {
        "action": "unknown",
        "score": score,
        "reasoning": text.strip(),
        "next_prompt": None,
    }


def write_summary(output_dir: Path, trace: list[dict]) -> Path:
    """Write summary.txt with the full iteration trace to output_dir."""
    path = output_dir / "summary.txt"
    sections = []
    for entry in trace:
        lines = [
            f"=== Iteration {entry['iteration']} ===",
            f"Prompt: {entry['prompt']}",
            f"Image:  {entry['image_path']}",
        ]
        if entry.get("score") is not None:
            lines.append(f"Hotness Score: {entry['score']}/10")
        lines.append(f"Evaluator Response:\n{entry['evaluator_response']}")
        sections.append("\n".join(lines))
    path.write_text("\n\n".join(sections) + "\n")
    return path
