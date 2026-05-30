"""Shared helpers for the hotdog heating loop scripts."""

from __future__ import annotations

import re
from pathlib import Path


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



EVALUATOR_PROMPT = """\
You are evaluating how visually "hot" this hotdog image is.
Rate its hotness from 1 to 10 (10 = blazing inferno hotdog, 1 = cold).
Then decide: can this hotdog get any hotter? If not, output exactly:
DONE: <your reasoning>
If it can get hotter, output exactly:
CONTINUE: <a refined image generation prompt that will make it hotter>
"""
