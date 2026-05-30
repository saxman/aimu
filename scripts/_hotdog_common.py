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
CONTINUE: <a full, natural-language description of how the next image should look
to make the hotdog hotter — describe the flames, char, spices, steam, colors, and
lighting in as much detail as you like>
The scene must depict exactly ONE single hotdog — never multiple hotdogs, a pile,
or a platter.
"""


# Second stage of the prompt chain: condense the evaluator's free-form description
# into a short prompt for image models with tight token limits (e.g. SDXL's CLIP
# encoder caps at 77 tokens). Run through a text client via summarize_for_image().
SUMMARIZER_PROMPT = """\
Condense the following description into a short text-to-image prompt.
Output ONLY the prompt — comma-separated visual descriptors, no full sentences,
under 40 words. It must depict exactly ONE single hotdog. Put the most important
"hot" details first; image encoders truncate long prompts.

Description:
{description}
"""


# Positive subject anchor prepended to every generation. CLIP negative prompts
# suppress a *concept*, not a *count* — listing "hotdogs" in the negative prompt
# removes hotdogs entirely. Stating a singular subject in the positive prompt is
# the reliable lever for "exactly one".
SUBJECT_ANCHOR = "a single hotdog, one sausage in one bun"

# Generic plurality/duplication cues to discourage. Deliberately omits the word
# "hotdog" (which would suppress the subject). HF uses it; providers without
# negative-prompt support (e.g. Gemini Nano Banana) silently ignore it.
NEGATIVE_PROMPT = "multiple, two, several, pile, platter, group, crowd, duplicate"


def build_image_prompt(prompt: str) -> str:
    """Prepend the single-hotdog subject anchor unless the prompt already states it."""
    if "single hotdog" in prompt.lower():
        return prompt
    return f"{SUBJECT_ANCHOR}, {prompt}"


def summarize_for_image(client, description: str) -> str:
    """Condense a free-form description into a short image-generation prompt.

    The second stage of the describe → summarize chain. Runs ``SUMMARIZER_PROMPT``
    through a text ``client`` (the eval model is reused — summarization is text-only)
    and returns the stripped short prompt. The client is reset first so the call
    doesn't inherit or pollute prior conversation state.
    """
    client.reset()
    return client.chat(SUMMARIZER_PROMPT.format(description=description)).strip()


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
      next_prompt -- text after CONTINUE: (or None); the full natural-language
                     description, which may span multiple lines
    """
    score = None
    score_match = re.search(r'\b(\d+)/10\b', text)
    if score_match:
        val = int(score_match.group(1))
        if 1 <= val <= 10:
            score = val

    done_match = re.search(r'^DONE\s*:\s*(.+)', text, re.DOTALL | re.IGNORECASE | re.MULTILINE)
    # DOTALL so the CONTINUE description can run across multiple lines.
    continue_match = re.search(r'^CONTINUE\s*:\s*(.+)', text, re.DOTALL | re.IGNORECASE | re.MULTILINE)

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
    """Write summary.txt with the full iteration trace to output_dir.

    Records exactly what is sent to each model: the constant image negative prompt
    and evaluator instruction once at the top, then per iteration the working prompt
    alongside the anchored prompt actually sent to the image model.
    """
    path = output_dir / "summary.txt"

    sections = [
        "\n".join([
            "=== Constant model inputs ===",
            f"Image negative prompt: {NEGATIVE_PROMPT}",
            f"Evaluator instruction:\n{EVALUATOR_PROMPT.strip()}",
            f"Summarizer instruction:\n{SUMMARIZER_PROMPT.strip()}",
        ])
    ]
    for entry in trace:
        lines = [
            f"=== Iteration {entry['iteration']} ===",
            f"Working prompt:    {entry['prompt']}",
            f"Image prompt sent: {entry.get('image_prompt', entry['prompt'])}",
            f"Image:             {entry['image_path']}",
        ]
        if entry.get("score") is not None:
            lines.append(f"Hotness Score: {entry['score']}/10")
        lines.append(f"Evaluator Response (full description):\n{entry['evaluator_response']}")
        if entry.get("summarized_prompt"):
            lines.append(f"Summarized → next image prompt: {entry['summarized_prompt']}")
        sections.append("\n".join(lines))
    path.write_text("\n\n".join(sections) + "\n")
    return path
