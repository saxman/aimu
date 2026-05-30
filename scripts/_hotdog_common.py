"""Shared helpers for the hotdog heating loop scripts."""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

from aimu.models import HuggingFaceImageModel

EVALUATOR_PROMPT = """\
You are evaluating how visually "hot" this hotdog image is.
Rate its hotness from 1 to 10 (10 = can't be conceivably hotter, 1 = cold).
Then decide: can this hotdog get any hotter? If not, output exactly:
DONE: <your reasoning>
If it can get hotter, output exactly:
CONTINUE: <a natural-language description of how the next image should look to make
the hotdog hotter — describe the flames, char, spices, steam, colors, and lighting>
The scene must depict exactly ONE single hotdog — never multiple hotdogs, a pile,
or a platter.
"""


# Second stage of the prompt chain: condense the evaluator's free-form description
# into a prompt that fits the image model's token budget. ``{max_words}`` is filled
# from the model's ImageSpec.max_prompt_tokens via build_summarizer_prompt().
SUMMARIZER_PROMPT = """\
Condense the following description into a text-to-image prompt.
Output ONLY the prompt — comma-separated visual descriptors, no full sentences,
under {max_words} words. It must depict exactly ONE single hotdog. Put the most
important "hot" details first; image encoders truncate prompts past their limit.
"""


# Positive subject anchor prepended to every generation. CLIP negative prompts
# suppress a *concept*, not a *count* — listing "hotdogs" in the negative prompt
# removes hotdogs entirely. Stating a singular subject in the positive prompt is
# the reliable lever for "exactly one".
SUBJECT_ANCHOR = "a single hotdog, one sausage in one bun, solo, centered, close-up shot"

# Generic plurality/duplication cues to discourage. Deliberately omits the word
# "hotdog" (which would suppress the subject). HF uses it; providers without
# negative-prompt support (e.g. Gemini Nano Banana) silently ignore it.
NEGATIVE_PROMPT = "multiple, two, several, pile, platter, group, crowd, duplicate"


def build_image_prompt(prompt: str) -> str:
    """Prepend the single-hotdog subject anchor unless the prompt already states it."""
    if "single hotdog" in prompt.lower():
        return prompt
    return f"{SUBJECT_ANCHOR}, {prompt}"


def prompt_word_budget(max_prompt_tokens: int) -> int:
    """A conservative word cap for a model's token budget (≈0.45×tokens, min 20).

    English averages ~1.3 tokens/word, so 0.45×tokens words ≈ 0.6×tokens — comfortably
    under the limit even if the model overshoots.
    """
    return max(20, int(max_prompt_tokens * 0.45))


def build_summarizer_prompt(max_prompt_tokens: int) -> str:
    """Format SUMMARIZER_PROMPT with a word budget derived from the model's token limit."""
    return SUMMARIZER_PROMPT.format(max_words=prompt_word_budget(max_prompt_tokens))


def summarize_for_image(client, description: str, max_prompt_tokens: int) -> str:
    """Condense a free-form description into an image prompt that fits ``max_prompt_tokens``.

    The second stage of the describe → summarize chain. Runs the budget-aware
    summarizer instruction through a text ``client`` (the eval model is reused —
    summarization is text-only) and returns the stripped prompt. The client is reset
    first so the call doesn't inherit or pollute prior conversation state.
    """
    client.reset()
    instruction = build_summarizer_prompt(max_prompt_tokens)
    return client.chat(f"{instruction}\nDescription:\n{description}").strip()


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """Build an argument parser with the options common to both hotdog scripts."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--image-model",
        default=HuggingFaceImageModel.SD_3_5_MEDIUM,
        help="Image model: 'provider:model_id' string or enum member "
        "(default: SD 3.5 Medium, a 256-token T5 model)",
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


def write_summary(
    output_dir: Path,
    trace: list[dict],
    *,
    image_model: str | None = None,
    eval_model: str | None = None,
    summarizer_instruction: str | None = None,
) -> Path:
    """Write summary.txt with the full iteration trace to output_dir.

    Records exactly what is sent to each model: the model names, the constant image
    negative prompt, the evaluator instruction, and (when summarizing) the budget-aware
    summarizer instruction, once at the top; then per iteration the working prompt
    alongside the anchored prompt actually sent.
    """
    path = output_dir / "summary.txt"

    header_lines = ["=== Constant model inputs ==="]
    if image_model:
        header_lines.append(f"Image model: {image_model}")
    if eval_model:
        header_lines.append(f"Eval model:  {eval_model}")
    header_lines += [
        f"Image negative prompt: {NEGATIVE_PROMPT}",
        f"Evaluator instruction:\n{EVALUATOR_PROMPT.strip()}",
    ]
    if summarizer_instruction:
        header_lines.append(f"Summarizer instruction:\n{summarizer_instruction.strip()}")
    sections = ["\n".join(header_lines)]
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


def build_collage(image_paths: list, output_dir: Path) -> Path | None:
    """Assemble the generated images into a near-square grid collage.

    Images are placed left-to-right, top-to-bottom in generation order. The grid
    uses the most-square layout that holds all of them (cols = ceil(sqrt(n))); any
    trailing empty cells stay blank. Returns the collage path, or None when there
    are no images. Requires Pillow (installed with the image extra).
    """
    from math import ceil, sqrt

    from PIL import Image

    paths = [Path(p) for p in image_paths if p and Path(p).exists()]
    if not paths:
        return None

    cols = ceil(sqrt(len(paths)))
    rows = ceil(len(paths) / cols)

    tiles = [Image.open(p).convert("RGB") for p in paths]
    cell_w = max(tile.width for tile in tiles)
    cell_h = max(tile.height for tile in tiles)

    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    for i, tile in enumerate(tiles):
        if tile.size != (cell_w, cell_h):
            tile = tile.resize((cell_w, cell_h))
        row, col = divmod(i, cols)
        canvas.paste(tile, (col * cell_w, row * cell_h))

    path = output_dir / "collage.png"
    canvas.save(path)
    return path


def collage_generated_images(output_dir: Path) -> Path | None:
    """Build the collage from the numbered image files saved in ``output_dir``.

    Scans for ``NN.png`` files (the per-iteration images) rather than relying on a
    trace, so the collage includes every generated image even if the run was
    interrupted before that image could be evaluated. Non-numeric names (the raw
    pre-rename file, ``collage.png``) are skipped. Returns the collage path, or None.
    """
    paths = sorted(
        (p for p in output_dir.glob("*.png") if p.stem.isdigit()),
        key=lambda p: int(p.stem),
    )
    return build_collage([str(p) for p in paths], output_dir)
