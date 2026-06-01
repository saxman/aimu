"""Shared helpers for the hotdog heating loop scripts."""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from aimu.models import HuggingFaceImageModel


def _image_model_enums() -> list[tuple[str, type]]:
    """Every installed image-provider ImageModel enum as ``(provider_prefix, enum)``.

    HuggingFace is always present (the package hard-imports it above); Gemini is
    added only when the [google] extra is installed."""
    enums: list[tuple[str, type]] = [("hf", HuggingFaceImageModel)]
    try:
        from aimu.models import GeminiImageModel

        enums.append(("gemini", GeminiImageModel))
    except ImportError:
        pass
    return enums


def resolve_image_model(model):
    """Normalize an ``--image-model`` value into something ``aimu.image_client()`` accepts.

    Pass-through for ImageModel enum members and ``'provider:model_id'`` strings —
    the factory handles both. A bare member name (e.g. ``'FLUX_2_KLEIN_4B'`` or
    ``'NANO_BANANA'``) is looked up across every installed image-provider enum,
    matching the ``--image-model`` help text's "or enum member" promise.

    A bare name that exists in more than one provider's enum is **ambiguous** and
    raises ``ValueError`` rather than silently picking one — pass the explicit
    ``'provider:model_id'`` string to disambiguate."""
    if not isinstance(model, str) or ":" in model:
        return model
    matches = [(prefix, enum_cls[model]) for prefix, enum_cls in _image_model_enums() if model in enum_cls.__members__]
    if len(matches) == 1:
        return matches[0][1]
    if not matches:
        available = sorted({name for _, enum_cls in _image_model_enums() for name in enum_cls.__members__})
        raise ValueError(f"Unknown image model {model!r}. Pass a 'provider:model_id' string or one of: {available}")
    options = ", ".join(f"{prefix}:{member.value}" for prefix, member in matches)
    providers = ", ".join(prefix for prefix, _ in matches)
    raise ValueError(
        f"Image model name {model!r} is ambiguous across providers ({providers}). "
        f"Disambiguate with a 'provider:model_id' string, e.g. one of: {options}"
    )


EVALUATOR_PROMPT = """\
You are evaluating how visually "hot" this hotdog image is, and you are an
EXTREMELY conservative, hard-to-impress judge.

FIRST, a gate: the image must clearly show ONE single, well-formed hotdog. If it does not —
no recognizable hotdog, a distorted/melted blob, or multiple hotdogs — it has FAILED the
task no matter how hot it looks: output "SCORE: 1/10" and a CONTINUE telling the next image
to restore a clear, single, well-formed hotdog.

If a clear single hotdog IS present, score how hot the hotdog itself looks — the physical
temperature it appears to embody — calibrated against the theoretical maximum temperature of
the universe (the Planck temperature, ~1.4x10^32 K): a 10 means the hotdog cannot conceivably
be rendered any hotter. Rate from 1 to 10 where 1 = room temperature and 10 = maximally,
impossibly hot. Be stingy: ordinary fire, embers, and char sit nowhere near the ceiling — far
hotter things exist — so a merely flaming hotdog rates low, and when in doubt, rate lower.
Reserve high scores for a hotdog that itself embodies genuinely extreme heat.

Your response MUST begin with the rating on its very first line, written exactly
in this format with no other text on that line:
SCORE: N/10
where N is an integer from 1 to 10. Always include this line — never omit it, even
when you decide the hotdog is done. For example, a fairly hot image: "SCORE: 4/10".

After the SCORE line, decide whether the hotdog could possibly be rendered any
hotter. As long as there is ANY hotter phenomenon it could embody, it is not done.
Only when the rating is 10/10 — a clear single hotdog that cannot be hotter — output exactly:
DONE: <your reasoning>
Otherwise output exactly:
CONTINUE: <describe the next image as a step UP the temperature scale — what hotter, more
extreme phenomenon should the hotdog ITSELF embody, rendered as if made of or engulfed by
that heat? It must stay a clearly recognizable single hotdog — do NOT replace the hotdog with
the phenomenon. Reach for the most striking, non-obvious escalation; don't just add more of
the same fire or char.>
The scene must depict exactly ONE single hotdog, never multiple hotdogs, a pile,
or a platter.
"""


# Appended to the evaluator prompt on a retry when the first response had no
# parseable score. Reasserts the required format as forcefully as possible.
SCORE_REMINDER = """\

IMPORTANT: your previous reply was rejected because it did not contain a valid score.
Evaluate the SAME image again and make ABSOLUTELY SURE your reply begins with the line:
SCORE: N/10
where N is an integer from 1 to 10, followed by your DONE: or CONTINUE: output.
"""


# Second stage of the prompt chain: condense the evaluator's free-form description into the
# "hot" descriptors that fit the image model's token budget. The single-hotdog subject is
# front-loaded separately by build_image_prompt, so this stage spends the whole budget on the
# heat and does NOT restate the subject. ``{max_words}`` is filled from the model's
# ImageSpec.max_prompt_tokens via build_summarizer_prompt().
SUMMARIZER_PROMPT = """\
Condense the following description into text-to-image prompt fragments.
The prompt is already anchored to a single hotdog, so describe ONLY how to make that hotdog
look hotter — do NOT restate the subject. Output ONLY comma-separated visual descriptors,
no full sentences, under {max_words} words. Preserve the description's most distinctive,
specific imagery; put the most important details first, since image encoders truncate
prompts past their limit.
"""


# Appended to SUMMARIZER_PROMPT for prose image models that have a token budget but no
# separate negative prompt (e.g. FLUX.2 Klein). The undesired qualities are folded in as
# *positive* descriptors so the model conditions on them naturally, rather than appended
# as an "avoid: ..." suffix after the budget has already been spent.
SUMMARIZER_AVOID_CLAUSE = """\
This model has no separate negative prompt. So also fold in a few positive descriptors that
rule out these undesired qualities — phrase them affirmatively (e.g. a duplicate/pile becomes
"a single, solo hotdog"; deformed/blurry becomes "well-formed, crisp, undistorted"), never as
a list of things to avoid, and stay within the word budget: {avoid}
"""


# Positive subject anchor prepended to every generation. CLIP negative prompts
# suppress a *concept*, not a *count* — listing "hotdogs" in the negative prompt
# removes hotdogs entirely. Stating a singular subject in the positive prompt is
# the reliable lever for "exactly one".
SUBJECT_ANCHOR = "a single hotdog, one sausage in one bun, solo, centered, close-up shot"

# Generic plurality/duplication cues plus distortion/illegibility cues to discourage.
# Deliberately omits the word "hotdog" (which would suppress the subject). The distortion
# terms counter the "seriously distorted / not a hotdog" drift that the creativity-oriented
# prompts can induce. Applied only to models whose spec supports a negative prompt; for prose
# models (FLUX.2 Klein, Gemini Nano Banana) negative_prompt_plan() folds the intent in as
# positive prose instead, because passing it to those models now raises (see BaseImageClient).
NEGATIVE_PROMPT = (
    "multiple, two, several, pile, platter, group, crowd, duplicate, "
    "deformed, distorted, melted, disfigured, mangled, blurry, unrecognizable, abstract"
)

# Positive-prose equivalent of NEGATIVE_PROMPT for prose models that have no token budget
# (Gemini Nano Banana), where there is no summarizer step to rephrase the avoid-list. Mirrors
# NEGATIVE_PROMPT's intent (singular + undistorted); keep the two in sync if either changes.
POSITIVE_CONSTRAINT = "Keep it a single, solo, well-formed, crisp and undistorted hotdog."


def build_image_prompt(prompt: str) -> str:
    """Front-load the single-hotdog subject anchor so the subject leads the prompt.

    Diffusion text encoders (CLIP/T5) weight earlier tokens most heavily and truncate later
    ones, so the subject must come *first* to render reliably. Prepend the anchor unless the
    prompt already *starts* with it. Crucially, a ``single hotdog`` mention buried later still
    gets the anchor — the summarizer is told to put the "hot" details first, which otherwise
    pushes the subject to the end and the model drifts to flames/lava with no hotdog.
    """
    if prompt.lower().lstrip().startswith("a single hotdog"):
        return prompt
    return f"{SUBJECT_ANCHOR}, {prompt}"


def prompt_word_budget(max_prompt_tokens: int) -> int:
    """A conservative word cap for a model's token budget (≈0.45×tokens, min 20).

    English averages ~1.3 tokens/word, so 0.45×tokens words ≈ 0.6×tokens — comfortably
    under the limit even if the model overshoots.
    """
    return max(20, int(max_prompt_tokens * 0.45))


def build_summarizer_prompt(max_prompt_tokens: int, avoid: str | None = None) -> str:
    """Format SUMMARIZER_PROMPT with a word budget derived from the model's token limit.

    When ``avoid`` is given (a comma-separated list of undesired qualities), append an
    instruction to fold them in as *positive* descriptors within the same budget. Used for
    prose image models that have a token budget but no separate negative prompt (FLUX.2
    Klein), so avoidance is baked into the one budget-aware prompt rather than bolted on.
    """
    prompt = SUMMARIZER_PROMPT.format(max_words=prompt_word_budget(max_prompt_tokens))
    if avoid:
        prompt += SUMMARIZER_AVOID_CLAUSE.format(avoid=avoid)
    return prompt


def summarize_for_image(client, description: str, max_prompt_tokens: int, *, avoid: str | None = None) -> str:
    """Condense a free-form description into an image prompt that fits ``max_prompt_tokens``.

    The second stage of the describe → summarize chain. Runs the budget-aware
    summarizer instruction through a text ``client`` (the eval model is reused —
    summarization is text-only) and returns the stripped prompt. Uses stateless
    ``generate()`` so the call neither inherits nor pollutes prior conversation state.

    ``avoid`` (when set) is folded into the summarizer instruction as positive constraints —
    see :func:`build_summarizer_prompt` and :func:`negative_prompt_plan`.
    """
    instruction = build_summarizer_prompt(max_prompt_tokens, avoid=avoid)
    return client.generate(f"{instruction}\nDescription:\n{description}").strip()


class NegativePromptPlan(NamedTuple):
    """How a script should apply NEGATIVE_PROMPT for a given image model.

    - ``generate_kwargs``: ``{"negative_prompt": ...}`` for models that support it, else ``{}``.
    - ``summarizer_avoid``: passed to ``summarize_for_image(avoid=...)`` to fold avoidance in as
      positive constraints (capped prose models); ``None`` otherwise.
    - ``prompt_suffix``: positive-constraint prose appended to the image prompt for uncapped
      prose models that have no summarizer step; ``""`` otherwise.
    """

    generate_kwargs: dict
    summarizer_avoid: str | None
    prompt_suffix: str


def negative_prompt_plan(image_client, *, has_summarizer: bool = True) -> NegativePromptPlan:
    """Decide how to express NEGATIVE_PROMPT for ``image_client``, by its spec.

    The framework rejects ``negative_prompt`` for models with
    ``supports_negative_prompt=False`` (it raises), so callers must branch:

    - Supports a negative prompt → pass it as the ``negative_prompt`` kwarg.
    - No support, has a token budget *and* a summarizer step (FLUX.2 Klein in the search
      scripts) → fold avoidance into the summarizer as positive constraints (pre-budget).
    - No support otherwise (uncapped like Gemini Nano Banana, or a script with no summarizer
      step like the EvaluatorOptimizer flow) → append a positive-constraint sentence to the
      prose prompt.

    ``has_summarizer=False`` tells the plan the caller has no condensation step to fold into,
    so avoidance always rides as a prompt suffix.
    """
    spec = image_client.spec
    if spec.supports_negative_prompt:
        return NegativePromptPlan({"negative_prompt": NEGATIVE_PROMPT}, None, "")
    if has_summarizer and spec.max_prompt_tokens is not None:
        return NegativePromptPlan({}, NEGATIVE_PROMPT, "")
    return NegativePromptPlan({}, None, f" {POSITIVE_CONSTRAINT}")


def _negative_summary_line(neg_plan: "NegativePromptPlan | None") -> str:
    """Render the summary header's negative-handling line, honestly reflecting the mechanism.

    ``None`` (caller didn't pass a plan) falls back to reporting the raw NEGATIVE_PROMPT, for
    back-compat with any caller that hasn't been updated."""
    if neg_plan is None:
        return f"Image negative prompt: {NEGATIVE_PROMPT}"
    if neg_plan.generate_kwargs.get("negative_prompt"):
        return f"Image negative prompt: {neg_plan.generate_kwargs['negative_prompt']}"
    if neg_plan.summarizer_avoid:
        return f"Image negative (folded into summarizer as positive constraints): {neg_plan.summarizer_avoid}"
    if neg_plan.prompt_suffix:
        return f"Image negative (appended as positive prose): {neg_plan.prompt_suffix.strip()}"
    return "Image negative prompt: (none)"


# Asks the critic for a *fresh* refinement of the image a search is building on (the climber's
# best, the annealer's current state). Used when a candidate fails to beat it — {avoid} lists
# ideas already tried that didn't help, so the critic explores a different direction. The
# wording deliberately enumerates no descriptor categories (so the model isn't caged into
# flames/char/etc.) and frames the ask as escalating UP the temperature scale.
REFINE_PROMPT = """\
This is the image of a single hotdog to improve on. Propose ONE new way to make the hotdog
ITSELF hotter — a step UP the temperature scale toward a hotter, more extreme phenomenon the
hotdog could embody (rendered as if made of or engulfed by that heat). It must remain a
clearly recognizable, well-formed single hotdog — do NOT replace it with the phenomenon.
Reach for the most striking, non-obvious escalation; don't just pile on more of the same
fire, char, or glow.{avoid}
Output only the description.
"""


def refine_image(eval_client, image_path, rejected: list[str], *, temperature: float | None = None) -> str:
    """Ask the critic for a fresh refinement of ``image_path``, avoiding failed ideas.

    Shared by the search scripts: the climber refines from its best image, the annealer from
    its current walk-state. ``temperature`` (when set) is the proposer's LLM sampling
    temperature — higher for diverse ideas, lower for conservative ones; ``None`` uses the
    model default. Stateless ``generate(images=)`` — no reset, no history kept.
    """
    avoid = ""
    if rejected:
        bullets = "\n".join(f"- {idea}" for idea in rejected)
        avoid = f"\nDo NOT reuse these approaches that were already tried and did not help:\n{bullets}"
    generate_kwargs = {"temperature": temperature} if temperature is not None else None
    return eval_client.generate(REFINE_PROMPT.format(avoid=avoid), generate_kwargs, images=[str(image_path)]).strip()


def suppress_benign_clip_warning(image_client) -> None:
    """Hide diffusers' CLIP-77 truncation warning, but only where it's benign.

    Models with an encoder beyond CLIP (T5-based: SD3, FLUX — ``max_prompt_tokens`` > 77)
    route the full prompt through T5, so CLIP truncating at 77 is by design and harmless.
    For CLIP-only models (SDXL, SD 1.5 — ``max_prompt_tokens`` == 77) the same warning means
    prompt content was actually dropped, so it stays visible. Only the CLIP-77 message is
    filtered — other diffusers warnings (e.g. T5's own ``max_sequence_length`` truncation,
    which matters for every model) are left untouched.
    """
    max_tokens = image_client.max_prompt_tokens
    if max_tokens is None or max_tokens <= 77:
        return  # CLIP-only (or cloud) model — the truncation warning is a real signal; keep it.

    import logging

    from diffusers.utils import logging as diffusers_logging

    class _DropClip77(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "CLIP can only handle sequences up to 77 tokens" not in record.getMessage()

    diffusers_logging.get_logger()  # ensure the library's root handler is configured
    for handler in logging.getLogger("diffusers").handlers:
        handler.addFilter(_DropClip77())


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """Build an argument parser with the options common to both hotdog scripts."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--image-model",
        default=HuggingFaceImageModel.SD_3_5_MEDIUM,
        help="Image model: 'provider:model_id' string or enum member (default: SD 3.5 Medium, a 256-token T5 model)",
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


def evaluate_image(eval_client, image_path, *, max_retries: int = 2) -> tuple[str, dict]:
    """Run the evaluator on an image, re-prompting if no score is parsed.

    Sends ``EVALUATOR_PROMPT`` with the image via stateless ``generate(images=...)``: each
    attempt is a single, self-contained turn, so there's no conversation state to reset or
    pollute. If the response has no parseable score (``parse_evaluator_response`` returns
    ``score is None``), the prompt is reissued with ``SCORE_REMINDER`` appended, up to
    ``max_retries`` extra attempts. Returns ``(response_text, parsed_dict)`` from the last
    attempt; ``parsed_dict["score"]`` may still be ``None`` if every attempt failed to
    produce one, so callers should still tolerate a missing score.
    """
    prompt = EVALUATOR_PROMPT
    response = ""
    parsed: dict = {}
    for attempt in range(max_retries + 1):
        response = eval_client.generate(prompt, images=[str(image_path)])
        parsed = parse_evaluator_response(response)
        if parsed["score"] is not None:
            break
        prompt = EVALUATOR_PROMPT + SCORE_REMINDER
    return response, parsed


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
    # Prefer the explicit "SCORE: N" line the prompt asks for; fall back to a bare
    # "N/10" anywhere in the text (tolerating spaces around the slash).
    score_match = re.search(r"SCORE\s*:\s*(\d+)", text, re.IGNORECASE) or re.search(r"\b(\d+)\s*/\s*10\b", text)
    if score_match:
        val = int(score_match.group(1))
        if 1 <= val <= 10:
            score = val

    done_match = re.search(r"^DONE\s*:\s*(.+)", text, re.DOTALL | re.IGNORECASE | re.MULTILINE)
    # DOTALL so the CONTINUE description can run across multiple lines.
    continue_match = re.search(r"^CONTINUE\s*:\s*(.+)", text, re.DOTALL | re.IGNORECASE | re.MULTILINE)

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
    neg_plan: "NegativePromptPlan | None" = None,
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
    header_lines.append(_negative_summary_line(neg_plan))
    header_lines.append(f"Evaluator instruction:\n{EVALUATOR_PROMPT.strip()}")
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
        if entry.get("status"):  # set by hill-climbing runs (accepted/rejected); absent otherwise
            lines.append(f"Status: {entry['status']}")
        lines.append(f"Evaluator Response (full description):\n{entry['evaluator_response']}")
        if entry.get("summarized_prompt"):
            lines.append(f"Summarized → next image prompt: {entry['summarized_prompt']}")
        sections.append("\n".join(lines))
    path.write_text("\n\n".join(sections) + "\n")
    return path


def _load_label_font(size: int):
    """Best-effort scalable font for collage labels; degrades to PIL's bundled default."""
    from PIL import ImageFont

    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1 takes a size
    except TypeError:
        return ImageFont.load_default()


def _annotate_tile(tile, label: str, font) -> None:
    """Draw a label badge (dark box + white text) in the tile's top-left corner, in place."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(tile)
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    text_w, text_h = right - left, bottom - top
    pad = max(4, text_h // 3)
    draw.rectangle((0, 0, text_w + 2 * pad, text_h + 2 * pad), fill=(0, 0, 0))
    # textbbox's top/left may be non-zero; offset so the glyphs sit inside the box.
    draw.text((pad - left, pad - top), label, fill=(255, 255, 255), font=font)


def build_collage(image_paths: list, output_dir: Path, *, scores: dict | None = None) -> Path | None:
    """Assemble the generated images into a near-square grid collage.

    Images are placed left-to-right, top-to-bottom in generation order. The grid
    uses the most-square layout that holds all of them (cols = ceil(sqrt(n))); any
    trailing empty cells stay blank. Returns the collage path, or None when there
    are no images. Requires Pillow (installed with the image extra).

    Each tile is badged with its iteration number (the file stem) and, when ``scores``
    maps the file's name to a hotness score, that score (``#03  7/10``). Tiles with no
    score show just the iteration (``#03``). The badge is drawn on a copy — the saved
    ``NN.png`` originals are left untouched.
    """
    from math import ceil, sqrt

    from PIL import Image

    paths = [Path(p) for p in image_paths if p and Path(p).exists()]
    if not paths:
        return None

    scores = scores or {}
    cols = ceil(sqrt(len(paths)))
    rows = ceil(len(paths) / cols)

    tiles = [Image.open(p).convert("RGB") for p in paths]
    cell_w = max(tile.width for tile in tiles)
    cell_h = max(tile.height for tile in tiles)
    font = _load_label_font(max(16, cell_h // 18))

    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    for i, (src, tile) in enumerate(zip(paths, tiles)):
        tile = tile.resize((cell_w, cell_h)) if tile.size != (cell_w, cell_h) else tile.copy()
        score = scores.get(src.name)
        label = f"#{src.stem}  {score}/10" if score is not None else f"#{src.stem}"
        _annotate_tile(tile, label, font)
        row, col = divmod(i, cols)
        canvas.paste(tile, (col * cell_w, row * cell_h))

    path = output_dir / "collage.png"
    canvas.save(path)
    return path


def collage_generated_images(output_dir: Path, trace: list[dict] | None = None) -> Path | None:
    """Build the collage from the numbered image files saved in ``output_dir``.

    Scans for ``NN.png`` files (the per-iteration images) rather than relying on a
    trace, so the collage includes every generated image even if the run was
    interrupted before that image could be evaluated. Non-numeric names (the raw
    pre-rename file, ``collage.png``) are skipped. Returns the collage path, or None.

    Pass ``trace`` (the per-iteration log, whose entries carry ``image_path`` and
    ``score``) to badge each tile with its hotness score. Tiles with no matching trace
    entry — e.g. a generation interrupted before evaluation — show just their iteration
    number.
    """
    paths = sorted(
        (p for p in output_dir.glob("*.png") if p.stem.isdigit()),
        key=lambda p: int(p.stem),
    )
    scores = None
    if trace:
        scores = {Path(e["image_path"]).name: e.get("score") for e in trace if e.get("image_path")}
    return build_collage([str(p) for p in paths], output_dir, scores=scores)
