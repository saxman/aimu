"""Shared helpers for the epic-sentence refinement scripts.

The text-only sibling of ``_hotdog_common.py``. The hotdog family iteratively makes an *image*
hotter; this family iteratively makes a *sentence* more epic. The control-flow lesson is
identical (code loop / agent / workflow class; greedy / hill-climb / anneal) but the modality
is pure text, so none of the image machinery (diffusers, vision eval, token-budget
summarizer, negative prompt, collage) is needed. What remains is just the orchestration and
search, which is the point of having a GPU-free twin.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path


# The mundane starting point. Everything the family does is render THIS errand epically; the
# judge's gate fails any rewrite that stops being a single grammatical sentence about it.
SEED_SENTENCE = "A man walks to the store to buy a carton of milk."

# The directive used to seed iteration 1, before the judge has proposed anything.
INITIAL_DIRECTION = "Give it a grand, cinematic, larger-than-life register."


# Mirrors hotdog's EVALUATOR_PROMPT: a conservative judge with a fidelity gate, a 1-10 score
# against a theoretical ceiling, and a DONE/CONTINUE contract that parse_judge_response reads.
JUDGE_PROMPT = """\
You are evaluating how EPIC this sentence is, and you are an EXTREMELY conservative,
hard-to-impress judge.

FIRST, a gate: the text must be ONE single grammatical sentence that still describes the SAME
mundane event as the original, a man going to a store to buy milk. If it does not (it is
multiple sentences, a fragment, ungrammatical, or it has drifted into a different event: no
man, no store, no buying milk) it has FAILED the task no matter how grand it sounds: output
"SCORE: 1/10" and a CONTINUE telling the next attempt to restore a single grammatical sentence
about that same errand.

If a valid single sentence about the errand IS present, score how EPIC it feels (the sense of
mythic, cinematic, world-shaking grandeur it embodies), calibrated against the most grandiose
myth conceivable: a 10 means the sentence could not possibly be made any more epic. Rate from 1
to 10 where 1 = flat and mundane and 10 = maximally, impossibly epic. Be stingy: ordinary
"dramatic" wording, a few adjectives, or a single grand metaphor sit nowhere near the ceiling
(far grander framings exist), so a merely florid sentence rates low, and when in doubt, rate
lower. Reserve high scores for a sentence that turns a trip for milk into something of
genuinely cosmic, legendary weight.

Your response MUST begin with the rating on its very first line, written exactly in this format
with no other text on that line:
SCORE: N/10
where N is an integer from 1 to 10. Always include this line; never omit it, even when you
decide the sentence is done. For example, a fairly epic sentence: "SCORE: 4/10".

After the SCORE line, decide whether the sentence could possibly be made any more epic. As long
as there is ANY grander framing it could embody, it is not done. Only when the rating is 10/10
(a valid single sentence about the errand that cannot be made more epic) output exactly:
DONE: <your reasoning>
Otherwise output exactly:
CONTINUE: <describe how the next rewrite should escalate UP the epic scale: what grander, more
mythic register the SAME errand should be told in. It must stay ONE grammatical sentence about
a man going to a store for milk; do NOT drop the errand or split it into multiple sentences.
Reach for the most striking, non-obvious escalation; don't just pile on more adjectives.>
"""


# Appended to the judge prompt on a retry when the first response had no parseable score.
SCORE_REMINDER = """\

IMPORTANT: your previous reply was rejected because it did not contain a valid score.
Evaluate the SAME sentence again and make ABSOLUTELY SURE your reply begins with the line:
SCORE: N/10
where N is an integer from 1 to 10, followed by your DONE: or CONTINUE: output.
"""


# The generator instruction. Mirrors build_image_prompt's job (anchor the subject so it can't
# drift): the seed errand and the single-sentence rule are restated every call so the rewrite
# escalates the *register* without abandoning what literally happens.
GENERATOR_PROMPT = """\
You rewrite a single mundane sentence to make it sound EPIC (mythic, cinematic, grand) while
keeping it ONE grammatical sentence about the exact same event.

The original errand is:
{seed}

Rewrite directive: {direction}

Rules:
- Output ONLY the rewritten sentence, with no preamble, no quotation marks, no explanation.
- It must remain ONE single grammatical sentence.
- It must still describe the same event: a man going to a store to buy milk. Render it grandly;
  do not change what literally happens.
"""


# Asks the judge for a *fresh* refinement direction for the sentence a search is building on
# (the climber's best, the annealer's current state). {avoid} lists directions already tried
# that didn't help, so the judge explores elsewhere. Mirrors hotdog's REFINE_PROMPT.
REFINE_PROMPT = """\
This is the current epic rewrite of "a man going to a store to buy milk" to improve on:
"{sentence}"

Propose ONE new way to make it MORE epic: a step UP the scale toward a grander, more mythic
register for the SAME errand. It must remain ONE grammatical sentence about that errand; do
NOT drop the errand or split it into multiple sentences. Reach for the most striking,
non-obvious escalation; don't just pile on more adjectives.{avoid}
Output only the directive describing the next rewrite.
"""


def _strip_sentence(text: str) -> str:
    """Strip whitespace and a single layer of wrapping quotes from a model's sentence output.

    Models routinely wrap a one-sentence answer in quotation marks; left in, they leak into the
    judge prompt and the summary. Removes one matched pair of straight or curly quotes.
    """
    out = text.strip()
    pairs = (('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"))
    for left, right in pairs:
        if len(out) >= 2 and out[0] == left and out[-1] == right:
            return out[1:-1].strip()
    return out


def generate_sentence(gen_client, seed: str, direction: str, generate_kwargs: dict | None = None) -> str:
    """Render ``seed`` epically following ``direction``; return the single rewritten sentence.

    The analog of ``image_client.generate(prompt)`` in the hotdog family. Uses stateless
    ``generate()`` so the call neither inherits nor pollutes prior conversation state.
    """
    prompt = GENERATOR_PROMPT.format(seed=seed, direction=direction)
    return _strip_sentence(gen_client.generate(prompt, generate_kwargs))


def evaluate_sentence(judge_client, sentence: str, *, max_retries: int = 2) -> tuple[str, dict]:
    """Run the judge on a sentence, re-prompting if no score is parsed.

    The text analog of ``evaluate_image``: the sentence rides *in the prompt* (no ``images=``).
    Each attempt is a self-contained stateless ``generate()`` turn. If the response has no
    parseable score, the prompt is reissued with ``SCORE_REMINDER`` appended, up to
    ``max_retries`` extra attempts. Returns ``(response_text, parsed_dict)`` from the last
    attempt; ``parsed_dict["score"]`` may still be ``None`` if every attempt failed.
    """
    base = f"{JUDGE_PROMPT}\nSENTENCE TO EVALUATE:\n{sentence}"
    prompt = base
    response = ""
    parsed: dict = {}
    for _ in range(max_retries + 1):
        response = judge_client.generate(prompt)
        parsed = parse_judge_response(response)
        if parsed["score"] is not None:
            break
        prompt = base + SCORE_REMINDER
    return response, parsed


def parse_judge_response(text: str) -> dict:
    """Parse DONE/CONTINUE signal and epicness score from a judge response.

    Same contract as hotdog's ``parse_evaluator_response`` (the prompts share the SCORE/DONE/
    CONTINUE structure), but the CONTINUE payload is a *rewrite direction* rather than an image
    description. Returns a dict with keys:
      action         -- "DONE", "CONTINUE", or "unknown"
      score          -- int 1-10 or None
      reasoning      -- text after DONE: (or None)
      next_direction -- text after CONTINUE: (or None); may span multiple lines
    """
    score = None
    score_match = re.search(r"SCORE\s*:\s*(\d+)", text, re.IGNORECASE) or re.search(r"\b(\d+)\s*/\s*10\b", text)
    if score_match:
        val = int(score_match.group(1))
        if 1 <= val <= 10:
            score = val

    done_match = re.search(r"^DONE\s*:\s*(.+)", text, re.DOTALL | re.IGNORECASE | re.MULTILINE)
    continue_match = re.search(r"^CONTINUE\s*:\s*(.+)", text, re.DOTALL | re.IGNORECASE | re.MULTILINE)

    if done_match:
        return {"action": "DONE", "score": score, "reasoning": done_match.group(1).strip(), "next_direction": None}
    if continue_match:
        return {
            "action": "CONTINUE",
            "score": score,
            "reasoning": None,
            "next_direction": continue_match.group(1).strip(),
        }
    return {"action": "unknown", "score": score, "reasoning": text.strip(), "next_direction": None}


def refine_sentence(judge_client, sentence: str, rejected: list[str], *, temperature: float | None = None) -> str:
    """Ask the judge for a fresh refinement direction for ``sentence``, avoiding failed ideas.

    Shared by the search scripts: the climber refines from its best sentence, the annealer from
    its current walk-state. ``temperature`` (when set) is the proposer's LLM sampling
    temperature; ``None`` uses the model default. Stateless ``generate()``, no history kept.
    The text twin of hotdog's ``refine_image`` (no ``images=``).
    """
    avoid = ""
    if rejected:
        bullets = "\n".join(f"- {idea}" for idea in rejected)
        avoid = f"\nDo NOT reuse these directions that were already tried and did not help:\n{bullets}"
    generate_kwargs = {"temperature": temperature} if temperature is not None else None
    return judge_client.generate(REFINE_PROMPT.format(sentence=sentence, avoid=avoid), generate_kwargs).strip()


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """Build an argument parser with the options common to every epic script."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--gen-model",
        default="ollama:gemma4:e4b",
        help="Text model that writes the sentence, in 'provider:model_id' form (default: ollama:gemma4:e4b)",
    )
    p.add_argument(
        "--judge-model",
        default="ollama:gemma4:e4b",
        help="Text model that scores epicness, in 'provider:model_id' form (default: ollama:gemma4:e4b)",
    )
    p.add_argument(
        "--seed-sentence",
        default=SEED_SENTENCE,
        help="The mundane sentence to make epic (default: a man buying milk)",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for the summary (default: output/epic/<timestamp>/)",
    )
    p.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Hard cap on iteration count (default: 10; 0 = run until the judge says DONE)",
    )
    return p


def resolve_output_dir(output_dir: str | None) -> Path:
    """Resolve the output directory, defaulting to output/epic/<timestamp>/."""
    if output_dir:
        return Path(output_dir)
    from aimu import paths as aimu_paths

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return aimu_paths.output / "epic" / timestamp


def write_summary(
    output_dir: Path,
    trace: list[dict],
    *,
    seed_sentence: str | None = None,
    gen_model: str | None = None,
    judge_model: str | None = None,
) -> Path:
    """Write summary.txt with the full iteration trace to output_dir.

    Records the constant inputs (seed sentence, the two model names, the judge instruction)
    once at the top, then per iteration the rewrite directive, the sentence produced, its score,
    and the judge's full response. The text analog of hotdog's ``write_summary`` (no image paths,
    negative prompt, or summarizer line, none of which exist for text).
    """
    path = output_dir / "summary.txt"

    header_lines = ["=== Constant inputs ==="]
    if seed_sentence:
        header_lines.append(f"Seed sentence: {seed_sentence}")
    if gen_model:
        header_lines.append(f"Generator model: {gen_model}")
    if judge_model:
        header_lines.append(f"Judge model:     {judge_model}")
    header_lines.append(f"Judge instruction:\n{JUDGE_PROMPT.strip()}")
    sections = ["\n".join(header_lines)]

    for entry in trace:
        lines = [
            f"=== Iteration {entry['iteration']} ===",
            f"Rewrite directive: {entry['direction']}",
            f"Sentence:          {entry['sentence']}",
        ]
        if entry.get("score") is not None:
            lines.append(f"Epicness Score: {entry['score']}/10")
        if entry.get("status"):  # set by hill-climbing / annealing runs; absent otherwise
            lines.append(f"Status: {entry['status']}")
        lines.append(f"Judge Response:\n{entry['judge_response']}")
        sections.append("\n".join(lines))

    path.write_text("\n\n".join(sections) + "\n")
    return path


def write_best(output_dir: Path, best: dict) -> Path:
    """Write the winning sentence (with its score) to best.txt; return the path.

    The text analog of copying ``best.png`` in the hotdog search scripts.
    """
    path = output_dir / "best.txt"
    score = best.get("score")
    score_label = f"{score}/10" if score is not None else "unscored"
    path.write_text(f"[iteration {best['iteration']}, score {score_label}]\n{best['sentence']}\n")
    return path
