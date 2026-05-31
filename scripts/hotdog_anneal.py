#!/usr/bin/env python3
"""Simulated-annealing variant of the hotdog heating loop.

`hotdog_loop_climbing.py` is a strict hill-climber: it only ever advances on a *higher*
score and reverts otherwise. This script generalises that into **simulated annealing** — the
climber is the ``T → 0`` limit of what's here. It keeps a ``current`` walk-state (distinct
from the best-ever image) and, controlled by a falling *temperature*, will probabilistically
accept a *worse* image to escape a local optimum, cooling into greedy behaviour over time.

Acceptance (Metropolis rule), with ``Δ = new_score − current_score``:
- ``Δ > 0`` (hotter): always accept.
- ``Δ == 0`` (tie): always accept — a free sideways move (the deliberate opposite of the
  climber, which treats a tie as no progress).
- ``Δ < 0`` (cooler): accept with probability ``exp(Δ / T)``. High ``T`` early ⇒ explores
  freely; as ``T`` cools toward 0 ⇒ behaves like the greedy climber.

The best-ever image is tracked separately and copied to ``best.png`` regardless of the walk.

Honest caveats for this domain:
- The judge is a coarse, conservative integer 1–10, so ``Δ`` is almost always ``0`` or ``±1``
  — annealing's fine temperature control is blunted (mostly "always accept ties, accept −1
  with probability ``p(T)``").
- Each step is a full image generation + vision eval, so runs are short (tens of steps, not
  thousands); annealing's asymptotic guarantees don't apply — the practical win over the
  climber is local-optimum escape, not convergence.
- The "neighbourhood" is the critic's prompt refinement, not a local perturbation of the
  previous image (no img2img). Annealing happens in prompt space.

Stops when the critic says DONE, or at --max-iterations.

Usage:
    python scripts/hotdog_anneal.py
    python scripts/hotdog_anneal.py --initial-temp 3.0 --cooling-rate 0.9 --seed 7
    python scripts/hotdog_anneal.py --max-iterations 0 --image-model hf:stabilityai/stable-diffusion-xl-base-1.0
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import shutil
import sys
from pathlib import Path

import aimu

sys.path.insert(0, str(Path(__file__).parent))
from _hotdog_common import (
    NEGATIVE_PROMPT,
    build_arg_parser,
    build_image_prompt,
    build_summarizer_prompt,
    collage_generated_images,
    evaluate_image,
    resolve_output_dir,
    summarize_for_image,
    suppress_benign_clip_warning,
    write_summary,
)

# Asks the critic for a *fresh* refinement of the current image. Used when a proposed step is
# rejected — {avoid} lists ideas already tried against the current state that didn't stick, so
# the critic explores a different direction. (Mirrors the climber's REFINE_PROMPT.)
REFINE_PROMPT = """\
This is the current image of a single hotdog. Propose ONE new way to make it look even
hotter — describe the flames, char, spices, steam, colors, and lighting. The scene must
depict exactly ONE single hotdog, never multiple.{avoid}
Output only the description.
"""


def _score(value: int | None) -> int:
    """Coerce a possibly-missing hotness score to a comparable int (unparsed → 0)."""
    return value if value is not None else 0


def _accept(delta: int, temperature: float, rng: random.Random) -> bool:
    """Metropolis acceptance rule.

    Always accept a non-worsening move (``delta >= 0``, which covers both a hotter image and a
    tie). Accept a worsening move (``delta < 0``) with probability ``exp(delta / temperature)``.
    At ``temperature → 0`` worsening moves are never accepted — the pure-greedy (hill-climbing)
    limit.
    """
    if delta >= 0:
        return True
    if temperature <= 1e-9:
        return False
    return rng.random() < math.exp(delta / temperature)


def refine_from_current(eval_client, current_image_path: str, rejected: list[str]) -> str:
    """Ask the critic for a fresh refinement of the current image, avoiding failed ideas."""
    avoid = ""
    if rejected:
        bullets = "\n".join(f"- {idea}" for idea in rejected)
        avoid = f"\nDo NOT reuse these approaches that were already tried and did not help:\n{bullets}"
    # Stateless one-shot vision call — no reset() needed, no history kept.
    return eval_client.generate(REFINE_PROMPT.format(avoid=avoid), images=[current_image_path]).strip()


def run_anneal(
    image_model_name: str,
    eval_model_id: str,
    output_dir: Path,
    max_iterations: int,
    initial_temp: float,
    cooling_rate: float,
    seed: int | None,
) -> None:
    image_client = aimu.image_client(image_model_name)
    suppress_benign_clip_warning(image_client)
    eval_client = aimu.client(eval_model_id)
    if not eval_client.is_vision_model:
        raise ValueError(f"Eval model {eval_model_id!r} does not support vision.")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Annealing: initial T={initial_temp}, cooling rate={cooling_rate}, seed={seed}\n")

    rng = random.Random(seed)

    # Summarize each candidate idea down to the image model's token budget (None → direct).
    max_prompt_tokens = image_client.max_prompt_tokens
    summarizer_instruction = build_summarizer_prompt(max_prompt_tokens) if max_prompt_tokens else None

    def to_prompt(idea: str) -> str:
        return summarize_for_image(eval_client, idea, max_prompt_tokens) if max_prompt_tokens else idea

    candidate_prompt = "a hot hotdog"
    candidate_idea: str | None = None  # the description that produced the current candidate
    current: dict | None = None  # the accepted walk-state (may be worse than best)
    best: dict | None = None  # the hottest image seen so far (copied to best.png)
    rejected: list[str] = []  # ideas tried against the current state that weren't accepted
    temperature = initial_temp
    trace: list[dict] = []

    iterations = itertools.count(1) if max_iterations == 0 else iter(range(1, max_iterations + 1))
    cap = "∞" if max_iterations == 0 else str(max_iterations)

    try:
        for i in iterations:
            print(f"--- Iteration {i}/{cap} (T={temperature:.3f}) ---")
            print(f"Prompt: {candidate_prompt}")

            image_prompt = build_image_prompt(candidate_prompt)
            raw_path = image_client.generate(
                image_prompt, negative_prompt=NEGATIVE_PROMPT, format="path", output_dir=output_dir
            )
            dest = output_dir / f"{i:02d}.png"
            Path(raw_path).rename(dest)

            response, parsed = evaluate_image(eval_client, dest)
            score = _score(parsed["score"])

            # Metropolis acceptance against the current walk-state (first image always accepted).
            delta = 0 if current is None else score - _score(current["score"])
            accept = current is None or _accept(delta, temperature, rng)
            improved_best = best is None or score > _score(best["score"])

            verdict = "accepted" if accept else "rejected"
            if improved_best:
                verdict += " (new best)"
            status = f"{verdict} [Δ={delta}, T={temperature:.3f}]"
            print(f"Evaluator: score={parsed['score']} action={parsed['action']} → {status}\n")

            trace.append(
                {
                    "iteration": i,
                    "prompt": candidate_prompt,
                    "image_prompt": image_prompt,
                    "image_path": str(dest),
                    "evaluator_response": response,
                    "score": parsed["score"],
                    "action": parsed["action"],
                    "next_prompt": parsed["next_prompt"],
                    "status": status,
                }
            )

            if improved_best:
                best = {"score": parsed["score"], "prompt": candidate_prompt, "image_path": str(dest), "iteration": i}

            if parsed["action"] == "DONE":
                print(f"Evaluator declared maximum hotness at iteration {i}.")
                break
            if parsed["action"] == "unknown":
                print("Could not parse evaluator response (no DONE/CONTINUE). Stopping.")
                break
            if i == max_iterations:
                print(f"Reached maximum iterations ({max_iterations}). Stopping.")
                break

            if accept:
                # Move the walk to this image and explore from its own suggested refinement.
                current = {"score": parsed["score"], "prompt": candidate_prompt, "image_path": str(dest)}
                rejected = []
                next_idea = parsed["next_prompt"]
            else:
                # Stay at current; remember the failed idea and ask for a different refinement of it.
                if candidate_idea is not None:
                    rejected.append(candidate_idea)
                print(f"Staying at current (iteration's score {parsed['score']} rejected).")
                next_idea = refine_from_current(eval_client, current["image_path"], rejected)

            candidate_idea = next_idea
            candidate_prompt = to_prompt(next_idea)
            temperature *= cooling_rate
    except KeyboardInterrupt:
        print("\nInterrupted — writing partial results so far...")
    finally:
        # Mark the winner and emit the same artifacts as the other scripts, even on interrupt.
        if best:
            shutil.copyfile(best["image_path"], output_dir / "best.png")
            print(f"\nBest: iteration {best['iteration']}, score {best['score']}/10 → {output_dir / 'best.png'}")
        if trace:
            summary_path = write_summary(
                output_dir,
                trace,
                image_model=image_client.spec.id,
                eval_model=eval_model_id,
                summarizer_instruction=summarizer_instruction,
            )
            print(f"Summary written to: {summary_path}")
        collage_path = collage_generated_images(output_dir)
        if collage_path:
            print(f"Collage written to: {collage_path}")


def build_parser() -> argparse.ArgumentParser:
    """Common hotdog options plus the simulated-annealing knobs."""
    p = build_arg_parser("Anneal a hotdog image: accept worsening steps early, cool to greedy.")
    p.add_argument(
        "--initial-temp",
        type=float,
        default=2.0,
        help="Starting temperature (default: 2.0). Higher = accepts worse images more readily early on",
    )
    p.add_argument(
        "--cooling-rate",
        type=float,
        default=0.85,
        help="Geometric cooling factor applied each iteration: T *= rate (default: 0.85)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducible acceptance decisions (default: None = nondeterministic)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    run_anneal(
        args.image_model,
        args.eval_model,
        output_dir,
        args.max_iterations,
        args.initial_temp,
        args.cooling_rate,
        args.seed,
    )


if __name__ == "__main__":
    main()
