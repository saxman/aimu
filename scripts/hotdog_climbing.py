#!/usr/bin/env python3
"""Hill-climbing variant of the hotdog heating loop.

`hotdog_loop.py` and `hotdog_agent.py` walk forward greedily: whatever prompt the critic
proposes next is generated and accepted, even if the new image is *worse* than a previous
one. This script instead keeps the best image seen so far and **only advances on a strictly
higher score** — if a generation fails to beat the best (a lower score *or* a tie; the coarse
1–10 judge makes ties common, and a tie isn't demonstrated progress), it's discarded and the
critic is asked for a *different* refinement of the current best (avoiding ideas that already
failed).

That mirrors the hill-climbing in `aimu.prompts` (best-state caching + revert when a candidate
doesn't improve), applied here to single-artifact image refinement rather than to a reusable
prompt over a dataset. Stops when the critic says DONE, after --patience consecutive
non-improvements, or at --max-iterations.

Usage:
    python scripts/hotdog_climbing.py
    python scripts/hotdog_climbing.py --max-iterations 0 --patience 4
    python scripts/hotdog_climbing.py --image-model hf:stabilityai/stable-diffusion-xl-base-1.0
"""

from __future__ import annotations

import itertools
import shutil
import sys
from pathlib import Path

import aimu

sys.path.insert(0, str(Path(__file__).parent))
from _hotdog_common import (
    build_arg_parser,
    build_image_prompt,
    build_summarizer_prompt,
    collage_generated_images,
    evaluate_image,
    negative_prompt_plan,
    refine_image,
    resolve_image_model,
    resolve_output_dir,
    summarize_for_image,
    suppress_benign_clip_warning,
    write_summary,
)


def _score(value: int | None) -> int:
    """Coerce a possibly-missing hotness score to a comparable int (unparsed → 0)."""
    return value if value is not None else 0


def run_climb(
    image_model_name: str,
    eval_model_id: str,
    output_dir: Path,
    max_iterations: int,
    patience: int,
) -> None:
    image_client = aimu.image_client(resolve_image_model(image_model_name))
    suppress_benign_clip_warning(image_client)
    eval_client = aimu.client(eval_model_id)
    if not eval_client.is_vision_model:
        raise ValueError(f"Eval model {eval_model_id!r} does not support vision.")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Summarize each candidate idea down to the image model's token budget (None → direct).
    # The plan decides how NEGATIVE_PROMPT is applied, based on the model's spec.
    neg_plan = negative_prompt_plan(image_client)
    max_prompt_tokens = image_client.max_prompt_tokens
    summarizer_instruction = (
        build_summarizer_prompt(max_prompt_tokens, avoid=neg_plan.summarizer_avoid) if max_prompt_tokens else None
    )

    def to_prompt(idea: str) -> str:
        if not max_prompt_tokens:
            return idea
        return summarize_for_image(eval_client, idea, max_prompt_tokens, avoid=neg_plan.summarizer_avoid)

    candidate_prompt = "a hot hotdog"
    candidate_idea: str | None = None  # the description that produced the current candidate
    best: dict | None = None
    rejected: list[str] = []  # ideas tried against the current best that didn't beat it
    stale = 0  # consecutive non-improving iterations
    trace: list[dict] = []

    iterations = itertools.count(1) if max_iterations == 0 else iter(range(1, max_iterations + 1))
    cap = "∞" if max_iterations == 0 else str(max_iterations)

    try:
        for i in iterations:
            print(f"--- Iteration {i}/{cap} ---")
            print(f"Prompt: {candidate_prompt}")

            image_prompt = build_image_prompt(candidate_prompt) + neg_plan.prompt_suffix
            raw_path = image_client.generate(
                image_prompt, format="path", output_dir=output_dir, **neg_plan.generate_kwargs
            )
            dest = output_dir / f"{i:02d}.png"
            Path(raw_path).rename(dest)

            response, parsed = evaluate_image(eval_client, dest)
            improved = best is None or _score(parsed["score"]) > _score(best["score"])
            status = "accepted (new best)" if improved else "rejected (no improvement)"
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

            if improved:
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

            if improved:
                # New best — explore from its own suggested refinement; clear the reject list.
                rejected = []
                stale = 0
                next_idea = parsed["next_prompt"]
            else:
                # No improvement (lower score or a tie) — keep the best, remember the failed
                # idea, and ask for a different one.
                if candidate_idea is not None:
                    rejected.append(candidate_idea)
                stale += 1
                if stale >= patience:
                    print(f"No improvement after {patience} attempt(s); stopping.")
                    break
                print(f"Reverting to best (iteration {best['iteration']}, score {best['score']}/10).")
                next_idea = refine_image(eval_client, best["image_path"], rejected)

            candidate_idea = next_idea
            candidate_prompt = to_prompt(next_idea)
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
                neg_plan=neg_plan,
            )
            print(f"Summary written to: {summary_path}")
        collage_path = collage_generated_images(output_dir, trace)
        if collage_path:
            print(f"Collage written to: {collage_path}")


def main() -> None:
    parser = build_arg_parser("Hill-climb a hotdog image: keep the best, revert on regression.")
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Stop after this many consecutive non-improving iterations (default: 3)",
    )
    args = parser.parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    run_climb(args.image_model, args.eval_model, output_dir, args.max_iterations, args.patience)


if __name__ == "__main__":
    main()
