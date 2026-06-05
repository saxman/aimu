#!/usr/bin/env python3
"""Iteratively generate and evaluate a hotdog image using local models.

Code-controlled: Python directs the loop; the LLMs only evaluate and evolve the prompt.
Two acceptance strategies are available via --strategy:

  greedy   (default) — always accept the evaluator's suggestion and move on, even if the
             new image scores lower than a previous one. Simple; the final image is whatever
             came last.

  climbing — keep the best image seen so far and only advance on a strictly higher score.
             A tie or lower score is discarded; the evaluator is asked for a different
             refinement of the current best (avoiding ideas that already failed). Stops
             after --patience consecutive non-improvements. Guarantees the final image is
             at least as good as every previous round.

The agent-directed counterpart is hotdog_agent.py. For Metropolis-acceptance annealing
see hotdog_anneal.py. For image-to-image refinement see hotdog_img2img.py.

Usage:
    python scripts/hotdog_loop.py
    python scripts/hotdog_loop.py --strategy climbing --patience 4
    python scripts/hotdog_loop.py --image-model hf:stabilityai/stable-diffusion-xl-base-1.0
    python scripts/hotdog_loop.py --max-iterations 0   # run until evaluator says DONE
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


def run(
    image_model_name: str,
    eval_model_id: str,
    output_dir: Path,
    max_iterations: int,
    strategy: str,
    patience: int,
) -> None:
    image_client = aimu.image_client(resolve_image_model(image_model_name))
    suppress_benign_clip_warning(image_client)
    eval_client = aimu.client(eval_model_id)
    if not eval_client.is_vision_model:
        raise ValueError(f"Eval model {eval_model_id!r} does not support vision.")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Strategy: {strategy}")
    print(f"Output directory: {output_dir}\n")

    neg_plan = negative_prompt_plan(image_client)
    max_prompt_tokens = image_client.max_prompt_tokens
    summarizer_instruction = (
        build_summarizer_prompt(max_prompt_tokens, avoid=neg_plan.summarizer_avoid) if max_prompt_tokens else None
    )
    budget_label = f"{max_prompt_tokens} tokens" if max_prompt_tokens else "uncapped"
    print(f"Image prompt budget: {budget_label} ({'summarizing' if summarizer_instruction else 'direct'})\n")

    def to_prompt(idea: str) -> str:
        if not max_prompt_tokens:
            return idea
        return summarize_for_image(eval_client, idea, max_prompt_tokens, avoid=neg_plan.summarizer_avoid)

    prompt = "a hot hotdog"
    # Climbing-only state — unused in greedy mode.
    best: dict | None = None
    rejected: list[str] = []
    candidate_idea: str | None = None
    stale = 0

    trace: list[dict] = []
    iterations = itertools.count(1) if max_iterations == 0 else iter(range(1, max_iterations + 1))
    cap = "∞" if max_iterations == 0 else str(max_iterations)

    try:
        for i in iterations:
            print(f"--- Iteration {i}/{cap} ---")
            print(f"Prompt: {prompt}")

            image_prompt = build_image_prompt(prompt) + neg_plan.prompt_suffix
            raw_path = image_client.generate(
                image_prompt, format="path", output_dir=output_dir, **neg_plan.generate_kwargs
            )
            dest = output_dir / f"{i:02d}.png"
            Path(raw_path).rename(dest)

            response, parsed = evaluate_image(eval_client, dest)

            entry: dict = {
                "iteration": i,
                "prompt": prompt,
                "image_prompt": image_prompt,
                "image_path": str(dest),
                "evaluator_response": response,
                "score": parsed["score"],
                "action": parsed["action"],
                "next_prompt": parsed["next_prompt"],
            }

            if strategy == "climbing":
                improved = best is None or _score(parsed["score"]) > _score(best["score"])
                entry["status"] = "accepted (new best)" if improved else "rejected (no improvement)"
                if improved:
                    best = {"score": parsed["score"], "prompt": prompt, "image_path": str(dest), "iteration": i}
                print(f"Evaluator: score={parsed['score']} action={parsed['action']} → {entry['status']}\n")
            else:
                print(f"Evaluator:\n{response}\n")

            trace.append(entry)

            if parsed["action"] == "DONE":
                print(f"Evaluator declared maximum hotness at iteration {i}.")
                break
            if parsed["action"] == "unknown":
                print("Could not parse evaluator response (no DONE/CONTINUE). Stopping.")
                break
            if i == max_iterations:
                print(f"Reached maximum iterations ({max_iterations}). Stopping.")
                break

            if strategy == "climbing":
                if improved:
                    rejected = []
                    stale = 0
                    next_idea = parsed["next_prompt"]
                else:
                    if candidate_idea is not None:
                        rejected.append(candidate_idea)
                    stale += 1
                    if stale >= patience:
                        print(f"No improvement after {patience} attempt(s); stopping.")
                        break
                    print(f"Reverting to best (iteration {best['iteration']}, score {best['score']}/10).")
                    next_idea = refine_image(eval_client, best["image_path"], rejected)
                candidate_idea = next_idea
                prompt = to_prompt(next_idea)
            else:
                next_idea = parsed["next_prompt"]
                prompt = to_prompt(next_idea)
                if summarizer_instruction:
                    entry["summarized_prompt"] = prompt
                    print(f"Summarized → next prompt: {prompt}\n")

    except KeyboardInterrupt:
        print("\nInterrupted — writing partial results so far...")
    finally:
        if strategy == "climbing" and best:
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
    parser = build_arg_parser(
        "Iteratively heat a hotdog image. --strategy greedy (default) always accepts the "
        "evaluator's suggestion; --strategy climbing keeps the best and reverts on regression."
    )
    parser.add_argument(
        "--strategy",
        choices=["greedy", "climbing"],
        default="greedy",
        help="greedy: always accept the evaluator's suggestion. "
        "climbing: keep the best image and revert on non-improvement. Default: greedy",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="(climbing only) Stop after this many consecutive non-improving iterations. Default: 3",
    )
    args = parser.parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    run(args.image_model, args.eval_model, output_dir, args.max_iterations, args.strategy, args.patience)


if __name__ == "__main__":
    main()
