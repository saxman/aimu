#!/usr/bin/env python3
"""Iteratively rewrite a mundane sentence to be as epic as possible using local models.

Code-controlled: Python directs the loop; the LLMs only generate and judge.
Two acceptance strategies are available via --strategy:

  greedy   (default) — always accept the judge's suggested directive and move on, even if the
             new sentence scores lower than a previous one. Simple; the final sentence is
             whatever came last.

  climbing — keep the best sentence seen so far and only advance on a strictly higher score.
             A tie or lower score is discarded; the judge is asked for a different refinement
             of the current best (avoiding directions that already failed). Stops after
             --patience consecutive non-improvements. Guarantees the final sentence is at
             least as good as every previous round.

The agent-directed counterpart is hotdog_agent.py's text twin, epic_agent.py. For
Metropolis-acceptance annealing see epic_anneal.py. This is the text-only twin of
hotdog_loop.py — same lesson, no image generation, vision model, or GPU required (runs on
Ollama alone).

Usage:
    python examples/text-refinement/epic_loop.py
    python examples/text-refinement/epic_loop.py --strategy climbing --patience 4
    python examples/text-refinement/epic_loop.py --gen-model ollama:qwen3:8b --judge-model anthropic:claude-sonnet-4-6
    python examples/text-refinement/epic_loop.py --seed-sentence "She parks the car." --max-iterations 0   # until DONE
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import aimu

sys.path.insert(0, str(Path(__file__).parent))
from _epic_common import (
    INITIAL_DIRECTION,
    build_arg_parser,
    evaluate_sentence,
    generate_sentence,
    refine_sentence,
    resolve_output_dir,
    write_best,
    write_summary,
)


def _score(value: int | None) -> int:
    """Coerce a possibly-missing epicness score to a comparable int (unparsed → 0)."""
    return value if value is not None else 0


def run(
    seed_sentence: str,
    gen_model_id: str,
    judge_model_id: str,
    output_dir: Path,
    max_iterations: int,
    strategy: str,
    patience: int,
) -> None:
    gen_client = aimu.client(gen_model_id)
    judge_client = aimu.client(judge_model_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Strategy: {strategy}")
    print(f"Seed sentence: {seed_sentence}")
    print(f"Output directory: {output_dir}\n")

    direction = INITIAL_DIRECTION
    # Climbing-only state — unused in greedy mode.
    best: dict | None = None
    rejected: list[str] = []
    stale = 0

    trace: list[dict] = []
    iterations = itertools.count(1) if max_iterations == 0 else iter(range(1, max_iterations + 1))
    cap = "∞" if max_iterations == 0 else str(max_iterations)

    try:
        for i in iterations:
            print(f"--- Iteration {i}/{cap} ---")
            print(f"Directive: {direction}")

            sentence = generate_sentence(gen_client, seed_sentence, direction)
            print(f"Sentence: {sentence}")

            response, parsed = evaluate_sentence(judge_client, sentence)

            entry: dict = {
                "iteration": i,
                "direction": direction,
                "sentence": sentence,
                "judge_response": response,
                "score": parsed["score"],
                "action": parsed["action"],
                "next_direction": parsed["next_direction"],
            }

            if strategy == "climbing":
                improved = best is None or _score(parsed["score"]) > _score(best["score"])
                entry["status"] = "accepted (new best)" if improved else "rejected (no improvement)"
                if improved:
                    best = {"score": parsed["score"], "sentence": sentence, "iteration": i}
                print(f"Judge: score={parsed['score']} action={parsed['action']} → {entry['status']}\n")
            else:
                print(f"Judge: score={parsed['score']} action={parsed['action']}\n")

            trace.append(entry)

            if parsed["action"] == "DONE":
                print(f"Judge declared maximum epicness at iteration {i}.")
                break
            if parsed["action"] == "unknown":
                print("Could not parse judge response (no DONE/CONTINUE). Stopping.")
                break
            if i == max_iterations:
                print(f"Reached maximum iterations ({max_iterations}). Stopping.")
                break

            # Acceptance rule — the only place the two strategies diverge.
            if strategy == "climbing":
                if improved:
                    rejected = []
                    stale = 0
                    direction = parsed["next_direction"]
                else:
                    rejected.append(direction)
                    stale += 1
                    if stale >= patience:
                        print(f"No improvement after {patience} attempt(s); stopping.")
                        break
                    print(f"Reverting to best (iteration {best['iteration']}, score {best['score']}/10).")
                    direction = refine_sentence(judge_client, best["sentence"], rejected)
            else:
                direction = parsed["next_direction"]

    except KeyboardInterrupt:
        print("\nInterrupted — writing partial results so far...")
    finally:
        if strategy == "climbing" and best:
            best_path = write_best(output_dir, best)
            print(f"\nBest: iteration {best['iteration']}, score {best['score']}/10 → {best_path}")
        if trace:
            summary_path = write_summary(
                output_dir, trace, seed_sentence=seed_sentence, gen_model=gen_model_id, judge_model=judge_model_id
            )
            print(f"Summary written to: {summary_path}")


def main() -> None:
    parser = build_arg_parser(
        "Iteratively make a sentence epic. --strategy greedy (default) always accepts the "
        "judge's suggestion; --strategy climbing keeps the best and reverts on regression."
    )
    parser.add_argument(
        "--strategy",
        choices=["greedy", "climbing"],
        default="greedy",
        help="greedy: always accept the judge's suggestion. "
        "climbing: keep the best sentence and revert on non-improvement. Default: greedy",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="(climbing only) Stop after this many consecutive non-improving iterations. Default: 3",
    )
    args = parser.parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    run(
        args.seed_sentence,
        args.gen_model,
        args.judge_model,
        output_dir,
        args.max_iterations,
        args.strategy,
        args.patience,
    )


if __name__ == "__main__":
    main()
