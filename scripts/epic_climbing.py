#!/usr/bin/env python3
"""Hill-climbing variant of the epic-sentence refinement loop.

`epic_loop.py` and `epic_agent.py` walk forward greedily: whatever directive the judge proposes
next is rendered and accepted, even if the new sentence scores *worse* than a previous one. This
script instead keeps the best sentence seen so far and **only advances on a strictly higher
score** — if a generation fails to beat the best (a lower score *or* a tie; the coarse 1–10
judge makes ties common, and a tie isn't demonstrated progress), it's discarded and the judge is
asked for a *different* refinement of the current best (avoiding directions that already failed).

That mirrors the hill-climbing in `aimu.prompts` (best-state caching + revert when a candidate
doesn't improve), applied here to a single sentence rather than to a reusable prompt over a
dataset. The text-only twin of `hotdog_climbing.py`. Stops when the judge says DONE, after
--patience consecutive non-improvements, or at --max-iterations.

Usage:
    python scripts/epic_climbing.py
    python scripts/epic_climbing.py --max-iterations 0 --patience 4
    python scripts/epic_climbing.py --gen-model ollama:qwen3:8b --judge-model ollama:qwen3:8b
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


def run_climb(
    seed_sentence: str,
    gen_model_id: str,
    judge_model_id: str,
    output_dir: Path,
    max_iterations: int,
    patience: int,
) -> None:
    gen_client = aimu.client(gen_model_id)
    judge_client = aimu.client(judge_model_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Seed sentence: {seed_sentence}\n")

    direction = INITIAL_DIRECTION
    best: dict | None = None
    rejected: list[str] = []  # directions tried against the current best that didn't beat it
    stale = 0  # consecutive non-improving iterations
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
            improved = best is None or _score(parsed["score"]) > _score(best["score"])
            status = "accepted (new best)" if improved else "rejected (no improvement)"
            print(f"Judge: score={parsed['score']} action={parsed['action']} → {status}\n")

            trace.append(
                {
                    "iteration": i,
                    "direction": direction,
                    "sentence": sentence,
                    "judge_response": response,
                    "score": parsed["score"],
                    "action": parsed["action"],
                    "next_direction": parsed["next_direction"],
                    "status": status,
                }
            )

            if improved:
                best = {"score": parsed["score"], "sentence": sentence, "iteration": i}

            if parsed["action"] == "DONE":
                print(f"Judge declared maximum epicness at iteration {i}.")
                break
            if parsed["action"] == "unknown":
                print("Could not parse judge response (no DONE/CONTINUE). Stopping.")
                break
            if i == max_iterations:
                print(f"Reached maximum iterations ({max_iterations}). Stopping.")
                break

            if improved:
                # New best — explore from its own suggested refinement; clear the reject list.
                rejected = []
                stale = 0
                direction = parsed["next_direction"]
            else:
                # No improvement (lower score or a tie) — keep the best, remember the failed
                # direction, and ask for a different one.
                rejected.append(direction)
                stale += 1
                if stale >= patience:
                    print(f"No improvement after {patience} attempt(s); stopping.")
                    break
                print(f"Reverting to best (iteration {best['iteration']}, score {best['score']}/10).")
                direction = refine_sentence(judge_client, best["sentence"], rejected)
    except KeyboardInterrupt:
        print("\nInterrupted — writing partial results so far...")
    finally:
        if best:
            best_path = write_best(output_dir, best)
            print(f"\nBest: iteration {best['iteration']}, score {best['score']}/10 → {best_path}")
        if trace:
            summary_path = write_summary(
                output_dir, trace, seed_sentence=seed_sentence, gen_model=gen_model_id, judge_model=judge_model_id
            )
            print(f"Summary written to: {summary_path}")


def main() -> None:
    parser = build_arg_parser("Hill-climb an epic sentence: keep the best, revert on regression.")
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Stop after this many consecutive non-improving iterations (default: 3)",
    )
    args = parser.parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    run_climb(args.seed_sentence, args.gen_model, args.judge_model, output_dir, args.max_iterations, args.patience)


if __name__ == "__main__":
    main()
