#!/usr/bin/env python3
"""Iteratively rewrite a mundane sentence to be as epic as possible using local models.

Code-controlled: Python directs the loop (a plain ``for`` loop over AIMU text clients); the
LLMs only generate and judge. The agent-driven counterpart is ``epic_agent.py``;
``epic_climbing.py`` adds best-state caching + revert-on-regression. Stops when the judge
declares the sentence cannot get any more epic.

This is the text-only twin of ``hotdog_loop.py`` — same generate → judge → refine loop, no
image generation, vision model, or GPU required (runs on Ollama alone).

Usage:
    python scripts/epic_loop.py
    python scripts/epic_loop.py --gen-model ollama:qwen3:8b --judge-model anthropic:claude-sonnet-4-6
    python scripts/epic_loop.py --seed-sentence "She parks the car." --max-iterations 5
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
    resolve_output_dir,
    write_summary,
)


def run_loop(
    seed_sentence: str,
    gen_model_id: str,
    judge_model_id: str,
    output_dir: Path,
    max_iterations: int,
) -> None:
    gen_client = aimu.client(gen_model_id)
    judge_client = aimu.client(judge_model_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Seed sentence: {seed_sentence}\n")

    direction = INITIAL_DIRECTION
    trace: list[dict] = []

    # max_iterations == 0 means run indefinitely until the judge says DONE.
    iterations = itertools.count(1) if max_iterations == 0 else iter(range(1, max_iterations + 1))
    cap_label = "∞" if max_iterations == 0 else str(max_iterations)

    try:
        for i in iterations:
            print(f"--- Iteration {i}/{cap_label} ---")
            print(f"Directive: {direction}")

            sentence = generate_sentence(gen_client, seed_sentence, direction)
            print(f"Sentence: {sentence}")

            judge_response, parsed = evaluate_sentence(judge_client, sentence)
            print(f"Judge: score={parsed['score']} action={parsed['action']}\n")
            trace.append(
                {
                    "iteration": i,
                    "direction": direction,
                    "sentence": sentence,
                    "judge_response": judge_response,
                    "score": parsed["score"],
                    "action": parsed["action"],
                    "next_direction": parsed["next_direction"],
                }
            )

            if parsed["action"] == "DONE":
                print(f"Judge declared maximum epicness after {i} iteration(s).")
                break
            if parsed["action"] == "unknown":
                print("Could not parse judge response (no DONE/CONTINUE). Stopping.")
                break
            if i == max_iterations:
                print(f"Reached maximum iterations ({max_iterations}). Stopping.")
                break

            direction = parsed["next_direction"]
    except KeyboardInterrupt:
        print("\nInterrupted — writing partial results so far...")
    finally:
        if trace:
            summary_path = write_summary(
                output_dir, trace, seed_sentence=seed_sentence, gen_model=gen_model_id, judge_model=judge_model_id
            )
            print(f"Summary written to: {summary_path}")


def main() -> None:
    args = build_arg_parser(
        "Iteratively make a sentence epic with a code-directed loop over local models."
    ).parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    run_loop(args.seed_sentence, args.gen_model, args.judge_model, output_dir, args.max_iterations)


if __name__ == "__main__":
    main()
