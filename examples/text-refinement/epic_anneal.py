#!/usr/bin/env python3
"""Simulated-annealing variant of the epic-sentence refinement loop.

``epic_loop.py --strategy climbing`` is a strict hill-climber: it only advances on a *higher*
score and reverts otherwise. This script generalises that into **simulated annealing** — the climber is the
``T → 0`` limit of what's here. It keeps a ``current`` walk-state (distinct from the best-ever
sentence) and, controlled by a falling *temperature*, will probabilistically accept a *worse*
sentence to escape a local optimum, cooling into greedy behaviour over time.

Acceptance (Metropolis rule), with ``Δ = new_score − current_score``:
- ``Δ > 0`` (more epic): always accept.
- ``Δ == 0`` (tie): always accept — a free sideways move (the deliberate opposite of the
  climber, which treats a tie as no progress).
- ``Δ < 0`` (less epic): accept with probability ``exp(Δ / T)``. High ``T`` early ⇒ explores
  freely; as ``T`` cools toward 0 ⇒ behaves like the greedy climber.

The best-ever sentence is tracked separately and written to ``best.txt`` regardless of the walk.

The same cooling schedule also drives the **proposer's** LLM sampling temperature: the judge
proposes refinement directions hot (diverse, exploratory) early and cold (conservative tweaks)
late — see ``_proposer_temperature``. Only proposals are annealed; the **judge that assigns the
score is always cold**, since the score is the objective and must stay stable.

This is the text-only twin of ``hotdog_anneal.py``. The same honest caveats apply: the coarse
integer judge makes ``Δ`` almost always ``0`` or ``±1`` (blunting fine temperature control),
and runs are short (tens of steps), so the practical win over the climber is local-optimum
escape, not asymptotic convergence. The "neighbourhood" is the judge's rewrite directive, so
annealing happens in directive space.

Stops when the judge says DONE, or at --max-iterations.

Usage:
    python examples/text-refinement/epic_anneal.py
    python examples/text-refinement/epic_anneal.py --initial-temp 3.0 --cooling-rate 0.9 --seed 7
    python examples/text-refinement/epic_anneal.py --max-iterations 0 --gen-model ollama:qwen3:8b
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
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


# Sampling-temperature band for the *proposer* (refinement-direction generation). The annealing
# schedule slides the proposer between these — diverse/creative directions while hot,
# conservative tweaks once cooled. The judge (scoring) is never annealed; only proposals get it.
PROPOSER_TEMP_HOT = 1.1
PROPOSER_TEMP_COLD = 0.3


def _score(value: int | None) -> int:
    """Coerce a possibly-missing epicness score to a comparable int (unparsed → 0)."""
    return value if value is not None else 0


def _proposer_temperature(temperature: float, initial_temp: float) -> float:
    """Map the current annealing temperature to an LLM sampling temperature for proposals.

    Shares the annealing *schedule* — via the cooling fraction ``f = T / T0`` (1 → 0) — but
    lives in the LLM's own units: hot early (``PROPOSER_TEMP_HOT``, diverse refinement
    directions) cooling to ``PROPOSER_TEMP_COLD`` (conservative tweaks).
    """
    fraction = temperature / initial_temp if initial_temp > 0 else 0.0
    fraction = max(0.0, min(1.0, fraction))
    return PROPOSER_TEMP_COLD + fraction * (PROPOSER_TEMP_HOT - PROPOSER_TEMP_COLD)


def _accept(delta: int, temperature: float, rng: random.Random) -> bool:
    """Metropolis acceptance rule.

    Always accept a non-worsening move (``delta >= 0``, which covers both a more-epic sentence
    and a tie). Accept a worsening move (``delta < 0``) with probability ``exp(delta /
    temperature)``. At ``temperature → 0`` worsening moves are never accepted — the pure-greedy
    (hill-climbing) limit.
    """
    if delta >= 0:
        return True
    if temperature <= 1e-9:
        return False
    return rng.random() < math.exp(delta / temperature)


def run_anneal(
    seed_sentence: str,
    gen_model_id: str,
    judge_model_id: str,
    output_dir: Path,
    max_iterations: int,
    initial_temp: float,
    cooling_rate: float,
    seed: int | None,
) -> None:
    gen_client = aimu.client(gen_model_id)
    judge_client = aimu.client(judge_model_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Seed sentence: {seed_sentence}")
    print(f"Annealing: initial T={initial_temp}, cooling rate={cooling_rate}, seed={seed}\n")

    rng = random.Random(seed)

    direction = INITIAL_DIRECTION
    current: dict | None = None  # the accepted walk-state (may be worse than best)
    best: dict | None = None  # the most-epic sentence seen so far (written to best.txt)
    rejected: list[str] = []  # directions tried against the current state that weren't accepted
    temperature = initial_temp
    trace: list[dict] = []

    iterations = itertools.count(1) if max_iterations == 0 else iter(range(1, max_iterations + 1))
    cap = "∞" if max_iterations == 0 else str(max_iterations)

    try:
        for i in iterations:
            print(f"--- Iteration {i}/{cap} (T={temperature:.3f}) ---")
            print(f"Directive: {direction}")

            sentence = generate_sentence(gen_client, seed_sentence, direction)
            print(f"Sentence: {sentence}")

            response, parsed = evaluate_sentence(judge_client, sentence)
            score = _score(parsed["score"])

            # Metropolis acceptance against the current walk-state (first sentence always accepted).
            delta = 0 if current is None else score - _score(current["score"])
            accept = current is None or _accept(delta, temperature, rng)
            improved_best = best is None or score > _score(best["score"])

            verdict = "accepted" if accept else "rejected"
            if improved_best:
                verdict += " (new best)"
            status = f"{verdict} [Δ={delta}, T={temperature:.3f}]"
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

            if improved_best:
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

            if accept:
                # Move the walk to this sentence; a fresh accept clears the failed-direction list.
                current = {"score": parsed["score"], "sentence": sentence}
                rejected = []
            else:
                # Stay at current; remember the failed direction so the proposer avoids it.
                rejected.append(direction)
                print(f"Staying at current (this score {parsed['score']} rejected).")

            # Propose the next refinement from the current sentence at the *annealed* proposer
            # temperature — diverse directions while hot, conservative once cooled. The score
            # itself stays cold.
            proposer_temp = _proposer_temperature(temperature, initial_temp)
            print(f"Proposing next refinement at sampling temperature {proposer_temp:.2f}.\n")
            direction = refine_sentence(judge_client, current["sentence"], rejected, temperature=proposer_temp)
            temperature *= cooling_rate
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


def build_parser() -> argparse.ArgumentParser:
    """Common epic options plus the simulated-annealing knobs."""
    p = build_arg_parser("Anneal an epic sentence: accept worsening steps early, cool to greedy.")
    p.add_argument(
        "--initial-temp",
        type=float,
        default=2.0,
        help="Starting temperature (default: 2.0). Higher = accepts worse sentences more readily early on",
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
        args.seed_sentence,
        args.gen_model,
        args.judge_model,
        output_dir,
        args.max_iterations,
        args.initial_temp,
        args.cooling_rate,
        args.seed,
    )


if __name__ == "__main__":
    main()
