#!/usr/bin/env python3
"""Iteratively generate and evaluate a hotdog image using local models.

Python controls the loop; LLMs handle evaluation and prompt evolution.
Stops when the vision evaluator declares the hotdog cannot get hotter.

Usage:
    python scripts/hotdog_loop.py
    python scripts/hotdog_loop.py --image-model hf:stabilityai/stable-diffusion-xl-base-1.0 --eval-model ollama:gemma4:26b
    python scripts/hotdog_loop.py --output-dir /tmp/hotdog --max-iterations 5
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import aimu
from aimu import paths as aimu_paths

sys.path.insert(0, str(Path(__file__).parent))
from _hotdog_common import parse_evaluator_response, write_summary, EVALUATOR_PROMPT


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Iteratively heat a hotdog image using local HF diffusers + Ollama vision."
    )
    p.add_argument(
        "--image-model",
        default="hf:black-forest-labs/FLUX.1-schnell",
        help="Image model string in 'provider:model_id' form (default: hf:black-forest-labs/FLUX.1-schnell)",
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
        help="Hard cap on iteration count (default: 10)",
    )
    return p


def run_loop(
    image_model_name: str,
    eval_model_id: str,
    output_dir: Path,
    max_iterations: int,
) -> None:
    image_client = aimu.image_client(image_model_name)
    eval_client = aimu.client(eval_model_id)
    if not eval_client.is_vision_model:
        raise ValueError(f"Eval model {eval_model_id!r} does not support vision.")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    prompt = "a hot hotdog"
    trace = []

    for i in range(1, max_iterations + 1):
        print(f"--- Iteration {i}/{max_iterations} ---")
        print(f"Prompt: {prompt}")

        raw_path = image_client.generate(prompt, format="path", output_dir=output_dir)
        dest = output_dir / f"{i:02d}.png"
        Path(raw_path).rename(dest)
        print(f"Image saved: {dest}")

        eval_client.reset()
        evaluator_response = eval_client.chat(EVALUATOR_PROMPT, images=[str(dest)])
        print(f"Evaluator:\n{evaluator_response}\n")

        parsed = parse_evaluator_response(evaluator_response)
        trace.append({
            "iteration": i,
            "prompt": prompt,
            "image_path": str(dest),
            "evaluator_response": evaluator_response,
            "score": parsed["score"],
            "action": parsed["action"],
            "next_prompt": parsed["next_prompt"],
        })

        if parsed["action"] == "DONE":
            print(f"Evaluator declared maximum hotness after {i} iteration(s).")
            break
        if parsed["action"] == "unknown":
            print("Could not parse evaluator response (no DONE/CONTINUE). Stopping.")
            break

        if i == max_iterations:
            print(f"Reached maximum iterations ({max_iterations}). Stopping.")
            break

        prompt = parsed["next_prompt"]

    summary_path = write_summary(output_dir, trace)
    print(f"Summary written to: {summary_path}")


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = aimu_paths.output / "hotdog" / timestamp
    run_loop(args.image_model, args.eval_model, output_dir, args.max_iterations)


if __name__ == "__main__":
    main()
