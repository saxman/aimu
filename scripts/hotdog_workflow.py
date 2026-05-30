#!/usr/bin/env python3
"""Iteratively generate and evaluate a hotdog image using local models.

A code-controlled **workflow**: Python directs the loop; the LLMs only evaluate and evolve
the prompt. The agent-driven counterpart is ``hotdog_agent.py``; ``hotdog_workflow_climbing.py``
adds best-state caching + revert-on-regression. Stops when the vision evaluator declares the
hotdog cannot get hotter.

Usage:
    python scripts/hotdog_workflow.py
    python scripts/hotdog_workflow.py --image-model hf:stabilityai/stable-diffusion-xl-base-1.0 --eval-model ollama:gemma4:26b
    python scripts/hotdog_workflow.py --output-dir /tmp/hotdog --max-iterations 5
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import aimu

sys.path.insert(0, str(Path(__file__).parent))
from _hotdog_common import (
    EVALUATOR_PROMPT,
    NEGATIVE_PROMPT,
    build_arg_parser,
    build_image_prompt,
    build_summarizer_prompt,
    collage_generated_images,
    parse_evaluator_response,
    resolve_output_dir,
    summarize_for_image,
    suppress_benign_clip_warning,
    write_summary,
)


def run_loop(
    image_model_name: str,
    eval_model_id: str,
    output_dir: Path,
    max_iterations: int,
) -> None:
    image_client = aimu.image_client(image_model_name)
    suppress_benign_clip_warning(image_client)
    eval_client = aimu.client(eval_model_id)
    if not eval_client.is_vision_model:
        raise ValueError(f"Eval model {eval_model_id!r} does not support vision.")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    prompt = "a hot hotdog"
    trace = []

    # Summarize the evaluator's description down to the image model's token budget.
    # Models with no cap (cloud, max_prompt_tokens is None) take the description directly.
    max_prompt_tokens = image_client.max_prompt_tokens
    summarizer_instruction = build_summarizer_prompt(max_prompt_tokens) if max_prompt_tokens else None
    budget_label = f"{max_prompt_tokens} tokens" if max_prompt_tokens else "uncapped"
    print(f"Image prompt budget: {budget_label} ({'summarizing' if summarizer_instruction else 'direct'})\n")

    # max_iterations == 0 means run indefinitely until the evaluator says DONE.
    iterations = itertools.count(1) if max_iterations == 0 else iter(range(1, max_iterations + 1))
    cap_label = "∞" if max_iterations == 0 else str(max_iterations)

    try:
        for i in iterations:
            print(f"--- Iteration {i}/{cap_label} ---")
            print(f"Prompt: {prompt}")

            image_prompt = build_image_prompt(prompt)
            raw_path = image_client.generate(
                image_prompt, negative_prompt=NEGATIVE_PROMPT, format="path", output_dir=output_dir
            )
            dest = output_dir / f"{i:02d}.png"
            Path(raw_path).rename(dest)
            print(f"Image saved: {dest}")

            eval_client.reset()
            evaluator_response = eval_client.chat(EVALUATOR_PROMPT, images=[str(dest)])
            print(f"Evaluator:\n{evaluator_response}\n")

            parsed = parse_evaluator_response(evaluator_response)
            entry = {
                "iteration": i,
                "prompt": prompt,
                "image_prompt": image_prompt,
                "image_path": str(dest),
                "evaluator_response": evaluator_response,
                "score": parsed["score"],
                "action": parsed["action"],
                "next_prompt": parsed["next_prompt"],
            }
            trace.append(entry)

            if parsed["action"] == "DONE":
                print(f"Evaluator declared maximum hotness after {i} iteration(s).")
                break
            if parsed["action"] == "unknown":
                print("Could not parse evaluator response (no DONE/CONTINUE). Stopping.")
                break

            if i == max_iterations:
                print(f"Reached maximum iterations ({max_iterations}). Stopping.")
                break

            if summarizer_instruction is None:
                # Uncapped model — feed the full description straight through.
                prompt = parsed["next_prompt"]
            else:
                # Second stage of the chain: condense the description to fit the budget.
                prompt = summarize_for_image(eval_client, parsed["next_prompt"], max_prompt_tokens)
                entry["summarized_prompt"] = prompt
                print(f"Summarized → next prompt: {prompt}\n")
    except KeyboardInterrupt:
        print("\nInterrupted — writing partial results so far...")
    finally:
        # Always emit summary + collage, even on interrupt. The collage scans the saved
        # image files so it includes every generated image, not just evaluated ones.
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


def main() -> None:
    args = build_arg_parser(
        "Iteratively heat a hotdog image using local HF diffusers + Ollama vision."
    ).parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    run_loop(args.image_model, args.eval_model, output_dir, args.max_iterations)


if __name__ == "__main__":
    main()
