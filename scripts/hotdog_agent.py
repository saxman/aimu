#!/usr/bin/env python3
"""Iteratively generate and evaluate a hotdog image using local models.

An AIMU Agent autonomously controls the loop via tool calls.
Stops when the vision evaluator declares the hotdog cannot get hotter.

Usage:
    python scripts/hotdog_agent.py
    python scripts/hotdog_agent.py --image-model hf:stabilityai/stable-diffusion-xl-base-1.0 --eval-model ollama:gemma4:26b
    python scripts/hotdog_agent.py --output-dir /tmp/hotdog --max-iterations 5
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import aimu
from aimu.agents import Agent
from aimu.tools.decorator import tool

sys.path.insert(0, str(Path(__file__).parent))
from _hotdog_common import (
    EVALUATOR_PROMPT,
    NEGATIVE_PROMPT,
    build_arg_parser,
    build_image_prompt,
    collage_generated_images,
    parse_evaluator_response,
    resolve_output_dir,
    summarize_for_image,
    write_summary,
)

AGENT_SYSTEM_PROMPT = """\
You are running a hotdog heating experiment. Your job is to iteratively make
an image of a single hotdog as hot as possible using image generation.

Procedure:
1. Call generate_hotdog_image with your current prompt (start: "a hot hotdog")
2. Call evaluate_hotness with the returned image path
3. If the evaluator outputs DONE, stop and summarise the final result
4. {continue_rule}
5. {iteration_limit_rule}
"""


def make_tools(image_client, eval_client, output_dir: Path) -> tuple:
    """Return (generate_hotdog_image, evaluate_hotness, summarize_description) tools.

    All three share the closure; ``eval_client`` backs both the vision evaluation
    and the text summarization step.
    """
    counter = {"value": 0}

    @tool
    def generate_hotdog_image(prompt: str) -> str:
        """Generate a hotdog image from a short text prompt and save it locally. Returns the saved file path."""
        counter["value"] += 1
        i = counter["value"]
        raw_path = image_client.generate(
            build_image_prompt(prompt), negative_prompt=NEGATIVE_PROMPT, format="path", output_dir=output_dir
        )
        dest = output_dir / f"{i:02d}.png"
        Path(raw_path).rename(dest)
        print(f"[Iteration {i}] Image saved: {dest}")
        return str(dest)

    @tool
    def evaluate_hotness(image_path: str) -> str:
        """Evaluate how hot a hotdog image is. Returns DONE, or CONTINUE with a detailed description."""
        eval_client.reset()
        response = eval_client.chat(EVALUATOR_PROMPT, images=[image_path])
        print(f"[Evaluator] {response}\n")
        return response

    @tool
    def summarize_description(description: str) -> str:
        """Condense a detailed natural-language description into a short text-to-image prompt. Returns the short prompt."""
        short = summarize_for_image(eval_client, description)
        print(f"[Summarized] {short}\n")
        return short

    return generate_hotdog_image, evaluate_hotness, summarize_description


def parse_agent_trace(messages: list[dict]) -> list[dict]:
    """Reconstruct the per-iteration trace from agent message history (OpenAI format)."""
    tool_results: dict[str, str] = {
        msg.get("tool_call_id", ""): msg.get("content", "")
        for msg in messages
        if msg.get("role") == "tool"
    }

    trace = []
    iteration = 0
    pending: dict | None = None
    last_entry: dict | None = None

    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            raw_args = fn.get("arguments", "{}")
            # Tool-call arguments arrive as a JSON string (OpenAI) or an already-parsed dict (Ollama).
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            result = tool_results.get(tc.get("id", ""), "")

            if name == "generate_hotdog_image":
                tool_prompt = args.get("prompt", "")
                pending = {
                    "prompt": tool_prompt,
                    # The tool anchors the prompt before generating; reconstruct what was sent.
                    "image_prompt": build_image_prompt(tool_prompt),
                    "image_path": result,
                }
            elif name == "evaluate_hotness" and pending is not None:
                iteration += 1
                parsed = parse_evaluator_response(result)
                last_entry = {
                    "iteration": iteration,
                    "prompt": pending["prompt"],
                    "image_prompt": pending["image_prompt"],
                    "image_path": pending["image_path"],
                    "evaluator_response": result,
                    "score": parsed["score"],
                    "action": parsed["action"],
                    "next_prompt": parsed["next_prompt"],
                }
                trace.append(last_entry)
                pending = None
            elif name == "summarize_description" and last_entry is not None:
                # The summarized short prompt feeds the next generate_hotdog_image call.
                last_entry["summarized_prompt"] = result

    return trace


def main() -> None:
    args = build_arg_parser(
        "Iteratively heat a hotdog image using an AIMU Agent + local models."
    ).parse_args()
    output_dir = resolve_output_dir(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    image_client = aimu.image_client(args.image_model)
    # Two separate client instances: one for agent reasoning, one for vision eval inside the tool.
    agent_client = aimu.client(args.eval_model)
    eval_client = aimu.client(args.eval_model)
    if not eval_client.is_vision_model:
        raise ValueError(f"Eval model {args.eval_model!r} does not support vision.")

    generate_fn, evaluate_fn, summarize_fn = make_tools(image_client, eval_client, output_dir)

    # Long-prompt models (T5-based: SD3, FLUX) take the full description directly;
    # CLIP-only models (SD/SDXL) need the summarize step to stay under 77 tokens.
    long_prompts = image_client.supports_long_prompts
    print(f"Long-prompt model: {long_prompts} (summarize step {'skipped' if long_prompts else 'enabled'})\n")

    if long_prompts:
        continue_rule = (
            "If the evaluator outputs CONTINUE, it gives a detailed natural-language "
            "description. Use that description directly as the prompt and repeat from step 1."
        )
        tools = [generate_fn, evaluate_fn]
        tools_per_iteration = 2
    else:
        continue_rule = (
            "If the evaluator outputs CONTINUE, it gives a detailed natural-language "
            "description. Call summarize_description with that full description to get a short "
            "image prompt, then repeat from step 1 using that short prompt."
        )
        tools = [generate_fn, evaluate_fn, summarize_fn]
        tools_per_iteration = 3

    if args.max_iterations == 0:
        # Run indefinitely: the agent stops on its own once the evaluator says DONE.
        # Agent.max_iterations has no unbounded sentinel, so use a very high cap as the safety net.
        iteration_limit_rule = "Continue until the evaluator outputs DONE; there is no fixed iteration limit"
        agent_max_iterations = 10**6
    else:
        iteration_limit_rule = f"Never exceed {args.max_iterations} iterations total"
        # Allow headroom: tools-per-iteration plus a decision round.
        agent_max_iterations = args.max_iterations * (tools_per_iteration + 1)

    agent = Agent(
        agent_client,
        name="hotdog-agent",
        system_message=AGENT_SYSTEM_PROMPT.format(
            continue_rule=continue_rule, iteration_limit_rule=iteration_limit_rule
        ),
        tools=tools,
        max_iterations=agent_max_iterations,
    )

    print("Starting hotdog heating experiment...\n")
    try:
        result = agent.run("Begin the hotdog heating experiment. Start with 'a hot hotdog'.")
        print(f"\nAgent final response:\n{result}")
    except KeyboardInterrupt:
        print("\nInterrupted — writing partial results so far...")
    finally:
        # Reconstruct from the live conversation (not agent.messages, whose snapshot is
        # only set when run() finishes) so an interrupted run still produces output.
        trace = parse_agent_trace(agent.model_client.messages)
        if trace:
            summary_path = write_summary(
                output_dir, trace, image_model=image_client.spec.id, eval_model=args.eval_model
            )
            print(f"\nSummary written to: {summary_path}")
        else:
            print("\nNo iterations recorded in agent messages.")
        # Collage scans the saved image files, so it works even when no trace was
        # reconstructed (e.g. interrupted before the first evaluation completed).
        collage_path = collage_generated_images(output_dir)
        if collage_path:
            print(f"Collage written to: {collage_path}")


if __name__ == "__main__":
    main()
