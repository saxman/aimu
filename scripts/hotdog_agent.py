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
    parse_evaluator_response,
    resolve_output_dir,
    write_summary,
)

AGENT_SYSTEM_PROMPT = """\
You are running a hotdog heating experiment. Your job is to iteratively make
an image of a single hotdog as hot as possible using image generation.

Procedure:
1. Call generate_hotdog_image with your current prompt (start: "a hot hotdog")
2. Call evaluate_hotness with the returned image path
3. If the evaluator outputs DONE, stop and summarise the final result
4. If the evaluator outputs CONTINUE, use the suggested prompt and repeat from step 1
5. {iteration_limit_rule}
"""


def make_tools(image_client, eval_client, output_dir: Path) -> tuple:
    """Return (generate_hotdog_image, evaluate_hotness) tools sharing a counter closure."""
    counter = {"value": 0}

    @tool
    def generate_hotdog_image(prompt: str) -> str:
        """Generate a hotdog image from a text prompt and save it locally. Returns the saved file path."""
        counter["value"] += 1
        i = counter["value"]
        raw_path = image_client.generate(
            prompt, negative_prompt=NEGATIVE_PROMPT, format="path", output_dir=output_dir
        )
        dest = output_dir / f"{i:02d}.png"
        Path(raw_path).rename(dest)
        print(f"[Iteration {i}] Image saved: {dest}")
        return str(dest)

    @tool
    def evaluate_hotness(image_path: str) -> str:
        """Evaluate how hot a hotdog image is. Returns DONE or CONTINUE with reasoning."""
        eval_client.reset()
        response = eval_client.chat(EVALUATOR_PROMPT, images=[image_path])
        print(f"[Evaluator] {response}\n")
        return response

    return generate_hotdog_image, evaluate_hotness


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
                pending = {"prompt": args.get("prompt", ""), "image_path": result}
            elif name == "evaluate_hotness" and pending is not None:
                iteration += 1
                parsed = parse_evaluator_response(result)
                trace.append({
                    "iteration": iteration,
                    "prompt": pending["prompt"],
                    "image_path": pending["image_path"],
                    "evaluator_response": result,
                    "score": parsed["score"],
                    "action": parsed["action"],
                    "next_prompt": parsed["next_prompt"],
                })
                pending = None

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

    generate_fn, evaluate_fn = make_tools(image_client, eval_client, output_dir)

    if args.max_iterations == 0:
        # Run indefinitely: the agent stops on its own once the evaluator says DONE.
        # Agent.max_iterations has no unbounded sentinel, so use a very high cap as the safety net.
        iteration_limit_rule = "Continue until the evaluator outputs DONE; there is no fixed iteration limit"
        agent_max_iterations = 10**6
    else:
        iteration_limit_rule = f"Never exceed {args.max_iterations} iterations total"
        # Each hotdog iteration calls ~2 tools; allow headroom for the agent's decision rounds.
        agent_max_iterations = args.max_iterations * 3

    agent = Agent(
        agent_client,
        name="hotdog-agent",
        system_message=AGENT_SYSTEM_PROMPT.format(iteration_limit_rule=iteration_limit_rule),
        tools=[generate_fn, evaluate_fn],
        max_iterations=agent_max_iterations,
    )

    print("Starting hotdog heating experiment...\n")
    result = agent.run("Begin the hotdog heating experiment. Start with 'a hot hotdog'.")
    print(f"\nAgent final response:\n{result}")

    trace = parse_agent_trace(agent.messages.get("hotdog-agent", []))
    if trace:
        summary_path = write_summary(output_dir, trace)
        print(f"\nSummary written to: {summary_path}")
    else:
        print("\nNo iterations recorded in agent messages.")


if __name__ == "__main__":
    main()
