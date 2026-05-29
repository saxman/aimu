#!/usr/bin/env python3
"""Iteratively generate and evaluate a hotdog image using local models.

An AIMU Agent autonomously controls the loop via tool calls.
Stops when the Ollama vision evaluator declares the hotdog cannot get hotter.

Usage:
    python scripts/hotdog_agent.py
    python scripts/hotdog_agent.py --image-model SDXL_BASE --eval-model gemma4:26b
    python scripts/hotdog_agent.py --output-dir /tmp/hotdog --max-iterations 5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from aimu.agents import Agent
from aimu.models.hf_image import HuggingFaceImageClient, HuggingFaceImageModel
from aimu.models.ollama import OllamaClient
from aimu.tools.decorator import tool

sys.path.insert(0, str(Path(__file__).parent))
from _hotdog_common import get_ollama_model, parse_evaluator_response, write_summary

EVALUATOR_PROMPT = """\
You are evaluating how visually "hot" this hotdog image is.
Rate its hotness from 1 to 10 (10 = blazing inferno hotdog, 1 = cold).
Then decide: can this hotdog get any hotter? If not, output exactly:
  DONE: <your reasoning>
If it can get hotter, output exactly:
  CONTINUE: <a refined image generation prompt that will make it hotter>
"""

AGENT_SYSTEM_PROMPT = """\
You are running a hotdog heating experiment. Your job is to iteratively make
a hotdog image as hot as possible using image generation.

Procedure:
1. Call generate_hotdog_image with your current prompt (start: "a hot hotdog")
2. Call evaluate_hotness with the returned image path
3. If the evaluator outputs DONE, stop and summarise the final result
4. If the evaluator outputs CONTINUE, use the suggested prompt and repeat from step 1
5. Never exceed {max_iterations} iterations total
"""


def make_tools(
    image_client: HuggingFaceImageClient,
    eval_client: OllamaClient,
    output_dir: Path,
) -> tuple:
    """Return (generate_hotdog_image, evaluate_hotness) tools sharing a counter closure."""
    counter = {"value": 0}

    @tool
    def generate_hotdog_image(prompt: str) -> str:
        """Generate a hotdog image from a text prompt and save it locally. Returns the saved file path."""
        counter["value"] += 1
        i = counter["value"]
        raw_path = image_client.generate(prompt, format="path", output_dir=output_dir)
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
            args = json.loads(fn.get("arguments", "{}"))
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


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Iteratively heat a hotdog image using an AIMU Agent + local models."
    )
    p.add_argument(
        "--image-model",
        default="FLUX_SCHNELL",
        help="HuggingFaceImageModel enum member name (default: FLUX_SCHNELL)",
    )
    p.add_argument(
        "--eval-model",
        default="gemma4:e4b",
        help="Ollama model id for agent reasoning and vision evaluation (default: gemma4:e4b)",
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
        help="Hard cap on hotdog generation iterations (default: 10)",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        from aimu import paths as aimu_paths
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = aimu_paths.output / "hotdog" / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    image_client = HuggingFaceImageClient(HuggingFaceImageModel[args.image_model])
    ollama_model = get_ollama_model(args.eval_model)
    # Two separate client instances: one for agent reasoning, one for vision eval inside the tool.
    agent_client = OllamaClient(ollama_model)
    eval_client = OllamaClient(ollama_model)

    generate_fn, evaluate_fn = make_tools(image_client, eval_client, output_dir)

    agent = Agent(
        agent_client,
        name="hotdog-agent",
        system_message=AGENT_SYSTEM_PROMPT.format(max_iterations=args.max_iterations),
        tools=[generate_fn, evaluate_fn],
        # Each hotdog iteration calls ~2 tools; allow headroom for the agent's decision rounds.
        max_iterations=args.max_iterations * 3,
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
