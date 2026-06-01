#!/usr/bin/env python3
"""Iteratively generate and evaluate a hotdog image using AIMU's EvaluatorOptimizer.

This is the same generate → evaluate → refine loop as ``hotdog_loop.py`` and
``hotdog_agent.py``, but expressed with the library's **EvaluatorOptimizer** workflow
instead of a hand-rolled loop or a single tool-calling agent. It composes two ``Agent``s:

- the **generator** holds ``generate_hotdog_image`` — it distils the task (or the critic's
  refinement feedback) into a short prompt, generates an image, and replies with the path.
- the **evaluator** holds ``evaluate_hotness`` — it reads that path, runs the vision model,
  and relays the verdict (``DONE`` or a ``CONTINUE`` description) back verbatim.

``EvaluatorOptimizer`` stops when ``pass_keyword="DONE"`` appears in the evaluator's reply or
``max_rounds`` is hit, then returns the last generation.

Why this is a *composition* and not a direct fit: ``EvaluatorOptimizer`` orchestrates text
``Runner``s — it shuttles each step's output to the next as a **string** (see
``aimu/agents/workflows/evaluator.py``). Image generation and vision evaluation therefore have
to live *inside the agents' tools*; the image path and the verdict travel between the two
agents as plain text. That round-trip is more indirect (and more fragile) than the variable
hand-off in ``hotdog_loop.py`` — which is exactly the trade-off worth seeing. ``evaluate_hotness``
falls back to the most recent image if the relayed path doesn't resolve, to keep that text
relay from breaking the run. Prompt summarization (token-budget fitting) folds into the
generator agent's own prompt-writing rather than a separate step.

Two caveats of the EvaluatorOptimizer shape, both visible in the output:
- The *final* generation is returned without being evaluated (the loop generates after the
  last evaluation, then stops), so the summary may have one fewer evaluation than images.
- Three client instances are needed: the generator brain, the evaluator brain, and a separate
  vision client for the tool (the tool calls ``reset()``, which would clobber an agent brain).

Usage:
    python scripts/hotdog_evaluator.py
    python scripts/hotdog_evaluator.py --image-model hf:stabilityai/stable-diffusion-xl-base-1.0 --eval-model ollama:gemma4:26b
    python scripts/hotdog_evaluator.py --output-dir /tmp/hotdog --max-iterations 5
"""

from __future__ import annotations

import sys
from pathlib import Path

import aimu
from aimu.agents import Agent, EvaluatorOptimizer
from aimu.tools.decorator import tool

sys.path.insert(0, str(Path(__file__).parent))
from _hotdog_common import (
    build_arg_parser,
    build_image_prompt,
    collage_generated_images,
    evaluate_image,
    negative_prompt_plan,
    prompt_word_budget,
    resolve_image_model,
    resolve_output_dir,
    write_summary,
)

GENERATOR_SYSTEM_PROMPT = """\
You generate images of a single hotdog, aiming to make it look as visually HOT as possible.

When given a task or refinement feedback:
1. Distil it into a short image-generation prompt{budget_clause}.
2. Call generate_hotdog_image with that prompt.
3. Reply with ONLY the exact file path the tool returned — no other text.

The scene must always depict exactly ONE single hotdog — never multiple.
"""

EVALUATOR_SYSTEM_PROMPT = """\
You judge how visually hot a generated hotdog image is, using your evaluate_hotness tool.

The message you receive contains a file path to the latest hotdog image.
1. Call evaluate_hotness with that path.
2. Reply with the tool's output copied EXACTLY — do not summarize, rephrase, or add commentary.

The tool replies either "DONE: <reasoning>" (the hotdog cannot get hotter) or
"CONTINUE: <description>" (how to make it hotter). Relay whichever it returns verbatim.
"""

INITIAL_TASK = (
    "Generate an image of a single hotdog and make it look as hot as possible. "
    "For this first attempt, use the prompt: a hot hotdog."
)


def make_tools(image_client, vision_client, output_dir: Path, records: list[dict], neg_plan) -> tuple:
    """Return (generate_hotdog_image, evaluate_hotness) tools sharing a live ``records`` log.

    ``records`` accumulates one dict per generated image; ``evaluate_hotness`` fills in the
    evaluation on the matching record. Building the trace live (rather than reconstructing it
    from merged agent message histories) keeps partial results available on interrupt.
    """

    @tool
    def generate_hotdog_image(prompt: str) -> str:
        """Generate a hotdog image from a short text prompt and save it locally. Returns the saved file path."""
        i = len(records) + 1
        image_prompt = build_image_prompt(prompt) + neg_plan.prompt_suffix
        raw_path = image_client.generate(
            image_prompt, format="path", output_dir=output_dir, **neg_plan.generate_kwargs
        )
        dest = output_dir / f"{i:02d}.png"
        Path(raw_path).rename(dest)
        records.append(
            {
                "iteration": i,
                "prompt": prompt,
                "image_prompt": image_prompt,
                "image_path": str(dest),
                "evaluator_response": None,
                "score": None,
                "action": None,
                "next_prompt": None,
            }
        )
        print(f"[Iteration {i}] Image saved: {dest}")
        return str(dest)

    @tool
    def evaluate_hotness(image_path: str) -> str:
        """Evaluate how hot a hotdog image is. Returns DONE, or CONTINUE with a detailed description."""
        # The path arrives via a text relay through two LLMs, so it may be mangled. Fall back
        # to the most recent generation if it doesn't resolve, rather than failing the run.
        path = image_path if Path(image_path).exists() else (records[-1]["image_path"] if records else image_path)
        response, parsed = evaluate_image(vision_client, path)
        for record in reversed(records):
            if record["image_path"] == str(path):
                record["evaluator_response"] = response
                record["score"] = parsed["score"]
                record["action"] = parsed["action"]
                record["next_prompt"] = parsed["next_prompt"]
                break
        print(f"[Evaluator] {response}\n")
        return response

    return generate_hotdog_image, evaluate_hotness


def main() -> None:
    args = build_arg_parser("Iteratively heat a hotdog image using AIMU's EvaluatorOptimizer workflow.").parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    image_client = aimu.image_client(resolve_image_model(args.image_model))
    # Three separate client instances: generator brain, evaluator brain, and the vision
    # client used inside the tool (it calls reset(), which would clobber an agent's brain).
    gen_client = aimu.client(args.eval_model)
    critic_client = aimu.client(args.eval_model)
    vision_client = aimu.client(args.eval_model)
    if not vision_client.is_vision_model:
        raise ValueError(f"Eval model {args.eval_model!r} does not support vision.")

    # Token-budget fitting folds into the generator's prompt-writing: cap its word count for
    # models with a prompt budget (CLIP/T5); uncapped cloud models get no constraint.
    max_prompt_tokens = image_client.max_prompt_tokens
    if max_prompt_tokens:
        word_budget = prompt_word_budget(max_prompt_tokens)
        budget_clause = (
            f" of at most {word_budget} words, written as comma-separated visual descriptors "
            "with the most important 'hot' details first"
        )
        budget_label = f"{max_prompt_tokens} tokens (≈{word_budget} words)"
    else:
        budget_clause = ""
        budget_label = "uncapped"
    print(f"Image prompt budget: {budget_label}\n")

    # No condensation step here (the generator LLM writes within budget itself), so the plan
    # expresses avoidance as a prompt suffix for prose models rather than via a summarizer.
    neg_plan = negative_prompt_plan(image_client, has_summarizer=False)

    records: list[dict] = []
    generate_fn, evaluate_fn = make_tools(image_client, vision_client, output_dir, records, neg_plan)

    generator = Agent(
        gen_client,
        name="hotdog-generator",
        system_message=GENERATOR_SYSTEM_PROMPT.format(budget_clause=budget_clause),
        tools=[generate_fn],
    )
    evaluator = Agent(
        critic_client,
        name="hotdog-evaluator",
        system_message=EVALUATOR_SYSTEM_PROMPT,
        tools=[evaluate_fn],
        # Judge each image independently — don't let prior verdicts bias the next.
        reset_messages_on_run=True,
    )

    # max_iterations == 0 means run until the evaluator says DONE; EvaluatorOptimizer needs a
    # finite bound, so use a very high cap as the safety net (range() is lazy, so this is cheap).
    max_rounds = 10**6 if args.max_iterations == 0 else args.max_iterations
    eo = EvaluatorOptimizer(
        generator=generator,
        evaluator=evaluator,
        name="hotdog-evaluator-optimizer",
        max_rounds=max_rounds,
        pass_keyword="DONE",
    )

    print("Starting hotdog heating experiment...\n")
    try:
        result = eo.run(INITIAL_TASK)
        print(f"\nWorkflow final response:\n{result}")
    except KeyboardInterrupt:
        print("\nInterrupted — writing partial results so far...")
    finally:
        # Only evaluated records have a verdict; the final generation may be unevaluated
        # (an EvaluatorOptimizer characteristic). The collage scans files, so it includes all.
        trace = [record for record in records if record["evaluator_response"] is not None]
        if trace:
            summary_path = write_summary(
                output_dir,
                trace,
                image_model=image_client.spec.id,
                eval_model=args.eval_model,
                summarizer_instruction=None,
                neg_plan=neg_plan,
            )
            print(f"\nSummary written to: {summary_path}")
        else:
            print("\nNo iterations recorded.")
        collage_path = collage_generated_images(output_dir, trace)
        if collage_path:
            print(f"Collage written to: {collage_path}")


if __name__ == "__main__":
    main()
