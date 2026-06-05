#!/usr/bin/env python3
"""Iteratively make a sentence epic using AIMU's EvaluatorOptimizer workflow.

The same generate → judge → refine loop as ``epic_loop.py`` and ``epic_agent.py``, but
expressed with the library's **EvaluatorOptimizer** workflow class instead of a hand-rolled
loop or a single tool-calling agent. It composes two ``Agent``s:

- the **generator** holds ``write_epic_sentence`` — it turns the task (or the judge's CONTINUE
  directive) into a rewrite and replies with the sentence.
- the **judge** holds ``judge_epicness`` — it scores that sentence and relays the verdict
  (``DONE`` or a ``CONTINUE`` directive) back verbatim.

``EvaluatorOptimizer`` stops when ``pass_keyword="DONE"`` appears in the judge's reply or
``max_rounds`` is hit, then returns the last generation.

Unlike hotdog_evaluator.py, the artifact relayed between the two agents *is* plain text (the
sentence itself), so the text relay is far less fragile than relaying an image file path — this
is the modality where the EvaluatorOptimizer shape fits most naturally. Two caveats of the shape
still apply, both visible in the output:
- The *final* generation is returned without being judged (the loop generates after the last
  judging, then stops), so the summary may have one fewer judgement than sentences.
- Three client instances are used: the generator brain, the judge brain, and a separate judge
  client for the tool (the tool is stateless ``generate()``, so a dedicated client keeps it from
  sharing turn history with the judge agent's brain).

Usage:
    python scripts/epic_evaluator.py
    python scripts/epic_evaluator.py --gen-model ollama:qwen3:8b --judge-model ollama:qwen3:8b
    python scripts/epic_evaluator.py --seed-sentence "She parks the car." --max-iterations 5
"""

from __future__ import annotations

import sys
from pathlib import Path

import aimu
from aimu.agents import Agent, EvaluatorOptimizer
from aimu.tools.decorator import tool

sys.path.insert(0, str(Path(__file__).parent))
from _epic_common import (
    build_arg_parser,
    evaluate_sentence,
    generate_sentence,
    resolve_output_dir,
    write_summary,
)

GENERATOR_SYSTEM_PROMPT = """\
You rewrite one mundane sentence to be as EPIC as possible, keeping it ONE grammatical sentence
about the same event. The sentence to escalate is: {seed}

When given the task or a refinement directive:
1. Call write_epic_sentence with that directive.
2. Reply with ONLY the exact sentence the tool returned — no other text.
"""

JUDGE_SYSTEM_PROMPT = """\
You judge how epic a rewritten sentence is, using your judge_epicness tool.

1. Call judge_epicness with the sentence in the message you received.
2. Reply with the tool's output copied EXACTLY — do not summarize, rephrase, or add commentary.

The tool replies either "DONE: <reasoning>" (the sentence cannot get more epic) or
"CONTINUE: <directive>" (how to make it more epic). Relay whichever it returns verbatim.
"""

INITIAL_TASK = (
    "Make the seed sentence as epic as possible. For this first attempt, use the directive: "
    "give it a grand, cinematic, larger-than-life register."
)


def make_tools(gen_client, judge_client, seed_sentence: str, records: list[dict]) -> tuple:
    """Return (write_epic_sentence, judge_epicness) tools sharing a live ``records`` log.

    ``records`` accumulates one dict per generated sentence; ``judge_epicness`` fills in the
    judgement on the matching record. Building the trace live (rather than reconstructing it
    from merged agent histories) keeps partial results available on interrupt.
    """

    @tool
    def write_epic_sentence(directive: str) -> str:
        """Rewrite the seed sentence more epically following the directive. Returns the new sentence."""
        i = len(records) + 1
        sentence = generate_sentence(gen_client, seed_sentence, directive)
        records.append(
            {
                "iteration": i,
                "direction": directive,
                "sentence": sentence,
                "judge_response": None,
                "score": None,
                "action": None,
                "next_direction": None,
            }
        )
        print(f"[Iteration {i}] Sentence: {sentence}")
        return sentence

    @tool
    def judge_epicness(sentence: str) -> str:
        """Judge how epic a sentence is. Returns DONE, or CONTINUE with a rewrite directive."""
        # The sentence is relayed through two LLMs; fall back to the latest generation if it
        # arrives empty, rather than failing the run.
        text = sentence if sentence.strip() else (records[-1]["sentence"] if records else sentence)
        response, parsed = evaluate_sentence(judge_client, text)
        for record in reversed(records):
            if record["sentence"] == text:
                record["judge_response"] = response
                record["score"] = parsed["score"]
                record["action"] = parsed["action"]
                record["next_direction"] = parsed["next_direction"]
                break
        print(f"[Judge] {response}\n")
        return response

    return write_epic_sentence, judge_epicness


def main() -> None:
    args = build_arg_parser("Make a sentence epic with AIMU's EvaluatorOptimizer workflow.").parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Seed sentence: {args.seed_sentence}\n")

    # Three separate client instances: generator brain, judge brain, and the judge client used
    # inside the tool (stateless generate(); a dedicated client avoids sharing the agent brain's
    # turn history).
    gen_client = aimu.client(args.gen_model)
    critic_client = aimu.client(args.judge_model)
    judge_client = aimu.client(args.judge_model)

    records: list[dict] = []
    write_fn, judge_fn = make_tools(gen_client, judge_client, args.seed_sentence, records)

    generator = Agent(
        gen_client,
        name="epic-generator",
        system_message=GENERATOR_SYSTEM_PROMPT.format(seed=args.seed_sentence),
        tools=[write_fn],
    )
    evaluator = Agent(
        critic_client,
        name="epic-judge",
        system_message=JUDGE_SYSTEM_PROMPT,
        tools=[judge_fn],
        # Judge each sentence independently — don't let prior verdicts bias the next.
        reset_messages_on_run=True,
    )

    # max_iterations == 0 means run until the judge says DONE; EvaluatorOptimizer needs a finite
    # bound, so use a very high cap as the safety net (range() is lazy, so this is cheap).
    max_rounds = 10**6 if args.max_iterations == 0 else args.max_iterations
    eo = EvaluatorOptimizer(
        generator=generator,
        evaluator=evaluator,
        name="epic-evaluator-optimizer",
        max_rounds=max_rounds,
        pass_keyword="DONE",
    )

    print("Starting epic-sentence experiment...\n")
    try:
        result = eo.run(INITIAL_TASK)
        print(f"\nWorkflow final response:\n{result}")
    except KeyboardInterrupt:
        print("\nInterrupted — writing partial results so far...")
    finally:
        # Only judged records have a verdict; the final generation may be unjudged (an
        # EvaluatorOptimizer characteristic).
        trace = [record for record in records if record["judge_response"] is not None]
        if trace:
            summary_path = write_summary(
                output_dir,
                trace,
                seed_sentence=args.seed_sentence,
                gen_model=args.gen_model,
                judge_model=args.judge_model,
            )
            print(f"\nSummary written to: {summary_path}")
        else:
            print("\nNo iterations recorded.")


if __name__ == "__main__":
    main()
