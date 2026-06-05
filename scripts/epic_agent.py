#!/usr/bin/env python3
"""Iteratively rewrite a mundane sentence to be as epic as possible using local models.

An AIMU Agent autonomously controls the loop via tool calls.
Stops when the judge declares the sentence cannot get any more epic.

The text-only twin of ``hotdog_agent.py`` — the diff from ``epic_loop.py`` is purely *who
drives the loop*: here an ``Agent``'s tool-calling loop decides when to write and when to judge.

Usage:
    python scripts/epic_agent.py
    python scripts/epic_agent.py --gen-model ollama:qwen3:8b --judge-model ollama:qwen3:8b
    python scripts/epic_agent.py --seed-sentence "She parks the car." --max-iterations 5
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import aimu
from aimu.agents import Agent
from aimu.tools.decorator import tool

sys.path.insert(0, str(Path(__file__).parent))
from _epic_common import (
    INITIAL_DIRECTION,
    build_arg_parser,
    evaluate_sentence,
    generate_sentence,
    parse_judge_response,
    resolve_output_dir,
    write_summary,
)

AGENT_SYSTEM_PROMPT = """\
You are running an experiment to make one mundane sentence as EPIC as possible.

The sentence to escalate is: {seed}

Procedure:
1. Call write_epic_sentence with your current rewrite directive (start: "{initial_direction}")
2. Call judge_epicness with the sentence the tool returned
3. If the judge outputs DONE, stop and report the final sentence
4. If the judge outputs CONTINUE, it gives a rewrite directive. Use that directive as the next
   write_epic_sentence call and repeat from step 1.
5. {iteration_limit_rule}
"""


def make_tools(gen_client, judge_client, seed_sentence: str) -> tuple:
    """Return (write_epic_sentence, judge_epicness) tools sharing a closure.

    ``gen_client`` writes each rewrite; ``judge_client`` scores it. The latest sentence is held
    in the closure so a mangled relay from the agent never loses the real text.
    """
    state = {"count": 0, "latest": ""}

    @tool
    def write_epic_sentence(directive: str) -> str:
        """Rewrite the seed sentence more epically following the directive. Returns the new sentence."""
        state["count"] += 1
        sentence = generate_sentence(gen_client, seed_sentence, directive)
        state["latest"] = sentence
        print(f"[Iteration {state['count']}] Sentence: {sentence}")
        return sentence

    @tool
    def judge_epicness(sentence: str) -> str:
        """Judge how epic a sentence is. Returns DONE, or CONTINUE with a rewrite directive."""
        # The sentence is relayed through the agent; fall back to the latest generation if it
        # arrives empty rather than judging nothing.
        text = sentence if sentence.strip() else state["latest"]
        response, _ = evaluate_sentence(judge_client, text)
        print(f"[Judge] {response}\n")
        return response

    return write_epic_sentence, judge_epicness


def parse_agent_trace(messages: list[dict]) -> list[dict]:
    """Reconstruct the per-iteration trace from agent message history (OpenAI format).

    The text twin of hotdog_agent's parse_agent_trace: a write_epic_sentence call (directive ->
    sentence) is paired with the judge_epicness call that follows it.
    """
    tool_results: dict[str, str] = {
        msg.get("tool_call_id", ""): msg.get("content", "") for msg in messages if msg.get("role") == "tool"
    }

    trace: list[dict] = []
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

            if name == "write_epic_sentence":
                pending = {"direction": args.get("directive", ""), "sentence": result}
            elif name == "judge_epicness" and pending is not None:
                iteration += 1
                parsed = parse_judge_response(result)
                trace.append(
                    {
                        "iteration": iteration,
                        "direction": pending["direction"],
                        "sentence": pending["sentence"],
                        "judge_response": result,
                        "score": parsed["score"],
                        "action": parsed["action"],
                        "next_direction": parsed["next_direction"],
                    }
                )
                pending = None

    return trace


def main() -> None:
    args = build_arg_parser("Make a sentence epic with an AIMU Agent driving the loop.").parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Seed sentence: {args.seed_sentence}\n")

    # Two separate client instances: one for agent reasoning, one for the judge inside the tool.
    agent_client = aimu.client(args.gen_model)
    judge_client = aimu.client(args.judge_model)
    gen_client = aimu.client(args.gen_model)

    write_fn, judge_fn = make_tools(gen_client, judge_client, args.seed_sentence)

    if args.max_iterations == 0:
        # Run indefinitely: the agent stops on its own once the judge says DONE. Agent has no
        # unbounded sentinel, so use a very high cap as the safety net.
        iteration_limit_rule = "Continue until the judge outputs DONE; there is no fixed iteration limit"
        agent_max_iterations = 10**6
    else:
        iteration_limit_rule = f"Never exceed {args.max_iterations} iterations total"
        # Headroom: two tool calls per iteration plus a decision round.
        agent_max_iterations = args.max_iterations * 3

    agent = Agent(
        agent_client,
        name="epic-agent",
        system_message=AGENT_SYSTEM_PROMPT.format(
            seed=args.seed_sentence,
            initial_direction=INITIAL_DIRECTION,
            iteration_limit_rule=iteration_limit_rule,
        ),
        tools=[write_fn, judge_fn],
        max_iterations=agent_max_iterations,
    )

    print("Starting epic-sentence experiment...\n")
    try:
        result = agent.run(f"Begin. Start with the directive: {INITIAL_DIRECTION}")
        print(f"\nAgent final response:\n{result}")
    except KeyboardInterrupt:
        print("\nInterrupted — writing partial results so far...")
    finally:
        # Reconstruct from the live conversation so an interrupted run still produces output.
        trace = parse_agent_trace(agent.model_client.messages)
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
            print("\nNo iterations recorded in agent messages.")


if __name__ == "__main__":
    main()
