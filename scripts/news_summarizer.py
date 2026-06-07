"""News summarizer — demonstrates several AIMU agent/workflow patterns.

Every method answers the same request ("summarize recent AI technical news") but
wires the building blocks together differently, so the script doubles as a tour of
AIMU's autonomous-agent and code-controlled-workflow surfaces:

    agent         One autonomous Agent with web tools — the LLM drives the whole loop
                  (search -> read -> summarize) by itself.
    chain         Prompt-chaining workflow: research -> summarize -> format, where each
                  step's output feeds the next.
    parallel      Parallelization: one worker per news topic runs concurrently, then an
                  aggregator merges their digests into one.
    orchestrator  Orchestrator-workers: an orchestrator LLM treats a "searcher" agent and
                  a "summarizer" agent as tools and decides how to combine them.

Usage::

    python scripts/news_summarizer.py --method agent
    python scripts/news_summarizer.py --method chain --model ollama:qwen3.5:9b
    python scripts/news_summarizer.py --method parallel --output-dir ./my-run
    python scripts/news_summarizer.py --list

Each run writes a Markdown ``summary.md`` and a ``trace.json`` into a timestamped folder
under ``output/news/<timestamp>/`` (like the hotdog and epic scripts), unless overridden
with ``--output-dir`` / ``--trace``.

Web search relies on a reachable SearXNG instance (``SEARXNG_BASE_URL``); the model must
support tool calling.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import aimu
from aimu.agents import Agent, Chain, OrchestratorAgent, Parallel
from aimu.tools import builtin

DEFAULT_MODEL = "ollama:qwen3.5:9b"


@aimu.tool
def search_recent_news(query: str, num_results: int = 8) -> str:
    """Search recent news articles for a query and return the top results.

    Restricted to news engines so results are news-grade and each one includes a
    "Published:" date when the engine reports it — use those dates to confirm an article
    is genuinely from the last day. Keep queries short (2-4 words); news engines match
    headlines, so long keyword-stuffed queries return nothing.

    Args:
        query: The search query string.
        num_results: Number of results to return (default 8).
    """
    # Prefer the freshest results (last day), but a day filter combined with a specific
    # query often returns nothing on news engines — fall back to recent news without the
    # hard 24h cutoff and let the caller filter by each result's "Published:" date.
    results = builtin.web_search(query, num_results=num_results, time_range="day", categories="news")
    if results.strip() == "No results found.":
        results = builtin.web_search(query, num_results=num_results, categories="news")
    return results


# Tools every news-gathering agent gets: a news-restricted search, page fetch (which
# prepends a "Published:" line from article metadata), the clock, and a Python sandbox
# so the model can compute the 24-hour cutoff deterministically instead of guessing.
TOOLS = [search_recent_news, builtin.get_webpage, builtin.get_current_date_and_time, builtin.execute_python]

# Cap output length per turn so a long multi-article digest isn't truncated mid-stream.
# 8192 leaves headroom for a thinking model's reasoning plus a ~10-article summary.
# max_tokens is provider-agnostic (the Ollama client maps it to num_predict).
GENERATE_KWARGS = {"max_tokens": 8192}

# Tool-using agents need many loop iterations to search, fetch, and verify ~10 articles
# before synthesizing — each search/fetch is one iteration. Generous so the agent isn't
# cut off mid-gather.
MAX_ITERATIONS = 100

# If a tool-using agent still hits the iteration cap, this forces one final tools-disabled
# turn so we always get a summary instead of an empty result (see Agent.final_answer_prompt).
FINAL_ANSWER_PROMPT = (
    "You have reached the limit on tool use. Do not call any more tools. Using only the "
    "articles you have already gathered and confirmed, write the final news summary now."
)

SYSTEM = (
    "You are a news summarizer specializing in AI technical developments. "
    "Summarize concisely and always include a publication date and source link for every article. "
    "Use search_recent_news with SHORT queries (2-4 words, e.g. 'AI model release' or "
    "'OpenAI') to find fresh, news-grade articles — long keyword-stuffed queries return "
    "nothing. Read each result's 'Published:' date. If a search result has no date, fetch the article with get_webpage to read "
    "its 'Published:' metadata. Determine the current time with get_current_date_and_time, and use "
    "execute_python with the datetime module to check whether each publication time is within the "
    "last 24 hours rather than estimating. Prefer primary news sources over social media posts. "
    "Continue searching until you find at least 10 articles published in the last 24 hours. "
    "Only include articles whose publication date you have confirmed is within the last 24 hours; "
    "if you cannot confirm an absolute date, leave the article out."
)

TASK = (
    "Find news about AI technical developments published in the last 24 hours. "
    "Summarize each article in a few sentences and provide its source link. "
    "Only include articles you can confirm were published no more than 24 hours ago."
)

# Appended to the system message of whichever agent produces the FINAL output, so the
# model writes in the requested format directly (no post-processing). Keyed by --format.
FORMAT_DIRECTIVE = {
    "text": (
        " Write your final answer as plain text only: no Markdown — no '#' headings, no "
        "'*'/'-' bullets or '**' emphasis, no backticks, and no '[label](url)' links. "
        "Write each source URL inline as plain text."
    ),
    "markdown": " Write your final answer as clean, well-structured Markdown.",
}


# --------------------------------------------------------------------------------------
# Progress / output helpers
# --------------------------------------------------------------------------------------


def _fmt_args(args: object) -> str:
    """Render tool-call arguments compactly for a progress line."""
    if not isinstance(args, dict):
        return ""
    parts = []
    for key, value in args.items():
        text = str(value).replace("\n", " ")
        if len(text) > 100:
            text = text[:97] + "..."
        parts.append(f'{key}="{text}"')
    return ", ".join(parts)


def run_with_progress(runner, task: str, *, combine: str = "last_step", generate_kwargs: dict | None = None) -> str:
    """Stream a Runner, print simple live progress, and return its final text.

    Progress is intentionally minimal: one line per tool call, and a quiet note the first
    time the model starts thinking. THINKING chunks are never captured into the result
    (``StreamChunk.is_text()`` is True for both THINKING and GENERATING, so we match the
    phase explicitly).

    ``combine`` picks how the final answer is extracted, because ``iteration`` means
    different things across patterns:

    - ``"after_tools"`` (agents): return everything generated *after the last tool call*.
      An :class:`Agent` runs one extra continuation turn once it has produced its answer
      (its stop condition sees the earlier tool calls), and that trailing turn can be a
      short restatement. Clearing the buffer on each tool call keeps the full final answer
      and drops pre-tool commentary.
    - ``"last_step"`` (chains/workflows): return the highest-``iteration`` block, i.e. the
      last step's output. A :class:`Chain`'s earlier steps legitimately emit text that is
      not the final answer, so we must not concatenate across steps.
    """
    texts: dict[int, list[str]] = {}
    after_tools: list[str] = []
    thinking_noted = False
    for chunk in runner.run(task, generate_kwargs=generate_kwargs, stream=True):
        if chunk.is_tool_call():
            after_tools.clear()  # text before a tool call is intermediate; the answer comes after
            name = chunk.content.get("name", "?")
            print(f"  · {name}({_fmt_args(chunk.content.get('arguments'))})", file=sys.stderr)
        elif chunk.phase == aimu.StreamingContentType.THINKING:
            if not thinking_noted:
                _note("  · thinking...")
                thinking_noted = True
        elif chunk.phase == aimu.StreamingContentType.GENERATING:
            texts.setdefault(chunk.iteration, []).append(chunk.content)
            after_tools.append(chunk.content)

    if combine == "after_tools" and after_tools:
        return "".join(after_tools)
    if not texts:
        return ""
    return "".join(texts[max(texts)])


def _note(message: str) -> None:
    print(message, file=sys.stderr)


def resolve_output_dir(output_dir: str | None) -> Path:
    """Resolve the run's output directory, defaulting to output/news/<timestamp>/.

    Mirrors the convention used by the hotdog and epic scripts so every run lands in a
    self-contained, timestamped folder under the shared output directory.
    """
    if output_dir:
        return Path(output_dir)
    from aimu import paths as aimu_paths

    return aimu_paths.output / "news" / datetime.now().strftime("%Y%m%d-%H%M%S")


# --------------------------------------------------------------------------------------
# Method builders — each returns a Runner ready to summarize `TASK`
# --------------------------------------------------------------------------------------


def build_agent(model: str, fmt: str) -> Agent:
    """Autonomous Agent: the LLM searches, reads, and summarizes in one tool-calling loop."""
    _note("Single autonomous agent with web tools.")
    return Agent(
        aimu.client(model),
        system_message=SYSTEM + FORMAT_DIRECTIVE[fmt],
        name="news-agent",
        tools=TOOLS,
        max_iterations=MAX_ITERATIONS,
        final_answer_prompt=FINAL_ANSWER_PROMPT,
    )


def build_chain(model: str, fmt: str) -> Chain:
    """Prompt-chaining workflow: research -> summarize -> format.

    The research step has web tools; the downstream steps are pure text transforms.
    Each step owns its own client so swapping tools/system prompts can't collide.
    """
    _note("Prompt chain: research -> summarize -> format.")
    research = Agent(
        aimu.client(model),
        system_message=(
            "Search the web for AI technical news published in the last 24 hours. "
            "For each article capture the title, source URL, publication time, and the key "
            "facts. Return a raw bulleted list — do not polish it yet."
        ),
        name="researcher",
        tools=TOOLS,
        max_iterations=MAX_ITERATIONS,
    )
    summarize = Agent(
        aimu.client(model),
        system_message=(
            "You are given raw notes about AI news articles. Write a concise 2-3 sentence "
            "summary for each, preserving the source URL. Drop anything older than 24 hours."
        ),
        name="summarizer",
    )
    formatter = Agent(
        aimu.client(model),
        system_message=(
            "Format the provided summaries into a clean, dated digest. Use a short headline "
            "and a source link per item. No preamble, just the digest." + FORMAT_DIRECTIVE[fmt]
        ),
        name="formatter",
    )
    return Chain(agents=[research, summarize, formatter], name="news-chain")


def build_parallel(model: str, fmt: str) -> Parallel:
    """Parallelization: one worker per topic runs concurrently, an aggregator merges them."""
    topics = [
        "large language models and foundation models",
        "AI chips, hardware, and accelerators",
        "AI research papers and new techniques",
    ]
    _note(f"Parallel workers, one per topic ({len(topics)} concurrent):")
    for topic in topics:
        _note(f"    - {topic}")

    workers = [
        Agent(
            aimu.client(model),
            system_message=(
                f"You research recent news (last 24 hours) specifically about {topic}. "
                "Search the web, summarize each finding concisely, and include source links. "
                "Only include articles confirmed to be from the last 24 hours."
            ),
            name=f"worker-{i}",
            tools=TOOLS,
            max_iterations=MAX_ITERATIONS,
        )
        for i, topic in enumerate(topics)
    ]
    aggregator = Agent(
        aimu.client(model),
        system_message=(
            "You are given several topic-specific AI news digests. Merge them into one "
            "deduplicated summary grouped by theme, preserving every source link. "
            "No preamble, just the combined digest." + FORMAT_DIRECTIVE[fmt]
        ),
        name="aggregator",
    )
    return Parallel(workers=workers, aggregator=aggregator, name="news-parallel")


def build_orchestrator(model: str, fmt: str) -> OrchestratorAgent:
    """Orchestrator-workers: the orchestrator LLM calls worker agents as tools."""
    _note("Orchestrator dispatching to 'searcher' and 'summarizer' worker agents.")
    searcher = Agent(
        aimu.client(model),
        system_message=(
            "Search the web for AI technical news published in the last 24 hours and return "
            "raw findings: title, source URL, publication time, and key facts."
        ),
        name="searcher",
        tools=TOOLS,
        max_iterations=MAX_ITERATIONS,
    )
    summarizer = Agent(
        aimu.client(model),
        system_message=(
            "Condense the provided article notes into concise summaries, preserving every "
            "source link. Drop anything older than 24 hours."
        ),
        name="summarizer",
    )
    return OrchestratorAgent.assemble(
        aimu.client(model),
        system_message=(
            "You coordinate AI-news summarization. First call 'searcher' to gather recent "
            "articles, then call 'summarizer' to condense them. Produce a final dated digest "
            "with a source link for every item. Only include articles from the last 24 hours." + FORMAT_DIRECTIVE[fmt]
        ),
        workers=[searcher, summarizer],
        name="news-orchestrator",
        final_answer_prompt=FINAL_ANSWER_PROMPT,
    )


METHODS = {
    "agent": build_agent,
    "chain": build_chain,
    "parallel": build_parallel,
    "orchestrator": build_orchestrator,
}

# How to extract the final answer from each method's stream (see run_with_progress).
# Autonomous agents run a trailing continuation turn, so we keep everything after the last
# tool call; workflows carry their result in the last step.
RESULT_COMBINE = {
    "agent": "after_tools",
    "orchestrator": "after_tools",
    "chain": "last_step",
    "parallel": "last_step",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize recent AI technical news using different AIMU patterns.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--method",
        choices=sorted(METHODS),
        default="agent",
        help="Which agent/workflow pattern to use (default: agent).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model string, e.g. 'anthropic:claude-sonnet-4-6' (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--format",
        choices=["text", "markdown"],
        default="markdown",
        help="Output format the model is asked to write in: plain text or Markdown (default: markdown).",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default=None,
        help="Directory for the run's artifacts (default: output/news/<timestamp>/).",
    )
    parser.add_argument(
        "--trace",
        metavar="FILE",
        help="Write the message trace to FILE instead of <output-dir>/trace.json.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the available methods and exit.",
    )
    args = parser.parse_args()

    if args.list:
        print("Available methods:\n")
        for name in sorted(METHODS):
            doc = (METHODS[name].__doc__ or "").strip().splitlines()[0]
            print(f"  {name:13} {doc}")
        return

    _note(f"Method: {args.method}  |  Model: {args.model}")
    runner = METHODS[args.method](args.model, args.format)
    _note("Working...\n")

    result = run_with_progress(runner, TASK, combine=RESULT_COMBINE[args.method], generate_kwargs=GENERATE_KWARGS)

    if not result.strip():
        _note(
            "\nWarning: the run produced no final summary — the agent likely exhausted its "
            "iterations on tool calls without writing an answer (inspect trace.json below). "
            "Try a different --method or --model."
        )

    # Each run lands in a self-contained, timestamped folder under output/news/ (like the
    # hotdog and epic scripts). The agent was instructed to write in the requested format
    # (see FORMAT_DIRECTIVE), so the result is already plain text or Markdown.
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / ("summary.md" if args.format == "markdown" else "summary.txt")
    summary_path.write_text(result, encoding="utf-8")
    _note(f"\nWrote {args.format} summary to {summary_path}")

    # runner.messages is a {agent_name: [messages]} dict in OpenAI format, merged across
    # every agent/step in the pattern (default=str guards any stray objects).
    trace_path = Path(args.trace) if args.trace else output_dir / "trace.json"
    trace_path.write_text(json.dumps(runner.messages, indent=2, default=str), encoding="utf-8")
    _note(f"Wrote message trace to {trace_path}")


if __name__ == "__main__":
    main()
