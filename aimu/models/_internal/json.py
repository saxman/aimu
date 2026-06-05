"""Utilities for extracting structured data from model responses."""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any


def parse_json_response(text: str, schema: type | None = None) -> dict | Any:
    """Parse JSON from an LLM response string.

    Tries three extraction strategies in order:
    1. Raw ``json.loads`` on the stripped text.
    2. Content of the first fenced code block (`` ```json `` or `` ``` ``).
    3. Substring from the first ``{`` to the last ``}``.

    If ``schema`` is provided (a dataclass class or Pydantic v2 ``BaseModel``
    subclass), the parsed dict is coerced into that type before returning.

    Raises ``ValueError`` if no strategy succeeds.
    """
    parsed: dict | None = None

    # Strategy 1: raw parse
    try:
        parsed = json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: fenced code block
    if parsed is None:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            try:
                parsed = json.loads(match.group(1).strip())
            except (json.JSONDecodeError, ValueError):
                pass

    # Strategy 3: {…} substring
    if parsed is None:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
            except (json.JSONDecodeError, ValueError):
                pass

    if parsed is None:
        raise ValueError(f"Could not parse JSON from response: {text[:200]!r}")

    if schema is None:
        return parsed

    if hasattr(schema, "model_validate"):
        # Pydantic v2
        return schema.model_validate(parsed)

    if dataclasses.is_dataclass(schema) and isinstance(schema, type):
        return schema(**parsed)

    raise TypeError(f"schema must be a dataclass or Pydantic BaseModel, got {type(schema).__name__}")


def generate_json(
    client: Any,
    prompt: str,
    schema: type | None = None,
    *,
    retries: int = 2,
    generate_kwargs: dict | None = None,
) -> dict | Any:
    """Call ``client.generate()`` and parse the result as JSON, retrying on failure.

    ``schema`` is forwarded to :func:`parse_json_response`.

    Raises ``ValueError`` after ``retries + 1`` failed attempts, chaining the
    last parse error.
    """
    last_error: Exception | None = None
    for _ in range(retries + 1):
        text = client.generate(prompt, generate_kwargs)
        try:
            return parse_json_response(text, schema)
        except ValueError as exc:
            last_error = exc
    raise ValueError(f"Failed to parse JSON after {retries + 1} attempt(s)") from last_error


def extract_tool_calls(messages: list[dict]) -> list[dict]:
    """Extract tool call/result pairs from an OpenAI-format message list.

    Returns a list of dicts with keys:

    - ``iteration`` — int, incremented each time an assistant turn contains
      tool calls.
    - ``tool`` — the function name.
    - ``arguments`` — the parsed argument dict (handles both ``arguments``
      and ``parameters`` key names).
    - ``result`` — the matching tool-role message content, or ``""`` if none.

    Does not mutate ``messages``.

    Example::

        from aimu import extract_tool_calls

        agent.run("Begin experiment")
        calls = extract_tool_calls(agent.messages["my-agent"])
        # [{"iteration": 1, "tool": "search", "arguments": {...}, "result": "..."}]
    """
    # Pre-build a lookup from tool_call_id → content
    tool_results: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") == "tool":
            tc_id = msg.get("tool_call_id", "")
            tool_results[tc_id] = msg.get("content", "")

    results: list[dict] = []
    iteration = 0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            continue
        iteration += 1
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            # Some models use "parameters" instead of "arguments"
            args_raw = fn.get("arguments") or fn.get("parameters") or "{}"
            if isinstance(args_raw, str):
                try:
                    arguments = json.loads(args_raw)
                except (json.JSONDecodeError, ValueError):
                    arguments = {"_raw": args_raw}
            else:
                arguments = args_raw
            tc_id = tc.get("id", "")
            results.append(
                {
                    "iteration": iteration,
                    "tool": name,
                    "arguments": arguments,
                    "result": tool_results.get(tc_id, ""),
                }
            )
    return results
