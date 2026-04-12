"""
aimu.prompts.mcp — MCP server for the prompt catalog.

Exposes PromptCatalog operations as MCP tools so agents can fetch and update
their own prompts at runtime.

The catalog DB path is configured with the PROMPT_CATALOG_PATH environment
variable (defaults to ./prompt_catalog.db).

Run as a standalone MCP server::

    python -m aimu.prompts.mcp

Or connect programmatically::

    from aimu.tools.client import MCPClient
    from aimu.prompts.mcp import mcp
    client = MCPClient(server=mcp)
"""

from __future__ import annotations

import os

from fastmcp import FastMCP

from aimu.prompts.catalog import Prompt, PromptCatalog

_DEFAULT_DB_PATH = os.environ.get("PROMPT_CATALOG_PATH", "./prompt_catalog.db")

mcp = FastMCP("AIMU Prompt Catalog")
_catalog = PromptCatalog(_DEFAULT_DB_PATH)


@mcp.tool()
def get_prompt(name: str, model_id: str) -> str:
    """
    Retrieve the latest version of a prompt by name and model.

    Args:
        name:     The task name (e.g. "disease_classifier", "summarizer").
        model_id: The model identifier string (e.g. "llama3.1").

    Returns:
        The prompt text, or an empty string if not found.
    """
    prompt = _catalog.retrieve_last(name, model_id)
    return prompt.prompt if prompt else ""


@mcp.tool()
def list_prompts() -> list[dict]:
    """
    List all stored prompt names and model IDs.

    Returns:
        List of dicts with 'name', 'model_id', 'version', and 'created_at' keys.
    """
    results = _catalog.session.query(Prompt.name, Prompt.model_id, Prompt.version, Prompt.created_at).all()
    seen: set[tuple] = set()
    latest = []
    for name, model_id, version, created_at in results:
        key = (name, model_id)
        if key not in seen:
            seen.add(key)
            latest.append({"name": name, "model_id": model_id, "version": version, "created_at": str(created_at)})
    return latest


@mcp.tool()
def store_prompt_version(name: str, model_id: str, prompt: str, metrics: dict | None = None) -> int:
    """
    Store a new version of a prompt, auto-incrementing the version number.

    Args:
        name:     The task name (e.g. "disease_classifier").
        model_id: The model identifier string.
        prompt:   The prompt text to store.
        metrics:  Optional dict of evaluation metrics (e.g. {"accuracy": 0.94}).

    Returns:
        The version number assigned to this prompt.
    """
    p = Prompt(name=name, model_id=model_id, prompt=prompt, metrics=metrics)
    _catalog.store_prompt(p)
    return p.version


if __name__ == "__main__":
    mcp.run()
