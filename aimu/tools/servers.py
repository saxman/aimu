import datetime
import os

import requests
from fastmcp import FastMCP

mcp = FastMCP("AIMU Tools")


@mcp.tool()
def echo(echo_string: str) -> str:
    """Returns echo_string."""
    return echo_string


@mcp.tool()
def get_current_date_and_time() -> str:
    """Returns the current date and time."""
    return str(datetime.datetime.now())


@mcp.tool()
def get_weather(location: str) -> str:
    """Returns the current weather for a given location."""
    # Stubbed for demo purposes
    return f"Sunny, 22°C in {location}"


@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluates a simple arithmetic expression and returns the result."""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"


SEARXNG_BASE_URL = os.environ.get("SEARXNG_BASE_URL", "http://localhost:8080")


@mcp.tool()
def search(query: str, num_results: int = 5) -> str:
    """Search the web using a SearXNG instance and return the top results.

    Args:
        query: The search query string.
        num_results: Number of results to return (default 5).

    Set SEARXNG_BASE_URL env var to point to your SearXNG instance
    (default: http://localhost:8080).
    """
    try:
        response = requests.get(
            f"{SEARXNG_BASE_URL}/search",
            params={"q": query, "format": "json"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        return f"Error contacting SearXNG: {e}"

    results = data.get("results", [])[:num_results]
    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")
        lines.append(f"{i}. {title}\n   {url}\n   {snippet}")
    return "\n\n".join(lines)


if __name__ == "__main__":
    mcp.run()
