"""
FastMCP server exposing AIMU's built-in tools for cross-process use.

The tool bodies live in [aimu/tools/builtin.py](aimu/tools/builtin.py); this module
just registers them with a FastMCP server. Run standalone with ``python -m aimu.tools.mcp``.

For in-process use, import the same callables directly:

    from aimu.tools import builtin
    agent = Agent(client, tools=[builtin.get_weather, builtin.calculate])
"""

from fastmcp import FastMCP

from . import builtin

mcp = FastMCP("AIMU Tools")

for _fn in builtin.ALL_TOOLS:
    mcp.tool()(_fn)


if __name__ == "__main__":
    mcp.run()
