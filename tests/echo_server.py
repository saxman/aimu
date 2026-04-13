"""Minimal FastMCP server used by test_mcp_client_with_file."""

from fastmcp import FastMCP

mcp = FastMCP("Echo")


@mcp.tool()
def echo(message: str) -> str:
    """Return the message unchanged."""
    return message


if __name__ == "__main__":
    mcp.run()
