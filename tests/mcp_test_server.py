from fastmcp import FastMCP

mcp = FastMCP("AIMU Tools")

@mcp.tool()
def echo(echo_string: str) -> str:
    """Returns echo_string."""
    return echo_string


if __name__ == "__main__":
    mcp.run()