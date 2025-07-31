from fastmcp import FastMCP

mcp = FastMCP("AIMU Tools")


@mcp.tool()
def echo(echo_string: str) -> str:
    """Returns echo_string."""
    return echo_string

@mcp.tool()
def get_current_data_and_time() -> str:
    """Returns the current date and time."""
    from datetime import datetime
    return datetime.now().isoformat()

if __name__ == "__main__":
    mcp.run()
