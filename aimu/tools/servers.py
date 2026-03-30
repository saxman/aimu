import datetime

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


if __name__ == "__main__":
    mcp.run()
