import datetime
import os
import re
from html.parser import HTMLParser
from pathlib import Path

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


_WMO_DESCRIPTIONS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight showers",
    81: "Moderate showers",
    82: "Violent showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


@mcp.tool()
def get_weather(location: str) -> str:
    """Returns the current weather for a given location.

    Args:
        location: City name or coordinates (e.g. "London", "48.8566,2.3522").
    """
    try:
        # Resolve location name to coordinates via Open-Meteo geocoding
        coord_match = re.match(r"^(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)$", location)
        if coord_match:
            lat, lon = float(coord_match.group(1)), float(coord_match.group(2))
            city, country = location, ""
        else:
            geo = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location, "count": 1},
                timeout=10,
            )
            geo.raise_for_status()
            results = geo.json().get("results")
            if not results:
                return f"Location not found: {location}"
            lat = results[0]["latitude"]
            lon = results[0]["longitude"]
            city = results[0]["name"]
            country = results[0].get("country", "")

        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        return f"Error fetching weather: {e}"

    current = data["current"]
    desc = _WMO_DESCRIPTIONS.get(current["weather_code"], f"WMO code {current['weather_code']}")
    temp_c = current["temperature_2m"]
    feels_c = current["apparent_temperature"]
    humidity = current["relative_humidity_2m"]
    wind_kmph = current["wind_speed_10m"]
    location_str = f"{city}, {country}".strip(", ")
    return (
        f"{desc}, {temp_c}°C (feels like {feels_c}°C) in {location_str}. Humidity: {humidity}%, Wind: {wind_kmph} km/h."
    )


@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluates a simple arithmetic expression and returns the result."""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"


class _TextExtractor(HTMLParser):
    """Strips HTML tags and decodes entities, collecting visible text."""

    SKIP_TAGS = {"script", "style", "head", "meta", "link", "noscript"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):  # noqa: ARG002
        if tag in self.SKIP_TAGS:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        text = " ".join(self._parts)
        return re.sub(r"\s+", " ", text).strip()


@mcp.tool()
def get_webpage(url: str) -> str:
    """Fetches a web page and returns its visible text content with HTML stripped.

    Args:
        url: The URL of the page to retrieve.
    """
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; aimu-tools/1.0)"},
            timeout=15,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        return f"Error fetching page: {e}"

    extractor = _TextExtractor()
    extractor.feed(response.text)
    return extractor.get_text()


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


@mcp.tool()
def list_directory(path: str) -> str:
    """Lists files and subdirectories at the given path.

    Args:
        path: Directory path to list.
    """
    p = Path(path)
    if not p.exists():
        return f"Path does not exist: {path}"
    if not p.is_dir():
        return f"Not a directory: {path}"
    entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
    lines = [f"{e.name}/" if e.is_dir() else e.name for e in entries]
    return "\n".join(lines) if lines else "(empty)"


@mcp.tool()
def read_file(path: str, max_lines: int = 200) -> str:
    """Reads a local file and returns its contents, capped at max_lines lines.

    Args:
        path: Path to the file to read.
        max_lines: Maximum number of lines to return (default 200).
    """
    p = Path(path)
    if not p.exists():
        return f"File does not exist: {path}"
    if not p.is_file():
        return f"Not a file: {path}"
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        return f"Error reading file: {e}"
    truncated = len(lines) > max_lines
    content = "\n".join(lines[:max_lines])
    if truncated:
        content += f"\n... (truncated at {max_lines} lines)"
    return content


if __name__ == "__main__":
    mcp.run()
