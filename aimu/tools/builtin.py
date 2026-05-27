"""
Built-in general-purpose tools, decorated with ``@tool`` for direct in-process use.

These same callables are registered on the FastMCP server in [aimu/tools/mcp.py](aimu/tools/mcp.py)
so that the tools are available either in-process (``Agent(client, tools=[get_weather, ...])``)
or cross-process (``python -m aimu.tools.mcp``).
"""

import datetime
import os
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import requests
from dotenv import load_dotenv

from .decorator import tool

load_dotenv()
SEARXNG_BASE_URL = os.environ.get("SEARXNG_BASE_URL", "http://localhost:8080")


@tool
def echo(echo_string: str) -> str:
    """Returns echo_string."""
    return echo_string


@tool
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


@tool
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


@tool
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


@tool
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


@tool
def search(query: str, num_results: int = 5) -> str:
    """Search the web using a SearXNG instance and return the top results.

    Args:
        query: The search query string.
        num_results: Number of results to return (default 5).

    Set SEARXNG_BASE_URL env var to point to your SearXNG instance (or a .env file)
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


@tool
def wikipedia(query: str) -> str:
    """Fetches a Wikipedia article summary for the given query.

    Args:
        query: Article title or search phrase (e.g. "Albert Einstein", "general relativity").
    """
    headers = {"User-Agent": "aimu-tools/1.0"}
    try:
        title = query.strip().replace(" ", "_")
        response = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title, safe='')}",
            headers=headers,
            timeout=10,
        )
        if response.status_code == 404:
            search = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "opensearch", "search": query, "limit": 1, "format": "json"},
                headers=headers,
                timeout=10,
            )
            search.raise_for_status()
            results = search.json()
            if not results[1]:
                return f"No Wikipedia article found for: {query}"
            best_title = results[1][0].replace(" ", "_")
            response = requests.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(best_title, safe='')}",
                headers=headers,
                timeout=10,
            )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        return f"Error fetching Wikipedia: {e}"

    if data.get("type") == "disambiguation":
        return f"'{data.get('title', query)}' is a disambiguation page. Try a more specific query."

    title = data.get("title", query)
    extract = data.get("extract", "")
    url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
    if not extract:
        return f"No summary available for {title}."
    return f"{title}\n\n{extract}\n\n{url}" if url else f"{title}\n\n{extract}"


@tool
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


@tool
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


# ---- Image generation (diffusion) --------------------------------------------
#
# Diffusion deps (``diffusers``, ``torch``, ``Pillow``) are heavy. The client is
# constructed lazily on first ``generate_image()`` call so that importing
# ``aimu.tools.builtin`` does not pull torch into ``sys.modules``.

_AIMU_IMAGE_DEFAULT = "hf:stabilityai/stable-diffusion-xl-base-1.0"
_image_client = None


def _get_image_client():
    """Return the lazy singleton :class:`ImageClient` for the built-in tool.

    Reads ``AIMU_IMAGE_MODEL`` from the environment (default: SDXL base). Accepts
    any model string supported by :func:`aimu.image_client` — ``"hf:..."`` for
    HuggingFace diffusers, ``"gemini:..."`` for Google Nano Banana.
    """
    global _image_client
    if _image_client is None:
        from aimu import image_client as _image_client_factory

        model_str = os.environ.get("AIMU_IMAGE_MODEL", _AIMU_IMAGE_DEFAULT)
        _image_client = _image_client_factory(model_str)
    return _image_client


@tool
def generate_image(prompt: str):
    """Generate an image from a text prompt and return the saved file path.

    This is a **streaming tool** — a generator that yields
    :attr:`~aimu.models.StreamingContentType.IMAGE_GENERATING` chunks during
    denoising, then returns the saved file path. When called via the agent's
    streaming path (``agent.run(stream=True)``), each step chunk flows through
    the agent's own stream and into the chat UI live.

    Uses an :class:`aimu.ImageClient`. The default model is controlled by the
    ``AIMU_IMAGE_MODEL`` env var (default: SDXL base; pass any HF diffusers repo
    via ``"hf:..."`` or ``"gemini:nano-banana"`` for Google Nano Banana).
    Override per-agent by constructing your own tool with :func:`make_image_tool`
    — it supports a ``preview_every=N`` kwarg for intermediate denoised-image
    previews.

    Args:
        prompt: A description of the desired image.
    """
    final_result = None
    for chunk in _get_image_client().generate(prompt, format="path", stream=True):
        yield chunk
        content = chunk.content
        if isinstance(content, dict) and content.get("final"):
            final_result = content.get("result")
    return final_result


def make_image_tool(client, *, preview_every: Optional[int] = None):
    """Build a ``generate_image`` tool bound to a specific image client.

    ``client`` may be an :class:`aimu.ImageClient` or any concrete
    :class:`aimu.BaseImageClient` (e.g. :class:`HuggingFaceImageClient`,
    :class:`GeminiImageClient`). Use this when an agent needs a different model
    from the default singleton, when several agents in one process shouldn't
    share a pipeline, or to opt into intermediate-image previews via
    ``preview_every=N`` (decode latents every N denoising steps).

    The returned tool is a **streaming tool** (generator) — its progress chunks
    flow through ``agent.run(stream=True)`` for live UI updates.

    Example::

        client = aimu.image_client(aimu.HuggingFaceImageModel.SDXL_BASE)
        my_tool = make_image_tool(client, preview_every=5)
        agent = Agent(text_client, tools=[my_tool])
    """

    @tool
    def generate_image(prompt: str):
        """Generate an image from a text prompt and return the saved file path.

        Streams per-step progress chunks during generation.

        Args:
            prompt: A description of the desired image.
        """
        final_result = None
        for chunk in client.generate(prompt, format="path", stream=True, preview_every=preview_every):
            yield chunk
            content = chunk.content
            if isinstance(content, dict) and content.get("final"):
                final_result = content.get("result")
        return final_result

    return generate_image


def make_describe_image_tool(
    client,
    *,
    default_instruction: str = "Describe this image in detail.",
):
    """Build a ``describe_image`` tool bound to a vision-capable chat client.

    The bound tool sends an image (file path / bytes / URL) plus an instruction to
    ``client.chat(..., images=[...])`` and returns the text response. Useful right
    after :func:`generate_image` — the agent can look at what it generated and
    tell the user what's in it.

    **Why a factory, not a singleton.** Unlike ``generate_image`` (which is a
    fixed-purpose tool with an env-var default model), ``describe_image`` is most
    useful when bound to the *same* model the agent is already using — same
    knowledge, same provider, no extra API key or model load. The agent's host
    code (e.g. the Streamlit chatbot, or your own ``Agent`` setup) creates this
    tool at agent-construction time, after picking a vision-capable chat client.

    **History isolation.** The tool snapshots ``client.messages`` before the
    vision call and restores it afterwards, so the agent's conversation log
    isn't polluted with one-off image bytes or vision-Q&A turns. ``use_tools=False``
    is passed to the inner ``chat()`` call to prevent the LLM from re-calling
    tools mid-vision.

    Args:
        client: A vision-capable :class:`aimu.BaseModelClient`. ``ValueError`` if
            ``client.model.supports_vision`` is False.
        default_instruction: Instruction passed to the vision model when the
            caller doesn't supply one. The agent can override per-call.

    Example::

        from aimu.tools.builtin import make_describe_image_tool

        text_client = aimu.client("anthropic:claude-sonnet-4-6")  # vision-capable
        describe_image = make_describe_image_tool(text_client)
        agent = Agent(text_client, tools=[builtin.generate_image, describe_image])
    """
    if not getattr(client.model, "supports_vision", False):
        raise ValueError(
            f"Client's model {client.model.value!r} does not support vision input. "
            "Pass a vision-capable client (e.g. anthropic:claude-sonnet-4-6, "
            "openai:gpt-4o-mini, gemini:gemini-2.5-flash, ollama:gemma4:e4b)."
        )

    @tool
    def describe_image(image_path: str, instruction: str = default_instruction) -> str:
        """Look at an image and return a text description.

        Use this when you (or the user) need to know what's in an image —
        right after ``generate_image``, or when the user references one by path.

        Args:
            image_path: Path to the image file (e.g. the return value of
                ``generate_image``), an http(s) URL, or a data URL.
            instruction: What to ask the vision model. Defaults to a generic
                "describe this image" request; pass a more specific question
                (e.g. ``"What text appears in this image?"``) for targeted reads.
        """
        # Snapshot the conversation state so the vision call doesn't pollute history.
        saved_messages = list(client.messages)
        try:
            client.messages = []
            return client.chat(instruction, images=[image_path], use_tools=False)
        finally:
            client.messages = saved_messages

    return describe_image


def make_tools(base_client, image_client=None, preview_every=None):
    """Assemble the standard tool list for a chat client.

    Starts from ALL_TOOLS, then applies two optional enhancements:
    - If *image_client* is provided, replaces the default ``generate_image``
      singleton with one bound to that client (honoring *preview_every*).
    - If *base_client* supports vision, appends a ``describe_image`` tool
      bound to the same client so the model can inspect generated images.
    """
    tools = list(ALL_TOOLS)
    if image_client is not None:
        bound = make_image_tool(image_client, preview_every=preview_every)
        tools = [t for t in tools if t is not generate_image] + [bound]
    if getattr(base_client.model, "supports_vision", False):
        tools.append(make_describe_image_tool(base_client))
    return tools


# Curated subsets — pass one of these to ``tools=`` instead of importing every function.
web = [get_weather, get_webpage, search, wikipedia]
fs = [list_directory, read_file]
compute = [calculate]
misc = [echo, get_current_date_and_time]
image = [generate_image]

ALL_TOOLS = [*misc, get_weather, *compute, get_webpage, search, wikipedia, *fs, *image]
