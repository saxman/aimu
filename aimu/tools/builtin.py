"""
Built-in general-purpose tools, decorated with ``@tool`` for direct in-process use.

These same callables are registered on the FastMCP server in [aimu/tools/mcp.py](aimu/tools/mcp.py)
so that the tools are available either in-process (``Agent(client, tools=[get_weather, ...])``)
or cross-process (``python -m aimu.tools.mcp``).
"""

import datetime
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional
from urllib.parse import quote, urljoin
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
from dotenv import load_dotenv

from aimu.events import EventSink

from . import _execute_python_worker
from .decorator import tool

_logger = logging.getLogger(__name__)

load_dotenv()
SEARXNG_BASE_URL = os.environ.get("SEARXNG_BASE_URL", "http://localhost:8080")


@tool
def echo(echo_string: str) -> str:
    """Returns echo_string."""
    return echo_string


_IANA_HINT = "Use an IANA name such as 'Asia/Tokyo' or 'America/New_York'."


def _resolve_zone(name: str) -> Optional[ZoneInfo]:
    """Returns the zone for an IANA key, or None if the key is unknown or malformed."""
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return None


def _local_zone_key() -> Optional[str]:
    """Best-effort IANA key for the system's zone.

    The stdlib exposes no API for this, so read the TZ environment variable and then
    fall back to the path the /etc/localtime symlink points into. Returns None when
    neither yields a key (notably on Windows), in which case callers report the UTC
    offset and abbreviation alone rather than guessing a zone.
    """
    tz = os.environ.get("TZ")
    if tz:
        return tz

    parts = Path(os.path.realpath("/etc/localtime")).parts
    if "zoneinfo" in parts:
        return "/".join(parts[parts.index("zoneinfo") + 1 :])
    return None


# A wall-clock time a model is likely to emit for a tool that documents ISO 8601: an unpadded hour
# ("2026-08-11T5:00:00") and a 12-hour clock ("2026-08-11 5:00 PM") both fail `fromisoformat`, which
# wants a zero-padded 24-hour time. Both are unambiguous once the date is there, so parse them rather
# than spending a round trip teaching the model to reformat. A date is still required: this tool exists
# for times other than now, so inferring today's date would be a silent guess (that is what
# get_current_date_and_time is for).
_LOOSE_DATETIME = re.compile(
    r"""^(?P<date>\d{4}-\d{2}-\d{2})
        [T ]\s*
        (?P<hour>\d{1,2}):(?P<minute>\d{2})
        (?::(?P<second>\d{2}))?
        (?:\s*(?P<meridiem>[AaPp])\.?[Mm]\.?)?
        \s*(?P<offset>Z|z|[+-]\d{2}:?\d{2})?$""",
    re.VERBOSE,
)


def _parse_datetime(text: str) -> Optional[datetime.datetime]:
    """Parses an ISO-8601 timestamp, tolerating an unpadded hour and a 12-hour clock.

    Returns None when the text is not a timestamp at all, so the caller can return its
    teaching string. Normalizes to a strict ISO string and defers to ``fromisoformat``
    for the actual parse, so calendar validation stays in one place.
    """
    text = text.strip()
    try:
        return datetime.datetime.fromisoformat(text).replace(microsecond=0)
    except ValueError:
        pass

    match = _LOOSE_DATETIME.match(text)
    if match is None:
        return None

    hour = int(match["hour"])
    meridiem = (match["meridiem"] or "").lower()
    if meridiem:
        if not 1 <= hour <= 12:
            return None  # "13:00 PM" is a contradiction, not a time to guess at
        hour = 0 if hour == 12 else hour
        if meridiem == "p":
            hour += 12
    elif hour > 23:
        return None

    offset = (match["offset"] or "").replace("z", "Z")
    normalized = f"{match['date']}T{hour:02d}:{match['minute']}:{match['second'] or '00'}{offset}"
    try:
        return datetime.datetime.fromisoformat(normalized).replace(microsecond=0)
    except ValueError:
        return None


def _format_instant(moment: datetime.datetime, zone_key: Optional[str]) -> str:
    """Renders an aware datetime as an ISO-8601 timestamp annotated with zone and UTC."""
    offset = moment.strftime("%z")
    labels = []
    for label in (zone_key, moment.tzname(), f"UTC{offset[:3]}:{offset[3:]}"):
        if label and label not in labels:
            labels.append(label)

    utc = moment.astimezone(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    return f"{moment.isoformat()} ({', '.join(labels)}; {utc})"


def _attach_zone(naive: datetime.datetime, zone: ZoneInfo, zone_key: str) -> tuple[datetime.datetime, Optional[str]]:
    """Attaches a zone to a wall-clock time, reporting any DST transition it lands on.

    ZoneInfo resolves both a nonexistent time (the spring-forward gap) and an ambiguous
    one (the fall-back overlap) silently, defaulting to fold=0. That silence is the very
    error a cross-zone calculation needs surfaced, so return a note alongside the result.
    A nonexistent time resolves to the instant ZoneInfo picks, so that a caller is never
    shown a wall-clock reading that did not occur.
    """
    attached = naive.replace(tzinfo=zone)

    resolved = attached.astimezone(datetime.timezone.utc).astimezone(zone)
    if resolved.replace(tzinfo=None) != naive:
        return resolved, (
            f"note: {naive.isoformat()} does not exist in {zone_key} (spring-forward gap); "
            f"interpreted as {resolved.isoformat()}"
        )

    if attached.replace(fold=1).utcoffset() != attached.utcoffset():
        return attached, (
            f"note: {naive.isoformat()} is ambiguous in {zone_key} (clocks repeat); "
            f"interpreted as the first occurrence, {attached.tzname()}"
        )
    return attached, None


@tool
def get_current_date_and_time(timezone: Optional[str] = None) -> str:
    """Returns the current date and time, with its UTC offset and timezone.

    Args:
        timezone: IANA timezone name (e.g. "Asia/Tokyo", "America/New_York") to report
            the time in. Omit for the local timezone.
    """
    if timezone is not None:
        zone = _resolve_zone(timezone)
        if zone is None:
            return f"Unknown timezone: '{timezone}'. {_IANA_HINT}"
        return _format_instant(datetime.datetime.now(zone).replace(microsecond=0), timezone)

    # Prefer building the time from the derived key so the reported zone and offset
    # always agree; astimezone() alone knows the offset but not which zone it is.
    key = _local_zone_key()
    zone = _resolve_zone(key) if key else None
    if zone is None:
        return _format_instant(datetime.datetime.now().astimezone().replace(microsecond=0), None)
    return _format_instant(datetime.datetime.now(zone).replace(microsecond=0), key)


@tool
def convert_time(datetime_str: str, from_timezone: str, to_timezone: str) -> str:
    """Converts a date and time from one timezone to another.

    Both timezones must be IANA names ("America/Los_Angeles"), not abbreviations ("PST")
    or spoken names ("Pacific Time"). Example call:
    convert_time("2026-08-11T05:00:00", "America/Los_Angeles", "Europe/Zurich").

    Handles daylight saving time, and flags times that a DST transition makes
    nonexistent or ambiguous.

    Args:
        datetime_str: Date and time, ISO 8601 preferred (e.g. "2026-11-02T15:00:00").
            A 12-hour clock and an unpadded hour are also accepted ("2026-11-02 3:00 PM").
            The date is required. If the time already carries a UTC offset, that offset
            is used and from_timezone is ignored.
        from_timezone: IANA timezone name the time is given in (e.g. "Europe/Berlin").
        to_timezone: IANA timezone name to convert to (e.g. "America/Denver").
    """
    parsed = _parse_datetime(datetime_str)
    if parsed is None:
        return (
            f"Could not parse '{datetime_str}'. Give a date and a time in ISO 8601 format, e.g. "
            f"'2026-11-02T15:00:00' (a 12-hour clock like '2026-11-02 3:00 PM' also works). "
            f"For the current time, call get_current_date_and_time instead."
        )

    source_zone = _resolve_zone(from_timezone)
    if source_zone is None:
        return f"Unknown timezone: '{from_timezone}'. {_IANA_HINT}"

    target_zone = _resolve_zone(to_timezone)
    if target_zone is None:
        return f"Unknown timezone: '{to_timezone}'. {_IANA_HINT}"

    notes = []
    if parsed.tzinfo is None:
        source, note = _attach_zone(parsed, source_zone, from_timezone)
        if note:
            notes.append(note)
        source_key = from_timezone
    else:
        offset = parsed.strftime("%z")
        notes.append(f"note: from_timezone ignored; input carried offset {offset[:3]}:{offset[3:]}")
        source = parsed
        source_key = None

    lines = [
        _format_instant(source, source_key),
        f"-> {_format_instant(source.astimezone(target_zone), to_timezone)}",
    ]
    return "\n".join(lines + notes)


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


# Restricted-exec logic (the shared "run this code, capture stdout + last expression"
# behavior for both execute_python backends) lives in the dependency-free
# _execute_python_worker leaf module -- see its docstring for why the subprocess
# backend can't just `import aimu.tools.builtin` in the child. Re-exported here mainly
# so existing callers/tests that reached for `_SANDBOX_ALLOWLIST` on this module still
# find it.
_SANDBOX_ALLOWLIST = _execute_python_worker.SANDBOX_ALLOWLIST


@tool
def execute_python_in_process(code: str) -> str:
    """Execute Python code in this process and return its output. Explicit opt-in.

    NOT A SECURITY BOUNDARY: this is isolation, not containment, and weaker isolation
    than `execute_python` (the default execute-code tool) -- code here shares this
    process's memory, imports, and environment variables (including any API keys sitting
    in os.environ), a hang blocks this process indefinitely, and a crash takes it down
    too. It runs with restricted builtins and an import allowlist, which stops accidents,
    not a determined attempt: a one-line expression reaches `subprocess.Popen` through the
    type hierarchy, and the filesystem through an allowlisted module's transitive
    attributes. Use this only for trusted code where the subprocess startup cost of
    `execute_python` matters; gate it with `tool_approval` otherwise, and reach for a
    container when you need real containment.

    Captures stdout and the value of the last expression. Pre-imports math, statistics,
    json, re, itertools, functools, datetime, zoneinfo, and numpy/pandas/scipy/matplotlib
    when installed.

    Args:
        code: Python code to execute.
    """
    return _execute_python_worker.run_restricted(code)


_EXECUTE_PYTHON_TIMEOUT_S = 10
# RLIMIT_AS caps the child's total address space. Genuinely enforced on Linux;
# best-effort elsewhere (macOS/XNU refuses to lower it at all -- confirmed
# independently via a plain shell's `ulimit -v`, so this is a platform limitation, not
# a bug here -- and Windows has no rlimit mechanism to speak of). See
# _build_execute_python_worker_source(): the child sets this itself, inside a
# try/except that reports failure back over stderr rather than raising, so a platform
# that can't enforce it degrades to "no cap" instead of refusing to run at all.
_EXECUTE_PYTHON_MEMORY_LIMIT_BYTES = 512 * 1024 * 1024
_EXECUTE_PYTHON_OUTPUT_LIMIT_BYTES = 20_000

# A private, unlikely-to-collide marker: written to the child's stderr by the rlimit
# try/except in the worker source when the memory cap could not be applied. The parent
# strips it out of the visible error text and turns it into a one-time logging warning
# (_warn_once_execute_python) instead of a docstring nobody reads at runtime.
_MEMORY_CAP_UNAVAILABLE_MARKER = "__AIMU_EXECUTE_PYTHON_MEMORY_CAP_UNAVAILABLE__"

# The directory this module lives in, baked in as a literal at import time. The worker
# script inserts it at the front of the child's sys.path so `import _execute_python_worker`
# resolves there directly -- independent of PYTHONPATH, site-packages, or the child's
# cwd (a throwaway temp directory), all of which the scrubbed environment and isolated
# cwd would otherwise break for a PYTHONPATH-based install or an unbuilt source checkout.
_EXECUTE_PYTHON_WORKER_DIR = str(Path(__file__).resolve().parent)


def _build_execute_python_worker_source() -> str:
    """Build the ``-c`` script for the execute_python child.

    Sets the memory cap as the very first thing the child does, before importing
    anything (including the allowlisted third-party modules `run_restricted` preloads,
    which are exactly the kind of heavy allocation the cap exists to bound) -- and
    never imports the `aimu` package (see `_execute_python_worker`'s module docstring).
    """
    limit = _EXECUTE_PYTHON_MEMORY_LIMIT_BYTES
    return (
        "import sys\n"
        f"sys.path.insert(0, {_EXECUTE_PYTHON_WORKER_DIR!r})\n"
        "try:\n"
        "    import resource\n"
        f"    resource.setrlimit(resource.RLIMIT_AS, ({limit}, {limit}))\n"
        "except Exception:\n"
        f"    sys.stderr.write({_MEMORY_CAP_UNAVAILABLE_MARKER + chr(10)!r})\n"
        "import _execute_python_worker\n"
        "sys.stdout.write(_execute_python_worker.run_restricted(sys.stdin.read()))\n"
    )


_EXECUTE_PYTHON_WORKER_SOURCE = _build_execute_python_worker_source()

# Environment variables passed through to the execute_python child. Everything else in
# this process's os.environ -- API keys included -- is deliberately left behind; only
# what the interpreter and the allowlisted libraries need to start is kept. Notably
# absent on purpose: PYTHONPATH. The child never needs it (see _EXECUTE_PYTHON_WORKER_DIR
# above), and leaving it out narrows the environment further.
_EXECUTE_PYTHON_ENV_ALLOWLIST = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "SYSTEMROOT",
    "SystemRoot",
    "COMSPEC",
    "TEMP",
    "TMP",
    "PYTHONIOENCODING",
)


def _execute_python_env() -> dict:
    """A minimal environment for the execute_python child: enough to start Python and
    import the allowlisted libraries, nothing else. This process's environment
    variables -- API keys included -- are never passed through. (Credentials living in
    *files* -- a `.env`, `~/.aws/credentials`, and so on -- are a different story; see
    `execute_python`'s docstring.)
    """
    return {name: value for name in _EXECUTE_PYTHON_ENV_ALLOWLIST if (value := os.environ.get(name)) is not None}


# What a command's environment adds to execute_python's allowlist. A Python snippet needs
# an interpreter to start; a command needs to look like it is running in a shell, which is
# a slightly larger set and a deliberately small one. Still absent, and the reason this is
# an allowlist rather than a denylist of secret-looking names: everything else in this
# process's environment, API keys included. `run_command("env")` therefore cannot lift a
# credential into a model's context.
_COMMAND_ENV_ALLOWLIST = _EXECUTE_PYTHON_ENV_ALLOWLIST + (
    "SHELL",
    "TERM",
    "TZ",
    "USER",
    "LOGNAME",
)


def _command_env(passthrough: tuple[str, ...] = ()) -> dict:
    """The environment for a `run_command` child: the allowlist, plus names the host asked for.

    *passthrough* is how a capability stays real without the default being unsafe. A host
    that wants `gh` or `git push` over ssh to work names `GH_TOKEN` or `SSH_AUTH_SOCK`
    itself (see `make_command_tool`); nothing here guesses that a variable is wanted, so
    naming one does not admit its neighbors. A name that is unset produces no key at all,
    rather than an empty-string value, since `SSH_AUTH_SOCK=""` misleads ssh in a way a
    missing variable does not.
    """
    names = _COMMAND_ENV_ALLOWLIST + tuple(passthrough)
    return {name: value for name in names if (value := os.environ.get(name)) is not None}


_execute_python_warned: set = set()


def _warn_once_execute_python(message: str) -> None:
    """Log *message* at WARNING the first time only.

    Mirrors `_ChatStateMixin._warn_once` (aimu/models/_internal/chat_state.py), which
    exists for the same reason: repeating the same warning on every call in an agent
    loop would just be noise. This is a module-level set rather than an instance
    attribute because `execute_python` is a plain function, not a client with `self`
    to hold the "have I warned" state.
    """
    if message in _execute_python_warned:
        return
    _execute_python_warned.add(message)
    _logger.warning(message)


def _strip_memory_cap_marker(stderr_text: str) -> str:
    """Remove the memory-cap-unavailable marker from *stderr_text* if present, firing a
    one-time discoverable warning first. Previously this failure was swallowed by a
    bare `except: pass` with the degradation documented only in a private docstring --
    discoverable by reading source, not by anyone actually running the tool.
    """
    if _MEMORY_CAP_UNAVAILABLE_MARKER in stderr_text:
        _warn_once_execute_python(
            "execute_python: could not apply the POSIX memory cap on this platform "
            "(RLIMIT_AS is not enforceable here); running without a memory cap. "
            "See execute_python's docstring."
        )
        stderr_text = stderr_text.replace(_MEMORY_CAP_UNAVAILABLE_MARKER + "\n", "")
        stderr_text = stderr_text.replace(_MEMORY_CAP_UNAVAILABLE_MARKER, "")
    return stderr_text


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Kill *proc* and every process in its process group.

    `_run_supervised` launches every child with `start_new_session=True`, making it the
    leader of a new process group; anything it spawns (a backgrounded process a snippet
    or a shell command starts and does not wait for) inherits that same group unless it
    detaches into a session of its own. A plain `proc.kill()` reaches only the direct
    child, which is exactly how a "hard timeout" used to leave a backgrounded
    grandchild running indefinitely. SIGKILL is uncatchable, so this can't race the
    child's (or grandchild's) own exception handling.

    `os.killpg`/`os.getpgid` don't exist on Windows, so this falls back to killing just
    the direct child there: worse than a real process-group kill (a backgrounded
    grandchild can still be left running), but it's the same guarantee
    `subprocess.run(timeout=...)` gave before this moved to a hand-rolled `Popen`, and
    it keeps the timeout/cleanup path returning rather than raising `AttributeError`
    partway through.
    """
    killpg = getattr(os, "killpg", None)
    getpgid = getattr(os, "getpgid", None)
    if killpg is None or getpgid is None:
        proc.kill()
        return
    try:
        killpg(getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass  # already gone


def _read_capped(path: str, limit_bytes: int) -> str:
    """Read at most *limit_bytes* from the file at *path*, decoding leniently.

    Two properties this buys over the previous pipe-based ``capture_output=True`` +
    post-hoc truncation:

    - **Bounded parent memory.** Reading only a capped prefix, rather than the whole
      file, means an output bomb (`print('x' * 200_000_000)`) cannot inflate this
      process's own RSS the way buffering the full child output first did (743 MB of
      *parent* memory, observed, before the old truncation ever ran) -- the excess
      stays on disk in the temp directory, deleted with it.
    - **No `UnicodeDecodeError` escaping this tool.** `errors="replace"` means a
      snippet that writes raw/invalid bytes to stdout still returns a string, never an
      uncaught exception; the tool's contract is "a string, or an error string."

    Returns "" if *path* doesn't exist (e.g. the child never got far enough to create
    it).
    """
    try:
        total_size = os.path.getsize(path)
    except OSError:
        return ""
    with open(path, "rb") as f:
        data = f.read(limit_bytes)
    text = data.decode("utf-8", errors="replace")
    if total_size > limit_bytes:
        text = f"{text}\n[... truncated {total_size - limit_bytes} bytes]"
    return text


class _SupervisedRun(NamedTuple):
    """What `_run_supervised` learned about one child: how it ended, and what it wrote."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool


def _run_supervised(
    argv: list[str],
    *,
    stdin_data: Optional[bytes],
    cwd: Optional[str],
    env: dict,
    timeout_s: float,
    output_limit_bytes: int,
) -> _SupervisedRun:
    """Run *argv* as a supervised child, and return how it ended plus what it wrote.

    Shared by `execute_python` and `run_command` so the properties that were expensive
    to get right hold identically for both, rather than being reimplemented once per
    tool and drifting:

    - **No path out leaves the direct child alive.** `Popen.__exit__` reaps it on a normal
      return; timeout and any interruption (`KeyboardInterrupt`, a `CancelledError`
      surfacing through a RunHandle) additionally SIGKILL the whole process group before
      reaping, since a normal return gives the child every chance to exit on its own but a
      timeout and an interruption do not. A grandchild the child backgrounded is deliberately
      exempt on the normal-return path: see the "Output goes to files" bullet below, and
      `test_a_fast_command_that_backgrounds_something_is_not_a_false_timeout`.
    - **Output goes to files, not pipes.** `wait()` depends only on the direct child's
      exit, never on every process holding a copy of the stdout/stderr fds closing
      them, so a backgrounded grandchild inheriting those fds cannot hold a fast,
      successful run hostage.
    - **Bounded parent memory,** since only a capped prefix of each file is ever read.

    *cwd* of None means the runner's own temp directory. That is what `execute_python`
    wants (a throwaway working directory the snippet cannot pollute) and never what
    `run_command` wants, which resolves its own default to `os.getcwd()` before calling
    here. Each caller's absent value therefore means its own thing, and neither has to
    know the other's.

    *stdin_data* of None gives the child `DEVNULL` rather than an open pipe, so a
    command that reads stdin sees EOF at once instead of blocking until the timeout.
    """
    with tempfile.TemporaryDirectory(prefix="aimu-supervised-") as tmpdir:
        stdout_path = os.path.join(tmpdir, "stdout.txt")
        stderr_path = os.path.join(tmpdir, "stderr.txt")
        timed_out = False

        with (
            open(stdout_path, "wb") as stdout_f,
            open(stderr_path, "wb") as stderr_f,
            subprocess.Popen(  # noqa: S603
                argv,
                stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
                stdout=stdout_f,
                stderr=stderr_f,
                cwd=tmpdir if cwd is None else cwd,
                env=env,
                start_new_session=True,
            ) as proc,
        ):
            # Everything from here to the end of the `with` runs under one guard: once the
            # child exists, no path out of this block -- normal, exceptional, or
            # interrupted -- may leave it alive. Feeding stdin is inside it too, because an
            # interruption during the write or the close is just as capable of orphaning a
            # child that is already running (and `Popen.__exit__` will not save us: its
            # KeyboardInterrupt branch waits only ~0.25s on the assumption the SIGINT
            # already reached the child, which is false once the child has its own group).
            try:
                if stdin_data is not None:
                    try:
                        proc.stdin.write(stdin_data)
                    except BrokenPipeError:
                        # The child exited before reading its input. Not an interruption:
                        # fall through to the normal path, where its output and exit code
                        # explain why.
                        pass
                    finally:
                        proc.stdin.close()
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                _kill_process_group(proc)
                proc.wait()
                timed_out = True
            except BaseException:
                # Kill, then reap, then re-raise: a kill without the wait leaves a zombie.
                # Deliberately BaseException, not Exception: KeyboardInterrupt and
                # CancelledError are exactly the cases this exists for.
                _kill_process_group(proc)
                proc.wait()
                raise

        # stdout_f/stderr_f are closed by the `with` above and the child has exited, so
        # its writes are guaranteed flushed to disk. Read on the timeout path too: what a
        # command printed before it hung is usually the most useful thing about it.
        stdout_text = _read_capped(stdout_path, output_limit_bytes)
        stderr_text = _read_capped(stderr_path, output_limit_bytes)

    return _SupervisedRun(proc.returncode, stdout_text, stderr_text, timed_out)


@tool
def execute_python(code: str) -> str:
    """Execute Python code in a fresh subprocess and return its output.

    NOT A SECURITY BOUNDARY: this is isolation, not containment. Running the code in a
    subprocess buys a hard timeout (an in-process exec could hang this process forever)
    and crash isolation (an ordinary crash or early exit in the code brings down only
    the child -- though nothing stops the code from reaching back out and taking this
    process down anyway, e.g. `os.kill(os.getppid(), signal.SIGKILL)`, since process
    signalling isn't confined any more than the filesystem is; see below). It also buys
    no mutation of this process's imports or global state, a memory cap on Linux
    (best-effort on other POSIX platforms, absent on Windows -- a one-time warning is logged
    if the cap can't be applied, rather than failing silently), and no access to
    this process's environment variables: `ANTHROPIC_API_KEY` and anything else set
    there is invisible to the child.

    That last point is about *environment variables specifically*, not credentials in general.
    It does not confine filesystem or network access: the child runs as the
    same user and can read, write, and make requests exactly as this process can, which
    means a `.env` file, `~/.aws/credentials`, `~/.config/gh/hosts.yml`, or any other
    credential sitting on disk is exactly as readable to the child as to this process's
    own user account. Treat any code reaching this tool as code you have chosen to run;
    gate it with `tool_approval` for untrusted callers, and reach for a container when
    you need real containment.

    The child also runs with restricted builtins and an import allowlist, the same
    accident guard `execute_python_in_process` uses (stops accidents, not a determined
    attempt) -- kept identical between both backends so switching between them changes
    only where the code runs, not what it's allowed to do.

    Captures stdout and the value of the last expression. Pre-imports math, statistics,
    json, re, itertools, functools, datetime, zoneinfo, and numpy/pandas/scipy/matplotlib
    when installed. For trusted code where the subprocess startup cost matters, see
    `execute_python_in_process` (same disclosed limits, weaker isolation).

    Args:
        code: Python code to execute.
    """
    run = _run_supervised(
        [sys.executable, "-c", _EXECUTE_PYTHON_WORKER_SOURCE],
        stdin_data=code.encode("utf-8"),
        cwd=None,  # the runner's temp directory: a throwaway cwd for the snippet
        env=_execute_python_env(),
        timeout_s=_EXECUTE_PYTHON_TIMEOUT_S,
        output_limit_bytes=_EXECUTE_PYTHON_OUTPUT_LIMIT_BYTES,
    )
    if run.timed_out:
        return f"Error: execution timed out after {_EXECUTE_PYTHON_TIMEOUT_S}s"

    stderr_text = _strip_memory_cap_marker(run.stderr)
    if run.exit_code != 0:
        detail = stderr_text.strip() or f"exit code {run.exit_code}"
        return f"Error: subprocess exited abnormally ({detail})"

    return run.stdout


_COMMAND_TIMEOUT_DEFAULT_S = 30
# A ceiling rather than a fixed duration: a model that knows it is running a test suite can
# ask for four minutes, and no model can ask for forever. 10s (execute_python's constant)
# would make the obvious use of this tool impossible.
_COMMAND_TIMEOUT_MAX_S = 600
# Same value as execute_python's cap today, kept under its own name so the two tiers can
# diverge later without a rename: this is the combined budget across stdout and stderr
# (see _cap_combined_output), where execute_python's is a per-stream one.
_COMMAND_OUTPUT_LIMIT_BYTES = _EXECUTE_PYTHON_OUTPUT_LIMIT_BYTES


def _shell_argv(command: str) -> list[str]:
    """The argv that hands *command* to a shell.

    Spelled out rather than passing `shell=True`, which would do this platform dispatch in
    less code: which shell interpreted a command decides what `&&`, quoting, and `$VAR`
    meant, so it is worth being able to read it here rather than inferring it from
    `subprocess`'s own rules. `/bin/sh` specifically, not `$SHELL`: a tool whose behavior
    changed with the user's login shell would be reproducible nowhere.
    """
    if os.name == "nt":
        return [os.environ.get("COMSPEC", "cmd.exe"), "/c", command]
    return ["/bin/sh", "-c", command]


def _truncate_text_to_bytes(text: str, encoded: bytes, budget: int) -> str:
    """Cut *text* (already available as *encoded*) to *budget* bytes, marking it if it shrank.

    Decodes with ``errors="replace"`` because *budget* can land mid-codepoint; the marker
    matches ``_read_capped``'s wording so a reader sees one consistent shape for "there was
    more than this" regardless of which cap produced it.
    """
    if len(encoded) <= budget:
        return text
    kept = encoded[:budget].decode("utf-8", errors="replace")
    return f"{kept}\n[... truncated {len(encoded) - budget} more bytes]"


def _cap_combined_output(stdout: str, stderr: str, limit_bytes: int) -> tuple[str, str]:
    """Trim *stdout* and *stderr* together to one shared byte budget.

    `_run_supervised` caps each stream independently (a property `execute_python` depends on
    and this function does not touch), so a command that writes near the cap to both streams
    can return roughly twice the documented limit. This is the other side of that call, where
    only `run_command` is affected: `execute_python` never reaches this function, since it
    returns stdout alone on success and capped stderr alone on failure.

    The budget splits evenly, except a stream that needs less than its half hands the
    difference to the other: a one-line stderr should not cost a talkative stdout half the
    budget, and a large stdout must not be able to push stderr out of the result entirely.
    """
    stdout_bytes = stdout.encode("utf-8")
    stderr_bytes = stderr.encode("utf-8")
    if len(stdout_bytes) + len(stderr_bytes) <= limit_bytes:
        return stdout, stderr

    half = limit_bytes // 2
    if len(stdout_bytes) <= half:
        stdout_budget, stderr_budget = len(stdout_bytes), limit_bytes - len(stdout_bytes)
    elif len(stderr_bytes) <= half:
        stderr_budget, stdout_budget = len(stderr_bytes), limit_bytes - len(stderr_bytes)
    else:
        stdout_budget = stderr_budget = half

    return (
        _truncate_text_to_bytes(stdout, stdout_bytes, stdout_budget),
        _truncate_text_to_bytes(stderr, stderr_bytes, stderr_budget),
    )


def _format_command_result(run: _SupervisedRun, timeout_s: float, limit_bytes: int) -> str:
    """Render a finished command for a model: how it ended, then what it wrote.

    The first line is always present, so the shape a model parses does not change with the
    outcome. Empty streams are omitted rather than shown as empty sections, and a command
    that wrote nothing at all says so, because a bare exit line reads like truncation.

    *limit_bytes* bounds stdout and stderr combined, via `_cap_combined_output`, rather than
    each stream separately: `_run_supervised` already caps each stream at *limit_bytes* on its
    own, so without this step a command writing near that cap to both streams would return
    close to double it.
    """
    header = f"Error: command timed out after {timeout_s:g}s" if run.timed_out else f"exit {run.exit_code}"
    stdout_text, stderr_text = _cap_combined_output(run.stdout, run.stderr, limit_bytes)
    sections = [header]
    if stdout_text.strip():
        sections.append(f"--- stdout ---\n{stdout_text.rstrip()}")
    if stderr_text.strip():
        sections.append(f"--- stderr ---\n{stderr_text.rstrip()}")
    if len(sections) == 1:
        sections.append("(no output)")
    return "\n".join(sections)


def make_command_tool(*, env_passthrough: tuple[str, ...] = ()) -> Callable:
    """Build a `run_command` whose child also sees the named environment variables.

    *env_passthrough* is what keeps the capability real without the default being unsafe.
    The child's environment is an allowlist with no API keys in it, which is what stops
    `run_command("env")` lifting a credential into a model's context, and which also makes
    `gh`, `ssh`, and `git push` over ssh fail. A host that wants one of those working names
    the variable it needs, typically from its own configuration, so the allowance is the
    user's to grant rather than this library's to assume.

    `builtin.compute`'s `run_command` is this factory called with no arguments, rather than
    a separately defined twin. `@tool` freezes the docstring into `__tool_spec__` at
    decoration time, so a closure with no docstring reaches a model with an empty
    description and a `__doc__` assigned afterwards never lands; one definition is the only
    way to keep one security disclosure.
    """
    if isinstance(env_passthrough, str):
        # tuple("GH_TOKEN") is ('G', 'H', '_', 'T', 'O', 'K', 'E', 'N'), none of which is set,
        # so a caller who meant one variable and passed a bare string would get a tool that
        # silently admits nothing rather than an error at the call that misconfigured it.
        raise TypeError(f"env_passthrough must be a tuple of variable names, not a bare string: {env_passthrough!r}.")

    @tool
    def run_command(command: str, cwd: str = "", timeout: int = _COMMAND_TIMEOUT_DEFAULT_S) -> str:
        """Run a unix command through a shell and return its exit code and output.

        NOT A SECURITY BOUNDARY: this is isolation, not containment, in exactly the sense
        `execute_python`'s docstring means it, and with one fewer step in between. The command
        runs as the same user this process does, so it reads, writes, and makes network
        requests exactly as this process can. A `.env` file, `~/.aws/credentials`, or
        `~/.config/gh/hosts.yml` is as readable to it as to this user's own account. Process
        signalling is not confined either: `kill -9` against this process's own pid is one
        command away. Treat anything reaching this tool as a command you have chosen to run,
        gate it with `tool_approval` for untrusted callers, and reach for a container when you
        need real containment.

        What the subprocess does buy: a hard timeout, so a hung command cannot hang this
        process; a process-group kill, so a command that backgrounds something does not leave
        it running after that timeout; crash isolation; and no access to this process's
        environment variables, so `ANTHROPIC_API_KEY` and anything else set there is invisible
        unless a host passed the name through deliberately (see `make_command_tool`). Note what
        that last one is not: it is about environment variables specifically, not credentials
        in general, which is what the paragraph above is about.

        Unlike `execute_python`, there is no memory cap. A 512 MB address-space limit would
        break compilers and test suites, and imposing one on a shell child needs `preexec_fn`,
        which is neither portable nor safe alongside threads.

        A nonzero exit is reported, not treated as a failure to be summarized away: `pytest`
        exits 1 with the answer on stdout, and `git diff --exit-code` exits 1 to mean "yes,
        there is a diff". Output is capped, stdout and stderr are labelled separately, and a
        command killed by its timeout still returns whatever it printed first.

        A command that backgrounds something returns as soon as the shell itself exits, not
        when the backgrounded work finishes: `run_command("./build.sh > log 2>&1 &")` can
        report `exit 0` and `(no output)` within milliseconds while the build keeps running.
        Only a run still alive at the timeout gets its whole process group killed; a fast
        return neither waits for nor cleans up anything left running in the background, and if
        that background process did not redirect its own output somewhere durable, it was
        inheriting this call's capture files, in a temp directory already deleted by the time
        you read the answer.

        Args:
            command: The command line to run. Interpreted by /bin/sh, so pipes, redirects,
                globs, and && work (on Windows, by COMSPEC instead, where quoting and glob
                rules differ).
            cwd: Directory to run in. Defaults to this process's working directory.
            timeout: Seconds to allow, clamped to 600.
        """
        if cwd and not os.path.isdir(cwd):
            return f"Error: working directory does not exist: {cwd}"

        timeout_s = max(1, min(int(timeout), _COMMAND_TIMEOUT_MAX_S))
        result = _run_supervised(
            _shell_argv(command),
            stdin_data=None,
            cwd=cwd or os.getcwd(),
            env=_command_env(tuple(env_passthrough)),
            timeout_s=timeout_s,
            output_limit_bytes=_COMMAND_OUTPUT_LIMIT_BYTES,
        )
        return _format_command_result(result, timeout_s, _COMMAND_OUTPUT_LIMIT_BYTES)

    return run_command


#: The command tool AIMU's own groups carry: no environment passthrough, so nothing beyond the
#: allowlist reaches a command. A host wanting more calls `make_command_tool` itself.
#: No `::: aimu.tools.builtin.run_command` directive in the API reference: this is a module
#: attribute rather than a `def`, so griffe resolves it as `Kind.ATTRIBUTE` with no docstring,
#: and the directive would render an empty stub. `docs/reference/api/tools.md` documents it
#: through `make_command_tool`'s directive instead.
run_command = make_command_tool()


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


# Publication timestamps hide in machine-readable HTML that _TextExtractor strips out
# (<head>/<meta>) or in attributes it ignores (<time datetime>). These patterns recover
# them in priority order: <meta> tags (either attribute ordering), JSON-LD datePublished,
# then <time>. The matched value is usually ISO 8601.
_META_DATE_KEYS = r"article:published_time|datePublished|pubdate|publishdate|date|dc\.date|sailthru\.date"
_PUBLISH_DATE_PATTERNS = [
    rf'<meta[^>]+(?:property|name)=["\'](?:{_META_DATE_KEYS})["\'][^>]*\bcontent=["\']([^"\']+)["\']',
    rf'<meta[^>]+\bcontent=["\']([^"\']+)["\'][^>]*(?:property|name)=["\'](?:{_META_DATE_KEYS})["\']',
    r'"datePublished"\s*:\s*"([^"]+)"',
    r'<time[^>]+datetime=["\']([^"\']+)["\']',
]


def _extract_publish_date(html: str) -> str:
    """Best-effort publication timestamp from raw article HTML.

    Checks <meta> tags, JSON-LD ``datePublished``, and ``<time datetime=...>`` in
    priority order. Returns the raw date string (usually ISO 8601) or "" if none found.
    """
    for pattern in _PUBLISH_DATE_PATTERNS:
        match = re.search(pattern, html, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


_DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; aimu-tools/1.0)"


def _truncate(text: str, limit: int) -> str:
    """Cap *text* at *limit* characters, appending a marker when it was cut.

    Raw HTML is token-heavy; an untruncated page can overflow a model's context window.
    """
    if limit is None or len(text) <= limit:
        return text
    dropped = len(text) - limit
    return f"{text[:limit]}\n[... truncated {dropped} chars]"


def _fetch_html(url: str, *, session=None, timeout: int = 15, method: str = "GET", **request_kwargs):
    """Issue an HTTP request and return the raw ``requests.Response``.

    Uses *session* (preserving cookies) when given, else a module-level request.
    ``request_kwargs`` forwards ``params`` / ``data`` to the underlying call.
    Raises ``requests.RequestException`` on transport/HTTP errors (callers translate
    to a tool-visible message).
    """
    requester = session or requests
    headers = {"User-Agent": _DEFAULT_USER_AGENT}
    response = requester.request(method, url, headers=headers, timeout=timeout, **request_kwargs)
    response.raise_for_status()
    return response


class _FormExtractor(HTMLParser):
    """Collects ``<form>`` elements and their fields from raw HTML.

    Each form is a dict with ``action``, ``method``, and ``fields`` (a list of
    ``{"name", "type", "value"}`` dicts). ``type=hidden`` inputs are kept so CSRF
    tokens surface for re-submission. ``<select>`` and ``<textarea>`` are recorded as
    fields with types ``"select"`` / ``"textarea"``.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.forms: list[dict] = []
        self._current: Optional[dict] = None

    def handle_starttag(self, tag, attrs):
        attr = dict(attrs)
        if tag == "form":
            self._current = {
                "action": attr.get("action", ""),
                "method": (attr.get("method") or "GET").upper(),
                "fields": [],
            }
        elif tag in ("input", "select", "textarea") and self._current is not None:
            field_type = "input" if tag == "input" else tag
            self._current["fields"].append(
                {
                    "name": attr.get("name", ""),
                    "type": attr.get("type", field_type),
                    "value": attr.get("value", ""),
                }
            )

    def handle_endtag(self, tag):
        if tag == "form" and self._current is not None:
            self.forms.append(self._current)
            self._current = None


def _parse_forms(html: str, base_url: str) -> list[dict]:
    """Parse forms from *html*, resolving each ``action`` against *base_url*."""
    extractor = _FormExtractor()
    extractor.feed(html)
    for form in extractor.forms:
        form["action"] = urljoin(base_url, form["action"]) if form["action"] else base_url
    return extractor.forms


def _format_forms(forms: list[dict]) -> str:
    """Render parsed forms as a model-readable text listing."""
    if not forms:
        return "No forms found on the page."
    lines = []
    for i, form in enumerate(forms):
        lines.append(f"Form {i}: {form['method']} {form['action']}")
        if not form["fields"]:
            lines.append("  (no fields)")
        for field in form["fields"]:
            name = field["name"] or "(unnamed)"
            value = f" = {field['value']!r}" if field["value"] else ""
            lines.append(f"  - {name} [{field['type']}]{value}")
    return "\n".join(lines)


@tool
def get_webpage(url: str) -> str:
    """Fetches a web page and returns its visible text content with HTML stripped.

    When the page exposes a publication timestamp (in <meta> tags, JSON-LD, or a
    <time> element), it is prepended as a "Published:" line so the date isn't lost
    when the HTML is stripped.

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

    published = _extract_publish_date(response.text)
    extractor = _TextExtractor()
    extractor.feed(response.text)
    text = extractor.get_text()
    return f"Published: {published}\n\n{text}" if published else text


@tool
def get_webpage_html(url: str) -> str:
    """Fetches a web page and returns its raw HTML markup (tags and all).

    Use this when you need to see the page structure, e.g. to locate links, attributes,
    or form markup. For readable article text instead, use ``get_webpage``. For inspecting
    and submitting forms with cookie/session persistence, use the tools from
    ``make_web_tools()``.

    Note: this fetches server-rendered HTML only; it does not execute JavaScript, so
    pages built client-side (SPAs) will return their pre-render markup. Long pages are
    truncated to keep the output within a model's context window.

    Args:
        url: The URL of the page to retrieve.
    """
    try:
        response = _fetch_html(url)
    except requests.RequestException as e:
        return f"Error fetching page: {e}"
    return _truncate(response.text, 20000)


@tool
def web_search(query: str, num_results: int = 5, time_range: str = "", categories: str = "") -> str:
    """Search the web using a SearXNG instance and return the top results.

    Each result includes its publication date when the search engine reports one
    (shown as a "Published:" line), useful for judging how recent an article is.

    Args:
        query: The search query string.
        num_results: Number of results to return (default 5).
        time_range: Optional recency filter, one of "day", "week", "month", or "year".
            Use "day" to restrict results to roughly the last 24 hours (best for fresh news).
        categories: Optional SearXNG category filter, e.g. "news" to restrict to news
            engines (which report publication dates far more reliably than general web
            engines). Comma-separated for multiple, e.g. "news,science".

    Set SEARXNG_BASE_URL env var to point to your SearXNG instance (or a .env file)
    (default: http://localhost:8080).
    """
    params = {"q": query, "format": "json"}
    if time_range:
        params["time_range"] = time_range
    if categories:
        params["categories"] = categories
    try:
        response = requests.get(
            f"{SEARXNG_BASE_URL}/search",
            params=params,
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
        published = r.get("publishedDate") or ""
        date_line = f"\n   Published: {published}" if published else ""
        lines.append(f"{i}. {title}\n   {url}{date_line}\n   {snippet}")
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
def read_file(path: str, max_lines: int = 2000) -> str:
    """Reads a local file and returns its contents, capped at max_lines lines.

    If the result says it was truncated, call again with a larger max_lines before drawing any
    conclusion from it: a partial document reads exactly like a complete one.

    Args:
        path: Path to the file to read.
        max_lines: Maximum number of lines to return (default 2000).
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
    content = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        # Name the total, so the model can see how much it is missing rather than just that
        # something was cut, and name the parameter that gets the rest.
        content += (
            f"\n... (truncated: showing {max_lines} of {len(lines)} lines; "
            f"call read_file with a larger max_lines to read the rest)"
        )
    return content


# ---- Image generation (diffusion) --------------------------------------------
#
# Diffusion deps (``diffusers``, ``torch``, ``Pillow``) are heavy. The client is
# constructed lazily on first ``generate_image()`` call so that importing
# ``aimu.tools.builtin`` does not pull torch into ``sys.modules``.


def _lazy_client_getter(global_name: str, factory_name: str):
    """Build a getter that lazily constructs and caches a modality client in a module global.

    On first call the client is built from its ``AIMU_*_MODEL`` env var via
    ``aimu.<factory_name>()`` (deferring the heavy provider import so importing this
    module stays cheap), cached in ``globals()[global_name]``, and reused after. If the
    factory raises (e.g. the env var is unset), the global is left ``None`` so a later
    call retries. The cached global stays a plain module attribute, so tests and
    callers can monkeypatch / replace it directly.
    """

    def _get():
        if globals()[global_name] is None:
            import aimu

            globals()[global_name] = getattr(aimu, factory_name)()  # resolves AIMU_*_MODEL or raises
        return globals()[global_name]

    _get.__name__ = f"_get{global_name}"
    return _get


_image_client = None
# Reads AIMU_IMAGE_MODEL (or "hf:..."/"gemini:..."); raises if unset, nothing downloaded implicitly.
_get_image_client = _lazy_client_getter("_image_client", "image_client")


@tool
def generate_image(prompt: str):
    """Generate an image from a text prompt and return the saved file path.

    This is a **streaming tool**: a generator that yields
    :attr:`~aimu.models.StreamingContentType.IMAGE_GENERATING` chunks during
    denoising, then returns the saved file path. When called via the agent's
    streaming path (``agent.run(stream=True)``), each step chunk flows through
    the agent's own stream and into the chat UI live.

    Uses an :class:`aimu.ImageClient`. The model is controlled by the
    ``AIMU_IMAGE_MODEL`` env var (required; pass any HF diffusers repo via
    ``"hf:..."`` or ``"gemini:nano-banana"`` for Google Nano Banana); the tool raises
    if it is unset. Override per-agent by constructing your own tool with
    :func:`make_image_tool`; it supports a ``preview_every=N`` kwarg for intermediate
    denoised-image previews.

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


def make_image_tool(client, *, preview_every: Optional[int] = None, num_inference_steps: Optional[int] = None):
    """Build a ``generate_image`` tool bound to a specific image client.

    ``client`` may be an :class:`aimu.ImageClient` or any concrete
    :class:`aimu.BaseImageClient` (e.g. :class:`HuggingFaceImageClient`,
    :class:`GeminiImageClient`). Use this when an agent needs a different model
    from the default singleton, when several agents in one process shouldn't
    share a pipeline, or to opt into intermediate-image previews via
    ``preview_every=N`` (decode latents every N denoising steps).

    The returned tool is a **streaming tool** (generator); its progress chunks
    flow through ``agent.run(stream=True)`` for live UI updates.

    Example::

        client = aimu.image_client(aimu.HuggingFaceImageModel.SDXL_BASE)
        my_tool = make_image_tool(client, preview_every=5, num_inference_steps=20)
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
        kw = {"format": "path", "stream": True, "preview_every": preview_every}
        if num_inference_steps is not None:
            kw["num_inference_steps"] = num_inference_steps
        for chunk in client.generate(prompt, **kw):
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
    after :func:`generate_image`; the agent can look at what it generated and
    tell the user what's in it.

    **Why a factory, not a singleton.** Unlike ``generate_image`` (which is a
    fixed-purpose tool with an env-var default model), ``describe_image`` is most
    useful when bound to the *same* model the agent is already using: same
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

        Use this when you (or the user) need to know what's in an image,
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


def make_tools(
    base_client,
    image_client=None,
    preview_every=None,
    audio_client=None,
    *,
    image_steps: Optional[int] = None,
    audio_steps: Optional[int] = None,
    speech_client=None,
    memory_store=None,
    allow_code_execution: bool = False,
):
    """Assemble the standard tool list for a chat client.

    Starts from ALL_TOOLS, then applies optional enhancements:
    - If *image_client* is provided, replaces the default ``generate_image``
      singleton with one bound to that client (honoring *preview_every* and
      *image_steps*).
    - If *base_client* supports vision, appends a ``describe_image`` tool
      bound to the same client so the model can inspect generated images.
    - If *audio_client* is provided, replaces the default ``generate_audio``
      singleton with one bound to that client (honoring *audio_steps*).
    - If *speech_client* is provided, replaces the default ``generate_speech``
      singleton with one bound to that client.
    - If *memory_store* is provided, appends ``store_memory``, ``search_memories``,
      and ``list_memories`` tools bound to that store.
    - If *allow_code_execution* is ``True``, appends :func:`execute_python` (the
      subprocess-backed tool). Read its docstring first: it is isolation, not
      containment.
    """
    tools = list(ALL_TOOLS)
    if image_client is not None:
        bound = make_image_tool(image_client, preview_every=preview_every, num_inference_steps=image_steps)
        tools = [t for t in tools if t is not generate_image] + [bound]
    if getattr(base_client.model, "supports_vision", False):
        tools.append(make_describe_image_tool(base_client))
    if audio_client is not None:
        bound_audio = make_audio_tool(audio_client, num_inference_steps=audio_steps)
        tools = [t for t in tools if t is not generate_audio] + [bound_audio]
    if speech_client is not None:
        bound_speech = make_speech_tool(speech_client)
        tools = [t for t in tools if t is not generate_speech] + [bound_speech]
    if memory_store is not None:
        tools.extend(make_memory_tools(memory_store))
    if allow_code_execution:
        tools.append(execute_python)
    return tools


# ---- Audio generation --------------------------------------------------------
#
# Audio deps (``soundfile``, ``torch``, ``transformers``, ``diffusers``) are heavy.
# The client is constructed lazily on first ``generate_audio()`` call so that
# importing ``aimu.tools.builtin`` does not pull torch into ``sys.modules``.

_audio_client = None
# Reads AIMU_AUDIO_MODEL (any "hf:<repo_id>"); raises if unset, nothing downloaded implicitly.
_get_audio_client = _lazy_client_getter("_audio_client", "audio_client")


@tool
def generate_audio(prompt: str):
    """Generate an audio clip from a text prompt and return the saved file path.

    This is a **streaming tool**: a generator that yields
    :attr:`~aimu.models.StreamingContentType.AUDIO_GENERATING` chunks during
    generation, then returns the saved WAV file path. When called via the agent's
    streaming path (``agent.run(stream=True)``), each step chunk flows through
    the agent's own stream and into the chat UI live.

    Uses an :class:`aimu.AudioClient`. The model is controlled by the
    ``AIMU_AUDIO_MODEL`` env var (required; the tool raises if it is unset).
    Override per-agent by constructing your own tool with :func:`make_audio_tool`.

    Args:
        prompt: A description of the desired audio (e.g. "upbeat lo-fi jazz loop").
    """
    final_result = None
    for chunk in _get_audio_client().generate(prompt, format="path", stream=True):
        yield chunk
        content = chunk.content
        if isinstance(content, dict) and content.get("final"):
            final_result = content.get("result")
    return final_result


def make_audio_tool(client, *, duration_s: Optional[float] = None, num_inference_steps: Optional[int] = None):
    """Build a ``generate_audio`` tool bound to a specific audio client.

    ``client`` may be an :class:`aimu.AudioClient` or any concrete
    :class:`aimu.BaseAudioClient` (e.g. :class:`HuggingFaceAudioClient`). Use this
    when an agent needs a different model from the default singleton, or to fix the
    generation duration via ``duration_s`` or the denoising step count via
    ``num_inference_steps`` (diffusers models only).

    The returned tool is a **streaming tool** (generator); its progress chunks
    flow through ``agent.run(stream=True)`` for live UI updates.

    Example::

        client = aimu.audio_client(aimu.HuggingFaceAudioModel.MUSICGEN_MEDIUM)
        my_tool = make_audio_tool(client, duration_s=15, num_inference_steps=100)
        agent = Agent(text_client, tools=[my_tool])
    """

    @tool
    def generate_audio(prompt: str):
        """Generate an audio clip from a text prompt and return the saved file path.

        Streams per-step progress chunks during generation (diffusers models only).

        Args:
            prompt: A description of the desired audio.
        """
        final_result = None
        kw = {"format": "path", "stream": True}
        if duration_s is not None:
            kw["duration_s"] = duration_s
        if num_inference_steps is not None:
            kw["num_inference_steps"] = num_inference_steps
        for chunk in client.generate(prompt, **kw):
            yield chunk
            content = chunk.content
            if isinstance(content, dict) and content.get("final"):
                final_result = content.get("result")
        return final_result

    return generate_audio


# ---- Speech generation (TTS) ------------------------------------------------
#
# Speech deps (``soundfile``, ``torch``, ``transformers``) and the ``openai``
# SDK are heavy. The client is constructed lazily on first ``generate_speech()``
# call so that importing ``aimu.tools.builtin`` does not pull them into
# ``sys.modules``.

_speech_client = None
# Reads AIMU_SPEECH_MODEL (any "provider:model_id"); raises if unset, nothing downloaded implicitly.
_get_speech_client = _lazy_client_getter("_speech_client", "speech_client")


@tool
def generate_speech(text: str):
    """Synthesise speech from text and return the saved WAV file path.

    This is a **streaming tool**: a generator that yields
    :attr:`~aimu.models.StreamingContentType.SPEECH_GENERATING` chunks during
    generation, then returns the saved WAV file path.

    Uses a :class:`aimu.SpeechClient`. The model is controlled by the
    ``AIMU_SPEECH_MODEL`` env var (required; the tool raises if it is unset).
    Override per-agent by constructing your own tool with :func:`make_speech_tool`.

    Args:
        text: The text to speak (e.g. "Hello, world!").
    """
    final_result = None
    for chunk in _get_speech_client().generate(text, format="path", stream=True):
        yield chunk
        content = chunk.content
        if isinstance(content, dict) and content.get("final"):
            final_result = content.get("result")
    return final_result


def make_speech_tool(client, *, voice: Optional[str] = None, speed: Optional[float] = None):
    """Build a ``generate_speech`` tool bound to a specific speech client.

    ``client`` may be a :class:`aimu.SpeechClient` or any concrete
    :class:`aimu.BaseSpeechClient`. Use this when an agent needs a different
    model or voice from the default singleton.

    The returned tool is a **streaming tool**; its progress chunks flow through
    ``agent.run(stream=True)`` for live UI updates.

    Example::

        client = aimu.speech_client("openai:tts-1-hd")
        my_tool = make_speech_tool(client, voice="nova")
        agent = Agent(text_client, tools=[my_tool])
    """

    @tool
    def generate_speech(text: str):
        """Synthesise speech from text and return the saved WAV file path.

        Streams progress chunks during generation.

        Args:
            text: The text to speak.
        """
        final_result = None
        kw = {"format": "path", "stream": True}
        if voice is not None:
            kw["voice"] = voice
        if speed is not None:
            kw["speed"] = speed
        for chunk in client.generate(text, **kw):
            yield chunk
            content = chunk.content
            if isinstance(content, dict) and content.get("final"):
                final_result = content.get("result")
        return final_result

    return generate_speech


# ---- Memory (semantic or document store) ------------------------------------
#
# Memory tools require an explicit store instance; there is no lazy singleton
# because persistence semantics (ephemeral vs. on-disk, semantic vs. document,
# persist_path, collection name) are caller-controlled. Use make_memory_tools()
# to get @tool-decorated functions that close over your store instance.


def make_memory_tools(store):
    """Build ``store_memory``, ``search_memories``, and ``list_memories`` tools bound to *store*.

    *store* may be any :class:`aimu.memory.MemoryStore` implementation:
    :class:`~aimu.memory.SemanticMemoryStore` (ChromaDB vector search),
    :class:`~aimu.memory.DocumentStore` (path-keyed), or a custom subclass.

    Unlike the image/audio/speech tools, there is no env-var singleton for memory
    because the choice of store (ephemeral vs. persistent, which persist_path) is
    meaningful and should be explicit. Construct the store, pass it here, and add
    the returned list to the agent::

        store = SemanticMemoryStore(persist_path="./.memory")
        agent = Agent(client, tools=make_memory_tools(store) + builtin.web)

    For cross-process or multi-agent memory, use the FastMCP servers in
    ``aimu.memory.mcp`` / ``aimu.memory.document_mcp`` instead.
    """

    @tool
    def store_memory(content: str) -> str:
        """Store a piece of information in memory for later retrieval.

        Args:
            content: The information to remember (a fact, note, or observation).
        """
        store.store(content)
        return "Stored."

    @tool
    def search_memories(query: str, n_results: int = 5) -> str:
        """Search memory for information relevant to a query.

        Args:
            query: The topic or question to search for.
            n_results: Maximum number of results to return (default 5).
        """
        results = store.search(query, n_results=n_results)
        if not results:
            return "No relevant memories found."
        return "\n".join(f"- {r}" for r in results)

    @tool
    def list_memories() -> str:
        """Return all stored memories."""
        items = store.list_all()
        if not items:
            return "Memory is empty."
        return "\n".join(f"- {item}" for item in items)

    return [store_memory, search_memories, list_memories]


def make_retrieval_tool(store, *, n_results: int = 5):
    """Build a ``retrieve_context`` tool bound to *store* for retrieval-augmented agents.

    *store* may be any :class:`aimu.memory.MemoryStore` (typically a
    :class:`~aimu.memory.SemanticMemoryStore` populated via :func:`aimu.rag.ingest`).
    The tool runs :func:`aimu.rag.retrieve` and returns the joined context, letting an
    agent fetch relevant background on demand::

        from aimu.rag import ingest
        store = SemanticMemoryStore()
        ingest(store, my_documents)
        agent = Agent(client, tools=[make_retrieval_tool(store)])

    Like :func:`make_memory_tools`, there is no env-var singleton; the store (and what was
    ingested into it) is a meaningful, explicit choice.
    """

    @tool
    def retrieve_context(query: str) -> str:
        """Retrieve background context relevant to a query from the knowledge base.

        Args:
            query: The topic or question to look up.
        """
        from aimu.rag import format_context, retrieve

        chunks = retrieve(store, query, n_results=n_results)
        if not chunks:
            return "No relevant context found."
        return format_context(chunks, numbered=True)

    return retrieve_context


# A whole document per match turns one search over a corpus of papers into a context-window
# flood. Excerpt instead, and name the tool that returns the rest.
SEARCH_EXCERPT_CHARS = 1500


def _excerpt(content: str, path: str) -> str:
    """Return *content*, or its head plus a pointer to the tool that returns the whole thing."""
    if len(content) <= SEARCH_EXCERPT_CHARS:
        return content
    return (
        content[:SEARCH_EXCERPT_CHARS]
        + f"\n... (excerpt of {len(content)} characters; "
        + f"call read_document('{path}') for the full text)"
    )


def make_document_tools(store):
    """Build ``save_document``, ``read_document``, ``list_documents``, and ``search_documents`` tools.

    Bound to a :class:`~aimu.memory.DocumentStore` (the path-keyed store), these expose its
    richer write/read/list/full-text-search API as agent tools. Their names are deliberately
    distinct from :func:`make_memory_tools` (``store_memory`` / ``search_memories`` / ``list_memories``)
    so an agent can carry **both** at once: a `SemanticMemoryStore` for short facts and a
    `DocumentStore` for longer documents the user provides as reference::

        facts = SemanticMemoryStore(persist_path="./.memory")
        docs = DocumentStore(persist_path="./.documents")
        agent = Agent(client, tools=make_memory_tools(facts) + make_document_tools(docs))

    For cross-process or multi-agent document memory, use the FastMCP server in
    ``aimu.memory.document_mcp`` instead.
    """

    @tool
    def save_document(path: str, content: str) -> str:
        """Save a document at a path for later retrieval (create or overwrite).

        Args:
            path: A document path, e.g. "/notes/standup.md".
            content: The full text of the document.
        """
        store.write(path, content)
        return f"Saved {path}."

    @tool
    def read_document(path: str) -> str:
        """Read the full contents of a stored document by path.

        Args:
            path: The document path to read, e.g. "/notes/standup.md".
        """
        try:
            return store.read(path)
        except KeyError:
            return f"No document found at {path}."

    @tool
    def list_documents() -> str:
        """List the paths of all stored documents.

        Call this before concluding that a document does not exist: the store is a directory the
        user can add files to directly, so it may hold documents you did not save yourself.
        """
        paths = store.list_paths()
        # Files the user placed in the directory that are not text can never become documents.
        # Reporting them here is the only way the model can tell the user why one is missing.
        unreadable = store.unreadable_paths() if hasattr(store, "unreadable_paths") else []
        listing = "\n".join(f"- {p}" for p in paths) if paths else "No documents stored."
        if unreadable:
            listing += (
                f"\n\nNote: {len(unreadable)} file(s) in the document directory could not be read, "
                "because documents must be UTF-8 text: "
                + ", ".join(unreadable)
                + ". Tell the user to export these to Markdown or plain text."
            )
        return listing

    @tool
    def search_documents(query: str, n_results: int = 5) -> str:
        """Search stored documents for a query, returning an excerpt of each match.

        Use this to locate which document discusses something. Long matches are truncated, so
        call `read_document` on a path when you need to analyze or summarize its full text.

        Args:
            query: The text to search for.
            n_results: Maximum number of matching documents to return (default 5).
        """
        matches = store.search_full_text(query, n_results=n_results)
        if not matches:
            return "No matching documents found."
        return "\n\n".join(f"{m['path']}:\n{_excerpt(m['content'], m['path'])}" for m in matches)

    return [save_document, read_document, list_documents, search_documents]


DEFAULT_SUBAGENT_SYSTEM_MESSAGE = (
    "You are a focused sub-agent handling one delegated subtask in isolation. You do not share "
    "memory or conversation history with other agents. Use the tools available to you to complete "
    "the task, and return a single complete, self-contained answer as plain text."
)


def _subagent_first_line(spec: dict) -> str:
    lines = spec["system_message"].strip().splitlines()
    return lines[0] if lines else "(no description)"


def _subagent_docstring(agent_types: Optional[dict[str, dict]]) -> str:
    """Build the ``spawn_subagent`` docstring; its first paragraph becomes the model-facing description.

    In typed mode the menu of ``agent_type`` names is folded into that first paragraph (the ``@tool``
    decorator keeps only the first paragraph), so the model sees the available specialists.
    """
    if agent_types is None:
        return (
            "Delegate an independent subtask to a fresh general-purpose sub-agent with its own "
            "isolated context, and return its final answer. The sub-agent shares no conversation "
            "history with you or with other sub-agents, so give it a complete, self-contained task. "
            "Emit multiple calls to this tool in a single turn to run subtasks in parallel.\n\n"
            "Args:\n"
            "    task: A complete, self-contained description of the subtask to delegate."
        )
    menu = "; ".join(f"{name} — {_subagent_first_line(spec)}" for name, spec in sorted(agent_types.items()))
    return (
        "Delegate an independent subtask to a fresh specialized sub-agent (chosen by agent_type) and "
        "return its final answer; each runs in isolation with its own context and tools. Emit multiple "
        "calls to this tool in a single turn to run subtasks in parallel. "
        f"Available agent_type values: {menu}.\n\n"
        "Args:\n"
        "    agent_type: Which specialist to delegate to (one of the names listed above).\n"
        "    task: A complete, self-contained description of the subtask to delegate."
    )


# Every key a typed ``agent_types`` spec may carry. Closed rather than open because an ignored key reads
# exactly like an applied one: a misspelled ``"thinkng"`` or a hopeful ``"temperature"`` would leave the
# spawned agent at its default with nothing raised anywhere, and the caller believing otherwise.
SUBAGENT_SPEC_KEYS = frozenset({"system_message", "tools", "model", "thinking", "generate_kwargs"})


def _validate_subagent_config(max_depth: int, agent_types: Optional[dict[str, dict]]) -> None:
    """Raise ``ValueError`` for programmer errors at factory-call time (failures apparent)."""
    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1 (got {max_depth}).")
    if agent_types is not None:
        if not agent_types:
            raise ValueError("agent_types must be a non-empty dict, or None for a generic sub-agent.")
        for type_name, spec in agent_types.items():
            if "system_message" not in spec:
                raise ValueError(f"agent_types[{type_name!r}] is missing the required 'system_message' key.")
            unknown = set(spec) - SUBAGENT_SPEC_KEYS
            if unknown:
                raise ValueError(
                    f"agent_types[{type_name!r}] has unknown key(s): {', '.join(sorted(unknown))}. "
                    f"A spec may carry: {', '.join(sorted(SUBAGENT_SPEC_KEYS))}."
                )


def make_subagent_tool(
    model,
    *,
    system_message: str = DEFAULT_SUBAGENT_SYSTEM_MESSAGE,
    tools: Optional[list[Callable]] = None,
    agent_types: Optional[dict[str, dict]] = None,
    max_depth: int = 1,
    max_iterations: int = 10,
    concurrent_tool_calls: bool = True,
    deps: Any = None,
    tool_approval: Optional[Callable] = None,
    tool_name: str = "spawn_subagent",
    events: Optional[EventSink] = None,
) -> Callable:
    """Build a ``spawn_subagent`` tool that delegates subtasks to fresh, isolated sub-agents.

    This is AIMU's answer to dynamic sub-agent spawning (as in Claude Code's ``Task`` tool): the
    LLM decides at runtime to hand an independent subtask to a brand-new :class:`~aimu.agents.Agent`
    with its own context, rather than choosing among a fixed roster. It is the *dynamic complement*
    to :class:`~aimu.agents.OrchestratorAgent` (which wires a known set of workers up front); both
    reduce to ``subagent.run(task)`` — the difference is who decides the roster.

    Each invocation builds a fresh ``ModelClient(model)`` (its own message history — the
    :func:`aimu.agents.prebuilt._base.make_workers` isolation idiom), so concurrent spawns share no
    state. **Parallelism is free**: give the *parent* ``Agent`` ``concurrent_tool_calls=True`` and,
    when the model emits several ``spawn_subagent`` calls in one turn, they run concurrently
    (``ThreadPoolExecutor``). Genuine overlap is for cloud models — a single local model serializes on
    the GIL/CUDA, and a shared ``deps`` object handed to concurrent spawns must be thread-safe.

    Two shapes, chosen by ``agent_types``:

    * Generic (``agent_types=None``): the tool is ``spawn_subagent(task)`` — a fresh general-purpose
      sub-agent using ``system_message`` + ``tools``.
    * Typed (``agent_types`` given): the tool is ``spawn_subagent(agent_type, task)`` over a registry
      of named specialists (each value a dict with ``"system_message"`` and optional ``"tools"`` /
      ``"model"`` / ``"thinking"`` / ``"generate_kwargs"``, and nothing else -- an unrecognized spec key
      raises at factory-call time rather than being ignored, since an ignored key reads exactly like an
      applied one); the available names are listed in the tool description. An unknown ``agent_type``, by
      contrast, is returned to the model as a tool result (self-correction), not raised: that one is the
      model's mistake to recover from, where a bad spec key is the programmer's.
      ``"thinking"`` takes the same values as :class:`~aimu.agents.Agent`'s field and is read with
      ``.get()``, so a spec omitting it leaves the spawned agent at ``None`` rather than inheriting
      anything from the caller: unlike ``"model"``, which falls back to the model this factory was
      built with, there is no factory-level thinking tier to fall back to.
      ``"generate_kwargs"`` is a dict assigned to the spawned client's ``default_generate_kwargs``, so it
      applies to every request that sub-agent makes. Like ``"thinking"`` and unlike ``"model"``, an
      omitted key inherits nothing: there is no factory-level generation tier, so a caller with one
      default across a roster writes it into each spec. Only the keys a spec names are set, which matters
      because this tier sits *above* the model card in the precedence chain -- a filled-in default would
      shadow a card's own tuned profile.

    ``max_depth`` (default 1) is the recursion guard: it counts the caller's agent as level 1, so the
    default gives spawned sub-agents *no* spawn tool of their own. ``max_depth=2`` lets one more level
    spawn, and so on; the nested tool is rebuilt with a decremented depth, so recursion is finite.

    Args:
        model: A ``Model`` enum member, a ``"provider:model_id"`` string, or a ``BaseModelClient``
            to clone the model from (a fresh client is built per spawn regardless — the live client
            is never shared).
        system_message: Persona for generic sub-agents.
        tools: Tools each generic sub-agent receives (``None`` = text-only).
        agent_types: Optional registry that switches the tool to typed mode.
        max_depth: Spawn levels permitted, counting the caller's agent as 1 (must be >= 1).
        max_iterations: Tool-loop cap forwarded to each spawned agent.
        concurrent_tool_calls: Applied to *spawned* agents (so nested spawns overlap). The parent's
            own concurrency is set by its author.
        deps: ``ToolContext.deps`` passed to each spawned agent.
        tool_approval: Callback ``(name, arguments) -> bool`` (may be a coroutine) run before each of
            the sub-agent's tool calls; returning False appends a refusal instead of executing the tool.
            Matches :class:`~aimu.agents.Agent`/:meth:`~aimu.agents.Agent.run`'s ``tool_approval`` semantics.
        tool_name: Name of the produced tool (mint several differently-named spawn tools on one agent).
        events: The sink each spawned child reports to. It has to be passed explicitly: a spawn
            builds a fresh client per call, which is outside the client family a caller's scoped
            per-run override reaches, so a delegated run is otherwise invisible to a caller
            measuring the whole turn. Set on the child :class:`~aimu.agents.Agent` (its own
            ``events`` field), not passed to its ``run``.

    Example::

        from aimu.tools.builtin import make_subagent_tool, web

        spawn = make_subagent_tool("anthropic:claude-sonnet-4-6", tools=web)
        agent = Agent(client, "Break the request into subtasks and spawn a sub-agent for each.",
                      tools=[spawn], concurrent_tool_calls=True)
        print(agent.run("Compare the GDP growth of France, Japan, and Brazil since 2019."))
    """
    from aimu.models.base import BaseModelClient

    _validate_subagent_config(max_depth, agent_types)

    # Normalize to an enum/string the sub-agent client is rebuilt from each call (never share a live client).
    default_model = model.model if isinstance(model, BaseModelClient) else model

    def _build_agent(
        sys_msg: str,
        agent_tools: Optional[list[Callable]],
        name: str,
        model_override=None,
        thinking=None,
        generate_kwargs=None,
    ):
        from aimu.agents.agent import Agent
        from aimu.models.model_client import ModelClient

        m = model_override if model_override is not None else default_model
        child_tools = list(agent_tools or [])
        if max_depth > 1:
            child_tools.append(
                make_subagent_tool(
                    m,
                    system_message=system_message,
                    tools=tools,
                    agent_types=agent_types,
                    max_depth=max_depth - 1,
                    max_iterations=max_iterations,
                    concurrent_tool_calls=concurrent_tool_calls,
                    deps=deps,
                    tool_approval=tool_approval,
                    tool_name=tool_name,
                    events=events,
                )
            )
        client = ModelClient(m)
        if generate_kwargs:
            # Copied, not aliased: the spec's dict is reused by every spawn of this agent_type, and a
            # client that later mutates its own defaults would otherwise edit the roster.
            client.default_generate_kwargs = dict(generate_kwargs)
        return Agent(
            client,
            system_message=sys_msg,
            name=name,
            tools=child_tools,
            max_iterations=max_iterations,
            concurrent_tool_calls=concurrent_tool_calls,
            deps=deps,
            tool_approval=tool_approval,
            thinking=thinking,
            events=events,
        )

    if agent_types is None:

        def spawn_subagent(task: str) -> str:
            return _build_agent(system_message, tools, name="subagent").run(task)

    else:

        def spawn_subagent(agent_type: str, task: str) -> str:
            spec = agent_types.get(agent_type)
            if spec is None:
                return (
                    f"Unknown agent_type {agent_type!r}. Available agent_type values: {', '.join(sorted(agent_types))}."
                )
            agent = _build_agent(
                spec["system_message"],
                spec.get("tools", tools),
                name=f"subagent-{agent_type}",
                model_override=spec.get("model"),
                thinking=spec.get("thinking"),
                generate_kwargs=spec.get("generate_kwargs"),
            )
            return agent.run(task)

    spawn_subagent.__name__ = tool_name
    spawn_subagent.__qualname__ = tool_name
    spawn_subagent.__doc__ = _subagent_docstring(agent_types)
    return tool(spawn_subagent)


def make_web_tools(
    *,
    session=None,
    timeout: int = 15,
    max_content_chars: int = 20000,
    user_agent: str = _DEFAULT_USER_AGENT,
):
    """Build ``find_forms`` and ``submit_form`` tools sharing a ``requests.Session``.

    The shared session preserves cookies across calls, so a GET-then-POST form flow
    works: ``find_forms`` scrapes hidden fields (including CSRF tokens) from the page,
    then ``submit_form`` echoes them back with the session's cookies intact. Pass these
    to an agent alongside the stateless ``get_webpage_html``::

        agent = Agent(client, tools=[get_webpage_html, *make_web_tools()])

    Pass your own *session* to control its lifecycle (e.g. pre-set auth headers) or to
    share one across several tool sets; otherwise a fresh ``requests.Session`` is created.

    Note: these fetch server-rendered HTML only and do not execute JavaScript, so
    JS-rendered (SPA) forms and anti-bot-protected pages are out of scope; a headless
    browser backend is a possible future addition.

    ``submit_form`` performs writes (POST). To require confirmation before it runs, gate
    it via the ``tool_approval`` hook, e.g. ``Agent(..., tool_approval=policy)`` where the
    policy inspects the tool name (see docs/how-to/gate-tool-calls.md).
    """
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", user_agent)

    @tool
    def find_forms(url: str) -> str:
        """Fetch a web page and list its HTML forms and fields.

        Returns each form's index, submission URL, HTTP method, and fields (name, type,
        and current value), including hidden fields such as CSRF tokens. Use the reported
        method and URL with ``submit_form``, echoing back any hidden field values.

        Args:
            url: The URL of the page containing the form(s).
        """
        try:
            response = _fetch_html(url, session=session, timeout=timeout)
        except requests.RequestException as e:
            return f"Error fetching page: {e}"
        return _format_forms(_parse_forms(response.text, response.url))

    @tool
    def submit_form(url: str, method: str = "POST", data: Optional[dict] = None) -> str:
        """Submit data to a URL via GET or POST, reusing the shared session's cookies.

        ``method="POST"`` sends *data* as a form-encoded body; ``method="GET"`` sends it
        as query parameters (and, with no data, simply fetches the page using the session's
        cookies, e.g. to read a page behind a login). Returns the response status, final URL
        after redirects, and the response body (raw HTML, truncated).

        Args:
            url: The URL to submit to (the form's action URL from ``find_forms``).
            method: "POST" (default) or "GET".
            data: Mapping of field names to values to submit.
        """
        method = method.upper()
        if method not in ("GET", "POST"):
            return f"Unsupported method {method!r}; use 'GET' or 'POST'."
        payload = "params" if method == "GET" else "data"
        try:
            response = _fetch_html(url, session=session, timeout=timeout, method=method, **{payload: data or {}})
        except requests.RequestException as e:
            return f"Error submitting form: {e}"
        body = _truncate(response.text, max_content_chars)
        return f"Status: {response.status_code}\nURL: {response.url}\n\n{body}"

    return [find_forms, submit_form]


# Curated subsets: pass one of these to ``tools=`` instead of importing every function.
web = [get_weather, get_webpage, get_webpage_html, web_search, wikipedia]
fs = [list_directory, read_file]
compute = [calculate, execute_python, run_command]
# Grouped apart from ``misc`` because an agent almost always wants a clock regardless of its role: an
# assistant scoped to filesystem work still has to resolve "by tomorrow morning". Bundled with ``echo``
# it could only be granted alongside it.
# Note: this name shadows the stdlib ``time`` module. Nothing here imports it (date work uses
# ``datetime``), and adding such an import would be silently rebound by this assignment.
time = [get_current_date_and_time, convert_time]
misc = [echo]
image = [generate_image]
audio = [generate_audio]
speech = [generate_speech]

# ---------------------------------------------------------------------------
# Transcription (speech-to-text)
# ---------------------------------------------------------------------------

# Lazy singleton for the built-in transcription tool. Populated on first call
# via AIMU_TRANSCRIPTION_MODEL; raises if unset, nothing downloaded implicitly.
_transcription_client = None
_get_transcription_client = _lazy_client_getter("_transcription_client", "transcription_client")


@tool
def transcribe_audio(audio_path: str) -> str:
    """Transcribe speech from an audio file to text.

    audio_path is a local file path (wav, mp3, ogg, flac, m4a, webm).
    Returns the transcribed text as a plain string.

    Args:
        audio_path: Path to the audio file to transcribe.
    """
    return _get_transcription_client().transcribe(audio_path)


def make_transcription_tool(client):
    """Return a ``transcribe_audio`` tool bound to *client*.

    Use this when you want to control which transcription model the agent uses
    rather than relying on the ``AIMU_TRANSCRIPTION_MODEL`` singleton.

    Example::

        from aimu.models import TranscriptionClient, OpenAITranscriptionModel
        tc = TranscriptionClient(OpenAITranscriptionModel.WHISPER_1)
        agent = Agent(text_client, tools=[make_transcription_tool(tc)])
    """

    @tool
    def transcribe_audio(audio_path: str) -> str:
        """Transcribe speech from an audio file to text.

        Args:
            audio_path: Path to the audio file to transcribe.
        """
        return client.transcribe(audio_path)

    return transcribe_audio


transcription = [transcribe_audio]

# execute_python (and its explicit in-process opt-in, execute_python_in_process) and
# run_command are excluded from ALL_TOOLS: isolation, not containment, so they're opt-in.
# execute_python via builtin.compute or make_tools(allow_code_execution=True); run_command
# via builtin.compute only, since that flag says *code* execution and widening it would
# hand a shell to every caller already passing it.
ALL_TOOLS = [
    *misc,
    *time,
    get_weather,
    calculate,
    get_webpage,
    get_webpage_html,
    web_search,
    wikipedia,
    *fs,
    *image,
    *audio,
    *speech,
    *transcription,
]
