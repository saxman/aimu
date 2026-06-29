from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from fastmcp import FastMCP

from aimu.skills.manager import SkillManager

# Skill scripts run as real subprocesses with the user's own privileges (no sandbox). This is
# intentional: a skill bundles executable helpers the agent is trusted to run. Discovery covers
# Python and shell scripts; the interpreter is chosen by file extension.
_SCRIPT_GLOBS = ("*.py", "*.sh")
_SCRIPT_TIMEOUT = 30


def _interpreter_for(script: Path) -> list[str]:
    """Return the argv prefix to run ``script``, chosen by extension.

    Raises :class:`ValueError` for an unsupported extension or a missing shell interpreter.
    """
    if script.suffix == ".py":
        return [sys.executable]
    if script.suffix == ".sh":
        bash = shutil.which("bash")
        if bash is None:
            raise ValueError("cannot run .sh script: 'bash' not found on PATH")
        return [bash]
    raise ValueError(f"unsupported script extension {script.suffix!r} (expected .py or .sh)")


def run_script_file(script: Path, args: str = "", *, fix_hint: str = "") -> str:
    """Run ``script`` with the interpreter for its extension and return its output.

    ``args`` is a shell-style string split with :func:`shlex.split` and appended to the script's
    argv. Returns stdout on success, or a formatted error string (non-zero exit, timeout,
    unsupported extension, bad quoting). ``fix_hint`` is appended to error results to tell the
    caller how to repair the script (the skills server passes the script's skill + filename, so a
    broken script can be overwritten in place rather than re-created under a new name). Note: this
    blocks for up to ``_SCRIPT_TIMEOUT`` seconds, which blocks the event loop on the async path;
    acceptable for an occasional skill invocation.
    """
    try:
        argv = _interpreter_for(script) + [str(script.resolve())] + shlex.split(args)
    except ValueError as exc:
        return str(exc)
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=_SCRIPT_TIMEOUT)
    except subprocess.TimeoutExpired:
        return f"Script timed out after {_SCRIPT_TIMEOUT} seconds.{fix_hint}"
    if result.returncode != 0:
        return (
            f"Script exited with code {result.returncode}.\nstdout: {result.stdout}\nstderr: {result.stderr}{fix_hint}"
        )
    return result.stdout


def build_skills_server(manager: SkillManager) -> FastMCP:
    """
    Build an in-process FastMCP server from a SkillManager.

    Registered tools:
      - activate_skill(name): returns the full SKILL.md body for the named skill
      - {skill_name}__{script_stem}(args=""): runs a Python or shell script from a skill's scripts/ dir

    The returned FastMCP instance can be passed directly to MCPClient(server=...).
    """
    server = FastMCP("AIMU Skills")

    # Capture manager in closure
    _manager = manager

    @server.tool()
    def activate_skill(name: str) -> str:
        """Load the full instructions for a named agent skill."""
        from aimu.skills.manager import SkillNotFoundError

        try:
            return _manager.get_skill_body(name)
        except SkillNotFoundError as exc:
            return str(exc)

    for skill in manager.skills.values():
        _register_script_tools(server, skill.name, skill.base_dir / "scripts")

    return server


def _register_script_tools(server: FastMCP, skill_name: str, scripts_dir: Path) -> None:
    """Register each ``*.py`` / ``*.sh`` file in scripts_dir as a tool on server."""
    if not scripts_dir.exists() or not scripts_dir.is_dir():
        return

    scripts = sorted(p for glob in _SCRIPT_GLOBS for p in scripts_dir.glob(glob))
    seen: set[str] = set()
    for script in scripts:
        # foo.py and foo.sh map to the same {skill}__foo tool name; .py sorts first and wins.
        if script.stem in seen:
            continue
        seen.add(script.stem)
        _register_script_tool(server, skill_name, script)


def _register_script_tool(server: FastMCP, skill_name: str, script: Path) -> None:
    """Register a single script (.py or .sh) as a FastMCP tool."""
    tool_name = f"{skill_name}__{script.stem}"
    script_path = script.resolve()
    filename = script.name
    # On failure, name the exact skill + file so a fix overwrites this script in place rather than
    # being written to a new filename (which would leave the broken script and its tool behind).
    fix_hint = (
        f"\n\nTo fix this script, overwrite it in place: call add_skill_script with "
        f"skill_name='{skill_name}', filename='{filename}', and the corrected content (reuse the "
        f"same filename so it replaces this script rather than creating a new one)."
    )

    # Build the tool function dynamically so each closure captures its own script path.
    def _make_tool(path: Path):
        def run_script(args: str = "") -> str:
            return run_script_file(path, args, fix_hint=fix_hint)

        run_script.__name__ = tool_name
        run_script.__doc__ = (
            f"Run the {filename} script from the '{skill_name}' skill. "
            "Optional `args` is a shell-style string forwarded to the script's arguments. "
            f"If it is broken, fix it with add_skill_script(skill_name='{skill_name}', "
            f"filename='{filename}', ...) using the same filename."
        )
        return run_script

    server.tool()(_make_tool(script_path))
