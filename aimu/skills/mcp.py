from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from fastmcp import FastMCP

from aimu.skills.manager import SkillManager


def build_skills_server(manager: SkillManager) -> FastMCP:
    """
    Build an in-process FastMCP server from a SkillManager.

    Registered tools:
      - activate_skill(name): returns the full SKILL.md body for the named skill
      - {skill_name}__{script_stem}(): runs a Python script from a skill's scripts/ dir

    The returned FastMCP instance can be passed directly to MCPClient(server=...).
    """
    server = FastMCP("AIMU Skills")

    # Capture manager in closure
    _manager = manager

    @server.tool()
    def activate_skill(name: str) -> str:
        """Load the full instructions for a named agent skill."""
        return _manager.get_skill_body(name)

    for skill in manager.skills.values():
        _register_script_tools(server, skill.name, skill.base_dir / "scripts")

    return server


def _register_script_tools(server: FastMCP, skill_name: str, scripts_dir: Path) -> None:
    """Register each *.py file in scripts_dir as a tool on server."""
    if not scripts_dir.exists() or not scripts_dir.is_dir():
        return

    for script in sorted(scripts_dir.glob("*.py")):
        _register_script_tool(server, skill_name, script)


def _register_script_tool(server: FastMCP, skill_name: str, script: Path) -> None:
    """Register a single Python script as a FastMCP tool."""
    tool_name = f"{skill_name}__{script.stem}"
    script_path = str(script.resolve())

    # Build the tool function dynamically so each closure captures its own script_path
    def _make_tool(path: str):
        def run_script() -> str:
            try:
                result = subprocess.run(
                    [sys.executable, path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except subprocess.TimeoutExpired:
                return f"Script timed out after 30 seconds."
            if result.returncode != 0:
                return f"Script exited with code {result.returncode}.\nstdout: {result.stdout}\nstderr: {result.stderr}"
            return result.stdout

        run_script.__name__ = tool_name
        run_script.__doc__ = f"Run the {script.name} script from the '{skill_name}' skill."
        return run_script

    server.tool()(_make_tool(script_path))
