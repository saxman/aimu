"""Author new Agent Skills at runtime.

AIMU *discovers* and *uses* skills via :class:`~aimu.skills.manager.SkillManager`; this
module is the inverse: it lets an agent *write* a new ``SKILL.md`` while running, so the
assistant grows reusable skills as it solves problems (the self-improvement pattern from
Hermes Agent). :func:`write_skill` is plain filesystem work shared by both surfaces;
:func:`make_skill_authoring_tool` wraps it as an async ``@tool`` an agent can call.
"""

from __future__ import annotations

import re
import stat
from pathlib import Path
from typing import Callable, Optional, Union

from aimu.skills.manager import SkillManager

# A skill name doubles as a directory name and a tool-name prefix, so restrict it to a
# safe slug: lowercase letters, digits, and single hyphens. This also blocks path
# traversal (no separators, no ``..``).
_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

# Scripts a skill may bundle. The extension selects the interpreter (.py -> python, .sh -> bash);
# the stem becomes the {skill}__{stem} tool name, so it allows lowercase letters, digits, and
# internal hyphens/underscores (matching common Python/shell filenames). No path separators.
_SCRIPT_EXTS = {".py", ".sh"}
_SCRIPT_STEM = re.compile(r"^[a-z0-9]+(?:[_-][a-z0-9]+)*$")


def _validate_script_filename(filename: str) -> None:
    """Raise :class:`ValueError` unless ``filename`` is ``<stem>.py`` or ``<stem>.sh``."""
    if "/" in filename or "\\" in filename or filename in (".", ".."):
        raise ValueError(f"invalid script filename {filename!r}: no path separators")
    stem = Path(filename).stem
    ext = Path(filename).suffix
    if ext not in _SCRIPT_EXTS:
        raise ValueError(f"invalid script {filename!r}: extension must be one of {sorted(_SCRIPT_EXTS)}")
    if not _SCRIPT_STEM.match(stem):
        raise ValueError(f"invalid script stem {stem!r}: use lowercase letters, digits, hyphens, or underscores")


def write_skill(
    name: str,
    description: str,
    body: str,
    *,
    skills_dir: Union[str, Path],
    overwrite: bool = False,
    metadata: Optional[dict] = None,
    scripts: Optional[dict[str, str]] = None,
) -> Path:
    """Write a new ``SKILL.md`` under ``skills_dir/<name>/`` and return its path.

    The file carries YAML frontmatter (``name``, ``description``, optional ``metadata``)
    followed by the markdown ``body``, matching the format
    :class:`~aimu.skills.manager.SkillManager` discovers.

    Validates that ``name`` is a slug (lowercase-with-hyphens, no path separators, which
    also prevents traversal) and that ``description`` is non-empty. Refuses to overwrite an
    existing skill unless ``overwrite=True``. The written file is round-tripped through the
    manager parser, so an authored skill is guaranteed discoverable (a parse failure raises
    :class:`~aimu.skills.manager.SkillLoadError`).

    ``scripts`` maps ``"<slug>.py"`` / ``"<slug>.sh"`` filenames to source, written into
    ``scripts/`` (each becomes a ``{skill}__{stem}`` tool). ``.sh`` files are marked executable.
    """
    if not _SLUG.match(name):
        raise ValueError(
            f"invalid skill name {name!r}: use a lowercase-with-hyphens slug "
            "(letters, digits, single hyphens; no spaces or path separators)"
        )
    if not description.strip():
        raise ValueError("skill description must be non-empty")

    skills_dir = Path(skills_dir).expanduser()
    skill_dir = skills_dir / name
    skill_md = skill_dir / "SKILL.md"
    if skill_md.exists() and not overwrite:
        raise FileExistsError(f"skill {name!r} already exists at {skill_md}; pass overwrite=True to replace it")

    # Validate every script filename up front so a bad name writes nothing.
    if scripts:
        for filename in scripts:
            _validate_script_filename(filename)

    skill_dir.mkdir(parents=True, exist_ok=True)

    frontmatter_lines = ["---", f"name: {name}", f"description: {description.strip()}"]
    if metadata:
        frontmatter_lines.append("metadata:")
        for key, value in metadata.items():
            frontmatter_lines.append(f"  {key}: {value}")
    frontmatter_lines.append("---")
    content = "\n".join(frontmatter_lines) + "\n\n" + body.strip() + "\n"

    skill_md.write_text(content, encoding="utf-8")

    # Round-trip through the parser so a malformed authored file fails loudly here, at the
    # write site, rather than silently later during discovery.
    SkillManager(skill_dirs=[str(skills_dir)])._parse(skill_md)

    if scripts:
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        for filename, source in scripts.items():
            # `overwrite` already gates the whole call via the SKILL.md check above, so a
            # script write here is either a fresh skill or an explicit overwrite (the
            # add_skill_script path), where replacing an existing script is intended.
            target = scripts_dir / filename
            target.write_text(source, encoding="utf-8")
            if target.suffix == ".sh":  # .py runs via the interpreter; only .sh needs +x
                target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    return skill_md


def make_skill_authoring_tool(manager: SkillManager, skills_dir: Union[str, Path]) -> Callable:
    """Return an async ``@tool`` that authors a skill and refreshes ``manager``.

    The tool writes a new ``SKILL.md`` under ``skills_dir`` via :func:`write_skill`, then
    calls :meth:`SkillManager.refresh` so the skill is discoverable in the same run. Both
    ``manager`` and ``skills_dir`` are captured by closure (no module globals).

    Note: after refresh, ``activate_skill`` (and any fresh-conversation catalog rebuild) will
    surface the new skill, but a skill catalog already injected into an in-flight system
    prompt is not retroactively updated. See :class:`~aimu.aio.SkillAgent`.
    """
    from aimu.tools import tool

    skills_dir = Path(skills_dir).expanduser()

    @tool
    async def author_skill(name: str, description: str, body: str) -> str:
        """Create a new reusable skill so you can apply it to future tasks.

        Use this after working out a repeatable procedure worth remembering. The skill is
        saved as instructions you can recall later.

        Args:
            name: Short slug identifying the skill (lowercase-with-hyphens, e.g. "format-standup").
            description: One line describing when to use the skill.
            body: Full markdown instructions for performing the skill.
        """
        path = write_skill(name, description, body, skills_dir=skills_dir, overwrite=False)
        manager.refresh()
        return f"Created skill '{name}' at {path}. It is now available."

    return author_skill


def make_skill_script_tool(agent, manager: SkillManager, skills_dir: Union[str, Path]) -> Callable:
    """Return an async ``@tool`` that adds a runnable script to an existing skill.

    The tool writes ``scripts/<filename>`` (``.py`` or ``.sh``) into the named skill via
    :func:`write_skill`, refreshes ``manager``, then calls ``await agent.reload_skills()`` so the
    new ``{skill}__{stem}`` tool is callable in the same turn. ``agent`` is a
    :class:`~aimu.aio.SkillAgent` (the tool is async); ``manager`` and ``skills_dir`` are captured
    by closure.

    **Full access**: the script runs as a real subprocess with the user's privileges, no sandbox.
    """
    from aimu.tools import tool

    skills_dir = Path(skills_dir).expanduser()

    @tool
    async def add_skill_script(skill_name: str, filename: str, content: str) -> str:
        """Attach a runnable script to an existing skill, then make it callable now.

        Scripts run with full access to this machine (no sandbox). Use this to automate a
        repeatable procedure as code you can invoke as a tool.

        Args:
            skill_name: Slug of an existing skill (create it first with author_skill).
            filename: Script file name, "<name>.py" or "<name>.sh" (lowercase-with-hyphens stem).
            content: Full source of the script.
        """
        skill = manager.skills.get(skill_name)
        if skill is None:
            return f"Skill {skill_name!r} not found. Create it first with author_skill."
        # Re-round-trip the existing SKILL.md unchanged plus the new script file.
        write_skill(
            skill_name,
            skill.description,
            skill.load_body(),
            skills_dir=skills_dir,
            overwrite=True,
            metadata=skill.metadata or None,
            scripts={filename: content},
        )
        manager.refresh()
        await agent.reload_skills()
        stem = Path(filename).stem
        return f"Added {filename} to '{skill_name}'. Tool '{skill_name}__{stem}' is now callable."

    return add_skill_script


__all__ = ["make_skill_authoring_tool", "make_skill_script_tool", "write_skill"]
