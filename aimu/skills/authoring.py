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

from aimu.skills.frontmatter import load_frontmatter, render_frontmatter, split_frontmatter
from aimu.skills.manager import SkillLoadError, SkillManager
from aimu.skills.skill import script_tool_name
from aimu.skills.validate import SkillSpecError, validate_frontmatter

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


def write_skill_script(name: str, filename: str, content: str, *, skills_dir: Union[str, Path]) -> Path:
    """Write ``skills_dir/<name>/scripts/<filename>`` and return its path.

    The unit of a script write, deliberately separate from :func:`write_skill`: attaching a script
    is not a change to the skill's own ``SKILL.md``, and rewriting that file to add one used to
    drop every frontmatter key :func:`write_skill` does not re-emit (``license``,
    ``compatibility``, ``allowed-tools``, and anything outside the spec).

    Validates the filename (``<stem>.py`` or ``<stem>.sh``, no path separators) before creating
    anything, and marks a ``.sh`` executable. An existing script of the same name is replaced,
    which is how a broken script is fixed: the ``{skill}__{stem}`` tool keeps its name.
    """
    _validate_script_filename(filename)
    scripts_dir = Path(skills_dir).expanduser() / name / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    target = scripts_dir / filename
    target.write_text(content, encoding="utf-8")
    if target.suffix == ".sh":  # .py runs via the interpreter; only .sh needs +x
        target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return target


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

    This is the **create** path, and it emits ``name``, ``description``, and ``metadata`` only. Aimed
    at an existing skill with ``overwrite=True`` it therefore drops the spec's optional ``license``,
    ``compatibility``, and ``allowed-tools``, along with any key outside the spec; :func:`update_skill`
    exists to revise a skill without that loss.
    """
    if not _SLUG.match(name):
        raise ValueError(
            f"invalid skill name {name!r}: use a kebab-case slug (lowercase words joined by hyphens, "
            "e.g. 'format-standup'); replace underscores or spaces with hyphens."
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

    fields = {"name": name, "description": description.strip()}
    if metadata:
        fields["metadata"] = dict(metadata)
    skill_md.write_text(_compose(fields, body), encoding="utf-8")

    # Round-trip through the parser so a malformed authored file fails loudly here, at the
    # write site, rather than silently later during discovery.
    SkillManager(skill_dirs=[str(skills_dir)])._parse(skill_md)

    for filename, source in (scripts or {}).items():
        write_skill_script(name, filename, source, skills_dir=skills_dir)

    return skill_md


def _compose(fields: dict, body: str) -> str:
    """Return the full text of a ``SKILL.md``: rendered frontmatter, blank line, stripped body."""
    return f"{render_frontmatter(fields)}\n\n{body.strip()}\n"


def update_skill(
    name: str,
    *,
    description: Optional[str] = None,
    body: Optional[str] = None,
    skills_dir: Union[str, Path],
) -> Path:
    """Revise an existing skill's description, body, or both, and return its ``SKILL.md`` path.

    The counterpart to :func:`write_skill`, which refuses to clobber, so before this there was no
    route to a skill's prose once it existed: an agent could fix a skill's *scripts* forever while
    the instructions a first attempt most often gets wrong stayed frozen.

    Only what is passed is changed. Everything else in the file is read and written back as it was,
    including the spec's optional ``license`` / ``compatibility`` / ``allowed-tools``, ``metadata``,
    and any key outside the spec, which is the whole reason this is not ``write_skill(overwrite=True)``.
    ``metadata`` is deliberately *not* a parameter: it is a host's provenance record (an installer's
    ``author`` field), and an update path that could rewrite it would be more capable than the create
    path, which cannot set it either.

    There is no rename: ``name`` locates the skill and is never written as new. It is the skill's
    address (its directory, its catalogue entry, and the prefix of every ``{skill}__{stem}`` script
    tool), and the spec requires the frontmatter name and the directory name to agree.

    Raises :class:`FileNotFoundError` if the skill does not exist, :class:`ValueError` if neither
    field is given or the new description is blank, and
    :class:`~aimu.skills.manager.SkillLoadError` if the file on disk cannot be parsed. Nothing is
    written unless every check passes, and the result is round-tripped through the manager parser
    the way :func:`write_skill`'s output is.
    """
    if description is None and body is None:
        raise ValueError("nothing to update: pass a new description, a new body, or both")
    if description is not None and not description.strip():
        raise ValueError("skill description must be non-empty")

    skills_dir = Path(skills_dir).expanduser()
    skill_md = skills_dir / name / "SKILL.md"
    if not skill_md.is_file():
        raise FileNotFoundError(f"skill {name!r} has no SKILL.md at {skill_md}; create it first with author_skill")

    frontmatter, current_body = split_frontmatter(skill_md.read_text(encoding="utf-8"))
    if frontmatter is None:
        raise SkillLoadError(f"{skill_md}: missing or unclosed YAML frontmatter, so there is nothing to update")
    try:
        fields = load_frontmatter(frontmatter)
    except ValueError as exc:
        raise SkillLoadError(f"{skill_md}: {exc}") from exc

    if description is not None:
        fields["description"] = description.strip()
    try:
        validate_frontmatter(fields, directory_name=name)
    except SkillSpecError as exc:
        raise SkillLoadError(f"{skill_md}: {exc}") from exc

    skill_md.write_text(_compose(fields, body if body is not None else current_body), encoding="utf-8")

    # Same round-trip guard write_skill uses: a file this wrote must be discoverable.
    SkillManager(skill_dirs=[str(skills_dir)])._parse(skill_md)
    return skill_md


# The tool make_skill_update_tool returns is itself named ``update_skill`` (a tool's name is its
# function's name), which makes the public function above unreachable by that name inside the factory.
_update_skill = update_skill


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
            name: A short kebab-case slug for the skill: lowercase words joined by hyphens,
                e.g. "format-standup". Use hyphens, never underscores or spaces.
            description: One line describing when to use the skill.
            body: Full markdown instructions for performing the skill.
        """
        path = write_skill(name, description, body, skills_dir=skills_dir, overwrite=False)
        manager.refresh()
        return f"Created skill '{name}' at {path}. It is now available."

    return author_skill


def make_skill_update_tool(manager: SkillManager, skills_dir: Union[str, Path]) -> Callable:
    """Return an async ``@tool`` that revises an existing skill's prose and refreshes ``manager``.

    Calls :func:`update_skill` then :meth:`SkillManager.refresh`. Unlike
    :func:`make_skill_script_tool` it needs no agent: the skill's tools are unchanged by an edit to
    its text, so there is nothing to reload.

    A missing skill and an empty call come back as sentences rather than exceptions, because both
    are the model's to correct in the next round (it mistyped a name, or called the tool with
    nothing to change); a malformed file on disk still raises, since nothing the model does next
    fixes it.

    Both surfaces of the refresh limit are worth knowing. A new **body** applies immediately:
    ``activate_skill`` reads it from disk. A new **description** reaches the model only in a fresh
    conversation, because the catalogue is injected into a system prompt that is not rewritten
    mid-run (see :class:`~aimu.aio.SkillAgent`).
    """
    from aimu.tools import tool

    skills_dir = Path(skills_dir).expanduser()

    @tool
    async def update_skill(skill_name: str, description: Optional[str] = None, body: Optional[str] = None) -> str:
        """Revise an existing skill's instructions or its one-line description.

        Use this when a skill turned out to be wrong, incomplete, or misleading, so the fix is
        remembered rather than repeated. Pass only the part you are changing; the other part, and
        everything else about the skill, is left alone. You cannot rename a skill this way, and
        creating one is author_skill.

        Args:
            skill_name: Slug of an existing skill.
            description: Replacement for the one line saying when to use the skill. Omit to keep it.
            body: Replacement markdown instructions, in full (this is not a patch). Omit to keep them.
        """
        if skill_name not in manager.skills:
            available = ", ".join(sorted(manager.skills)) or "(none yet)"
            return f"Skill {skill_name!r} not found. Create it first with author_skill. Existing skills: {available}."
        try:
            _update_skill(skill_name, description=description, body=body, skills_dir=skills_dir)
        except ValueError as exc:  # nothing to change, or a blank description
            return f"Nothing was written: {exc}."
        manager.refresh()
        changed = " and ".join(part for part, given in (("description", description), ("body", body)) if given)
        return (
            f"Updated the {changed} of '{skill_name}'. New instructions apply the next time you "
            "activate it; a new description reaches your skill catalogue in a fresh conversation."
        )

    return update_skill


def make_skill_script_tool(agent, manager: SkillManager, skills_dir: Union[str, Path]) -> Callable:
    """Return an async ``@tool`` that adds a runnable script to an existing skill.

    The tool writes ``scripts/<filename>`` (``.py`` or ``.sh``) into the named skill via
    :func:`write_skill_script`, refreshes ``manager``, then calls ``await agent.reload_skills()`` so
    the new ``{skill}__{stem}`` tool is callable in the same turn. ``agent`` is a
    :class:`~aimu.aio.SkillAgent` (the tool is async); ``manager`` and ``skills_dir`` are captured
    by closure.

    The skill's own ``SKILL.md`` is never opened, which is the point of writing the script directly
    rather than through :func:`write_skill`: a script write is not a change to the skill's prose, and
    routing it through a full rewrite silently dropped frontmatter keys the rewrite does not re-emit.
    Revising the prose is :func:`make_skill_update_tool`'s job.

    **Full access**: the script runs as a real subprocess with the user's privileges, no sandbox.
    """
    from aimu.tools import tool

    skills_dir = Path(skills_dir).expanduser()

    @tool
    async def add_skill_script(skill_name: str, filename: str, content: str) -> str:
        """Create or update a runnable script in an existing skill, then make it callable now.

        Scripts run with full access to this machine (no sandbox). Use this to automate a
        repeatable procedure as code you can invoke as a tool. **To fix a broken script, call this
        again with the SAME filename**: it overwrites the script in place (the `{skill}__{stem}`
        tool keeps its name). Using a different filename instead creates a second script and leaves
        the broken one in place.

        Args:
            skill_name: Slug of an existing skill (create it first with author_skill).
            filename: Script file name, "<stem>.py" or "<stem>.sh". The stem is lowercase and may
                use hyphens or underscores, but the two are equivalent in the tool name, so
                "backup-db.sh" and "backup_db.sh" would collide in one skill: pick a distinct stem.
                Reuse the exact filename of an existing script to replace (fix) it.
            content: Full source of the script.
        """
        skill = manager.skills.get(skill_name)
        if skill is None:
            available = ", ".join(sorted(manager.skills)) or "(none yet)"
            return f"Skill {skill_name!r} not found. Create it first with author_skill. Existing skills: {available}."
        existed = (skills_dir / skill_name / "scripts" / filename).exists()
        write_skill_script(skill_name, filename, content, skills_dir=skills_dir)
        manager.refresh()
        await agent.reload_skills()
        verb = "Updated" if existed else "Added"
        tool_name = script_tool_name(skill_name, Path(filename).stem)
        return f"{verb} {filename} in '{skill_name}'. Tool '{tool_name}' is now callable."

    return add_skill_script


__all__ = [
    "make_skill_authoring_tool",
    "make_skill_script_tool",
    "make_skill_update_tool",
    "update_skill",
    "write_skill",
    "write_skill_script",
]
