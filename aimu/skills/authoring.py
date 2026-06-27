"""Author new Agent Skills at runtime.

AIMU *discovers* and *uses* skills via :class:`~aimu.skills.manager.SkillManager`; this
module is the inverse: it lets an agent *write* a new ``SKILL.md`` while running, so the
assistant grows reusable skills as it solves problems (the self-improvement pattern from
Hermes Agent). :func:`write_skill` is plain filesystem work shared by both surfaces;
:func:`make_skill_authoring_tool` wraps it as an async ``@tool`` an agent can call.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Optional, Union

from aimu.skills.manager import SkillManager

# A skill name doubles as a directory name and a tool-name prefix, so restrict it to a
# safe slug: lowercase letters, digits, and single hyphens. This also blocks path
# traversal (no separators, no ``..``).
_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def write_skill(
    name: str,
    description: str,
    body: str,
    *,
    skills_dir: Union[str, Path],
    overwrite: bool = False,
    metadata: Optional[dict] = None,
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


__all__ = ["make_skill_authoring_tool", "write_skill"]
