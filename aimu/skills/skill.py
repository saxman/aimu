from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Any run of characters outside [a-z0-9] collapses to a single underscore, so a tool name is a
# valid identifier whatever the skill name and script stem contain. Both halves still need this
# after spec validation: a valid skill name is [a-z0-9-], whose hyphens are not legal in an
# identifier, and a script stem legitimately allows hyphens as well as underscores. Leaving the
# double underscore as the only separator is the point.
_NON_SLUG_CHARS = re.compile(r"[^a-z0-9]+")


def _slugify(text: str) -> str:
    return _NON_SLUG_CHARS.sub("_", text.lower()).strip("_")


def script_tool_name(skill_name: str, script_stem: str) -> str:
    """Return the tool name a skill's script is registered under: ``{skill}__{stem}``, slugified.

    The single source of this name. Three surfaces need it and must agree, because the catalogue
    is what the model reads: :meth:`AgentSkill.script_tool_names` advertises it,
    :func:`aimu.skills.mcp.build_skills_server` registers it, and ``add_skill_script`` reports it
    back after writing a script. A name advertised but not registered is a tool call that cannot
    succeed, so they are built here rather than formatted in three places.

    Because both halves are slugified, two stems that differ only in their separator
    (``backup-db`` and ``backup_db``) map to one tool name; callers dedupe on the result.
    """
    return f"{_slugify(skill_name)}__{_slugify(script_stem)}"


@dataclass
class AgentSkill:
    """A single discovered Agent Skill from the filesystem.

    The fields mirror the Agent Skills specification's frontmatter, with two renames where the
    spec's key is not a usable Python attribute: ``license`` becomes ``license_info`` and
    ``allowed-tools`` becomes ``allowed_tools``.
    """

    name: str
    description: str
    path: Path  # absolute path to SKILL.md
    compatibility: str = ""
    license_info: str = ""
    metadata: dict = field(default_factory=dict)
    # The spec's `allowed-tools`, split on whitespace. Carried for a host to act on, and acted on
    # by nothing here: the spec marks the field experimental, and which tools an agent may run is
    # the host's policy rather than a library's.
    allowed_tools: tuple[str, ...] = ()

    @property
    def base_dir(self) -> Path:
        return self.path.parent

    def script_tool_names(self) -> list[str]:
        """Return the :func:`script_tool_name` for every ``.py`` / ``.sh`` in ``scripts/``.

        Scripts whose names collapse to the same tool name are listed once, first sorted path
        winning, matching the skills-server registration. That covers ``foo.py`` / ``foo.sh``
        (``.py`` sorts first) and ``foo-bar.py`` / ``foo_bar.py`` (the hyphen sorts first).
        """
        scripts_dir = self.base_dir / "scripts"
        if not scripts_dir.is_dir():
            return []
        scripts = sorted(list(scripts_dir.glob("*.py")) + list(scripts_dir.glob("*.sh")))
        names: list[str] = []
        seen: set[str] = set()
        for p in scripts:
            name = script_tool_name(self.name, p.stem)
            if name in seen:
                continue
            seen.add(name)
            names.append(name)
        return names

    def load_body(self) -> str:
        """Read SKILL.md, strip YAML frontmatter, return the markdown body."""
        content = self.path.read_text(encoding="utf-8")
        if content.startswith("---"):
            end = content.find("---", 3)
            if end != -1:
                return content[end + 3 :].strip()
        return content.strip()
