from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

from aimu.skills.skill import AgentSkill

logger = logging.getLogger(__name__)


_DEFAULT_SKILL_DIRS = [
    ".agents/skills",
    ".claude/skills",
    "~/.agents/skills",
    "~/.claude/skills",
]


class SkillLoadError(ValueError):
    """Raised when a ``SKILL.md`` file is malformed and cannot be parsed."""


class SkillNotFoundError(KeyError):
    """Raised when a requested skill name does not exist."""


class SkillManager:
    """Discovers and manages Agent Skills from the filesystem.

    With no ``skill_dirs`` argument, scans the standard search paths at project and
    user scope: ``.agents/skills/``, ``.claude/skills/``, ``~/.agents/skills/``,
    ``~/.claude/skills/``. Project-level paths win on name collision. Pass explicit
    ``skill_dirs`` to override all defaults.

    Discovery logs (at INFO) the number of skills found and the paths searched, so a
    missing skill directory is easy to spot. Malformed ``SKILL.md`` files raise
    :class:`SkillLoadError` rather than being silently skipped.

    Usage::

        manager = SkillManager()                                # auto-discover
        manager = SkillManager(skill_dirs=["/path/to/skills"])  # explicit
        print(manager.catalog_prompt())
        body = manager.get_skill_body("pdf-processing")
    """

    def __init__(self, skill_dirs: Optional[list[str]] = None):
        self._custom_dirs = [Path(d).expanduser().resolve() for d in skill_dirs] if skill_dirs else []
        self._skills: Optional[dict[str, AgentSkill]] = None

    @property
    def skills(self) -> dict[str, AgentSkill]:
        if self._skills is None:
            self._skills = self._discover()
        return self._skills

    def refresh(self) -> dict[str, AgentSkill]:
        """Invalidate the cache and re-discover skills, returning the new map.

        Lets a skill authored at runtime (see :func:`aimu.skills.write_skill`) become
        visible mid-run without constructing a fresh manager.
        """
        self._skills = None
        return self.skills

    def _search_dirs(self) -> list[Path]:
        if self._custom_dirs:
            return self._custom_dirs
        return [Path(d).expanduser().resolve() for d in _DEFAULT_SKILL_DIRS]

    def _discover(self) -> dict[str, AgentSkill]:
        skills: dict[str, AgentSkill] = {}
        searched = self._search_dirs()
        for search_dir in searched:
            if not search_dir.exists() or not search_dir.is_dir():
                continue
            for skill_dir in sorted(search_dir.iterdir()):
                if not skill_dir.is_dir():
                    continue
                skill_md = skill_dir / "SKILL.md"
                if not skill_md.exists():
                    continue
                skill = self._parse(skill_md)
                if skill.name in skills:
                    logger.debug("Skill '%s' at %s shadowed by higher-precedence skill", skill.name, skill_md)
                else:
                    skills[skill.name] = skill
        logger.info(
            "SkillManager discovered %d skill(s) across %d path(s): %s",
            len(skills),
            len(searched),
            ", ".join(str(p) for p in searched),
        )
        return skills

    def _parse(self, skill_md: Path) -> AgentSkill:
        try:
            content = skill_md.read_text(encoding="utf-8")
        except OSError as exc:
            raise SkillLoadError(f"could not read {skill_md}: {exc}") from exc

        if not content.startswith("---"):
            raise SkillLoadError(f"{skill_md}: missing YAML frontmatter (must start with ---)")

        end = content.find("---", 3)
        if end == -1:
            raise SkillLoadError(f"{skill_md}: unclosed YAML frontmatter (no closing ---)")

        fm_str = content[3:end]
        fm = self._load_yaml(fm_str, skill_md)

        name = str(fm.get("name", skill_md.parent.name))
        description = str(fm.get("description", "")).strip()
        if not description:
            raise SkillLoadError(f"{skill_md}: missing required 'description' field")

        return AgentSkill(
            name=name,
            description=description,
            path=skill_md.resolve(),
            compatibility=str(fm.get("compatibility", "")),
            license_info=str(fm.get("license", "")),
            metadata=dict(fm.get("metadata") or {}),
        )

    @staticmethod
    def _load_yaml(fm_str: str, skill_md: Path) -> dict:
        try:
            return yaml.safe_load(fm_str) or {}
        except yaml.YAMLError:
            pass
        # Lenient fallback: quote bare values that contain colons (common authoring mistake).
        fixed_lines = []
        for line in fm_str.splitlines():
            stripped = line.lstrip()
            if ":" in stripped and not stripped.startswith("-") and not stripped.startswith("#"):
                key, _, value = stripped.partition(":")
                value = value.strip()
                if value and not (value.startswith('"') or value.startswith("'") or value.startswith("|")):
                    indent = " " * (len(line) - len(stripped))
                    line = f'{indent}{key}: "{value}"'
            fixed_lines.append(line)
        try:
            return yaml.safe_load("\n".join(fixed_lines)) or {}
        except yaml.YAMLError as exc:
            raise SkillLoadError(f"{skill_md}: unparseable YAML frontmatter: {exc}") from exc

    def catalog_prompt(self) -> str:
        """Return an XML skill catalog suitable for injection into a system prompt.

        Each entry lists the skill name, description, and any script-derived tool names
        the model can call directly (without first calling ``activate_skill``).
        """
        if not self.skills:
            return ""
        lines = ["<available_skills>"]
        for skill in self.skills.values():
            lines.append("  <skill>")
            lines.append(f"    <name>{skill.name}</name>")
            lines.append(f"    <description>{skill.description}</description>")
            tool_names = skill.script_tool_names()
            if tool_names:
                lines.append("    <tools>")
                for tn in tool_names:
                    lines.append(f"      <tool>{tn}</tool>")
                lines.append("    </tools>")
            lines.append("  </skill>")
        lines.append("</available_skills>")
        return "\n".join(lines)

    def get_skill_body(self, name: str) -> str:
        """Return the full instructions body of a named skill.

        Raises :class:`SkillNotFoundError` if the skill doesn't exist.
        """
        if name not in self.skills:
            available = ", ".join(self.skills.keys()) or "(none)"
            raise SkillNotFoundError(f"Skill '{name}' not found. Available skills: {available}")
        return self.skills[name].load_body()
