from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

from aimu.skills.skill import Skill

logger = logging.getLogger(__name__)


class SkillManager:
    """
    Discovers and manages Agent Skills from the filesystem.

    Scans standard locations for skill directories (each containing a SKILL.md file).
    Project-level skills take precedence over user-level skills on name collision.

    Default scan paths:
        Project: <cwd>/.agents/skills/, <cwd>/.claude/skills/
        User:    ~/.agents/skills/,     ~/.claude/skills/

    Usage::

        manager = SkillManager()
        print(manager.catalog_prompt())   # XML catalog for system prompt
        body = manager.get_skill_body("pdf-processing")

    With custom directories::

        manager = SkillManager(skill_dirs=["/path/to/my/skills"])
    """

    _PROJECT_DIRS = [".agents/skills", ".claude/skills"]
    _USER_DIRS = ["~/.agents/skills", "~/.claude/skills"]

    def __init__(
        self,
        skill_dirs: Optional[list[str]] = None,
        cwd: Optional[str] = None,
    ):
        self._cwd = Path(cwd).resolve() if cwd else Path.cwd()
        self._custom_dirs = [Path(d).expanduser().resolve() for d in skill_dirs] if skill_dirs else None
        self._skills: Optional[dict[str, Skill]] = None

    @property
    def skills(self) -> dict[str, Skill]:
        if self._skills is None:
            self._skills = self._discover()
        return self._skills

    def _search_dirs(self) -> list[Path]:
        if self._custom_dirs is not None:
            return self._custom_dirs
        dirs: list[Path] = []
        for d in self._PROJECT_DIRS:
            dirs.append(self._cwd / d)
        for d in self._USER_DIRS:
            dirs.append(Path(d).expanduser())
        return dirs

    def _discover(self) -> dict[str, Skill]:
        skills: dict[str, Skill] = {}
        for search_dir in self._search_dirs():
            if not search_dir.exists() or not search_dir.is_dir():
                continue
            for skill_dir in sorted(search_dir.iterdir()):
                if not skill_dir.is_dir():
                    continue
                skill_md = skill_dir / "SKILL.md"
                if not skill_md.exists():
                    continue
                try:
                    skill = self._parse(skill_md)
                except Exception as exc:
                    logger.warning("Skipping skill at %s: %s", skill_md, exc)
                    continue
                if skill.name in skills:
                    logger.debug("Skill '%s' at %s shadowed by higher-precedence skill", skill.name, skill_md)
                else:
                    skills[skill.name] = skill
        return skills

    def _parse(self, skill_md: Path) -> Skill:
        content = skill_md.read_text(encoding="utf-8")
        if not content.startswith("---"):
            raise ValueError("missing YAML frontmatter (must start with ---)")

        end = content.find("---", 3)
        if end == -1:
            raise ValueError("unclosed YAML frontmatter (no closing ---)")

        fm_str = content[3:end]
        fm = self._load_yaml(fm_str, skill_md)

        name = str(fm.get("name", skill_md.parent.name))
        description = str(fm.get("description", "")).strip()
        if not description:
            raise ValueError("missing required 'description' field")

        return Skill(
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
        # Lenient fallback: quote bare values that contain colons (common authoring mistake)
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
            raise ValueError(f"unparseable YAML frontmatter in {skill_md}: {exc}") from exc

    def catalog_prompt(self) -> str:
        """Return an XML skill catalog suitable for injection into a system prompt."""
        if not self.skills:
            return ""
        lines = ["<available_skills>"]
        for skill in self.skills.values():
            lines.append("  <skill>")
            lines.append(f"    <name>{skill.name}</name>")
            lines.append(f"    <description>{skill.description}</description>")
            lines.append("  </skill>")
        lines.append("</available_skills>")
        return "\n".join(lines)

    def get_skill_body(self, name: str) -> str:
        """Load and return the full instructions body of a named skill."""
        if name not in self.skills:
            available = ", ".join(self.skills.keys()) or "(none)"
            return f"Skill '{name}' not found. Available skills: {available}"
        return self.skills[name].load_body()
