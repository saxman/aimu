from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentSkill:
    """A single discovered Agent Skill from the filesystem."""

    name: str
    description: str
    path: Path  # absolute path to SKILL.md
    compatibility: str = ""
    license_info: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def base_dir(self) -> Path:
        return self.path.parent

    def script_tool_names(self) -> list[str]:
        """Return ``{skill}__{script_stem}`` tool names for every ``.py`` / ``.sh`` in ``scripts/``.

        ``foo.py`` and ``foo.sh`` share the ``{skill}__foo`` name; ``.py`` sorts first and wins, so
        the name is listed once (matching the skills-server registration).
        """
        scripts_dir = self.base_dir / "scripts"
        if not scripts_dir.is_dir():
            return []
        scripts = sorted(list(scripts_dir.glob("*.py")) + list(scripts_dir.glob("*.sh")))
        names: list[str] = []
        seen: set[str] = set()
        for p in scripts:
            if p.stem in seen:
                continue
            seen.add(p.stem)
            names.append(f"{self.name}__{p.stem}")
        return names

    def load_body(self) -> str:
        """Read SKILL.md, strip YAML frontmatter, return the markdown body."""
        content = self.path.read_text(encoding="utf-8")
        if content.startswith("---"):
            end = content.find("---", 3)
            if end != -1:
                return content[end + 3 :].strip()
        return content.strip()
