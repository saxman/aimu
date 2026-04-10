from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Skill:
    """Represents a single discovered Agent Skill."""

    name: str
    description: str
    path: Path  # absolute path to SKILL.md
    compatibility: str = ""
    license_info: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def base_dir(self) -> Path:
        return self.path.parent

    def load_body(self) -> str:
        """Read SKILL.md, strip YAML frontmatter, and return the markdown body."""
        content = self.path.read_text(encoding="utf-8")
        if content.startswith("---"):
            end = content.find("---", 3)
            if end != -1:
                return content[end + 3 :].strip()
        return content.strip()
