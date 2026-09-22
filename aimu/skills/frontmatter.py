"""The ``SKILL.md`` file format: splitting one, loading its frontmatter, rendering it back.

Three surfaces touch this format and used to carry their own copy of it: discovery
(:class:`~aimu.skills.manager.SkillManager`), body loading
(:meth:`~aimu.skills.skill.AgentSkill.load_body`), and authoring (:mod:`aimu.skills.authoring`).
Editing a skill needs all three at once (read the file, change one field, write the rest back
unchanged), and one place is what makes "unchanged" true: a writer that re-emits only the keys it
knows about silently drops every other one, which is exactly how ``license`` and ``compatibility``
used to disappear from a skill an agent attached a script to.

Deliberately no dependency on the rest of the package, so a parse error raises plain
:class:`ValueError` and the caller decides what to wrap it in
(:class:`~aimu.skills.manager.SkillLoadError`, which adds the file it came from).
"""

from __future__ import annotations

from typing import Optional

import yaml

FENCE = "---"


def split_frontmatter(content: str) -> tuple[Optional[str], str]:
    """Split ``SKILL.md`` text into its raw frontmatter block and its markdown body.

    Returns ``(None, content)`` when there is no closed ``---`` block, rather than raising: for
    discovery that is a malformed file, but for :meth:`AgentSkill.load_body` it is a body-only file
    to read as-is, so which one it is belongs to the caller.
    """
    if content.startswith(FENCE):
        end = content.find(FENCE, len(FENCE))
        if end != -1:
            return content[len(FENCE) : end], content[end + len(FENCE) :].strip()
    return None, content.strip()


def load_frontmatter(frontmatter: str) -> dict:
    """Parse a raw frontmatter block into a mapping, raising :class:`ValueError` if it cannot.

    Falls back to quoting bare values that contain a colon, a common authoring mistake that makes
    an otherwise readable block invalid YAML. The fallback is for hand-written files;
    :func:`render_frontmatter` quotes what AIMU writes itself.
    """
    try:
        return yaml.safe_load(frontmatter) or {}
    except yaml.YAMLError:
        pass
    fixed_lines = []
    for line in frontmatter.splitlines():
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
        raise ValueError(f"unparseable YAML frontmatter: {exc}") from exc


def render_frontmatter(fields: dict) -> str:
    """Render ``fields`` as a fenced YAML frontmatter block, keys in the order given.

    ``yaml.safe_dump`` rather than hand-built lines, so a value needing quotes gets them: a
    description containing ``": "`` is the ordinary case, and writing it bare produced a file that
    only loaded through :func:`load_frontmatter`'s fallback. ``width`` is set past any real
    description so a long one stays on one line, since folding it would be valid YAML that a
    person reads as a mistake.
    """
    body = yaml.safe_dump(fields, sort_keys=False, allow_unicode=True, default_flow_style=False, width=10**6)
    return f"{FENCE}\n{body}{FENCE}"


__all__ = ["FENCE", "load_frontmatter", "render_frontmatter", "split_frontmatter"]
