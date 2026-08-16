"""Frontmatter validation for the Agent Skills specification.

The rules come from https://agentskills.io/specification and are enforced on discovery, so a
skill either loads as a spec-valid skill or raises. That strictness is deliberate: the name is
how everything addresses a skill (the catalogue the model reads, ``activate_skill``, and the
``{skill}__{stem}`` script tool names), so a skill loaded under a name the spec forbids is
reachable by nothing and says nothing about why.

Kept separate from :class:`~aimu.skills.manager.SkillManager` so the rules can be tested
without a filesystem, and so the one function here mirrors what ``skills-ref validate`` checks
for an author who runs it directly.
"""

from __future__ import annotations

import re

# Encodes four of the spec's name rules at once: only lowercase alphanumerics and hyphens, no
# leading or trailing hyphen, and no consecutive hyphens. The length limit is checked separately
# so its error can name the actual length.
_NAME = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

_NAME_MAX = 64
_DESCRIPTION_MAX = 1024
_COMPATIBILITY_MAX = 500


class SkillSpecError(ValueError):
    """A ``SKILL.md``'s frontmatter violates the Agent Skills specification.

    Raised by :func:`validate_frontmatter` and wrapped in
    :class:`~aimu.skills.manager.SkillLoadError` by the discovery path, which adds the file it
    came from.
    """


def validate_frontmatter(frontmatter: dict, *, directory_name: str) -> None:
    """Raise :class:`SkillSpecError` if ``frontmatter`` violates the Agent Skills spec.

    ``directory_name`` is the skill directory's own name, which the spec requires ``name`` to
    match: the two identify the same skill, and a disagreement leaves it addressable under one
    and discoverable under the other.
    """
    name = frontmatter.get("name")
    if not isinstance(name, str) or not name:
        raise SkillSpecError(
            f"missing required 'name' field (the spec requires it, and it must be {directory_name!r} "
            "to match the skill's directory)"
        )
    if len(name) > _NAME_MAX:
        raise SkillSpecError(f"'name' is {len(name)} characters; the spec allows at most {_NAME_MAX}")
    if not _NAME.match(name):
        raise SkillSpecError(
            f"'name' {name!r} is not a valid skill name: the spec allows lowercase letters, digits and "
            "single hyphens, and forbids a leading or trailing hyphen"
        )
    if name != directory_name:
        raise SkillSpecError(
            f"'name' {name!r} does not match its directory {directory_name!r}; the spec requires them to agree"
        )

    description = frontmatter.get("description")
    if not isinstance(description, str) or not description.strip():
        raise SkillSpecError("missing required 'description' field")
    if len(description) > _DESCRIPTION_MAX:
        raise SkillSpecError(
            f"'description' is {len(description)} characters; the spec allows at most {_DESCRIPTION_MAX}"
        )

    compatibility = frontmatter.get("compatibility")
    if compatibility is not None:
        if not isinstance(compatibility, str):
            raise SkillSpecError("'compatibility' must be a string")
        if len(compatibility) > _COMPATIBILITY_MAX:
            raise SkillSpecError(
                f"'compatibility' is {len(compatibility)} characters; the spec allows at most {_COMPATIBILITY_MAX}"
            )

    metadata = frontmatter.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise SkillSpecError("'metadata' must be a mapping of string keys to string values")
        for key, value in metadata.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise SkillSpecError(
                    f"'metadata' entry {key!r} is not a string-to-string pair; the spec allows only string "
                    "values, so quote a number or a version ('1.0')"
                )

    allowed_tools = frontmatter.get("allowed-tools")
    if allowed_tools is not None and not isinstance(allowed_tools, str):
        raise SkillSpecError("'allowed-tools' must be a space-separated string, e.g. 'Bash(git:*) Read'")
