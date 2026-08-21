from importlib import import_module as _import_module

from aimu.skills.authoring import make_skill_authoring_tool, make_skill_script_tool, write_skill
from aimu.skills.manager import SkillLoadError, SkillManager, SkillNotFoundError
from aimu.skills.skill import AgentSkill, script_tool_name
from aimu.skills.validate import SkillSpecError, validate_frontmatter

# build_skills_server lives in .mcp, which imports fastmcp (and its own dependency tree --
# mcp, jsonschema, ...) at module level. fastmcp is a *required* (not optional) dependency,
# so this is the same plain lazy import aimu.tools.__init__ uses for MCPClient: deferring
# fastmcp's real cost until a caller actually builds a skills server, rather than paying it
# on every `import aimu.skills` (and everything that imports SkillManager alongside it, e.g.
# aimu.agents.skill_agent / aimu.aio.skill_agent).
_LAZY_MCP_SYMBOLS = frozenset({"build_skills_server"})


def __getattr__(name: str):
    if name in _LAZY_MCP_SYMBOLS:
        return getattr(_import_module(".mcp", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_MCP_SYMBOLS})


__all__ = [
    "AgentSkill",
    "SkillLoadError",
    "SkillManager",
    "SkillNotFoundError",
    "SkillSpecError",
    "build_skills_server",
    "make_skill_authoring_tool",
    "make_skill_script_tool",
    "script_tool_name",
    "validate_frontmatter",
    "write_skill",
]
