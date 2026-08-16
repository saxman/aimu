# Use skills

A *skill* is a directory containing a `SKILL.md` file with YAML frontmatter and markdown instructions. `SkillAgent` discovers skills from the filesystem and injects them into the agent's system message, letting the model load full instructions on demand via the `activate_skill` MCP tool.

## SKILL.md format

```markdown
---
name: pdf-processing
description: Extract pages, merge PDFs, and convert PDFs to text.
---

# PDF processing

When asked to work with a PDF:

1. Use `pdfplumber` to extract text...
2. ...
```

AIMU implements the [Agent Skills specification](https://agentskills.io/specification), and validates frontmatter against it on discovery:

| Field | Required | Rule |
| --- | --- | --- |
| `name` | yes | 1-64 characters, lowercase letters, digits and single hyphens, no leading or trailing hyphen, **and it must match the skill's directory name** |
| `description` | yes | 1-1024 characters |
| `compatibility` | no | at most 500 characters |
| `license` | no | exposed as `AgentSkill.license_info` |
| `metadata` | no | a mapping of string to string, so quote a version (`version: "1.0"`) |
| `allowed-tools` | no | a space-separated string, exposed as `AgentSkill.allowed_tools`. Experimental in the spec; AIMU carries it and acts on it nowhere, since which tools an agent may run is your policy rather than the library's |

A malformed or non-compliant `SKILL.md` raises `SkillLoadError` naming the rule and the fix, rather than being silently skipped or loaded under a name nothing can address. `aimu.skills.validate_frontmatter` is that same check as a callable, for validating before you write a file.

## Discovery paths

`SkillManager()` with no arguments scans the standard search paths in this order (project-level wins on collision):

1. `.agents/skills/`
2. `.claude/skills/`
3. `~/.agents/skills/`
4. `~/.claude/skills/`

Skills are logged at `INFO` level on first discovery (count + paths searched). Pass `skill_dirs=[...]` to override the defaults.

## Use SkillAgent

```python
import aimu
from aimu.agents import SkillAgent

client = aimu.client("ollama:qwen3.5:9b")
agent = SkillAgent(client, "You are a helpful assistant.")
result = agent.run("Use the pdf-processing skill to extract pages from report.pdf.")
```

On the first `run()`, the agent injects the skill catalogue into the system message and attaches a skills MCP client. The model can either:

- Call `activate_skill("pdf-processing")` to load the full instructions and then act, or
- Call a script-derived tool (`pdf_processing__extract_pages(...)`) directly if the skill ships executable scripts.

## Skill scripts

A skill can include scripts in a `scripts/` subdirectory. Each `*.py` and `*.sh` file is auto-registered as an MCP tool named `{skill_name}__{script_stem}`, with **both halves slugified**: lowercased, with every run of other characters collapsed to `_`. So the `pdf-processing` skill yields `pdf_processing__*` tools, and the `__` stays the only separator.

```
.agents/skills/pdf-processing/
├── SKILL.md
└── scripts/
    ├── extract_pages.py     # → pdf_processing__extract_pages tool
    └── merge-all.sh         # → pdf_processing__merge_all tool
```

Slugifying is what keeps the tool name a valid identifier: a spec-valid skill name is `[a-z0-9-]` and a script stem allows either separator, and hyphens are not legal in an identifier, so both halves fold to `[a-z0-9_]` and the `__` stays the only separator. (A name like `My Skill` no longer reaches this stage at all: spec validation rejects it on discovery.) Use [`aimu.skills.script_tool_name`](../reference/api/skills.md) rather than formatting the name yourself.

Names collapse, so two scripts in one skill can collide: `merge.py` / `merge.sh` (same stem) and `merge-all.py` / `merge_all.py` (same slug) each register once, the first sorted path winning. Give each script a distinct slug.

Scripts run via `subprocess`; their stdout becomes the tool result. The catalogue lists script tool names inline, so the model can call them directly without first invoking `activate_skill`.

### Self-contained scripts

A `.py` script may declare its own dependencies inline with [PEP 723](https://peps.python.org/pep-0723/), which is the [spec's recommended way](https://agentskills.io/skill-creation/using-scripts) to make a skill self-contained. A script containing a `# /// script` block runs through `uv run --script`, which resolves those dependencies into an isolated environment:

```python
# /// script
# dependencies = [
#   "beautifulsoup4",
# ]
# ///
from bs4 import BeautifulSoup
...
```

A script with no inline block runs on the same interpreter as AIMU, so a skill relying on packages already installed in your environment keeps working. Inline dependencies need `uv` on `PATH`; without it the tool result names uv rather than surfacing an import error.

### Two constraints worth designing around

**Scripts run non-interactively.** stdin is closed, so a script that prompts gets `EOFError` immediately rather than hanging. Take input through flags or environment variables, and make `--help` describe the interface: that output is how a model learns to call the script.

**Scripts have a 30-second budget** and run synchronously, so a long job blocks the event loop on the async path. Keep script work short, and put anything model-driven or long-running in a tool rather than a script.

## Use a SkillManager directly

For inspection or programmatic use:

```python
from aimu.skills import SkillManager

manager = SkillManager(skill_dirs=["./my-skills"])
print(manager.catalog_prompt())                 # XML block listing all skills + scripts
print(manager.get_skill_body("pdf-processing")) # full markdown body

# Missing skills raise SkillNotFoundError
try:
    manager.get_skill_body("nonexistent")
except SkillNotFoundError as exc:
    print(exc)
```

### Give one agent a subset of the skills

`include` narrows discovery to the skills you name, which is how you hand different agents different skills from one directory:

```python
researcher = SkillAgent(client, skill_manager=SkillManager(include=["citation-check"]))
```

Because `catalog_prompt()` and `build_skills_server()` both read `skills`, filtering here scopes the catalogue *and* the callable script tools together, so an agent cannot be advertised a skill it cannot activate. A name that no search path provides raises `SkillLoadError` listing what was discovered, rather than handing the agent a shorter list than it asked for.

## See also

- [Tutorial: first agent with tools](../tutorials/02-first-agent-with-tools.md): start here if `SkillAgent` is overkill
- [`aimu.skills` API reference](../reference/api/skills.md)
- Notebook [08 - Agent Skills](https://github.com/saxman/aimu/blob/main/notebooks/08-agent-skills.qmd)
