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

Required fields: `name`, `description`. Optional: `compatibility`, `license`, free-form `metadata`.

Malformed `SKILL.md` files raise `SkillLoadError` (no silent skipping).

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

A skill can include Python scripts in a `scripts/` subdirectory. Each `*.py` file is auto-registered as an MCP tool named `{skill_name}__{script_stem}`:

```
.agents/skills/pdf-processing/
├── SKILL.md
└── scripts/
    ├── extract_pages.py     # → pdf_processing__extract_pages tool
    └── merge.py             # → pdf_processing__merge tool
```

Scripts run via `subprocess`; their stdout becomes the tool result. The catalogue lists script tool names inline, so the model can call them directly without first invoking `activate_skill`.

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

## See also

- [Tutorial: first agent with tools](../tutorials/02-first-agent-with-tools.md): start here if `SkillAgent` is overkill
- [`aimu.skills` API reference](../reference/api/skills.md)
- Notebook [08 - Agent Skills](https://github.com/saxman/aimu/blob/main/notebooks/08-agent-skills.qmd)
