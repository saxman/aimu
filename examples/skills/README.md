# Example agent skills

Two demo skills used to show how [`SkillAgent`](../../aimu/agents/skill_agent.py) discovers and
injects filesystem skills. A skill is a directory with a `SKILL.md` (YAML frontmatter `name` +
`description`, then an instructions body); optional `scripts/*.py` files become callable tools.

| Skill | What it does |
|---|---|
| [`haiku-poet/`](haiku-poet/SKILL.md) | Write haiku poems in the classic 5-7-5 syllable format. |
| [`unit-converter/`](unit-converter/SKILL.md) | Convert between units of measurement (length, weight, temperature, volume). |

## Use them

These live outside the standard auto-discovery paths (`.agents/skills/`, `.claude/skills/`, and
their `~` equivalents), so point a `SkillManager` at this directory explicitly. The path is exposed
as [`aimu.paths.skills`](../../aimu/paths.py):

```python
from aimu import paths
from aimu.skills import SkillManager

manager = SkillManager(skill_dirs=[str(paths.skills)])   # paths.skills == examples/skills
```

Then pass it to a `SkillAgent`:

```python
import aimu
from aimu.agents import SkillAgent

agent = SkillAgent(aimu.client(), skill_manager=manager)
agent.run("Write a haiku about autumn.")
```

## Learn more

- [08 - Agent Skills](../../notebooks/08%20-%20Agent%20Skills.ipynb): discovery, the skills MCP server, `SkillAgent`
- [Use skills](../../docs/how-to/use-skills.md)
