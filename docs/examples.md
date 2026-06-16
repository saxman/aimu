# Examples

The [`examples/`](https://github.com/saxman/aimu/tree/main/examples) directory ships larger,
real-world AIMU programs — bigger than a notebook cell, organized by theme. Each subdirectory has
its own README with run commands.

| Example | What it shows | Guide |
|---|---|---|
| [text-refinement/](https://github.com/saxman/aimu/tree/main/examples/text-refinement) | A generate → judge → refine loop over **text**, implemented four ways: a code loop, an `Agent`, the `EvaluatorOptimizer` workflow, and simulated annealing. GPU-free, Ollama-only. | [Iterative text refinement](how-to/iterative-text-refinement.md) |
| [image-refinement/](https://github.com/saxman/aimu/tree/main/examples/image-refinement) | The same loop over **images** (diffusion + a vision evaluator), in five variants including image-to-image. Needs `aimu[hf]` and a GPU. | [Iterative image refinement](how-to/iterative-image-refinement.md) |
| [news-summarizer/](https://github.com/saxman/aimu/tree/main/examples/news-summarizer) | One task — *summarize recent AI news* — solved four ways via `--method`: `Agent`, `Chain`, `Parallel`, and `OrchestratorAgent`. | [Build an orchestrator](how-to/build-orchestrator.md) |
| [skills/](https://github.com/saxman/aimu/tree/main/examples/skills) | Demo `SKILL.md` skills (`haiku-poet`, `unit-converter`) for `SkillAgent` discovery, exposed as `aimu.paths.skills`. | [Use skills](how-to/use-skills.md) |

The text-refinement and image-refinement families are deliberately the **same task in two
modalities**. Read them side by side: the difference between *who drives the loop* (code vs. agent
vs. workflow) and *which search strategy* wins (greedy vs. hill-climbing vs. annealing) is
independent of whether the artifact is a string or an image — a concrete take on
[agents vs. workflows](explanation/agents-vs-workflows.md).
