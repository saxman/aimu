# Examples

Runnable, real-world AIMU programs, larger than the notebook cells, organized by theme. Each
subdirectory has its own README with run commands and a link to the relevant how-to guide.

| Example | What it shows | Start with |
|---|---|---|
| [text-refinement/](text-refinement/) | A generate, judge, refine loop over **text**, implemented four ways (code loop, `Agent`, `EvaluatorOptimizer`, simulated annealing). GPU-free, Ollama-only. | `python examples/text-refinement/epic_loop.py` |
| [image-refinement/](image-refinement/) | The same loop over **images** (diffusion + vision evaluator), five variants including img2img. Needs the `[hf]` extra and a GPU. | `python examples/image-refinement/hotdog_loop.py` |
| [news-summarizer/](news-summarizer/) | One task (*summarize recent AI news*) solved with `Agent`, `Chain`, `Parallel`, and `OrchestratorAgent`, selected via `--method`. | `python examples/news-summarizer/news_summarizer.py --method agent` |
| [personal-assistant/](personal-assistant/) | A single-user, always-on assistant (OpenClaw / Hermes style) from AIMU primitives: a `Channel` (CLI or a WebSocket `web_assistant.py`), a `Scheduler` for proactive messages, and a `SkillAgent` that authors its own skills and scripts. | `python examples/personal-assistant/assistant.py` |
| [web/](web/) | Streamlit and Gradio chat apps (showcase + full-featured). Needs `pip install aimu[web]`. | `streamlit run examples/web/streamlit_chatbot.py` |
| [skills/](skills/) | Demo `SKILL.md` skills (`haiku-poet`, `unit-converter`) for `SkillAgent` discovery, exposed as `aimu.paths.skills`. | [08 - Agent Skills](../notebooks/08%20-%20Agent%20Skills.ipynb) |

The text-refinement and image-refinement families are deliberately the **same task in two
modalities**. Read them side by side to see that the diff between "who drives the loop"
(code vs. agent vs. workflow) and "which search strategy" (greedy vs. hill-climbing vs. annealing)
is independent of whether the artifact is a string or an image.

## Tests

The example test suites are kept out of the default `pytest` run. Run them explicitly:

```bash
pytest examples/                       # all example suites
pytest examples/text-refinement/tests  # one suite
```

## See also

- [`notebooks/`](../notebooks/): interactive, subsystem-by-subsystem demos (numbered 01-24)
- [How-to guides](https://saxman.github.io/aimu/how-to/): task-oriented recipes
