# Image refinement (the "hotdog" family)

A **generate → evaluate → refine** loop over *images*: a diffusion model generates an image, a
vision model scores it and proposes a better prompt, and the loop repeats. The worked task is
*"make the hottest hotdog imaginable"* — each round pushes toward a single, well-formed,
maximally-appetising hotdog while a vision evaluator gates on "is this still one hotdog?"

This is the image counterpart of [`../text-refinement/`](../text-refinement/) — the same control-flow
and search lessons, but with image-specific scaffolding (a diffusers pipeline, a vision evaluator, a
token-budget prompt summarizer, negative prompts). All variants share `_hotdog_common.py`.

> **Requires** the `[hf]` extra (HuggingFace `diffusers`) and ideally a GPU, plus a vision-capable
> evaluator model. See [Generate images](../../docs/how-to/generate-images.md).

| Script | Who/what drives the loop | AIMU feature |
|---|---|---|
| `hotdog_loop.py` | Python `for` loop (`--strategy greedy`\|`climbing`) | `image_client()` + vision `generate(images=...)` |
| `hotdog_agent.py` | an autonomous `Agent` via tool calls | [`Agent`](../../aimu/agents/agent.py) + image/vision tools |
| `hotdog_evaluator.py` | the `EvaluatorOptimizer` workflow class | [`EvaluatorOptimizer`](../../aimu/agents/workflows/evaluator.py) |
| `hotdog_anneal.py` | simulated annealing (Metropolis acceptance) | temperature-annealed proposer |
| `hotdog_img2img.py` | hill-climb in image space + strength annealing | `reference_image=` / `strength=` img2img |

## Run

```bash
python examples/image-refinement/hotdog_loop.py                       # greedy walk (default)
python examples/image-refinement/hotdog_loop.py --strategy climbing --patience 4
python examples/image-refinement/hotdog_agent.py                      # agent-directed
python examples/image-refinement/hotdog_evaluator.py                  # EvaluatorOptimizer workflow class
python examples/image-refinement/hotdog_anneal.py --seed 7            # simulated annealing
python examples/image-refinement/hotdog_img2img.py                    # img2img hill-climb + strength annealing
```

Each run writes images, a `summary.txt` trace, and a collage under `output/hotdog/<timestamp>/`.
Pass `--help` to any script for the full flag list (`--image-model`, `--eval-model`, etc.).

## Tests

```bash
pytest examples/image-refinement/tests -q
```

## Learn more

Full walkthrough: [Iterative image refinement](../../docs/how-to/iterative-image-refinement.md).
