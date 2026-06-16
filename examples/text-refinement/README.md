# Text refinement (the "epic" family)

A **generate → judge → refine** loop over *text*: a model writes a sentence, a judge model scores
it and proposes a better rewrite, and the loop repeats until the judge is satisfied. The worked
task is *"take a mundane sentence and make it as **epic** as possible"* — starting from
`A man walks to the store to buy a carton of milk.`, each round makes it grander while keeping it
**one grammatical sentence about the same errand**.

These scripts are the GPU-free, text-only sibling of [`../image-refinement/`](../image-refinement/):
the same orchestration and search lessons with no image generation, vision model, or `[hf]` extra.
They run on a local text model (Ollama) alone. All four share every prompt and helper in
`_epic_common.py`, so the difference between them is purely **who drives the loop** and **which
search strategy** is used.

| Script | Who drives the loop | AIMU feature |
|---|---|---|
| `epic_loop.py` | Python `for` loop (`--strategy greedy`\|`climbing`) | plain `client.generate()` |
| `epic_agent.py` | an autonomous `Agent` via tool calls | [`Agent`](../../aimu/agents/agent.py) + `@aimu.tool` |
| `epic_evaluator.py` | the `EvaluatorOptimizer` workflow class | [`EvaluatorOptimizer`](../../aimu/agents/workflows/evaluator.py) |
| `epic_anneal.py` | simulated annealing (Metropolis acceptance) | temperature-annealed `generate()` |

## Run

```bash
python examples/text-refinement/epic_loop.py                          # code-directed greedy walk (default)
python examples/text-refinement/epic_loop.py --strategy climbing      # hill-climb: keep best, revert on regression
python examples/text-refinement/epic_agent.py                         # agent-directed
python examples/text-refinement/epic_evaluator.py                     # EvaluatorOptimizer workflow class
python examples/text-refinement/epic_anneal.py --seed 7               # simulated annealing
python examples/text-refinement/epic_loop.py --seed-sentence "She parks the car."
```

Each run writes a `summary.txt` trace under `output/epic/<timestamp>/`; `--strategy climbing` and
`epic_anneal.py` also write `best.txt`. Pass `--help` to any script for the full flag list.

## Tests

```bash
pytest examples/text-refinement/tests -q
```

## Learn more

Full walkthrough: [Iterative text refinement](../../docs/how-to/iterative-text-refinement.md)
(the *why* behind each variant, the greedy → hill-climbing → annealing progression, and the
one-to-one mapping back to the hotdog scripts).
