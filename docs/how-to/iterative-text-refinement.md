# Iterative text refinement

Build a **generate → judge → refine** loop over *text*: a model writes a sentence, a judge model scores it and proposes a better rewrite, and the loop repeats until the judge is satisfied. This is the text-only sibling of [iterative image refinement](iterative-image-refinement.md), sharing the same orchestration and search lessons, but with no image generation, vision model, or GPU. Everything here runs on a local text model (Ollama) alone.

The worked example: *"take a mundane sentence and make it as **epic** as possible."* Starting from `A man walks to the store to buy a carton of milk.`, each round rewrites it into a grander, more mythic register while it stays **one grammatical sentence about the same errand**. A judge scores epicness 1–10 against an "epic ceiling," gates on fidelity (still one sentence about the same event, else 1/10), and emits `DONE` or `CONTINUE: <a grander rewrite direction>`.

The full, runnable code lives in four scripts that share one helper module:

- `examples/text-refinement/epic_loop.py`: **Python directs the loop** (a plain code-controlled `for` loop). A `--strategy` flag selects the search: `greedy` (default, accept every step) or `climbing` (hill-climb: keep the best sentence, revert when a step doesn't beat it). Greedy is the climber with "always accept, never revert," so the two share one loop.
- `examples/text-refinement/epic_agent.py`: **an `Agent` directs the loop** via tool calls (autonomous).
- `examples/text-refinement/epic_evaluator.py`: **`EvaluatorOptimizer` directs the loop** (the library workflow class, composing a generator + judge agent).
- `examples/text-refinement/epic_anneal.py`: **simulated annealing**: generalises the climber by accepting worse sentences early (high temperature) to escape local optima, cooling to greedy.
- `examples/text-refinement/_epic_common.py`: shared prompts and helpers used by all of them.

```bash
python examples/text-refinement/epic_loop.py                                            # code-directed greedy walk (default)
python examples/text-refinement/epic_loop.py --strategy climbing                           # hill-climb: keep best, revert on regression
python examples/text-refinement/epic_agent.py                                             # agent-directed greedy walk
python examples/text-refinement/epic_evaluator.py                                         # EvaluatorOptimizer workflow class
python examples/text-refinement/epic_anneal.py --seed 7                                   # simulated annealing: explore early, cool to greedy
python examples/text-refinement/epic_loop.py --seed-sentence "She parks the car."       # any mundane seed sentence
python examples/text-refinement/epic_agent.py --max-iterations 0                          # run until the judge says DONE
python examples/text-refinement/epic_loop.py --gen-model ollama:qwen3:8b --judge-model anthropic:claude-sonnet-4-6
```

Every script produces a `summary.txt` trace; `--strategy climbing` and `epic_anneal.py` also write `best.txt`.

## Why a text twin of the hotdog scripts?

The [hotdog scripts](iterative-image-refinement.md) teach the same control-flow lesson, but each one carries image-specific scaffolding: a diffusers pipeline, a vision evaluator, a token-budget summarizer, a negative prompt, a collage. That machinery is valuable to see *once*, but it obscures the orchestration lesson and forces a GPU and the `[hf]` extra on anyone who just wants to compare **who drives the loop** and **which search strategy** wins.

The epic family strips all of it away. The artifact is a string, the "generator" and "judge" are plain text `generate()` calls, and there is no encoder budget or negative prompt to manage. What's left in each script is *only* the orchestration and search, so the diff between `epic_loop.py` and `epic_agent.py` is purely the loop driver, and the diff between `epic_loop.py`'s two strategies and `epic_anneal.py` is purely the acceptance rule. The mapping back to hotdog is one-to-one:

| hotdog (image) | epic (text) |
|---|---|
| hotness vs the Planck temperature | epicness vs the grandest myth conceivable |
| gate: one well-formed hotdog | gate: one grammatical sentence about the same errand |
| "don't replace the hotdog with the phenomenon" | "don't drop the errand or split the sentence" |
| `image_client.generate(prompt)` → image | `gen_client.generate(...)` → sentence |
| vision eval via `generate(images=...)` | text judge via `generate(sentence-in-prompt)` |

## Three ways to run the same loop

The same task is implemented three times on purpose, a concrete take on [agents vs workflows](../explanation/agents-vs-workflows.md):

- **Code-directed** (`epic_loop.py`): you write the `for` loop. Flow is explicit and deterministic, and easiest to read, debug, and bound. Reach for this when the control flow is fixed and you want full visibility.
- **Agent-directed** (`epic_agent.py`): you hand an [`Agent`](../reference/api/agents.md) two tools (`write_epic_sentence`, `judge_epicness`) and let its tool-calling loop decide when to call what. Reach for this when you want the model to own the control flow.
- **Workflow-class** (`epic_evaluator.py`): you express the loop with [`EvaluatorOptimizer`](../reference/api/agents.md), composing a generator agent and a judge agent. This is the modality where `EvaluatorOptimizer` fits *most* naturally: the artifact relayed between the two agents is plain text (the sentence itself), so there's none of the fragile image-path relay the hotdog version has to work around.

All three share every prompt and helper in `_epic_common.py`, so the difference between them is purely *who drives the loop*.

## Search strategy: greedy → hill-climbing → simulated annealing

`epic_loop.py` defaults to **greedy** (and the agent script is greedy too): each round it accepts whatever rewrite the judge's directive produces and moves on, even if the new sentence scored *lower* than a previous one.

`epic_loop.py --strategy climbing` borrows the strategy from [`aimu.prompts`](tune-prompts.md) tuning: **best-state caching + revert when a step doesn't beat the best**. Track the highest-scoring sentence; if a generation beats it, adopt it and refine from there; if it doesn't (a lower score *or* a tie), discard it, revert to the best, and ask the judge for a *different* refinement (passing the directions that already failed so it explores elsewhere). Stop on `DONE`, after `--patience` consecutive non-improvements, or at `--max-iterations`. The winner is written to `best.txt`. Because greedy *is* this loop with "always accept, never revert," both strategies live in the one script and differ only in the acceptance rule.

`epic_anneal.py` **generalises the climber into simulated annealing**: the climber is annealing's `T → 0` limit. It keeps a `current` walk-state (distinct from the best-ever) and accepts the next sentence by the Metropolis rule, where `Δ = new_score − current_score`:

- `Δ > 0` (more epic): always accept.
- `Δ == 0` (tie): always accept, a free sideways move, the deliberate *opposite* of the climber's revert-on-tie.
- `Δ < 0` (less epic): accept with probability `exp(Δ / T)`. A falling temperature (`--initial-temp`, `--cooling-rate`) explores freely early, then cools into greedy behaviour. `--seed` makes the acceptance decisions reproducible.

The same cooling schedule also drives the **proposer's** sampling temperature: the judge generates refinement directions at a high LLM temperature while hot (diverse) and a low one once cooled (conservative), via `_proposer_temperature`. Only proposals are annealed; the **judge that assigns the score stays cold**, since the score is the optimisation objective and must stay stable.

The same honest caveats as the hotdog version apply: the coarse integer judge makes `Δ` almost always `0` or `±1` (blunting fine temperature control), and runs are short, so the practical win over the climber is local-optimum escape, not asymptotic convergence.
