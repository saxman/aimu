# Iterative image refinement

Build a **generate → evaluate → refine** loop: a model produces an image, a vision model critiques it and proposes an improved prompt, and the loop repeats until the critic is satisfied. This guide walks through a worked example — *"make a hotdog as visually hot as possible"* — that composes image generation, vision input, prompt chaining, and either an agent, a plain Python loop, or a library workflow class.

The full, runnable code lives in five scripts that share one helper module:

- `scripts/hotdog_loop.py` — **Python directs the loop**. Two strategies via `--strategy`:
  - `greedy` (default) — always accept the evaluator's suggestion and move on.
  - `climbing` — keep the best image and revert on non-improvement (`--patience` controls the stop condition).
- `scripts/hotdog_agent.py` — **an `Agent` directs the loop** via tool calls (autonomous).
- `scripts/hotdog_evaluator.py` — **`EvaluatorOptimizer` directs the loop** (the library workflow class, composing a generator + critic agent).
- `scripts/hotdog_anneal.py` — **simulated annealing**: generalises the climber — accepts worse images early (high temperature) to escape local optima, cooling to greedy.
- `scripts/hotdog_img2img.py` — **image-to-image refinement**: each iteration starts from the current best image rather than pure noise, combining hill-climbing with strength annealing (high `strength` early to explore, low `strength` late to polish).
- `scripts/_hotdog_common.py` — shared prompts and helpers used by all of them.

```bash
python scripts/hotdog_loop.py                                        # greedy walk (default)
python scripts/hotdog_loop.py --strategy climbing --patience 4       # hill-climb: keep best, revert on regression
python scripts/hotdog_loop.py --max-iterations 0                     # run until the critic says DONE
python scripts/hotdog_agent.py                                       # agent-directed greedy walk
python scripts/hotdog_evaluator.py                                   # EvaluatorOptimizer workflow class
python scripts/hotdog_anneal.py --seed 7                             # simulated annealing: explore early, cool to greedy
python scripts/hotdog_img2img.py                                     # img2img hill-climbing + strength annealing
python scripts/hotdog_img2img.py --image-model FLUX_2_KLEIN_4B      # unified pipeline — no strength knob (warning printed)
python scripts/hotdog_img2img.py --initial-strength 0.85 --final-strength 0.2 --patience 3
```

## Three ways to run the same loop

The same task is implemented three times on purpose — it's a concrete take on [agents vs workflows](../explanation/agents-vs-workflows.md):

- **Code-directed** (`hotdog_loop.py`): you write the `for` loop. Flow is explicit and deterministic — easiest to read, debug, and bound. Reach for this when the control flow is fixed and you want full visibility. Supports `--strategy greedy` (always accept) and `--strategy climbing` (keep best, revert on regression) — the two differ only in their acceptance policy.
- **Agent-directed** (`hotdog_agent.py`): you hand an [`Agent`](../reference/api/agents.md) three tools (`generate_hotdog_image`, `evaluate_hotness`, `summarize_description`) and let its tool-calling loop decide when to call what. Reach for this when you want the model to own the control flow, or as a stepping stone to more open-ended behaviour.
- **Workflow-class** (`hotdog_evaluator.py`): you express the loop with [`EvaluatorOptimizer`](../reference/api/agents.md), the library's generate → evaluate → revise workflow, by composing a generator agent and a critic agent. Reach for this when your task already matches a named pattern and you'd rather configure it than hand-write the loop.

All three produce the same artifacts and share every prompt and helper, so the diff between them is purely *who drives the loop*.

## Search strategy: greedy → hill-climbing → simulated annealing

`--strategy greedy` (the default) and the agent script are **greedy**: each round they accept whatever prompt the critic proposes and move on — even if the new image scored *lower* than a previous one. That's simple and often fine, but the "best" image can be somewhere in the middle of the run rather than at the end.

`--strategy climbing` borrows the strategy from [`aimu.prompts`](tune-prompts.md) tuning — **best-state caching + revert when a step doesn't beat the best**:

- Track the highest-scoring image so far.
- If a generation **beats** the best (strictly higher score), adopt it and refine from there.
- If it **doesn't** (a lower score *or* a tie — the coarse 1–10 judge makes ties common, and a tie isn't demonstrated progress), discard it, revert to the best, and ask the critic for a *different* refinement (passing the ideas that already failed so it explores a new direction).
- Stop on `DONE`, after `--patience` consecutive non-improvements, or at `--max-iterations`. The winner is copied to `best.png`.

This makes the search monotonic (the result is never worse than the best seen) at the cost of extra critic calls on non-improving steps. It's the single-artifact analog of how prompt tuning hill-climbs a reusable prompt — see the comparison in [Tune prompts](tune-prompts.md).

`hotdog_anneal.py` **generalises the climber into simulated annealing** — in fact the climber is annealing's `T → 0` limit. It keeps a `current` walk-state (distinct from the best-ever) and accepts the critic's proposed step by the Metropolis rule, where `Δ = new_score − current_score`:

- `Δ > 0` (hotter): always accept.
- `Δ == 0` (tie): always accept — a free sideways move, the deliberate *opposite* of the climber's revert-on-tie.
- `Δ < 0` (cooler): accept with probability `exp(Δ / T)`. A falling temperature (`--initial-temp`, `--cooling-rate`) means it explores freely early — taking downhill steps to escape a local optimum — then cools into greedy behaviour. `--seed` makes the acceptance decisions reproducible. The best-ever image is still tracked separately and copied to `best.png`.

The same cooling schedule also drives the **proposer's** sampling temperature: the critic generates refinement ideas at a high LLM temperature while hot (diverse, exploratory) and a low one once cooled (conservative tweaks), via `_proposer_temperature`. Only proposals are annealed — the **judge that assigns the score stays cold**, since the score is the optimisation objective and must stay stable (annealing it would inject noise into the very signal you're climbing). This is the difference between annealing the *acceptance* and annealing the *proposal distribution* too; the script does both off one schedule.

Annealing is worth reaching for when the climber gets stuck on a plateau, but mind the domain: the judge's coarse integer score means `Δ` is almost always `0` or `±1` (so temperature control is blunt), and each step is an expensive image generation, so runs are short — the practical win is local-optimum escape, not asymptotic convergence.

Note the *opposite* direction (using prompt tuning to find a reusable image prompt across a dataset) is a different, also-valid use of the `Scorer` abstraction.

### Image-to-image: refining in image space, not prompt space

All five scripts above refine by updating the *prompt* and regenerating from pure noise — the image can change completely between rounds (different composition, different colours, different character). `hotdog_img2img.py` takes a different axis: it uses [image-to-image generation](generate-images.md#image-to-image) so each round starts from the current best image rather than Gaussian noise.

The `strength` parameter controls how far each round can stray from the reference:

- **High strength (≈0.8)** — the reference is a loose constraint; large changes are possible. Used early in the run to explore: push lighting, texture, and heat intensity while preserving rough composition.
- **Low strength (≈0.3)** — the reference is a tight constraint; only fine adjustments get through. Used late to polish what's already working without blowing up the composition.

The script anneals `strength` **linearly** from `--initial-strength` to `--final-strength` over the run, and combines this with hill-climbing: it always refines from the *best* image seen so far, not just the most recent. A rejected round leaves the reference unchanged.

Compared with the prompt-refinement scripts, img2img preserves things that are already good (framing, a specific sear pattern, a particular colour palette) at the cost of constraining the search: if the reference is pulling in the wrong direction, low strength slows the escape. The tradeoff is deliberate — "make this hotter, don't change the composition" vs "try a completely different take".

**Models without a `strength` knob** (e.g. `FLUX_2_KLEIN_4B`, which uses a unified `Flux2KleinPipeline` that conditions on the reference image directly): the script detects this via `spec.img2img_uses_strength` and prints a warning at startup. The annealing schedule is a no-op — all iterations condition at the model's fixed degree of influence — but hill-climbing in image space still applies.

## The pieces

### Two clients: a generator and a vision critic

```python
import aimu

image_client = aimu.image_client("hf:stabilityai/stable-diffusion-3.5-medium")
eval_client = aimu.client("ollama:gemma4:e4b")          # any vision-capable chat model
if not eval_client.is_vision_model:
    raise ValueError("the eval model must support image input")
```

The critic is a normal chat client used with `images=` (see [Handle vision input](handle-vision.md)); the generator is an image client (see [Generate images](generate-images.md)).

### The evaluator contract

The critic is prompted to return a score plus a machine-readable verdict — `DONE:` (stop) or `CONTINUE:` (here's how to make it hotter):

```text
Rate its hotness from 1 to 10. Then decide: can it get hotter? If not, output exactly:
DONE: <reasoning>
If it can, output exactly:
CONTINUE: <a natural-language description of how the next image should look>
```

A small regex parser turns that into `{"action": "DONE"|"CONTINUE"|"unknown", "score", "next_prompt"}`, so the loop has a clean stop condition.

### Code-directed loop

```python
prompt = "a hot hotdog"
for i in range(1, max_iterations + 1):
    path = image_client.generate(build_image_prompt(prompt), format="path", output_dir=out)
    verdict = eval_client.generate(EVALUATOR_PROMPT, images=[path])  # stateless one-shot — no reset()
    parsed = parse_evaluator_response(verdict)
    if parsed["action"] == "DONE":
        break
    prompt = summarize_for_image(eval_client, parsed["next_prompt"], image_client.max_prompt_tokens)
```

### Agent-directed loop

Wrap the same three operations as `@tool`s and let the agent sequence them:

```python
from aimu.agents import Agent

agent = Agent(
    agent_client,
    name="hotdog-agent",
    system_message=AGENT_SYSTEM_PROMPT,   # "generate → evaluate → (summarize) → repeat until DONE"
    tools=[generate_hotdog_image, evaluate_hotness, summarize_description],
    max_iterations=max_iterations * 4,    # ~3 tool calls per round + headroom
)
agent.run("Begin the hotdog heating experiment. Start with 'a hot hotdog'.")
```

The agent stops on its own once the critic returns `DONE` (it simply stops calling tools). `--max-iterations 0` removes the hard cap so the *critic* decides when to stop.

### Workflow-class loop ([`EvaluatorOptimizer`](../reference/api/agents.md))

The hand-written loop *is* generate → evaluate → revise — which is exactly what `EvaluatorOptimizer` automates. So the same task drops onto the workflow class by composing two agents: a **generator** holding `generate_hotdog_image`, and an **evaluator** holding `evaluate_hotness`. The workflow stops when `pass_keyword` shows up in the critic's reply.

```python
from aimu.agents import Agent, EvaluatorOptimizer

eo = EvaluatorOptimizer(
    generator=Agent(gen_client, "Write a prompt, generate, reply with the path.", tools=[generate_hotdog_image]),
    evaluator=Agent(critic_client, "Evaluate the image; relay DONE or CONTINUE verbatim.", tools=[evaluate_hotness]),
    max_rounds=max_iterations,
    pass_keyword="DONE",
)
eo.run("Make a single hotdog look as hot as possible. Start from: a hot hotdog.")
```

The catch worth seeing: `EvaluatorOptimizer` orchestrates *text* `Runner`s — it passes each step's output to the next as a **string** (see `aimu/agents/workflows/evaluator.py`). Image generation and vision evaluation therefore live *inside the tools*, and the image path travels between the two agents as text — two extra LLM hops for what `hotdog_loop.py` hands off as a plain variable. That's more indirect and more fragile (the script falls back to the latest image if the relayed path doesn't resolve). It also needs **three** client instances — a generator brain, a critic brain, and a separate vision client for the tool (which calls `reset()`) — and the *final* generation is returned unevaluated, an artifact of the loop's shape. Reach for the workflow class when the pattern fit is clean; reach for the hand-written loop when you want every hand-off explicit.

### Prompt chaining: describe richly, then summarize to the model's budget

The critic writes a detailed, free-form description (the "hot" it imagines). But image models cap prompt length — SDXL's CLIP encoder at 77 tokens, SD 3.5's T5 at 256. So a second step condenses the description to fit, sized from the model's own [`max_prompt_tokens`](generate-images.md#prompt-length):

```python
def summarize_for_image(client, description, max_prompt_tokens):
    client.reset()
    instruction = build_summarizer_prompt(max_prompt_tokens)   # "...under N words..."
    return client.chat(f"{instruction}\nDescription:\n{description}").strip()
```

This is [prompt chaining](../explanation/agents-vs-workflows.md): step N's output (a rich description) becomes step N+1's input (a tight prompt). When `max_prompt_tokens` is `None` (uncapped cloud models), the description is fed straight through and the summarize step is skipped.

### Keeping it to one subject

Diffusion models drift toward "more of a good thing." Two levers keep it to a single hotdog:

- **Positive anchor** — every prompt is prefixed with `"a single hotdog, one sausage in one bun, solo, centered, close-up shot"`. Stating the subject and framing is the reliable lever for count.
- **Negative prompt** — `"multiple, two, several, pile, platter, ..."` discourages plurality. It deliberately omits the word *hotdog* (a CLIP negative suppresses the *concept*, so "no hotdogs" removes the subject entirely).

### Placement and dtype are automatic

Nothing in any script pins a GPU or sets a dtype. `HuggingFaceImageClient` measures free VRAM across your GPUs and the model's size, then [places it for you](generate-images.md#gpu-placement-huggingface) — pinning to the freest card or falling back to CPU offload — and defaults to a memory-efficient dtype (bf16 on CUDA). So SD 3.5 loads alongside a local LLM without manual juggling.

### Durable output: summary + collage, even on Ctrl-C

Every script writes results in a `finally` block, so an interrupted run still produces output:

- `summary.txt` — the constant model inputs (negative prompt, evaluator/summarizer instructions) plus a per-iteration trace of exactly what was sent.
- `collage.png` — a near-square grid of every generated image, in order. It scans the saved files rather than the trace, so even an image generated just before you hit Ctrl-C is included.

### Quieting the benign CLIP warning — but only when it's benign

On SD 3.5, the two CLIP encoders always truncate prompts past 77 tokens and warn about it — harmless, because T5 carries the full prompt. The scripts filter *just that message* via `suppress_benign_clip_warning(image_client)`, and **only for models whose `max_prompt_tokens > 77`**. On a CLIP-only model (SDXL), the same warning means content was actually dropped, so it stays visible.

## See also

- [Generate images](generate-images.md) — the image client, placement, and prompt length
- [Handle vision input](handle-vision.md) — passing images to a chat model
- [Agents vs workflows](../explanation/agents-vs-workflows.md) — when to let the model drive
- [Build an orchestrator](build-orchestrator.md) — coordinating multiple worker agents
