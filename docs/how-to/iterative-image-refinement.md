# Iterative image refinement

Build a **generate → evaluate → refine** loop: a model produces an image, a vision model critiques it and proposes an improved prompt, and the loop repeats until the critic is satisfied. This guide walks through a worked example — *"make a hotdog as visually hot as possible"* — that composes image generation, vision input, prompt chaining, and either an agent or a plain Python loop.

The full, runnable code lives in three scripts that share one helper module:

- `scripts/hotdog_loop.py` — **Python directs the loop** (a plain code-controlled `for` loop).
- `scripts/hotdog_agent.py` — **an `Agent` directs the loop** via tool calls (autonomous).
- `scripts/hotdog_loop_climbing.py` — like the loop, but **hill-climbs**: keeps the best image and reverts on regression.
- `scripts/_hotdog_common.py` — shared prompts and helpers used by all three.

```bash
python scripts/hotdog_loop.py                 # code-directed greedy walk
python scripts/hotdog_agent.py                # agent-directed greedy walk
python scripts/hotdog_loop_climbing.py                # hill-climb: keep best, revert on regression
python scripts/hotdog_agent.py --max-iterations 0   # run until the critic says DONE
```

## Two ways to run the same loop

The same task is implemented twice on purpose — it's a concrete take on [agents vs workflows](../explanation/agents-vs-workflows.md):

- **Code-directed** (`hotdog_loop.py`): you write the `for` loop. Flow is explicit and deterministic — easiest to read, debug, and bound. Reach for this when the control flow is fixed and you want full visibility.
- **Agent-directed** (`hotdog_agent.py`): you hand an [`Agent`](../reference/api/agents.md) three tools (`generate_hotdog_image`, `evaluate_hotness`, `summarize_description`) and let its tool-calling loop decide when to call what. Reach for this when you want the model to own the control flow, or as a stepping stone to more open-ended behaviour.

Both produce the same artifacts and share every prompt and helper, so the diff between them is purely *who drives the loop*.

## Greedy walk vs hill-climbing

The loop and agent scripts are **greedy**: each round they accept whatever prompt the critic proposes and move on — even if the new image scored *lower* than a previous one. That's simple and often fine, but the "best" image can be somewhere in the middle of the run rather than at the end.

`hotdog_loop_climbing.py` borrows the strategy from [`aimu.prompts`](tune-prompts.md) tuning — **best-state caching + revert-on-regression**:

- Track the highest-scoring image so far.
- If a generation **improves** on the best, adopt it and refine from there.
- If it **regresses**, discard it, revert to the best, and ask the critic for a *different* refinement (passing the ideas that already failed so it explores a new direction).
- Stop on `DONE`, after `--patience` consecutive non-improvements, or at `--max-iterations`. The winner is copied to `best.png`.

This makes the search monotonic (the result is never worse than the best seen) at the cost of extra critic calls on regressions. It's the single-artifact analog of how prompt tuning hill-climbs a reusable prompt — see the comparison in [Tune prompts](tune-prompts.md). Note the *opposite* direction (using prompt tuning to find a reusable image prompt across a dataset) is a different, also-valid use of the `Scorer` abstraction.

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
    eval_client.reset()
    verdict = eval_client.chat(EVALUATOR_PROMPT, images=[path])
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

Nothing in either script pins a GPU or sets a dtype. `HuggingFaceImageClient` measures free VRAM across your GPUs and the model's size, then [places it for you](generate-images.md#gpu-placement-huggingface) — pinning to the freest card or falling back to CPU offload — and defaults to a memory-efficient dtype (bf16 on CUDA). So SD 3.5 loads alongside a local LLM without manual juggling.

### Durable output: summary + collage, even on Ctrl-C

Both scripts write results in a `finally` block, so an interrupted run still produces output:

- `summary.txt` — the constant model inputs (negative prompt, evaluator/summarizer instructions) plus a per-iteration trace of exactly what was sent.
- `collage.png` — a near-square grid of every generated image, in order. It scans the saved files rather than the trace, so even an image generated just before you hit Ctrl-C is included.

### Quieting the benign CLIP warning — but only when it's benign

On SD 3.5, the two CLIP encoders always truncate prompts past 77 tokens and warn about it — harmless, because T5 carries the full prompt. The scripts filter *just that message* via `suppress_benign_clip_warning(image_client)`, and **only for models whose `max_prompt_tokens > 77`**. On a CLIP-only model (SDXL), the same warning means content was actually dropped, so it stays visible.

## See also

- [Generate images](generate-images.md) — the image client, placement, and prompt length
- [Handle vision input](handle-vision.md) — passing images to a chat model
- [Agents vs workflows](../explanation/agents-vs-workflows.md) — when to let the model drive
- [Build an orchestrator](build-orchestrator.md) — coordinating multiple worker agents
