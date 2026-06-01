"""hotdog_img2img.py — iterative hotdog refinement via image-to-image + strength annealing.

Strategy: hill climbing in image space with strength annealing.

- Every iteration refines the current *best* image (not just the most recent one),
  so a bad round never pulls the next generation in the wrong direction.
- `strength` starts high (explore: big changes from the reference) and anneals toward
  a floor (exploit: polish what's already working). The schedule is linear.
- The first iteration is always plain txt2img — there is no reference image yet.
  All subsequent iterations use img2img conditioned on the current best.
- On acceptance (new best): the new image becomes the reference, `rejected` clears,
  staleness resets. The evaluator's CONTINUE description drives the next prompt.
- On rejection: the best image stays as the reference. `refine_image()` is called on
  that best image to get a fresh direction, avoiding ideas already tried.
- Stops on: evaluator returning DONE, `--patience` consecutive non-improvements,
  or `--max-iterations` reached.

Usage::

    python scripts/hotdog_img2img.py
    python scripts/hotdog_img2img.py --image-model FLUX_2_KLEIN_4B --max-iterations 12
    python scripts/hotdog_img2img.py --initial-strength 0.85 --final-strength 0.2 --patience 3
    python scripts/hotdog_img2img.py --seed 42
"""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import aimu
from aimu import image_client

from _hotdog_common import (
    NEGATIVE_PROMPT,
    build_arg_parser,
    build_image_prompt,
    build_summarizer_prompt,
    collage_generated_images,
    evaluate_image,
    refine_image,
    resolve_image_model,
    resolve_output_dir,
    summarize_for_image,
    suppress_benign_clip_warning,
    write_summary,
)

_DEFAULT_INITIAL_STRENGTH = 0.8
_DEFAULT_FINAL_STRENGTH = 0.3
_DEFAULT_PATIENCE = 4
_DEFAULT_PROMPT = (
    "a single hotdog sizzling on a grill, dramatic flames and charred marks, "
    "intense heat haze rising, photorealistic close-up"
)

# When --max-iterations 0 (unlimited), use this many steps for the annealing schedule
# so strength still decays rather than staying fixed at initial forever.
_UNLIMITED_ANNEAL_STEPS = 20


def _anneal_strength(iteration: int, steps: int, initial: float, final: float) -> float:
    """Linear decay: strength(0) = initial, strength(steps-1) ≈ final."""
    if steps <= 1:
        return initial
    frac = min(1.0, iteration / (steps - 1))
    return max(final, initial - (initial - final) * frac)


def run(
    *,
    image_model_name: str,
    eval_model: str,
    output_dir: Path,
    max_iterations: int,
    initial_strength: float,
    final_strength: float,
    patience: int,
    seed: int | None,
) -> None:
    rng = random.Random(seed)

    img_client = image_client(resolve_image_model(image_model_name))
    eval_client = aimu.client(eval_model)
    suppress_benign_clip_warning(img_client)

    max_prompt_tokens = img_client.max_prompt_tokens
    use_summarizer = max_prompt_tokens is not None
    anneal_steps = max_iterations if max_iterations > 0 else _UNLIMITED_ANNEAL_STEPS
    uses_strength = getattr(img_client.spec, "img2img_uses_strength", True)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Image model : {image_model_name}")
    print(f"Eval model  : {eval_model}")
    if uses_strength:
        print(f"Strength    : {initial_strength:.2f} → {final_strength:.2f} (linear over {anneal_steps} steps)")
    else:
        print(
            f"Strength    : not supported by {image_model_name} — annealing schedule has no effect. "
            f"The script will still run hill climbing in image space, but all img2img iterations "
            f"condition on the reference at a fixed model-determined degree."
        )
    print(f"Patience    : {patience}")
    print(f"Output      : {output_dir}\n")

    prompt = _DEFAULT_PROMPT
    best: dict | None = None        # best["score"], best["path"], best["prompt"]
    reference_path: str | None = None
    rejected: list[str] = []
    stale = 0
    trace: list[dict] = []

    loop = range(1, max_iterations + 1) if max_iterations > 0 else iter(range(1, 10_001))

    try:
        for i in loop:
            # Strength for this iteration (0-indexed for the schedule)
            strength = _anneal_strength(i - 1, anneal_steps, initial_strength, final_strength)

            image_prompt = build_image_prompt(prompt)

            gen_kwargs: dict = dict(
                negative_prompt=NEGATIVE_PROMPT,
                format="path",
                output_dir=output_dir,
            )
            if seed is not None:
                gen_kwargs["seed"] = rng.randint(0, 2**31)
            if reference_path is not None:
                gen_kwargs["reference_image"] = reference_path
                if uses_strength:
                    gen_kwargs["strength"] = strength

            raw_path = img_client.generate(image_prompt, **gen_kwargs)

            dest = output_dir / f"{i:02d}.png"
            Path(raw_path).rename(dest)

            if not reference_path:
                mode = "txt2img (bootstrap)"
            elif uses_strength:
                mode = f"img2img  strength={strength:.2f}"
            else:
                mode = "img2img"
            print(f"[{i:02d}] {mode}")
            print(f"      prompt: {prompt[:90]}...")

            evaluator_response, parsed = evaluate_image(eval_client, str(dest))
            score = parsed.get("score")
            action = parsed.get("action", "unknown")

            current_score = score or 0
            best_score = (best["score"] or 0) if best else -1

            if current_score > best_score:
                best = {"score": score, "path": str(dest), "prompt": prompt}
                reference_path = str(dest)
                rejected = []
                stale = 0
                status = "accepted (new best)"
                print(f"      score={score}/10  ✓ new best")
            else:
                stale += 1
                status = "rejected"
                if parsed.get("next_prompt"):
                    rejected.append(parsed["next_prompt"])
                print(f"      score={score}/10  ✗ no improvement (stale {stale}/{patience})")

            record: dict = {
                "iteration": i,
                "strength": round(strength, 3),
                "prompt": prompt,
                "image_prompt": image_prompt,
                "image_path": str(dest),
                "evaluator_response": evaluator_response,
                "score": score,
                "action": action,
                "next_prompt": parsed.get("next_prompt"),
                "reference_path": reference_path,
                "status": status,
            }

            trace.append(record)

            if action == "DONE":
                print("\nEvaluator satisfied — stopping.")
                break

            if stale >= patience:
                print("\nPatience exhausted — stopping.")
                break

            # Derive next prompt
            if status.startswith("accepted"):
                # Evaluator's CONTINUE suggestion describes what to push further
                raw_next = parsed.get("next_prompt") or prompt
            else:
                # Ask for a fresh direction on the best image, excluding already-tried ideas
                assert best is not None
                raw_next = refine_image(eval_client, best["path"], rejected=rejected)

            if use_summarizer:
                prompt = summarize_for_image(eval_client, raw_next, max_prompt_tokens)
                record["summarized_prompt"] = prompt
            else:
                prompt = raw_next

    finally:
        write_summary(
            output_dir,
            trace,
            image_model=image_model_name,
            eval_model=eval_model,
            summarizer_instruction=build_summarizer_prompt(max_prompt_tokens) if use_summarizer else None,
        )
        collage_generated_images(output_dir, trace)
        if best:
            shutil.copy(best["path"], output_dir / "best.png")
            print(f"\nBest : {best['score']}/10  →  {best['path']}")
        print(f"Output: {output_dir}")


def main() -> None:
    parser = build_arg_parser(
        "Iterative hotdog refinement via image-to-image with strength annealing. "
        "Keeps the best image as the reference and anneals strength from high "
        "(explore) to low (polish) over the run."
    )
    parser.add_argument(
        "--initial-strength",
        type=float,
        default=_DEFAULT_INITIAL_STRENGTH,
        metavar="F",
        help=f"Starting img2img strength (0–1). Higher = bigger changes from reference. "
             f"Default: {_DEFAULT_INITIAL_STRENGTH}",
    )
    parser.add_argument(
        "--final-strength",
        type=float,
        default=_DEFAULT_FINAL_STRENGTH,
        metavar="F",
        help=f"Minimum strength after annealing. Default: {_DEFAULT_FINAL_STRENGTH}",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=_DEFAULT_PATIENCE,
        metavar="N",
        help=f"Stop after N consecutive non-improvements. Default: {_DEFAULT_PATIENCE}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Random seed for reproducible generation (optional).",
    )
    args = parser.parse_args()

    run(
        image_model_name=args.image_model,
        eval_model=args.eval_model,
        output_dir=resolve_output_dir(args.output_dir),
        max_iterations=args.max_iterations,
        initial_strength=args.initial_strength,
        final_strength=args.final_strength,
        patience=args.patience,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
