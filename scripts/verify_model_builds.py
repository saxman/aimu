"""Ask the HuggingFace Hub which builds exist for each catalogued model.

Evidence-gathering for the catalog parity fill: AIMU never ships a model id it has not
shown to exist (the curated-catalog policy in CLAUDE.md). Run:

    python scripts/verify_model_builds.py > /tmp/builds.md

KNOWN UNRELIABLE: hand-verify every row, not just the ones printed as NO / UNRESOLVED / ERROR.
`classify()` picks the first search hit whose repo *owner* is in a hardcoded allowlist. It never
checks that the hit's *name* matches the model you searched for, so a same-family sibling repo
from an allowlisted owner comes back looking exactly as confident as a correct answer -- an
owner match is not a name match. Concretely, `search("Qwen3-4B")` can and does rank
`Qwen/Qwen3.5-4B` (a different generation) ahead of `Qwen/Qwen3-4B` (the one you meant); both
have owner "qwen", so `classify()` cannot tell them apart and just returns the first one.

The 2026-08-22 model-catalog-parity sweep (see this repo's `docs/superpowers/specs/` for the
full appendix, if present -- that directory is gitignored, so it may not be checked out) hand-
verified all 27 rows against the raw Hub JSON and found this heuristic wrong on 5 of them,
each returning a plausible, wrong, unflagged answer:
  - QWEN_3_4B: returned `Qwen/Qwen3.5-4B` instead of `Qwen/Qwen3-4B` (wrong generation).
  - PHI_4_14B: returned `microsoft/Phi-4-mini-instruct` instead of `microsoft/phi-4` (wrong
    model entirely -- the mini variant, already a separate catalog entry).
  - NEMOTRON_3_NANO_30B: returned `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` instead
    of `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (a different, multimodal "Omni" model).
  - MINISTRAL_3_14B: returned `mistralai/Ministral-3-14B-Reasoning-2512` instead of
    `mistralai/Ministral-3-14B-Instruct-2512` (the Reasoning variant, not Instruct).
  - MUSE_GLIMMER_30B: printed **UNRESOLVED** rather than a wrong answer, but for the same root
    cause -- the canonical owner, "meta-models", was simply missing from the allowlist below.
That is an 18.5% wrong-but-confident rate on output that otherwise reads as "resolved." Treat
every printed row -- "yes", a repo id, `NO`, `UNRESOLVED`, or `ERROR` alike -- as a lead to
confirm with a narrower manual query (`?search=<tighter term>`, checked against `hit["id"]` and,
where two similarly-named repos exist, the model card), never as a citable fact on its own.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request

from aimu.models._catalog import MODEL_FACTS

API = "https://huggingface.co/api/models"


class LookupFailed(Exception):
    """A Hub query for one row failed (network / rate limit / bad response).

    Distinct from a resolved-empty result on purpose: a row that errored has not been shown to
    lack a build, it simply wasn't checked. Conflating "the Hub said no" with "we couldn't ask
    the Hub" would silently turn a network hiccup into a false claim of absence -- exactly the
    kind of confident-wrong answer this script exists to avoid.
    """


# Search terms per canonical name: what to type into the Hub to find this model's builds.
# Seeded from the ids already in the catalogs; edit as the sweep reveals better terms.
SEARCH_TERMS: dict[str, str] = {
    "QWEN_3_8_27B": "Qwen3.8-27B",
    "QWEN_3_6_35B": "Qwen3.6-35B-A3B",
    "QWEN_3_6_27B": "Qwen3.6-27B",
    "QWEN_3_5_9B": "Qwen3.5-9B",
    "QWEN_3_32B": "Qwen3-32B",
    "QWEN_3_8B": "Qwen3-8B",
    "QWEN_3_4B": "Qwen3-4B",
    "GEMMA_4_E4B": "gemma-4-E4B-it",
    "GEMMA_4_12B": "gemma-4-12b-it",
    "GEMMA_4_26B": "gemma-4-26B-A4B-it",
    "GEMMA_4_31B": "gemma-4-31B-it",
    "GEMMA_3_12B": "gemma-3-12b-it",
    "LLAMA_3_1_8B": "Llama-3.1-8B-Instruct",
    "LLAMA_3_2_3B": "Llama-3.2-3B-Instruct",
    "MISTRAL_7B": "Mistral-7B-Instruct-v0.3",
    "PHI_4_MINI_3_8B": "Phi-4-mini-instruct",
    "PHI_4_14B": "phi-4",
    "DEEPSEEK_R1_7B": "DeepSeek-R1-Distill-Qwen-7B",
    "DEEPSEEK_R1_8B": "DeepSeek-R1-Distill-Llama-8B",
    "GPT_OSS_20B": "gpt-oss-20b",
    "SMOLLM2_1_7B": "SmolLM2-1.7B-Instruct",
    "MAGISTRAL_SMALL_24B": "Magistral-Small",
    "MINISTRAL_3_14B": "Ministral-3-14B",
    "NEMOTRON_CASCADE_2_30B": "Nemotron-Cascade-2-30B",
    "NEMOTRON_3_NANO_30B": "Nemotron-3-Nano-30B",
    "GLM_4_7_FLASH_31B_Q4": "GLM-4.7-Flash",
    "MUSE_GLIMMER_30B": "Muse-Glimmer-30B",
}
# The starting point for each term is the id an existing catalog already uses: the vLLM /
# SGLang / HF-Serve entries are literal repo paths, so their id minus the owner prefix is the
# search term. The five names no compat catalog carries yet (the Nemotron pair, GLM, Magistral,
# Ministral) are derived from their Ollama tag and must be confirmed by hand in Step 3.


def search(term: str, extra: str = "") -> list[dict]:
    url = f"{API}?search={urllib.parse.quote(term + extra)}&limit=50&full=false"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        # Covers urllib.error.HTTPError too (a URLError subclass): a 429 rate-limit, a transient
        # DNS/connection failure, or a non-JSON body all land here. Re-raised as LookupFailed so
        # main() can report this row as errored rather than silently resolving it to nothing.
        raise LookupFailed(f"{term!r}: {exc}") from exc


def classify(term: str) -> dict[str, list[str]]:
    # See the KNOWN UNRELIABLE section of the module docstring before trusting this function's
    # output: it matches on repo *owner* only, never on whether the hit's *name* is the model
    # you searched for, so an allowlisted owner's same-family sibling repo is indistinguishable
    # from a correct answer. Every result needs hand-verification, not only NO/UNRESOLVED ones.
    hits = search(term)
    out = {"canonical": [], "gguf": [], "mlx": []}
    for hit in hits:
        repo = hit["id"]
        owner, _, _ = repo.partition("/")
        tags = set(hit.get("tags", []))
        if "gguf" in tags or repo.upper().endswith("GGUF"):
            out["gguf"].append(repo)
        elif owner == "mlx-community":
            out["mlx"].append(repo)
        elif owner.lower() in {
            "qwen",
            "google",
            "meta-llama",
            "microsoft",
            "deepseek-ai",
            "mistralai",
            "nvidia",
            "openai",
            "zai-org",
            "huggingfacetb",
        }:
            out["canonical"].append(repo)
    return out


def main() -> int:
    missing = sorted(set(MODEL_FACTS) - set(SEARCH_TERMS))
    if missing:
        print(f"<!-- no SEARCH_TERMS entry for: {missing} -->")
    print("| model | canonical repo | GGUF | MLX |")
    print("|---|---|:---:|:---:|")
    for name in sorted(SEARCH_TERMS):
        try:
            found = classify(SEARCH_TERMS[name])
        except LookupFailed as exc:
            # A distinct third outcome from "resolved" and "NO"/"UNRESOLVED": this row was never
            # actually checked, so it must not be read as evidence of absence. The sweep
            # continues so one failed lookup doesn't cost the other 26 rows' worth of output.
            print(f"| `{name}` | **ERROR: {exc}** | ERROR | ERROR |")
            continue
        canonical = found["canonical"][0] if found["canonical"] else "**UNRESOLVED**"
        print(f"| `{name}` | `{canonical}` | {'yes' if found['gguf'] else 'NO'} | {'yes' if found['mlx'] else 'NO'} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
