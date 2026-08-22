"""Ask the HuggingFace Hub which builds exist for each catalogued model.

Evidence-gathering for the catalog parity fill: AIMU never ships a model id it has not
shown to exist (the curated-catalog policy in CLAUDE.md). Run:

    python scripts/verify_model_builds.py > /tmp/builds.md
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request

from aimu.models._catalog import MODEL_FACTS

API = "https://huggingface.co/api/models"

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
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def classify(term: str) -> dict[str, list[str]]:
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
        found = classify(SEARCH_TERMS[name])
        canonical = found["canonical"][0] if found["canonical"] else "**UNRESOLVED**"
        print(f"| `{name}` | `{canonical}` | {'yes' if found['gguf'] else 'NO'} | {'yes' if found['mlx'] else 'NO'} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
