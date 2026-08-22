"""Regenerate the tables in docs/reference/model-matrix.md from the model catalogs.

The prose is hand-written and untouched; only the marker-delimited tables (``<!-- generated:X -->
... <!-- /generated -->``) are emitted. Run with --write to update the file in place, or with no
arguments to print it (which is what tests/test_docs_model_matrix.py compares against).

This script is tooling, not part of the package: nothing under aimu/ imports it, and it adds no
new dependency.

**Scope of what is derived vs. hand-encoded.** Every id, membership, and capability flag comes
live from the catalogs (aimu/models/providers/*, aimu/models/_catalog.py) -- that is the parity
concern this script exists to prevent from drifting again. A handful of footnote *symbols* in the
"OpenAI-compatible local servers" table point at prose paragraphs a generator cannot write (why
GEMMA_3_12B's tools flag differs on Ollama, why Muse Glimmer needs its own parser, why MLX ships
per-quantization ids): those symbols are attached by small, explicit rules below (see
"_servers_markers" and "_canonical_and_marker"), not by re-deriving the prose itself. Where the
catalog data disagrees with the previously hand-maintained doc (see the task-12 report), this
script's output is the corrected value.
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Optional

from aimu import paths
from aimu.models._catalog import MODEL_FACTS
from aimu.models.providers.anthropic import AnthropicModel, ThinkingStyle
from aimu.models.providers.gemini.text import GeminiModel
from aimu.models.providers.hf.text import HuggingFaceModel
from aimu.models.providers.llamacpp import LlamaCppModel
from aimu.models.providers.ollama import OllamaModel
from aimu.models.providers.openai.text import OpenAIModel
from aimu.models.providers.openai_compat import (
    HFOpenAIModel,
    LlamaServerOpenAIModel,
    LMStudioOpenAIModel,
    OllamaOpenAIModel,
    OMLXOpenAIModel,
    SGLangOpenAIModel,
    VLLMOpenAIModel,
)

MATRIX_PATH = paths.root / "docs" / "reference" / "model-matrix.md"

# Tables keyed by enum-id-column tables, in the order they appear in the doc.
ID_TABLES = {
    "AnthropicModel": AnthropicModel,
    "OpenAIModel": OpenAIModel,
    "GeminiModel": GeminiModel,
    "OllamaModel": OllamaModel,
    "HuggingFaceModel": HuggingFaceModel,
    "LlamaCppModel": LlamaCppModel,
}

# The id column header varies by table: Ollama/Anthropic/OpenAI/Gemini call it a "Model id"
# (a real, servable identifier); HuggingFace calls it a "Repo id"; llama-cpp calls it a "Hint id"
# because the value is ignored at load time (model_path= is what actually loads).
ID_COLUMN_LABEL = {
    "AnthropicModel": "Model id",
    "OpenAIModel": "Model id",
    "GeminiModel": "Model id",
    "OllamaModel": "Model id",
    "HuggingFaceModel": "Repo id",
    "LlamaCppModel": "Hint id",
}

# The OpenAI-compat table is keyed by enum member NAME (ids differ per server). Order matters:
# it is both the default "list servers explicitly" order and defines the NON_MLX_SERVERS set.
SERVER_ENUMS = {
    "Ollama": OllamaOpenAIModel,
    "LM Studio": LMStudioOpenAIModel,
    "vLLM": VLLMOpenAIModel,
    "HF Serve": HFOpenAIModel,
    "llama-server": LlamaServerOpenAIModel,
    "SGLang": SGLangOpenAIModel,
    "oMLX": OMLXOpenAIModel,
}
NON_MLX_SERVERS = [s for s in SERVER_ENUMS if s != "oMLX"]

# Suffixes that mark a per-quantization MLX sibling (oMLX / LM Studio's MLX engine), in the order
# they are listed when a base model has more than one. See the "GPT_OSS_20B_MXFP4_Q4" /
# "PHI_4_MINI_3_8B_FP16" footnote in the doc for why these aren't a uniform "-4bit/-8bit/-bf16".
_QUANT_SUFFIXES = ["_4BIT", "_8BIT", "_BF16", "_FP16", "_MXFP4_Q4", "_MXFP4_Q8"]

BLOCK = re.compile(r"(<!-- generated:(?P<id>[\w-]+) -->\n).*?(\n<!-- /generated -->)", re.DOTALL)


def _tick(supported: bool) -> str:
    return "✅" if supported else "✗"  # ✅ / ✗


def _thinking_legend_marks(member) -> str:
    """◆ = accepts thinking= effort levels, ◇ = always reasons (thinking_optional=False)."""
    marks = ""
    if member.thinking_levels:
        marks += " ◆"  # ◆
    if not member.thinking_optional:
        marks += " ◇"  # ◇
    return marks


def _fp8_mark(name: str) -> str:
    """§: the e4m3 FP8 checkpoint footnote (hardware-gated, see the doc paragraph)."""
    return " §" if name.endswith("_FP8") else ""


def _vision_override_mark(name: str, member) -> str:
    """*: llama-cpp overrides an intrinsically vision-capable model to vision=False.

    True exactly when MODEL_FACTS says the weights are vision-capable but this catalog's Wire
    overrides it off (no mmproj projector loaded by default). Reading the raw Wire (rather than
    re-deriving "no mmproj" from prose) is what keeps this rule a fact-check instead of a guess.
    """
    wire = getattr(member, "_wire", None)
    if wire is None:
        return ""
    facts = MODEL_FACTS.get(name)
    if facts is not None and facts.vision and wire.overrides.get("vision") is False:
        return " *"
    return ""


def _thinking_cell(table_id: str, member) -> str:
    """Anthropic spells out which request shape (budget vs. adaptive) thinking=True uses."""
    if table_id == "AnthropicModel":
        style = "adaptive" if member.thinking_style is ThinkingStyle.ADAPTIVE else "budget"
        return f"{_tick(member.supports_thinking)} ({style})"
    return _tick(member.supports_thinking)


def _id_table(table_id: str, enum) -> str:
    id_label = ID_COLUMN_LABEL[table_id]
    lines = [
        f"| Enum member | {id_label} | Tools | Thinking | Vision |",
        "|---|---|:---:|:---:|:---:|",
    ]
    for member in enum:
        marks = _thinking_legend_marks(member) + _fp8_mark(member.name)
        if table_id == "LlamaCppModel":
            marks += _vision_override_mark(member.name, member)
        lines.append(
            f"| `{member.name}`{marks} | `{member.value}` | "
            f"{_tick(member.supports_tools)} | {_thinking_cell(table_id, member)} | "
            f"{_tick(member.supports_vision)} |"
        )
    return "\n".join(lines)


def _quant_base(name: str) -> Optional[str]:
    """The bare model name a per-quantization member id is a sibling of, or None if not one."""
    for suffix in _QUANT_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def _collect_server_members() -> dict[str, dict[str, object]]:
    """member name -> {server label: enum member}, across every OpenAI-compat server catalog."""
    by_name: dict[str, dict[str, object]] = {}
    for label, enum in SERVER_ENUMS.items():
        for member in enum:
            by_name.setdefault(member.name, {})[label] = member
    return by_name


def _row_order(by_name: dict[str, dict[str, object]]) -> list[str]:
    """Bases in VLLMOpenAIModel's definition order, each followed by its quant siblings.

    vLLM (like llama-server/HF Serve/SGLang, which all agree with it) enumerates the bare model
    set with no per-quantization members, so its definition order is the natural "one row per
    model family" reference the per-quant oMLX/LM Studio ids get grouped under.
    """
    bases = {name for name in by_name if _quant_base(name) is None}
    quant_children: dict[str, list[str]] = {}
    for name in by_name:
        base = _quant_base(name)
        if base is not None:
            quant_children.setdefault(base, []).append(name)

    def quant_key(name: str) -> int:
        for i, suffix in enumerate(_QUANT_SUFFIXES):
            if name.endswith(suffix):
                return i
        return len(_QUANT_SUFFIXES)

    ordered_bases = [m.name for m in VLLMOpenAIModel if m.name in bases]
    leftover_bases = sorted(bases - set(ordered_bases))
    if leftover_bases:
        # A base that exists only via its quant siblings (no vLLM member at all) would silently
        # vanish from a vLLM-order walk; surface it instead of dropping the row.
        ordered_bases += leftover_bases

    orphans = sorted(set(quant_children) - bases)
    if orphans:
        raise SystemExit(
            f"quant sibling(s) with no base row anywhere in SERVER_ENUMS: {orphans}. "
            f"Add a footnote/handling rule before regenerating."
        )

    rows: list[str] = []
    for base in ordered_bases:
        rows.append(base)
        rows.extend(sorted(quant_children.get(base, []), key=quant_key))
    return rows


def _canonical_and_marker(carriers: dict[str, object], attr: str, symbol: str) -> str:
    """Majority-vote a flag across the servers that actually carry this member.

    Returns the tick, with the disagreement symbol appended when the servers don't all agree
    (e.g. GEMMA_3_12B: Ollama's in-process tool-call parser can't handle Gemma 3, everything
    else can). A tie has no established convention, so it raises rather than guessing.
    """
    values = {label: getattr(member, attr) for label, member in carriers.items()}
    distinct = set(values.values())
    if len(distinct) == 1:
        return _tick(next(iter(distinct)))
    true_labels = [label for label, value in values.items() if value]
    false_labels = [label for label, value in values.items() if not value]
    if len(true_labels) == len(false_labels):
        raise SystemExit(f"tie in {attr!r} across servers, no majority: {values}")
    canonical = len(true_labels) > len(false_labels)
    return f"{_tick(canonical)} {symbol}"


def _servers_text(carriers: set) -> str:
    non_mlx = set(NON_MLX_SERVERS)
    if carriers == non_mlx:
        return "all"
    missing = non_mlx - carriers
    if "oMLX" not in carriers and len(missing) == 1:
        return "all except " + ", ".join(s for s in SERVER_ENUMS if s in missing)
    if carriers <= {"oMLX", "LM Studio"}:
        return ", ".join(s for s in ("oMLX", "LM Studio") if s in carriers)
    return ", ".join(s for s in SERVER_ENUMS if s in carriers)


def _servers_markers(name: str, carriers: set) -> str:
    marks = []
    if name.startswith("MUSE_GLIMMER"):
        # ‡: needs a dedicated parser most servers don't have; see the doc paragraph.
        marks.append("‡")
    if "oMLX" in carriers and (carriers <= {"oMLX", "LM Studio"} or carriers == set(NON_MLX_SERVERS) | {"oMLX"}):
        # ¶: this row is part of the MLX per-quantization family.
        marks.append("¶")
    return "".join(f" {mark}" for mark in marks)


def _server_table() -> str:
    by_name = _collect_server_members()
    lines = [
        "| Enum member | Tools | Thinking | Vision | Servers |",
        "|---|:---:|:---:|:---:|---|",
    ]
    for name in _row_order(by_name):
        carriers = by_name[name]
        thinking_values = {getattr(member, "supports_thinking") for member in carriers.values()}
        if len(thinking_values) > 1:
            raise SystemExit(f"unhandled 'thinking' disagreement for {name}: {carriers.keys()}")
        sample = next(iter(carriers.values()))

        name_cell = f"`{name}`{_thinking_legend_marks(sample)}"
        tools_cell = _canonical_and_marker(carriers, "supports_tools", "†")  # †
        thinking_cell = _tick(next(iter(thinking_values)))
        vision_cell = _canonical_and_marker(carriers, "supports_vision", "※")  # ※
        servers_cell = _servers_text(set(carriers)) + _servers_markers(name, set(carriers))

        lines.append(f"| {name_cell} | {tools_cell} | {thinking_cell} | {vision_cell} | {servers_cell} |")
    return "\n".join(lines)


def render() -> str:
    text = MATRIX_PATH.read_text()

    def replace(match: re.Match) -> str:
        table_id = match.group("id")
        if table_id == "servers":
            body = _server_table()
        elif table_id in ID_TABLES:
            body = _id_table(table_id, ID_TABLES[table_id])
        else:
            raise SystemExit(f"unknown generated block id: {table_id!r}")
        return match.group(1) + body + match.group(3)

    rendered, count = BLOCK.subn(replace, text)
    expected = len(ID_TABLES) + 1  # + the cross-server table
    if count != expected:
        raise SystemExit(f"expected {expected} generated blocks, found {count}")
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="update the file in place")
    args = parser.parse_args()

    rendered = render()
    if args.write:
        MATRIX_PATH.write_text(rendered)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
