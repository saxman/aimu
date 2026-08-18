"""The model matrix claims to be kept up to date with the enums; this holds it to that.

docs/reference/model-matrix.md is maintained by hand, so it drifts silently: a new catalog
member gets no row, or a corrected capability flag gets fixed in the enum and not in the doc.
Both happened. These tests read the doc and compare it against the catalogs, so the next drift
fails the suite instead of misleading a reader.
"""

from __future__ import annotations

import re

import pytest

from aimu import paths
from aimu.models.providers.anthropic import AnthropicModel
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

# Tables that carry a model-id column, keyed by the heading that opens each one.
ID_TABLES = [
    ("## Anthropic", AnthropicModel),
    ("## OpenAI (", OpenAIModel),
    ("## Google Gemini", GeminiModel),
    ("## Ollama native", OllamaModel),
    ("## HuggingFace (", HuggingFaceModel),
    ("## llama-cpp", LlamaCppModel),
]

# The OpenAI-compat table is keyed by enum member NAME, since ids differ per server.
SERVER_ENUMS = {
    "Ollama": OllamaOpenAIModel,
    "LM Studio": LMStudioOpenAIModel,
    "vLLM": VLLMOpenAIModel,
    "HF Serve": HFOpenAIModel,
    "llama-server": LlamaServerOpenAIModel,
    "SGLang": SGLangOpenAIModel,
    "oMLX": OMLXOpenAIModel,
}

NON_MLX_SERVERS = set(SERVER_ENUMS) - {"oMLX"}

ROW = re.compile(r"\| `(\w+)`[^|]*\|[^|]*\|\s*([✅✗])[^|]*\|\s*([✅✗])[^|]*\|\s*([✅✗])[^|]*\|")


def _tick(supported: bool) -> str:
    return "✅" if supported else "✗"


def _matrix() -> str:
    return MATRIX_PATH.read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    """The lines from ``heading`` up to the next level-2 heading."""
    start = text.index(heading)
    rest = text[start + len(heading) :]
    end = rest.find("\n## ")
    return rest if end == -1 else rest[:end]


@pytest.mark.parametrize("heading,enum", ID_TABLES, ids=[h.strip("# (") for h, _ in ID_TABLES])
def test_every_catalog_member_has_a_matrix_row(heading, enum):
    section = _section(_matrix(), heading)

    missing = [member.name for member in enum if f"`{member.value}`" not in section]

    assert missing == []


@pytest.mark.parametrize("heading,enum", ID_TABLES, ids=[h.strip("# (") for h, _ in ID_TABLES])
def test_matrix_flags_match_the_catalog(heading, enum):
    section = _section(_matrix(), heading)

    wrong = []
    for name, tools, thinking, vision in ROW.findall(section):
        member = enum[name]
        documented = (tools, thinking, vision)
        actual = (_tick(member.supports_tools), _tick(member.supports_thinking), _tick(member.supports_vision))
        if documented != actual:
            wrong.append(f"{name}: documented {documented}, catalog {actual}")

    assert wrong == []


def test_openai_compat_table_covers_every_server_member():
    text = _matrix()
    section = _section(text, "## OpenAI-compatible local servers")
    documented = set(re.findall(r"\| `(\w+)`", section))

    missing = sorted({m.name for enum in SERVER_ENUMS.values() for m in enum} - documented)

    assert missing == []


def test_openai_compat_servers_column_matches_membership():
    """The Servers column must match which server enums actually carry the member.

    The table defines "all" as the six non-MLX servers, because oMLX ships only MLX conversions
    and carries none of the GGUF-era models. Writing "all" for a member oMLX *does* carry would
    still fail here, since oMLX has to be named explicitly on those rows.
    """
    section = _section(_matrix(), "## OpenAI-compatible local servers")

    wrong = []
    for line in section.splitlines():
        match = re.match(r"\| `(\w+)`[^|]*\|[^|]*\|[^|]*\|[^|]*\|([^|]*)\|", line)
        if not match:
            continue
        name, described = match.group(1), match.group(2)
        actual = {label for label, enum in SERVER_ENUMS.items() if name in enum.__members__}
        described_clean = described.strip().rstrip("¶‡†§ ").strip()
        if described_clean.startswith("all except "):
            claimed = NON_MLX_SERVERS - {s.strip() for s in described_clean[len("all except ") :].split(",")}
        elif described_clean == "all":
            claimed = set(NON_MLX_SERVERS)
        else:
            claimed = {s.strip() for s in described_clean.split(",")}
        if claimed != actual:
            wrong.append(f"{name}: documented {sorted(claimed)}, catalog {sorted(actual)}")

    assert wrong == []
