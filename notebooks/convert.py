#!/usr/bin/env python3
"""Convert the notebook collection between Quarto ``.qmd`` (the committed source of
truth) and Jupyter ``.ipynb`` (a local, git-ignored working copy for inline
cell execution in VS Code or Jupyter).

Usage (paths resolve relative to this file, so run it from anywhere)::

    python notebooks/convert.py to-ipynb            # every .qmd -> .ipynb
    python notebooks/convert.py to-ipynb 06-tools   # just notebooks/06-tools.qmd
    python notebooks/convert.py to-qmd  06-tools    # sync edits back: .ipynb -> .qmd

Requires the Quarto CLI on PATH (https://quarto.org/docs/get-started/) -- this
shells out to ``quarto convert``. Generated ``.ipynb`` files are git-ignored;
``.qmd`` is the source of truth, so run ``to-qmd`` after editing a notebook.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent


def _require_quarto() -> str:
    exe = shutil.which("quarto")
    if not exe:
        sys.exit(
            "quarto not found on PATH. Install it (https://quarto.org/docs/get-started/): "
            "`brew install quarto`, the installer, or `pip install quarto-cli`."
        )
    return exe


def _targets(stems: list[str], suffix: str) -> list[Path]:
    if stems:
        return [NB_DIR / (Path(s).stem + suffix) for s in stems]
    return sorted(NB_DIR.glob(f"[0-9]*{suffix}"))


def to_ipynb(quarto: str, stems: list[str]) -> None:
    for src in _targets(stems, ".qmd"):
        out = src.with_suffix(".ipynb")
        subprocess.run([quarto, "convert", str(src), "--output", str(out)], check=True)
        print(f"{src.name} -> {out.name}")


def to_qmd(quarto: str, stems: list[str]) -> None:
    for src in _targets(stems, ".ipynb"):
        out = src.with_suffix(".qmd")
        # `quarto convert` drops newlines from string-typed cell sources when writing
        # .qmd. VS Code saves list-typed sources (so this is usually a no-op), but
        # normalize via a temp copy defensively without mutating the working .ipynb.
        nb = json.loads(src.read_text())
        needs_fix = any(isinstance(c.get("source"), str) for c in nb.get("cells", []))
        if not needs_fix:
            subprocess.run([quarto, "convert", str(src), "--output", str(out)], check=True)
        else:
            for cell in nb["cells"]:
                if isinstance(cell.get("source"), str):
                    cell["source"] = cell["source"].splitlines(keepends=True)
            fd, tmp = tempfile.mkstemp(suffix=".ipynb")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(nb, f)
                subprocess.run([quarto, "convert", tmp, "--output", str(out)], check=True)
            finally:
                os.unlink(tmp)
        print(f"{src.name} -> {out.name}")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"to-ipynb", "to-qmd"}:
        sys.exit("usage: python notebooks/convert.py {to-ipynb|to-qmd} [stem ...]")
    quarto = _require_quarto()
    (to_ipynb if sys.argv[1] == "to-ipynb" else to_qmd)(quarto, sys.argv[2:])


if __name__ == "__main__":
    main()
