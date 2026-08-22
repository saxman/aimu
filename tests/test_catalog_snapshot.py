"""Temporary characterization test pinning every catalog member's resolved spec.

Written against the pre-refactor tree so the shared-facts migration can be proven inert:
the snapshot is regenerated only when membership changes *intentionally*. Delete this file
once the catalogs start gaining members (Task 7 of the parity plan).
"""

from __future__ import annotations

import json
from pathlib import Path

import aimu.models as models

SNAPSHOT = Path(__file__).parent / "catalog_snapshot.json"

FIELDS = (
    "value", "supports_tools", "supports_thinking", "supports_vision", "supports_audio",
    "supports_structured_output", "thinking_levels", "thinking_optional",
    "generation_kwargs", "nonthinking_generation_kwargs",
)


def _capture() -> dict:
    out: dict[str, dict] = {}
    for attr in sorted(dir(models)):
        if not attr.endswith("Model") or attr == "Model":
            continue
        enum = getattr(models, attr)
        if enum is None:
            continue
        try:
            members = list(enum)
        except TypeError:
            continue
        if not members or not hasattr(members[0], "supports_tools"):
            continue
        out[attr] = {m.name: {f: getattr(m, f) for f in FIELDS} for m in members}
    return out


PROFILE_FIELDS = ("generation_kwargs", "nonthinking_generation_kwargs")


def test_catalog_specs_are_unchanged():
    current = _capture()
    if not SNAPSHOT.exists():
        SNAPSHOT.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        return

    before = json.loads(SNAPSHOT.read_text())
    assert set(current) == set(before), "a catalog appeared or vanished"

    for catalog, members in before.items():
        assert set(current[catalog]) == set(members), f"{catalog} membership changed"
        for member, was in members.items():
            now = current[catalog][member]
            for field, old in was.items():
                new = now[field]
                if old == new:
                    continue
                # The single sanctioned delta: a catalog that declared no card profile gains
                # the shared one. Everything else -- a capability flag, an id, or a profile
                # changing from one non-empty value to a different one -- is drift.
                assert field in PROFILE_FIELDS and not old and new, (
                    f"{catalog}.{member}.{field} changed: {old!r} -> {new!r}"
                )
