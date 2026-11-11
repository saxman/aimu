"""Models a provider has deprecated or retired, kept so that naming one fails with a migration path.

A catalog drops a model the moment its provider deprecates it, and records it here instead. Every
route a caller can use to name a model consults this table on a miss, so the caller gets a
``ModelDeprecatedError`` naming the replacement rather than an "unknown model" error that reads
like a typo. Those routes are a model string, an enum attribute or subscript, a value lookup, and
a bare enum-member name.

Each provider module registers its own entries right after defining its catalog, with
:func:`register_deprecated`. Registration checks the entries against the live catalog, so a
replacement that does not exist, or a deprecated entry that collides with a live member, fails at
import rather than at the moment a user needs the message.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional

from .._base.shared import ModelDeprecatedError


@dataclass(frozen=True)
class DeprecatedModel:
    """One deprecated or retired model.

    ``name`` is the enum-member name the model has (or would have had) in the catalog. ``ids``
    lists every wire id that names it, typically the dateless alias and the dated snapshot.
    ``replacement`` is the enum-member name of the provider's recommended successor in the same
    catalog. ``retires_on`` is ``None`` when the provider has not announced a date.
    """

    name: str
    ids: tuple[str, ...]
    replacement: str
    deprecated_on: date
    retires_on: Optional[date] = None


_REGISTRY: dict[type[Enum], tuple[DeprecatedModel, ...]] = {}


def register_deprecated(catalog: type[Enum], entries: tuple[DeprecatedModel, ...]) -> None:
    """Record ``entries`` as the deprecated models of ``catalog``. Raises ``ValueError`` on a bad entry."""
    live_names = set(catalog.__members__)
    live_ids = {member.value for member in catalog}
    for entry in entries:
        if entry.replacement not in live_names:
            raise ValueError(
                f"{catalog.__name__} deprecation entry {entry.name} names replacement {entry.replacement!r}, "
                "which is not a member of the catalog."
            )
        if entry.name in live_names or live_ids.intersection(entry.ids):
            raise ValueError(
                f"{catalog.__name__} deprecation entry {entry.name} is still a live catalog member; remove the "
                "member when deprecating it."
            )
    _REGISTRY[catalog] = entries


def find_deprecated(catalog: type[Enum], *, name: Optional[str] = None, model_id: Optional[str] = None):
    """The entry of ``catalog`` matching ``name`` or ``model_id``, or ``None``."""
    for entry in _REGISTRY.get(catalog, ()):
        if (name is not None and entry.name == name) or (model_id is not None and model_id in entry.ids):
            return entry
    return None


def deprecation_error(catalog: type[Enum], entry: DeprecatedModel, named: str) -> ModelDeprecatedError:
    """Build the error for a caller who named ``entry`` as ``named``."""
    replacement = catalog[entry.replacement]
    if entry.retires_on is None:
        status = f"was deprecated on {entry.deprecated_on.isoformat()} (retirement date not yet announced)"
    elif entry.retires_on <= date.today():
        status = f"was retired on {entry.retires_on.isoformat()}, and requests to it fail"
    else:
        status = f"was deprecated on {entry.deprecated_on.isoformat()} and retires on {entry.retires_on.isoformat()}"
    return ModelDeprecatedError(
        f"{catalog.__name__} no longer catalogs {named!r}: it {status}. "
        f"Use {catalog.__name__}.{replacement.name} ({replacement.value!r}) instead.",
        model=entry.ids[0],
        replacement=replacement.value,
        deprecated_on=entry.deprecated_on,
        retires_on=entry.retires_on,
    )


def raise_if_deprecated(catalog: type[Enum], *, name: Optional[str] = None, model_id: Optional[str] = None) -> None:
    """Raise ``ModelDeprecatedError`` if ``name`` or ``model_id`` names a deprecated model, else return."""
    entry = find_deprecated(catalog, name=name, model_id=model_id)
    if entry is not None:
        raise deprecation_error(catalog, entry, name if name is not None else model_id)
