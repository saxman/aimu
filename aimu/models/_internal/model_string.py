"""Parse the extended model-string grammar: ``provider:model_id[@base_url][;flags]``.

Only structural splitting lives here. Provider validation, flag validation, and the
enum-vs-ad-hoc decision are the caller's job (see
``aimu.models.model_client.resolve_model``) because those need the provider registry
and the ``Model`` catalogs.

Split order is fixed so ambiguity never arises: flags (``;``) first, then base_url
(``@``), then provider (``:``). A model id may contain ``:`` (e.g. an ollama tag) and a
URL may contain ``:``, ``/`` and even ``@`` (userinfo), so each split consumes only the
first delimiter of its kind on the remaining left-hand segment. Documented constraint:
a model id contains no ``@`` or ``;``, and a base_url contains no ``;``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ParsedModelString:
    provider: str
    model_id: str
    base_url: Optional[str]
    flags: tuple[str, ...]


def parse_model_string(model_str: str) -> ParsedModelString:
    if ":" not in model_str:
        raise ValueError(f"Model string must be in 'provider:model_id' form, got: {model_str!r}.")
    main, _, flag_str = model_str.partition(";")
    provider_model, _, base_url = main.partition("@")
    provider, _, model_id = provider_model.partition(":")

    flags = tuple(f.strip() for f in flag_str.split(",") if f.strip())
    return ParsedModelString(
        provider=provider,
        model_id=model_id,
        base_url=base_url or None,
        flags=flags,
    )
