"""Shared dispatch logic for the per-modality client factories.

The image / audio / speech / transcription / embedding factories (``ImageClient``,
``AudioClient``, …) all do the same three things: parse a ``"provider:model_id"``
string, dispatch an enum member to its concrete provider client, and delegate a few
read-only properties to that inner client. This module factors that out so each
factory is just a provider table plus a thin ``__init__`` and its one modality method.

It is deliberately plain Python: a small dataclass describing each provider, two
functions, and a delegation mixin. No registry singletons, no metaclasses. The text
factory (:mod:`aimu.models.model_client`) is intentionally *not* built on this; its
bare-name/local-availability resolution is richer and lives on its own.

``ProviderEntry`` holds module and symbol *names* rather than classes, so building a
factory's table costs no import at all: a caller can list providers, check
availability, and build error messages without ever loading a provider SDK. Only
:meth:`ProviderEntry.load` (called for the one provider actually requested) imports
anything.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Optional


def installed(dep: str) -> bool:
    """True if ``dep`` is importable, decided without importing it.

    ``find_spec`` raises rather than returning None when a *parent* package is missing
    (e.g. probing ``google.genai`` with no ``google``), so both outcomes are folded into
    one boolean here.

    Called through the module (``importlib.util.find_spec``) rather than a ``from``-import,
    so a test simulating an absent dependency can patch it. A name bound at import time
    would be immune to that patch.
    """
    try:
        return importlib.util.find_spec(dep) is not None
    except (ImportError, ValueError):
        return False


@dataclass(frozen=True)
class ProviderEntry:
    """One provider for a modality, described rather than imported.

    Holding module and symbol *names* instead of classes is what keeps
    ``import aimu`` from loading every provider SDK: the table can be built, searched,
    and reported on with no import at all, and :meth:`load` runs only for the provider
    a caller actually asked for.

    ``requires`` is the third-party module probed for availability (``"openai"``,
    ``"sentence_transformers"``), not AIMU's own module. ``install_hint`` is the
    ``ImportError`` text shown when a recognized-but-uninstalled provider is requested.

    ``direct_kwargs`` names the constructor parameters this client declares in its own
    signature. A factory kwarg named here is forwarded as a real keyword argument; every
    other one is bundled into ``model_kwargs`` (which is what a weight-loading client wants
    -- ``device=`` reaching ``from_pretrained``). Listing them explicitly beats inspecting
    the signature: the split is visible in the provider table, and a client that declares a
    parameter the table forgot fails loudly rather than having the kwarg vanish into an
    ignored ``model_kwargs``.
    """

    prefix: str
    module: str
    enum_name: str
    client_name: str
    requires: str
    install_hint: str = ""
    direct_kwargs: frozenset[str] = frozenset()

    def split_kwargs(self, kwargs: Optional[dict]) -> tuple[dict, Optional[dict]]:
        """Split factory kwargs into ``(direct, model_kwargs)`` per :attr:`direct_kwargs`."""
        if not kwargs:
            return {}, None
        direct = {k: v for k, v in kwargs.items() if k in self.direct_kwargs}
        model_kwargs = {k: v for k, v in kwargs.items() if k not in self.direct_kwargs}
        return direct, model_kwargs or None

    @property
    def available(self) -> bool:
        return installed(self.requires)

    def load(self) -> tuple[type, type]:
        """Import the provider and return ``(enum_cls, client_cls)``.

        Raises if the dependency is installed but broken. That is deliberate: a silent
        ``None`` here would make a present-but-unusable provider look absent, which is
        harder to diagnose than the underlying ImportError.
        """
        module = import_module(self.module)
        return getattr(module, self.enum_name), getattr(module, self.client_name)


def available_prefixes(entries: list[ProviderEntry]) -> list[str]:
    """Sorted prefixes of installed providers. Imports nothing."""
    return sorted(e.prefix for e in entries if e.available)


def available_registry(entries: list[ProviderEntry]) -> dict[str, tuple]:
    """``{prefix: (enum_cls, client_cls)}`` for installed providers only.

    This *does* import every installed provider, so it is reserved for the explicit
    ``available_*_registry()`` discovery helpers. Dispatch paths use
    :func:`available_prefixes` plus a single :meth:`ProviderEntry.load`.
    """
    return {e.prefix: e.load() for e in entries if e.available}


def resolve_model_string(model_str: str, entries: list[ProviderEntry], *, modality: str) -> Any:
    """Look up a model enum member from a ``"provider:model_id"`` string.

    Matches *exact* enum-member values only; ad-hoc ids are handled inside each
    concrete client's ``__init__`` (pass the string straight to the factory).

    Which is why an uncatalogued id does not report as simply unknown. It may well be valid
    (a HuggingFace repo, a provider alias); what is true is narrower, and is what the error
    says: this function returns an enum member, and an uncatalogued id has none. The text
    modality's :func:`aimu.resolve_model_enum` refuses its own unrepresentable forms in the
    same terms, from a parse that can name them exactly. Here the parse cannot tell an
    ad-hoc id from a typo, so the message offers the catalog *and* the way past it.
    """
    label = modality.capitalize()
    prefixes = available_prefixes(entries)
    if ":" not in model_str:
        raise ValueError(
            f"{label} model string must be in 'provider:model_id' form, got: {model_str!r}. "
            f"Available providers: {prefixes}"
        )
    provider, _, model_id = model_str.partition(":")
    entry = next((e for e in entries if e.prefix == provider and e.available), None)
    if entry is None:
        raise ValueError(
            f"Unknown {modality} provider {provider!r}. Available providers (with installed deps): {prefixes}"
        )
    model_enum, _ = entry.load()
    for member in model_enum:
        if member.value == model_id:
            return member
    available = sorted(m.value for m in model_enum)
    raise ValueError(
        f"Provider {provider!r} has no catalogued {modality} model id {model_id!r}. Available: {available}. "
        f"An uncatalogued id has no enum member to return, but may still be usable: pass the whole "
        f"string to {label}Client instead."
    )


def build_client(
    model: Any,
    model_kwargs: Optional[dict],
    entries: list[ProviderEntry],
    *,
    modality: str,
    model_base: type,
    spec_base: type,
) -> Any:
    """Construct the concrete provider client for ``model`` (enum / spec / string).

    String form routes by prefix (so ad-hoc ids reach the concrete client); a bare
    ``spec_base`` instance is rejected (it's the enum's value type, not a selector);
    an enum member dispatches by the defining module of its class, which imports
    nothing (an enum member cannot exist unless its module was already imported).
    """
    label = modality.capitalize()

    if isinstance(model, str):
        if ":" not in model:
            raise ValueError(f"{label} model string must be in 'provider:model_id' form, got: {model!r}")
        provider, _, _model_id = model.partition(":")
        entry = next((e for e in entries if e.prefix == provider), None)
        if entry is None:
            raise ValueError(f"Unknown {modality} provider {provider!r}. Available: {available_prefixes(entries)}")
        if not entry.available:
            raise ImportError(entry.install_hint)
        _enum_cls, client_cls = entry.load()
        direct, rest = entry.split_kwargs(model_kwargs)
        return client_cls(model, model_kwargs=rest, **direct)

    if isinstance(model, spec_base) and not isinstance(model, model_base):
        raise TypeError(
            f"Pass a {model_base.__name__} enum member or a 'provider:model_id' string. "
            f"{spec_base.__name__} is the value type held by enum members."
        )

    # Dispatch by the model enum's defining module + class name. Matching on the module
    # alone is not enough: aimu.models.providers.ollama defines both OllamaModel and
    # OllamaEmbeddingModel, so a module-only match would misroute one as the other (see
    # aimu/models/model_client.py's _TEXT_PROVIDERS dispatch, which has the same rule).
    member_module = type(model).__module__
    member_enum = type(model).__name__
    for entry in entries:
        if entry.module == member_module and entry.enum_name == member_enum:
            _enum_cls, client_cls = entry.load()
            direct, rest = entry.split_kwargs(model_kwargs)
            return client_cls(model, model_kwargs=rest, **direct)

    raise ValueError(
        f"No available client for {modality}-model type {type(model).__name__!r}. "
        "Ensure the required optional dependency is installed."
    )


class FactoryDelegate:
    """Mixin delegating the common read-only properties to ``self._client``.

    Modality factories add their one generate/embed/transcribe method and any extra
    property (e.g. ``ImageClient.max_prompt_tokens``, ``EmbeddingClient.dimensions``).
    """

    _client: Any

    @property
    def model(self) -> Any:
        return self._client.model

    @property
    def spec(self) -> Any:
        return self._client.spec

    @property
    def model_kwargs(self) -> Optional[dict]:
        return self._client.model_kwargs
