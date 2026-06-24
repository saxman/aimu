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
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Optional


def factory_model_kwargs(kwargs: dict) -> Optional[dict]:
    """Map a modality factory's ``**kwargs`` to the concrete client's ``model_kwargs`` dict.

    The factories (``ImageClient``, ``AudioClient``, …) accept provider construction kwargs
    directly, mirroring ``ModelClient(model, base_url=...)``::

        ImageClient(HuggingFaceImageModel.SDXL_BASE, variant="fp16")

    The legacy ``model_kwargs={...}`` form is still accepted but deprecated: it is unwrapped
    here with a :class:`DeprecationWarning`. Returns ``None`` when no kwargs were given.
    """
    if "model_kwargs" in kwargs:
        inner = kwargs.pop("model_kwargs")
        if kwargs:
            raise TypeError(
                "Pass provider kwargs directly to the factory (e.g. variant='fp16'); "
                "do not mix them with the deprecated model_kwargs= argument."
            )
        warnings.warn(
            "Passing model_kwargs= to a modality factory is deprecated; pass provider kwargs "
            "directly, e.g. ImageClient(model, variant='fp16').",
            DeprecationWarning,
            stacklevel=3,
        )
        return inner
    return kwargs or None


@dataclass(frozen=True)
class ProviderEntry:
    """One provider for a modality.

    ``enum_cls`` / ``client_cls`` are ``None`` when the provider's optional dependency
    isn't installed (``available is False``); ``install_hint`` is the ``ImportError``
    text shown when a recognized-but-uninstalled provider is requested by string.
    """

    prefix: str
    available: bool
    enum_cls: Optional[type]
    client_cls: Optional[type]
    install_hint: str


def available_registry(entries: list[ProviderEntry]) -> dict[str, tuple]:
    """``{prefix: (enum_cls, client_cls)}`` for installed providers only.

    Used by the ``resolve_*_model_string`` helpers and discovery.
    """
    return {e.prefix: (e.enum_cls, e.client_cls) for e in entries if e.available}


def resolve_model_string(model_str: str, entries: list[ProviderEntry], *, modality: str) -> Any:
    """Look up a model enum member from a ``"provider:model_id"`` string.

    Matches *exact* enum-member values only; ad-hoc ids are handled inside each
    concrete client's ``__init__`` (pass the string straight to the factory).
    """
    label = modality.capitalize()
    registry = available_registry(entries)
    if ":" not in model_str:
        raise ValueError(
            f"{label} model string must be in 'provider:model_id' form, got: {model_str!r}. "
            f"Available providers: {sorted(registry)}"
        )
    provider, _, model_id = model_str.partition(":")
    if provider not in registry:
        raise ValueError(
            f"Unknown {modality} provider {provider!r}. Available providers (with installed deps): {sorted(registry)}"
        )
    model_enum, _ = registry[provider]
    for member in model_enum:
        if member.value == model_id:
            return member
    available = sorted(m.value for m in model_enum)
    raise ValueError(f"Provider {provider!r} has no {modality} model id {model_id!r}. Available: {available}")


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
    an enum member dispatches by ``isinstance`` against each installed provider enum.
    """
    label = modality.capitalize()

    if isinstance(model, str):
        if ":" not in model:
            raise ValueError(f"{label} model string must be in 'provider:model_id' form, got: {model!r}")
        provider, _, _model_id = model.partition(":")
        entry = next((e for e in entries if e.prefix == provider), None)
        if entry is None:
            raise ValueError(
                f"Unknown {modality} provider {provider!r}. Available: {sorted(available_registry(entries))}"
            )
        if not entry.available:
            raise ImportError(entry.install_hint)
        return entry.client_cls(model, model_kwargs=model_kwargs)

    if isinstance(model, spec_base) and not isinstance(model, model_base):
        raise TypeError(
            f"Pass a {model_base.__name__} enum member or a 'provider:model_id' string. "
            f"{spec_base.__name__} is the value type held by enum members."
        )

    for entry in entries:
        if entry.available and entry.enum_cls is not None and isinstance(model, entry.enum_cls):
            return entry.client_cls(model, model_kwargs=model_kwargs)

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
