"""A recorded request payload as plain JSON data, for ``RequestPrepared``.

Payloads are built for an SDK, not for storage, so they can hold values with no JSON form:
Anthropic's ``NOT_GIVEN`` sentinel, Pydantic classes and instances, an in-process provider's
rendered inputs. Every host persisting a request would otherwise hit this on its own.

This is serialization, not redaction: nothing is hidden. Where a conversion is lossless it is
silent (a ``NOT_GIVEN`` key is dropped, because the SDK omits it from the wire; a Pydantic
instance becomes its own JSON form). Where it loses information it leaves a visible marker such
as ``"<bytes 3>"``, never a bare ``repr``, so a reader can see that something was summarized.
``client.last_request`` keeps the raw object, so in-process inspection loses nothing.
"""

from __future__ import annotations

from typing import Any

_PLAIN = (str, int, float, bool, type(None))


def _is_not_given(value: Any) -> bool:
    # Matched by class name rather than isinstance: anthropic and openai each define their own
    # NotGiven, and importing either SDK here would make every provider depend on both.
    return type(value).__name__ == "NotGiven"


def _marker(value: Any) -> str:
    kind = type(value)
    return f"<{kind.__module__}.{kind.__qualname__}>"


def json_safe(value: Any) -> Any:
    """``value`` with every non-JSON part converted or marked; see the module docstring."""
    if isinstance(value, _PLAIN):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items() if not _is_not_given(item)}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (bytes, bytearray)):
        return f"<bytes {len(value)}>"
    if isinstance(value, type):
        return f"<class {value.__module__}.{value.__qualname__}>"
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return json_safe(model_dump(mode="json"))
        except Exception:
            return _marker(value)
    size = getattr(value, "size", None)
    if type(value).__module__.startswith("PIL.") and isinstance(size, tuple) and len(size) == 2:
        return f"<PIL.Image {size[0]}x{size[1]}>"
    return _marker(value)
