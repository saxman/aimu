"""Convert a schema class to a JSON Schema dict for structured-output requests.

The base client converts the caller's ``schema=`` (a dataclass type or a Pydantic v2
``BaseModel`` subclass) into a JSON Schema *once*, here, and hands it to providers, which
wrap it in their native envelope (OpenAI ``response_format``, Ollama ``format=``, Anthropic
forced-tool ``input_schema``). Coercion of the response back into the schema type is done
separately by :func:`aimu.models.parse_json_response`.
"""

from __future__ import annotations

import dataclasses
from typing import Any, get_type_hints

# Reuse the Python-type -> JSON Schema type mapping that backs the @tool decorator,
# so dataclass field conversion stays consistent with tool-argument conversion.
from aimu.tools.decorator import _json_type_for


def schema_name(schema: type) -> str:
    """A stable name for the schema (used by providers that require a named schema)."""
    return getattr(schema, "__name__", "Response")


def schema_to_json_schema(schema: type) -> dict:
    """Return a JSON Schema dict for *schema* (a dataclass type or Pydantic v2 model).

    Raises ``TypeError`` if *schema* is neither.
    """
    # Pydantic v2
    if hasattr(schema, "model_json_schema"):
        return schema.model_json_schema()

    if dataclasses.is_dataclass(schema) and isinstance(schema, type):
        # Resolve string annotations (e.g. under `from __future__ import annotations`).
        hints = get_type_hints(schema)
        properties: dict[str, Any] = {}
        required: list[str] = []
        for field in dataclasses.fields(schema):
            properties[field.name] = {"type": _json_type_for(hints.get(field.name, field.type))}
            has_default = field.default is not dataclasses.MISSING or field.default_factory is not dataclasses.MISSING
            if not has_default:
                required.append(field.name)
        return {
            "title": schema_name(schema),
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    raise TypeError(f"schema must be a dataclass type or Pydantic BaseModel subclass, got {type(schema).__name__}")


def json_schema_instruction(json_schema: dict) -> str:
    """Prompt suffix used on the parse path (providers without native enforcement).

    Asks the model to emit a bare JSON object matching the schema; the response is then
    parsed by :func:`aimu.models.parse_json_response` (which tolerates fences / surrounding
    prose), so this is a best-effort nudge, not a guarantee.
    """
    import json

    return (
        "Respond with ONLY a JSON object matching this JSON Schema. "
        "Do not include any prose, explanation, or markdown code fences.\n\n"
        f"{json.dumps(json_schema)}"
    )
