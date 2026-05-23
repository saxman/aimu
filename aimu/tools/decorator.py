import inspect
from typing import Callable, Union, get_args, get_origin, get_type_hints

_PY_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


class ToolSignatureError(TypeError):
    """Raised by ``@tool`` when a function signature can't be converted to a tool spec."""


def tool(func: Callable) -> Callable:
    """Mark a Python function as an AIMU tool.

    Inspects the signature and docstring at decoration time and attaches an OpenAI-format
    tool spec to ``func.__tool_spec__``. The function itself is unchanged and remains
    directly callable.

    Each parameter must either have a type hint or a default value. Variadic parameters
    (``*args`` / ``**kwargs``) are not supported — declare each argument explicitly.

    Supported parameter types: ``str``, ``int``, ``float``, ``bool``, ``list``, ``dict``,
    plus ``Optional[T]`` and ``T | None`` (which unwrap to the inner type).

    Usage::

        @tool
        def letter_counter(word: str, letter: str) -> int:
            \"\"\"Count occurrences of a letter in a word.\"\"\"
            return word.lower().count(letter.lower())

        agent = Agent(client, tools=[letter_counter])
    """
    func.__tool_spec__ = _build_spec(func)
    return func


def _unwrap_optional(annotation):
    """Reduce ``Optional[T]`` / ``T | None`` to ``T``. Leaves other annotations alone."""
    origin = get_origin(annotation)
    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _json_type_for(annotation) -> str:
    """Map a Python type annotation to its JSON Schema type name."""
    annotation = _unwrap_optional(annotation)
    # Map subscripted generics (list[str], dict[str, int]) by their origin.
    origin = get_origin(annotation)
    if origin is not None:
        annotation = origin
    return _PY_TO_JSON.get(annotation, "string")


def _build_spec(func: Callable) -> dict:
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}
    doc = (inspect.getdoc(func) or "").strip()
    description = doc.split("\n\n", 1)[0] if doc else ""

    properties: dict[str, dict] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise ToolSignatureError(
                f"@tool: function '{func.__name__}' uses variadic parameter '*{name}'. "
                "Declare each argument explicitly — variadic args cannot be described as JSON Schema."
            )
        annotation = hints.get(name, param.annotation)
        has_default = param.default is not inspect.Parameter.empty
        has_annotation = annotation is not inspect.Parameter.empty and annotation is not None
        if not has_annotation and not has_default:
            raise ToolSignatureError(
                f"@tool: parameter '{name}' on '{func.__name__}' has no type hint and no default. "
                "Add a type hint (e.g. `name: str`) or a default value."
            )
        properties[name] = {"type": _json_type_for(annotation) if has_annotation else "string"}
        if not has_default:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }
