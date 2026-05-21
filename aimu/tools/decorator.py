import inspect
from typing import Callable, get_type_hints

_PY_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def tool(func: Callable) -> Callable:
    """Mark a Python function as an AIMU tool.

    Attaches ``__tool_spec__`` (an OpenAI-format tool spec derived from the
    function's signature and docstring) to the function. The function itself
    is returned unchanged and remains directly callable.

    Usage::

        @tool
        def letter_counter(word: str, letter: str) -> int:
            \"\"\"Count occurrences of a letter in a word.\"\"\"
            return word.lower().count(letter.lower())

        agent = Agent(client, tools=[letter_counter])
    """
    func.__tool_spec__ = _build_spec(func)
    return func


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
        properties[name] = {"type": _PY_TO_JSON.get(hints.get(name, str), "string")}
        if param.default is inspect.Parameter.empty:
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
