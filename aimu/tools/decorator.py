import inspect
import types as _types
from typing import Callable, Union, get_args, get_origin, get_type_hints

from pydantic import TypeAdapter, ValidationError

from .context import ToolContext

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


class ToolArgumentError(ValueError):
    """Raised when model-supplied tool-call arguments fail validation/coercion.

    Caught at dispatch and surfaced to the model as a tool result so it can self-correct.
    """


def tool(func: Callable) -> Callable:
    """Mark a Python function as an AIMU tool.

    Inspects the signature and docstring at decoration time and attaches an OpenAI-format
    tool spec to ``func.__tool_spec__``. The function itself is unchanged and remains
    directly callable.

    Each parameter must either have a type hint or a default value. Variadic parameters
    (``*args`` / ``**kwargs``) are not supported; declare each argument explicitly.

    Supported parameter types: ``str``, ``int``, ``float``, ``bool``, ``list``, ``dict``,
    plus ``Optional[T]`` and ``T | None`` (which unwrap to the inner type).

    A tool may be plain (``def fn() -> T``), async (``async def fn() -> T``), a generator
    (``def fn(): yield ...; return T``), or an async generator (``async def fn(): yield ...``).
    Generator and async-generator tools stream :class:`~aimu.models.StreamChunk` objects
    during execution; the agent forwards each yielded chunk through its own stream and
    treats the final yielded ``TOOL_CALLING`` chunk's ``content["response"]`` as the
    canonical tool result. The decorator sets discriminator attributes:

    - ``func.__tool_is_async__``: True for ``async def`` *or* ``async def`` + ``yield``.
    - ``func.__tool_is_streaming__``: True for generator functions (sync or async).

    Usage::

        import aimu

        @aimu.tool
        def letter_counter(word: str, letter: str) -> int:
            \"\"\"Count occurrences of a letter in a word.\"\"\"
            return word.lower().count(letter.lower())

        agent = Agent(client, tools=[letter_counter])
    """
    spec, injected, adapters, allowed = _build_spec(func)
    func.__tool_spec__ = spec
    func.__tool_injected__ = injected
    func.__tool_param_adapters__ = adapters
    func.__tool_allowed_args__ = allowed
    func.__tool_required__ = spec["function"]["parameters"]["required"]
    func.__tool_is_async__ = inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func)
    func.__tool_is_streaming__ = inspect.isgeneratorfunction(func) or inspect.isasyncgenfunction(func)
    return func


def _is_injected(annotation) -> bool:
    """True if a parameter annotation marks it as a framework-injected ``ToolContext``."""
    annotation = _unwrap_optional(annotation)
    return annotation is ToolContext or get_origin(annotation) is ToolContext


def _unwrap_optional(annotation):
    """Reduce ``Optional[T]`` / ``T | None`` to ``T``. Leaves other annotations alone."""
    origin = get_origin(annotation)
    # typing.Union covers Optional[T]; types.UnionType covers Python 3.10+ T | None syntax.
    if origin is Union or isinstance(annotation, _types.UnionType):
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


def _build_spec(func: Callable) -> tuple[dict, list[str], dict[str, TypeAdapter], set[str]]:
    """Build the OpenAI-format tool spec plus the metadata used at dispatch.

    Returns ``(spec, injected, param_adapters, allowed_args)``:

    * ``injected`` -- framework-injected (``ToolContext``) parameter names, filled by the
      agent at call time and kept out of the model-facing schema.
    * ``param_adapters`` -- a Pydantic :class:`TypeAdapter` per annotated, model-facing
      parameter, built once here so dispatch-time coercion needs no reflection.
    * ``allowed_args`` -- every model-facing parameter name (annotated or default-only),
      used to reject unknown model-supplied arguments.
    """
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}
    doc = (inspect.getdoc(func) or "").strip()
    description = doc.split("\n\n", 1)[0] if doc else ""

    properties: dict[str, dict] = {}
    required: list[str] = []
    injected: list[str] = []
    param_adapters: dict[str, TypeAdapter] = {}
    allowed_args: set[str] = set()
    for name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise ToolSignatureError(
                f"@aimu.tool: function '{func.__name__}' uses variadic parameter '*{name}'. "
                "Declare each argument explicitly. Variadic args cannot be described as JSON Schema."
            )
        annotation = hints.get(name, param.annotation)
        if _is_injected(annotation):
            injected.append(name)
            continue
        has_default = param.default is not inspect.Parameter.empty
        has_annotation = annotation is not inspect.Parameter.empty and annotation is not None
        if not has_annotation and not has_default:
            raise ToolSignatureError(
                f"@aimu.tool: parameter '{name}' on '{func.__name__}' has no type hint and no default. "
                "Add a type hint (e.g. `name: str`) or a default value."
            )
        allowed_args.add(name)
        properties[name] = {"type": _json_type_for(annotation) if has_annotation else "string"}
        if has_annotation:
            # Full annotation (not unwrapped) so Optional[T]/T|None still accepts None.
            param_adapters[name] = TypeAdapter(annotation)
        if not has_default:
            required.append(name)

    spec = {
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
    return spec, injected, param_adapters, allowed_args


def coerce_tool_arguments(fn: Callable, arguments: dict) -> dict:
    """Validate and lax-coerce model-supplied tool arguments against ``fn``'s type hints.

    Returns a new dict of model-facing arguments coerced to their declared types
    (``"5"`` -> ``5``, ``"true"`` -> ``True``). Raises :class:`ToolArgumentError` with a
    single self-contained message when arguments are unknown, a required one is missing,
    or a value can't be coerced.

    Callables not built by ``@tool`` (e.g. MCP ``as_tools()`` wrappers) carry no
    ``__tool_param_adapters__``; their arguments pass through unchanged (the MCP server
    validates them).
    """
    adapters = getattr(fn, "__tool_param_adapters__", None)
    if adapters is None:
        return arguments

    allowed = getattr(fn, "__tool_allowed_args__", set())
    required = getattr(fn, "__tool_required__", [])

    errors: list[str] = []
    unknown = [name for name in arguments if name not in allowed]
    errors.extend(f"unexpected argument '{name}'" for name in unknown)
    errors.extend(f"missing required argument '{name}'" for name in required if name not in arguments)

    coerced: dict = {}
    for name, value in arguments.items():
        if name in unknown:
            continue
        adapter = adapters.get(name)
        if adapter is None:  # default-only param with no annotation: accept as-is
            coerced[name] = value
            continue
        try:
            coerced[name] = adapter.validate_python(value)
        except ValidationError as exc:
            detail = exc.errors()[0].get("msg", "is invalid") if exc.errors() else "is invalid"
            errors.append(f"'{name}' {detail.lower()}")

    if errors:
        raise ToolArgumentError(f"invalid arguments for tool '{fn.__name__}': " + "; ".join(errors))
    return coerced
