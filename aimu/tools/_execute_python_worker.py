"""Dependency-free restricted-exec worker for the ``execute_python`` tool family.

Deliberately imports nothing outside the standard library. The ``execute_python``
subprocess reaches this module via an explicit ``sys.path.insert(0, <literal>)``
baked into its ``-c`` script, never ``import aimu`` -- importing the full ``aimu``
package in the child would (a) pull in ``requests``, ``dotenv``, and every other
dependency ``aimu.tools.builtin`` carries, slowing every call, (b) run
``aimu.tools.builtin``'s module-scope ``load_dotenv()``, silently reloading a host
repo's ``.env`` file into the "credential-scrubbed" child, and (c) fail outright for
any install that resolves ``aimu`` via ``PYTHONPATH`` rather than site-packages, since
the child's scrubbed environment drops ``PYTHONPATH`` and its cwd is a throwaway temp
directory.

``execute_python_in_process`` (in ``aimu.tools.builtin``) imports and calls
``run_restricted`` directly, in-process; the ``execute_python`` subprocess worker
imports it too, after the path insert. One implementation, two call sites, so the two
backends can't drift on what "run this code" means or what it's allowed to do.
"""

import ast
import builtins as _builtins_module
import contextlib
import importlib
import io
import traceback

# Modules allowlisted for both execute_python backends. This stops accidents, not a
# determined attempt (a one-line expression reaches `subprocess.Popen` through the type
# hierarchy, and the filesystem through an allowlisted module's transitive attributes).
SANDBOX_ALLOWLIST = frozenset(
    [
        "math",
        "statistics",
        "json",
        "re",
        "itertools",
        "functools",
        "datetime",
        "zoneinfo",
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
    ]
)

# Restricted builtins: copy all stdlib builtins, then block dangerous ones.
_SANDBOX_BUILTINS = {
    k: v for k, v in vars(_builtins_module).items() if k not in ("open", "breakpoint", "input", "__import__")
}


def run_restricted(code: str) -> str:
    """Parse and run *code* against a namespace with restricted builtins and an
    import allowlist, returning captured stdout plus the last expression's ``repr``
    (or an error string).
    """
    namespace = {}
    for mod_name in SANDBOX_ALLOWLIST:
        try:
            namespace[mod_name] = importlib.import_module(mod_name)
        except ImportError:
            pass

    _real_import = _builtins_module.__import__

    def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".")[0] not in SANDBOX_ALLOWLIST:
            raise ImportError(f"'{name}' is not in the execute_python import allowlist")
        return _real_import(name, globals, locals, fromlist, level)

    namespace["__builtins__"] = {**_SANDBOX_BUILTINS, "__import__": _restricted_import}

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError: {exc}"

    # If the last statement is an expression, compile preamble + last separately
    # so we can eval() the expression and capture its value without AST mutation.
    stdout_buf = io.StringIO()
    result = None
    try:
        with contextlib.redirect_stdout(stdout_buf):
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                preamble = ast.Module(body=tree.body[:-1], type_ignores=[])
                ast.fix_missing_locations(preamble)
                expr = ast.Expression(body=tree.body[-1].value)
                ast.fix_missing_locations(expr)
                exec(compile(preamble, "<execute_python>", "exec"), namespace)  # noqa: S102
                result = eval(compile(expr, "<execute_python>", "eval"), namespace)  # noqa: S307
            else:
                exec(compile(tree, "<execute_python>", "exec"), namespace)  # noqa: S102
    except Exception:
        return f"Error:\n{traceback.format_exc()}"

    parts = []
    stdout = stdout_buf.getvalue()
    if stdout:
        parts.append(stdout.rstrip())
    if result is not None:
        parts.append(repr(result))
    return "\n".join(parts) if parts else "(no output)"
