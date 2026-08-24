"""Tests for the subprocess-backed ``execute_python`` tool.

``execute_python`` used to run user code with ``exec()`` inside this process, behind
restricted builtins and an import allowlist that were documented as a sandbox but never
were one (two one-line escapes reached ``subprocess.Popen`` and the filesystem). This
task replaces the default backend with a real subprocess: it buys isolation
properties an in-process ``exec`` cannot -- a hard timeout, crash isolation, no
mutation of this process, a memory cap on POSIX, and no access to this process's
environment (so no leaked API keys) -- but it does not confine the filesystem or the
network, and none of these tests should be read as claiming otherwise.

The restricted-builtins/import-allowlist accident guard is kept, unchanged, on BOTH
backends: it never provided containment (the same historical escapes still work here,
see test_child_cannot_see_the_parents_api_keys below), so the subprocess boundary adds
new isolation properties on top of it rather than replacing it. A caller switching
between ``execute_python`` and ``execute_python_in_process`` (the explicit in-process
opt-in) gets the same rules either way -- only where the code runs differs.
"""

import time

import pytest

from aimu.tools import builtin
from aimu.tools.builtin import execute_python, execute_python_in_process


def test_child_cannot_see_the_parents_api_keys(monkeypatch):
    """The concrete win.

    ``import os`` is blocked by the accident-guard allowlist (both backends), so this
    reaches ``os.environ`` the same way the historical sandbox-escape docs describe:
    through an allowlisted module's transitive attributes (``json`` -> its ``codecs``
    submodule -> that module's own ``sys`` reference -> ``sys.modules["os"]``). That
    escape defeats the accident guard exactly as documented -- the point of this test
    is that it *still* doesn't recover the credential, because the credential was never
    in the child's environment in the first place. That's a property of the subprocess
    boundary + the scrubbed env=, not of the allowlist, and it holds even against a
    determined attempt to route around the allowlist.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    out = execute_python("import json; print(json.codecs.sys.modules['os'].environ.get('ANTHROPIC_API_KEY'))")
    assert "sk-should-not-leak" not in out
    assert "None" in out  # the child prints None: the var truly isn't there, not just unprinted


def test_a_hanging_program_is_killed_by_the_timeout(monkeypatch):
    """In-process exec could hang the host forever.

    Uses a short timeout so this test doesn't tax every suite run; production default
    lives in `_EXECUTE_PYTHON_TIMEOUT_S` and is untouched by this monkeypatch.
    """
    monkeypatch.setattr(builtin, "_EXECUTE_PYTHON_TIMEOUT_S", 0.5)
    start = time.monotonic()
    out = execute_python("while True: pass")
    elapsed = time.monotonic() - start
    assert "timed out" in out.lower()
    assert elapsed < 5  # generous upper bound; proves the hang was actually killed, not outlasted


def test_a_crash_does_not_take_down_the_host():
    """A child that dies abnormally must return an error string, not take pytest down
    with it.

    Uses ``raise SystemExit(1)`` rather than ``os._exit(1)``: ``os`` is outside the
    import allowlist enforced by both backends (see
    test_child_cannot_see_the_parents_api_keys and tests/test_tools.py), so reaching
    for it here would require the same escape-through-an-allowlisted-module trick,
    which is beside the point of this test. ``SystemExit`` needs no import (it's a
    builtin exception, present in the restricted builtins) and is a ``BaseException``,
    so the shared ``except Exception:`` inside ``_exec_restricted_code`` does not catch
    it -- it propagates out of the child's ``-c`` script uncaught, and the interpreter
    exits with that status exactly as an unrecoverable crash would. What's actually
    under test is ``execute_python``'s handling of a nonzero child exit code, which is
    the same code path a real crash (a segfault, ``os._exit``) would take; this is the
    clean, fast, deterministic way to reach it without needing a non-allowlisted import
    or a platform-specific crash mechanism.
    """
    out = execute_python("raise SystemExit(1)")
    assert "error" in out.lower()
    # If crash isolation didn't hold, this line would never execute.
    assert True


def test_output_is_capped():
    huge = execute_python("print('x' * 1_000_000)")
    assert len(huge) < 1_000_000
    assert "truncated" in huge


def test_the_docstring_still_refuses_to_claim_containment():
    """The docstring is what the model (and a human wiring tools) reads.

    It must state the real tradeoff -- isolation, not containment -- and must not
    claim the subprocess confines the filesystem or the network, since it doesn't.
    """
    doc = execute_python.__doc__.lower()
    assert "sandbox" not in doc
    assert "not a security boundary" in doc
    assert "isolation, not containment" in doc
    assert "does not confine" in doc


def test_in_process_backend_is_still_reachable_explicitly():
    """Trusted code that cares about subprocess startup cost can still opt into the
    old in-process behavior by name.
    """
    assert execute_python_in_process is not builtin.execute_python
    assert "4" in execute_python_in_process("2 + 2")
    # Opt-in, not a default: neither backend ships in ALL_TOOLS, and only the
    # subprocess-backed tool is in the curated `compute` subgroup.
    assert execute_python_in_process not in builtin.ALL_TOOLS
    assert execute_python_in_process not in builtin.compute
    assert execute_python in builtin.compute


def test_in_process_backend_docstring_is_equally_honest():
    doc = execute_python_in_process.__doc__.lower()
    assert "sandbox" not in doc
    assert "not a security boundary" in doc


def test_subprocess_backend_still_enforces_the_import_allowlist():
    """Consistency requirement: a caller switching between execute_python and
    execute_python_in_process must not get different rules. The subprocess boundary
    buys isolation properties (timeout, crash isolation, no leaked environment) that
    don't depend on restricting imports, but the restricted-builtins/import-allowlist
    accident guard applies to both backends identically -- it is not a security
    boundary either way (see test_child_cannot_see_the_parents_api_keys for what
    actually makes the credential-leak property hold), but dropping it from just one
    backend would make `import os` succeed by accident in ordinary (non-adversarial)
    model output there and not the other, which is its own kind of surprising.
    """
    out = execute_python("import os")
    assert "Error" in out or "ImportError" in out


def test_subprocess_backend_still_returns_stdout_and_last_expression():
    """The return contract is unchanged: a caller upgrading from the in-process
    backend shouldn't have to change how it reads the result.
    """
    out = execute_python('print("x =", 5)\n5 * 5')
    assert "x = 5" in out
    assert "25" in out


def test_subprocess_backend_reports_syntax_errors():
    out = execute_python("def bad syntax")
    assert "SyntaxError" in out


def test_subprocess_backend_no_output_returns_sentinel():
    assert execute_python("x = 5") == "(no output)"


def test_memory_cap_failure_on_this_platform_does_not_crash_the_launch(monkeypatch):
    """Regression: on macOS, ``resource.setrlimit(RLIMIT_AS, ...)`` raises even though
    the ``resource`` module imports fine (confirmed independently: a plain shell's
    ``ulimit -v N`` fails the same way on this platform). A ``preexec_fn`` that raises
    turns every ``execute_python`` call into an unhandled ``subprocess.SubprocessError``,
    so the memory cap must degrade silently instead of taking the whole tool down.
    """
    if builtin.resource is None:
        pytest.skip("no resource module on this platform")

    def _always_raises(*_args, **_kwargs):
        raise OSError("simulated: this platform refuses to lower RLIMIT_AS")

    monkeypatch.setattr(builtin.resource, "setrlimit", _always_raises)
    assert "4" in execute_python("2 + 2")


def test_windows_has_no_resource_module_and_execute_python_still_runs(monkeypatch):
    """Windows has no ``resource`` module at all (guarded import at the top of
    builtin.py). Simulates that here without needing a Windows machine: execute_python
    must degrade to no memory cap rather than fail to run.
    """
    monkeypatch.setattr(builtin, "resource", None)
    assert "4" in execute_python("2 + 2")
