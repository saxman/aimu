"""Tests for the subprocess-backed ``execute_python`` tool.

``execute_python`` used to run user code with ``exec()`` inside this process, behind
restricted builtins and an import allowlist that were documented as a sandbox but never
were one (two one-line escapes reached ``subprocess.Popen`` and the filesystem). This
task replaces the default backend with a real subprocess: it buys isolation
properties an in-process ``exec`` cannot -- a hard timeout, crash isolation, no
mutation of this process, a memory cap on Linux (best-effort elsewhere), and no access
to this process's *environment variables* (so no leaked API keys) -- but it does not
confine the filesystem or the network, and none of these tests should be read as
claiming otherwise.

The restricted-builtins/import-allowlist accident guard is kept, unchanged, on BOTH
backends: it never provided containment (the same historical escapes still work here,
see test_child_cannot_see_the_parents_api_keys below), so the subprocess boundary adds
new isolation properties on top of it rather than replacing it. A caller switching
between ``execute_python`` and ``execute_python_in_process`` (the explicit in-process
opt-in) gets the same rules either way -- only where the code runs differs.

The restricted-exec logic itself lives in the dependency-free
``aimu.tools._execute_python_worker`` module, reached by the subprocess backend via an
explicit ``sys.path.insert`` -- never ``import aimu`` -- so the child never depends on
how ``aimu`` itself is installed and never re-runs ``aimu.tools.builtin``'s module-scope
``load_dotenv()``.
"""

import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from aimu.tools import builtin
from aimu.tools.builtin import execute_python, execute_python_in_process

# Reaches os.environ / os.write / os.system without an `import os` statement, which the
# accident-guard allowlist blocks on both backends. `os` (like every stdlib module the
# interpreter itself has already imported) sits in sys.modules from process startup;
# `json` is allowlisted and happens to import `codecs`, whose own `sys` reference gets
# us the rest of the way. This is the same one-line escape the v0.19 docs cite as proof
# the allowlist was never containment -- used here deliberately, to prove properties
# that must hold even against it.
_REACH_REAL_OS = "json.codecs.sys.modules['os']"


def test_child_cannot_see_the_parents_api_keys(monkeypatch):
    """The concrete win.

    ``import os`` is blocked by the accident-guard allowlist (both backends), so this
    reaches ``os.environ`` via the escape described above. That escape defeats the
    accident guard exactly as documented -- the point of this test is that it *still*
    doesn't recover the credential, because the credential was never in the child's
    environment in the first place. That's a property of the subprocess boundary + the
    scrubbed env=, not of the allowlist, and it holds even against a determined attempt
    to route around the allowlist.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    out = execute_python(f"import json; print({_REACH_REAL_OS}.environ.get('ANTHROPIC_API_KEY'))")
    assert "sk-should-not-leak" not in out
    assert "None" in out  # the child prints None: the var truly isn't there, not just unprinted


def test_child_can_still_read_a_dotenv_file_off_disk(tmp_path):
    """The docstring's own disclosure: env-variable scrubbing is not filesystem
    confinement. A `.env` (or `~/.aws/credentials`, or anything else on disk) is exactly
    as readable to the child as to this process's own user account -- proven here so the
    claim isn't just prose. Uses the same os-escape as above, reached via `os.open`/`os.read`
    since `open()` itself is blocked by the allowlist but raw fd reads via `os.open`/`os.read`
    are not. The child's cwd is its own throwaway temp directory (see `execute_python`'s
    `cwd=tmpdir`), not this process's cwd, so the file is addressed by absolute path rather
    than relying on `monkeypatch.chdir` -- which would only change the parent's cwd, not the
    subprocess's.
    """
    secret_file = tmp_path / ".env"
    secret_file.write_text("SOME_SECRET=this-is-on-disk-and-readable\n")
    code = (
        "import json\n"
        f"real_os = {_REACH_REAL_OS}\n"
        f"fd = real_os.open({str(secret_file)!r}, real_os.O_RDONLY)\n"
        "print(real_os.read(fd, 4096).decode())\n"
        "real_os.close(fd)\n"
    )
    out = execute_python(code)
    assert "this-is-on-disk-and-readable" in out


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
    import allowlist enforced by both backends, so reaching for it here would require
    the same escape-through-an-allowlisted-module trick used above, which is beside the
    point of this test. ``SystemExit`` needs no import (it's a builtin exception,
    present in the restricted builtins) and is a ``BaseException``, so the shared
    ``except Exception:`` inside ``run_restricted`` does not catch it -- it propagates
    out of the child's ``-c`` script uncaught, and the interpreter exits with that
    status exactly as an unrecoverable crash would. What's actually under test is
    ``execute_python``'s handling of a nonzero child exit code, which is the same code
    path a real crash (a segfault, ``os._exit``) would take; this is the clean, fast,
    deterministic way to reach it without needing a non-allowlisted import or a
    platform-specific crash mechanism.
    """
    out = execute_python("raise SystemExit(1)")
    assert "error" in out.lower()
    # If crash isolation didn't hold, this line would never execute.
    assert True


def test_output_is_capped():
    huge = execute_python("print('x' * 1_000_000)")
    assert len(huge) < 1_000_000
    assert "truncated" in huge


def test_read_capped_never_loads_more_than_the_cap(tmp_path):
    """Minor 2: the old design captured the entire child stdout via a pipe (buffered
    fully in this process) before truncating -- one `print('x' * 200_000_000)` produced
    743 MB of *parent* RSS before the cap ever ran, and macOS applies no child-side cap
    at all, so nothing bounded it. Reading a bounded prefix directly from a file, as
    execute_python now does, means the excess never enters this process at all --
    verified here against a modest 10 KB file (fast; no real gigabytes needed to prove
    the mechanism, per the "these tests must be fast" rule).
    """
    path = tmp_path / "big.txt"
    payload = "x" * 10_000
    path.write_text(payload)

    capped = builtin._read_capped(str(path), 100)

    assert len(capped) < 10_000
    assert capped.startswith("x" * 100)
    assert "truncated" in capped
    assert "9900" in capped  # exact dropped-byte count, proving this is a real stat, not a guess


def test_invalid_utf8_from_the_child_does_not_raise():
    """Minor 3: the return contract is "a string, or an error string" -- never an
    exception. Raw/invalid bytes written straight to the stdout fd (bypassing the
    blocked `open()`/`print`'s text encoding) used to reach a text-mode decode and
    raise UnicodeDecodeError; reading bytes and decoding with errors="replace" means
    the tool always returns *something* instead.
    """
    code = f"import json\nreal_os = {_REACH_REAL_OS}\nreal_os.write(1, b'\\xff\\xfe not valid utf-8')\n"
    out = execute_python(code)
    assert isinstance(out, str)


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


def test_the_docstring_scopes_the_credential_claim_to_environment_variables():
    """Important 1: the docstring used to say the child has no access to
    "ANTHROPIC_API_KEY and every other credential this host holds", which is false --
    a `.env`, `~/.aws/credentials`, or `~/.config/gh/hosts.yml` is readable exactly as
    it would be to this process. The claim must be scoped to environment variables,
    with the filesystem paragraph doing the rest of the disclosure. It must also
    disclose that the child can reach back and kill this process (os.kill on the
    parent pid), since nothing here confines inter-process signalling either.
    """
    doc = execute_python.__doc__.lower()
    assert "environment variable" in doc
    assert "credentials in general" in doc or "not credentials in general" in doc
    assert "os.kill" in doc


def test_the_docstring_says_linux_for_the_memory_cap_and_mentions_the_warning():
    """Important 2: "a memory cap on POSIX" was silently false on macOS. The public
    wording must name Linux specifically and disclose that a failure to apply the cap
    is logged, not swallowed.
    """
    doc = execute_python.__doc__.lower()
    assert "memory cap on linux" in doc
    assert "warning is logged" in doc or "warning if the cap" in doc


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


@pytest.mark.parametrize("backend", [execute_python, execute_python_in_process], ids=["subprocess", "in_process"])
def test_both_backends_enforce_the_same_import_allowlist(backend):
    """Important 5: the docstrings promise both backends enforce identical rules, and
    nothing checked that promise held for both callables at once. (Reviewer note: a
    mutation that let `execute_python_in_process` allow `os`/`subprocess`/`open` while
    preserving the output contract left all other tests green -- this closes that gap.)
    """
    out_import = backend("import os")
    assert "Error" in out_import or "ImportError" in out_import
    out_open = backend("open('/etc/passwd')")
    assert "Error" in out_open or "NameError" in out_open


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


def test_worker_source_never_imports_the_aimu_package():
    """Important 4: importing `aimu.tools.builtin` in the child (a) makes the child
    depend on PYTHONPATH/site-packages resolution that the scrubbed env + temp cwd can
    break for a PYTHONPATH-based install or an unbuilt source checkout (every call would
    fail with ModuleNotFoundError), and (b) runs `aimu.tools.builtin`'s module-scope
    `load_dotenv()`, which -- once the worker is backed by a real file with a real
    `__file__` -- would silently reload a host repo's `.env` into the child, undoing
    the credential-scrubbing test above. The worker must only ever reach the
    dependency-free `_execute_python_worker` leaf module via an explicit
    `sys.path.insert`.
    """
    source = builtin._EXECUTE_PYTHON_WORKER_SOURCE
    assert "aimu.tools.builtin" not in source
    assert "import aimu" not in source
    assert "load_dotenv" not in source
    assert "sys.path.insert" in source
    assert "_execute_python_worker" in source


def test_memory_cap_unavailable_marker_triggers_a_one_time_warning(caplog):
    """Important 2: a platform that can't apply the memory cap (macOS, on this dev
    machine, confirmed independently of aimu via a plain shell's `ulimit -v`) must not
    swallow that fact -- it needs to be discoverable by whoever runs the tool, not just
    by reading a docstring. Tests the parent-side marker handling directly (fast,
    deterministic, platform-independent) rather than depending on the real subprocess
    launch to reproduce a platform-specific rlimit failure.
    """
    builtin._execute_python_warned.clear()
    with caplog.at_level("WARNING"):
        cleaned = builtin._strip_memory_cap_marker(f"{builtin._MEMORY_CAP_UNAVAILABLE_MARKER}\nother stderr text")
    assert builtin._MEMORY_CAP_UNAVAILABLE_MARKER not in cleaned
    assert "other stderr text" in cleaned
    assert any("memory cap" in rec.message.lower() for rec in caplog.records)

    # Warn-once: a second call with the same message must not log again.
    caplog.clear()
    with caplog.at_level("WARNING"):
        builtin._strip_memory_cap_marker(builtin._MEMORY_CAP_UNAVAILABLE_MARKER)
    assert len(caplog.records) == 0


def test_memory_cap_actually_failing_on_this_platform_is_end_to_end_discoverable(caplog):
    """The unit test above proves the marker-handling logic; this proves the real
    subprocess launch on this machine actually produces that marker and the warning
    fires end-to-end. Skips itself on a platform where the cap might genuinely succeed
    (Linux), since the point specific to *this* dev machine is macOS's confirmed
    refusal to lower RLIMIT_AS -- a platform where the cap works has nothing to warn
    about, which is the intended behavior, not a gap in the test.
    """
    if sys.platform != "darwin":
        pytest.skip("memory-cap failure is a confirmed macOS/XNU limitation; not expected to fire on this platform")
    builtin._execute_python_warned.clear()
    with caplog.at_level("WARNING"):
        assert "4" in execute_python("2 + 2")
    assert any("memory cap" in rec.message.lower() for rec in caplog.records)


def test_kill_process_group_also_kills_a_grandchild():
    """Important 3, mechanism-level: the direct-child-only `proc.kill()` historically
    left a backgrounded grandchild running past the timeout. Exercises the actual kill
    helper against a real process tree shaped like the one execute_python launches
    (`start_new_session=True`), independent of the code's own import allowlist, for a
    fast and fully deterministic check of exactly what gets killed.
    """
    proc = subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            "-c",
            "import subprocess, sys, time\n"
            "gc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
            "print(gc.pid)\n"
            "sys.stdout.flush()\n"
            "time.sleep(30)\n",
        ],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        grandchild_pid = int(proc.stdout.readline().strip())
    finally:
        builtin._kill_process_group(proc)
        proc.wait(timeout=5)

    for _ in range(50):
        try:
            os.kill(grandchild_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail(f"grandchild pid {grandchild_pid} was not killed")


def test_timeout_kills_the_whole_process_group_including_a_backgrounded_grandchild(monkeypatch, tmp_path):
    """Important 3, end-to-end: a snippet that both hangs itself and backgrounds
    another process must not leave the backgrounded process running once we report the
    timeout. `start_new_session=True` + `os.killpg(...)` on `TimeoutExpired` reaches the
    whole group, not just the direct child a plain `proc.kill()` would reach.

    The timeout is 5s rather than the sub-second one the neighbouring tests use: the
    subject here is *what the kill reaches*, not how quickly it fires, and the child has to
    get far enough to spawn the grandchild first. Cold-start of the child interpreter has
    been measured above 1.2s on a loaded machine, so a 0.5s budget SIGKILLed it mid-preload
    and the test failed for reasons unrelated to process groups.
    """
    monkeypatch.setattr(builtin, "_EXECUTE_PYTHON_TIMEOUT_S", 5.0)
    marker = tmp_path / "grandchild_pid.txt"
    shell_command = f"sleep 30 & echo $! > {marker}"
    code = f"import json\nreal_os = {_REACH_REAL_OS}\nreal_os.system({shell_command!r})\nwhile True: pass\n"

    out = execute_python(code)
    assert "timed out" in out.lower()

    grandchild_pid = None
    for _ in range(50):
        if marker.exists() and marker.read_text().strip():
            grandchild_pid = int(marker.read_text().strip())
            break
        time.sleep(0.05)
    assert grandchild_pid is not None, "grandchild never started"

    for _ in range(50):
        try:
            os.kill(grandchild_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail(f"grandchild pid {grandchild_pid} was not killed")


def test_kill_process_group_falls_back_to_proc_kill_without_killpg(monkeypatch):
    """Regression: `os.killpg`/`os.getpgid` don't exist on Windows. Before the
    fallback, `_kill_process_group` raised `AttributeError` on a
    platform lacking them instead of killing anything -- proved here by removing both
    attributes (simulating Windows without needing one) and confirming the direct
    child is still killed via `proc.kill()` rather than the call raising.
    """
    monkeypatch.delattr(builtin.os, "killpg", raising=False)
    monkeypatch.delattr(builtin.os, "getpgid", raising=False)

    proc = subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    try:
        builtin._kill_process_group(proc)
        proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    assert proc.returncode is not None


def test_timeout_path_still_returns_a_string_without_killpg(monkeypatch):
    """Important 1, end-to-end: with `os.killpg`/`os.getpgid` unavailable (simulating
    Windows), the timeout path must still return the timeout error string -- not raise
    `AttributeError` out of `execute_python` -- and the hung child must still be dead
    afterward, just via the direct-child-only `proc.kill()` fallback.
    """
    monkeypatch.delattr(builtin.os, "killpg", raising=False)
    monkeypatch.delattr(builtin.os, "getpgid", raising=False)
    monkeypatch.setattr(builtin, "_EXECUTE_PYTHON_TIMEOUT_S", 0.5)

    captured = {}
    real_popen_init = subprocess.Popen.__init__

    def _capturing_init(self, *args, **kwargs):
        real_popen_init(self, *args, **kwargs)
        captured["proc"] = self

    monkeypatch.setattr(subprocess.Popen, "__init__", _capturing_init)

    out = execute_python("while True: pass")

    assert "timed out" in out.lower()
    proc = captured["proc"]
    for _ in range(50):
        try:
            os.kill(proc.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail(f"child pid {proc.pid} was not killed by the proc.kill() fallback")


def test_interrupted_wait_kills_the_child_instead_of_orphaning_it(monkeypatch):
    """Important 2: `start_new_session=True` detaches the child from this process's
    controlling terminal, so a Ctrl-C (or any `BaseException` escaping `proc.wait()`,
    e.g. `asyncio.CancelledError` surfacing through a `RunHandle`) does not reach the
    child for free the way it would a foreground child. Before the `except
    BaseException` handler, an interrupted run left the child running indefinitely --
    orphaned, burning CPU. Simulates the interruption without a real Ctrl-C: the first
    `proc.wait()` call raises `KeyboardInterrupt`; `execute_python` must kill the child
    and re-raise rather than swallow the interrupt or leave the child alive. The second
    `proc.wait()` call (this handler's own cleanup) is left to run for real, so by the
    time `execute_python` re-raises, the child is actually reaped, not just signalled.
    """
    real_wait = subprocess.Popen.wait
    state = {"calls": 0, "proc": None}

    def _wait_raise_once(self, *args, **kwargs):
        state["calls"] += 1
        state["proc"] = self
        if state["calls"] == 1:
            raise KeyboardInterrupt()
        return real_wait(self, *args, **kwargs)

    monkeypatch.setattr(subprocess.Popen, "wait", _wait_raise_once)

    with pytest.raises(KeyboardInterrupt):
        execute_python("import time\ntime.sleep(30)\n")

    assert state["calls"] >= 2, "the cleanup proc.wait() after the kill was never reached"
    proc = state["proc"]
    assert proc.poll() is not None, "child was not reaped -- it would be orphaned and still running"


class _StdinRaisingOnClose:
    """Wraps a live ``Popen.stdin`` and raises ``KeyboardInterrupt`` from the first
    ``close()``, letting the ``write()`` before it genuinely succeed -- so the child is
    really running when the interrupt lands.
    """

    def __init__(self, real, state):
        self._real = real
        self._state = state

    def write(self, data):
        return self._real.write(data)

    def close(self):
        self._state["closes"] += 1
        if self._state["closes"] == 1:
            raise KeyboardInterrupt()
        return self._real.close()


def test_interrupt_while_feeding_stdin_kills_the_child_instead_of_orphaning_it(monkeypatch):
    """The kill-on-interrupt guard has to cover the child's whole lifetime, not just
    `proc.wait()`. The stdin write/close between spawning the child and reaching the
    wait was outside it: an interrupt there propagated in a fraction of a second while
    the child ran on to completion. `Popen.__exit__` is no backstop -- its
    KeyboardInterrupt branch waits only ~0.25s, on the assumption that the SIGINT
    already reached the child, which `start_new_session=True` makes false.

    Forces the interrupt with monkeypatch rather than racing a real signal, and asserts
    on the child's liveness rather than sleeping and hoping.
    """
    state = {"closes": 0, "proc": None}
    real_popen = subprocess.Popen

    class _InterruptingPopen(real_popen):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            state["proc"] = self
            self.stdin = _StdinRaisingOnClose(self.stdin, state)

    monkeypatch.setattr(subprocess, "Popen", _InterruptingPopen)

    with pytest.raises(KeyboardInterrupt):
        execute_python("import time\ntime.sleep(20)\n")

    proc = state["proc"]
    assert proc is not None, "the child was never spawned -- the interrupt landed too early to be this bug"
    pid = proc.pid
    try:
        assert proc.poll() is not None, "child was not reaped -- it would be orphaned and still running"
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        # Only reachable when the guard regressed and the child is still alive.
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def test_a_fast_backgrounding_snippet_returns_its_real_output_not_a_false_timeout():
    """Important 3, the other half: `subprocess.run`'s pipe-based `capture_output`
    waits for EOF, which requires every process holding a copy of the write end to
    close it -- including a backgrounded grandchild the snippet spawns and does not
    wait for. That made a fast, successful run falsely report "execution timed out"
    and discard its real output. Reading from files (this tool's actual mechanism)
    instead of pipes doesn't have this dependency: `Popen.wait()` only waits for the
    direct child.
    """
    code = f"import json\nreal_os = {_REACH_REAL_OS}\nreal_os.system('sleep 2 &')\nprint('done fast')\n"
    out = execute_python(code)
    assert "done fast" in out
    assert "timed out" not in out.lower()


def test_command_env_carries_no_parent_secrets(monkeypatch):
    """The property the allowlist exists for: a credential in this process's environment
    is absent from a command's, so `run_command("env")` cannot lift it into a model's
    context. Asserted on the mapping directly rather than end to end, because that is
    where the policy lives.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("KOKUA_EMAIL_PASSWORD", "should-not-leak-either")
    env = builtin._command_env()
    assert "ANTHROPIC_API_KEY" not in env
    assert "KOKUA_EMAIL_PASSWORD" not in env


def test_command_env_carries_the_variables_a_shell_needs(monkeypatch):
    """PATH alone is not enough to be a usable shell environment: a command that renders
    a table wants TERM, a timestamp wants TZ, and `whoami`-style output wants USER.
    """
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    monkeypatch.setenv("USER", "someone")
    env = builtin._command_env()
    assert env["PATH"] == "/usr/bin"
    assert env["TERM"] == "xterm-256color"
    assert env["TZ"] == "America/Los_Angeles"
    assert env["USER"] == "someone"


def test_command_env_passes_through_exactly_the_named_variables(monkeypatch):
    """The opt-in half. A host that wants `gh` or `ssh` to work names the variable, and
    naming one does not admit its neighbors.
    """
    monkeypatch.setenv("GH_TOKEN", "ghp-deliberate")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "not-asked-for")
    env = builtin._command_env(("GH_TOKEN",))
    assert env["GH_TOKEN"] == "ghp-deliberate"
    assert "AWS_SECRET_ACCESS_KEY" not in env


def test_command_env_omits_a_named_variable_that_is_unset(monkeypatch):
    """A passthrough name with nothing behind it produces no key, rather than a key whose
    value is the empty string: `SSH_AUTH_SOCK=""` is worse for ssh than no SSH_AUTH_SOCK.
    """
    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    assert "SSH_AUTH_SOCK" not in builtin._command_env(("SSH_AUTH_SOCK",))


def test_run_command_returns_stdout_and_the_exit_code():
    """The ordinary case. The exit line is always present, not only on failure, so the
    format a model has to parse is the same every time.
    """
    out = builtin.run_command("echo hello")
    assert "hello" in out
    assert "exit 0" in out


def test_a_nonzero_exit_returns_the_output_rather_than_an_error_string():
    """The deliberate divergence from execute_python, which collapses a nonzero exit into
    "Error: subprocess exited abnormally" and discards the output. `pytest` exits 1 with
    the answer on stdout, and `git diff --exit-code` exits 1 to mean "yes, there is a
    diff", so a nonzero exit here is information rather than a fault.
    """
    out = builtin.run_command("echo before; exit 3")
    assert "before" in out
    assert "exit 3" in out
    assert "exited abnormally" not in out


def test_stderr_comes_back_labelled_separately():
    """A command's diagnostics are worth having and worth distinguishing: a model that
    cannot tell them apart cannot tell a warning from an answer.
    """
    out = builtin.run_command("echo out; echo err 1>&2")
    assert "--- stdout ---" in out
    assert "--- stderr ---" in out
    assert out.index("--- stdout ---") < out.index("--- stderr ---")


def test_a_silent_command_says_so_rather_than_returning_a_bare_exit_line():
    out = builtin.run_command("true")
    assert "exit 0" in out
    assert "no output" in out


def test_shell_features_work():
    """The reason the tool takes a shell string at all. If a pipe does not work, a model
    writes `sh -c` itself and lands back here with an extra layer.

    Keyed off the stdout section specifically, and checking equality rather than substring
    membership, because `"a" in out` also passes for a bare `printf` that never reached the
    pipe (its unsorted output still contains the letter "a") and for a broken pipeline whose
    error message happens to contain the word "command".
    """
    out = builtin.run_command("printf 'b\\na\\n' | sort | head -1")
    assert "exit 0" in out
    assert "--- stdout ---" in out
    stdout_section = out.split("--- stdout ---", 1)[1]
    assert stdout_section.strip() == "a"


def test_a_command_reading_stdin_gets_eof_instead_of_hanging():
    """stdin is DEVNULL, not an open pipe. A command that reads it sees EOF at once; an
    open pipe nobody writes to would block until the timeout and report a hang.
    """
    start = time.monotonic()
    out = builtin.run_command("cat", timeout=5)
    assert time.monotonic() - start < 5
    assert "exit 0" in out


def test_cwd_is_honored():
    out = builtin.run_command("pwd", cwd="/tmp")
    assert "/tmp" in out or "/private/tmp" in out  # macOS resolves /tmp through a symlink


def test_cwd_defaults_to_the_process_working_directory(tmp_path, monkeypatch):
    """The default is the shell a user would have had, not execute_python's throwaway temp
    directory: `ls` and `git status` in an empty temp dir are useless.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "marker.txt").write_text("here")
    assert "marker.txt" in builtin.run_command("ls")


def test_a_nonexistent_cwd_returns_a_message_naming_the_path():
    """Popen raises FileNotFoundError for a missing cwd, and a model passing a path that
    does not exist is a routine mistake rather than a program fault: it should be told,
    so it can correct itself, not have an exception surface through tool dispatch.
    """
    out = builtin.run_command("pwd", cwd="/no/such/directory")
    assert "/no/such/directory" in out
    assert "does not exist" in out


def test_a_hanging_command_is_killed_by_its_timeout():
    start = time.monotonic()
    out = builtin.run_command("sleep 30", timeout=1)
    assert time.monotonic() - start < 15
    assert "timed out" in out.lower()


def test_a_timed_out_command_still_returns_what_it_printed_first():
    """The second deliberate divergence from execute_python, which discards the output on
    timeout. A test run killed at its ceiling has usually already printed the failure that
    mattered, and the file-based capture has it on disk at that point, so returning it is
    free.
    """
    out = builtin.run_command("echo progress; sleep 30", timeout=1)
    assert "timed out" in out.lower()
    assert "progress" in out


def test_a_timeout_above_the_ceiling_clamps(monkeypatch):
    """A model can ask for four minutes when it knows it needs four minutes, and cannot
    ask for forever.

    The real ceiling is 600s, which this test cannot wait out, so it lowers the ceiling
    itself: the property under test is that the *reported* number is the ceiling, not the
    requested value, and that holds at any ceiling.
    """
    monkeypatch.setattr(builtin, "_COMMAND_TIMEOUT_MAX_S", 2)
    start = time.monotonic()
    out = builtin.run_command("sleep 30", timeout=10_000)
    assert time.monotonic() - start < 15
    assert "timed out" in out.lower()
    assert str(builtin._COMMAND_TIMEOUT_MAX_S) in out


def test_a_nonpositive_timeout_becomes_one_second_rather_than_an_immediate_kill():
    """timeout=0 read literally kills the child before it can run, which reports a hang
    for a command that never had a chance. One second is the smallest honest floor.
    """
    out = builtin.run_command("echo quick", timeout=0)
    assert "quick" in out


def test_command_output_is_capped():
    """The cap is combined across both streams, not per stream: a test that only wrote to
    stdout would pass even if stderr doubled the return size on top of it, which is exactly
    the gap that let a two-stream command return roughly twice the documented limit.
    """
    limit = builtin._COMMAND_OUTPUT_LIMIT_BYTES
    out = builtin.run_command(f"printf 'o%.0s' $(seq 1 {limit}); printf 'e%.0s' $(seq 1 {limit}) 1>&2")
    assert "truncated" in out
    assert "--- stdout ---" in out
    assert "--- stderr ---" in out
    assert len(out) < limit + 500


def test_the_command_child_cannot_see_the_parents_api_keys(monkeypatch):
    """Task 3 asserts the policy on the mapping; this asserts the tool actually applies it,
    which is the half a wiring mistake would break.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    out = builtin.run_command("printenv ANTHROPIC_API_KEY")
    assert "sk-should-not-leak" not in out


def test_a_timeout_kills_a_backgrounded_grandchild_too(tmp_path):
    """The property _run_supervised exists for, asserted through this tool: SIGKILL reaches
    the whole process group, so a command that backgrounds something and hangs does not
    leave that something running after the timeout. The grandchild writes its marker at
    t=2 and the parent is killed at t=1, so the marker appears only if the kill failed to
    reach the group; a kill of the direct child alone would let it survive and write.
    """
    marker = tmp_path / "grandchild.txt"
    out = builtin.run_command(f"(sleep 2; echo alive > {marker}) & sleep 30", timeout=1)
    assert "timed out" in out.lower()
    time.sleep(4)
    assert not marker.exists()


def test_a_fast_command_that_backgrounds_something_is_not_a_false_timeout():
    """The mirror of the test above, and the reason output goes to files rather than pipes:
    wait() depends only on the direct child's exit, so a grandchild holding the inherited
    fds cannot make a run that finished in milliseconds report a hang.
    """
    start = time.monotonic()
    out = builtin.run_command("(sleep 20) & echo done", timeout=5)
    assert time.monotonic() - start < 5
    assert "done" in out
    assert "timed out" not in out.lower()


def test_make_command_tool_admits_a_named_variable(monkeypatch):
    """The factory is how a host turns the passthrough into a real setting: the tool a model
    sees takes only the arguments a model should choose, so a config-driven allowance needs
    a closure.
    """
    monkeypatch.setenv("GH_TOKEN", "ghp-deliberate")
    tool_fn = builtin.make_command_tool(env_passthrough=("GH_TOKEN",))
    assert "ghp-deliberate" in tool_fn("printenv GH_TOKEN")


def test_make_command_tool_admits_nothing_by_default(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "ghp-not-asked-for")
    assert "ghp-not-asked-for" not in builtin.make_command_tool()("printenv GH_TOKEN")


def test_the_module_level_tool_admits_nothing(monkeypatch):
    """`run_command` is `make_command_tool()` with no arguments, and AIMU's other consumers
    take it from `builtin.compute` without ever calling the factory, so the default has to be
    the safe one.
    """
    monkeypatch.setenv("GH_TOKEN", "ghp-not-asked-for")
    assert "ghp-not-asked-for" not in builtin.run_command("printenv GH_TOKEN")


def test_every_built_tool_keeps_the_name_and_the_description_a_model_reads():
    """@tool parses the docstring at decoration time and freezes it into __tool_spec__, so a
    closure with no docstring reaches a model as a tool with an empty description, and a
    __doc__ assigned afterwards never lands. A wrong __name__ is worse than cosmetic too: an
    approval gate keyed on "run_command" stops matching a tool called "wrapper".
    """
    for tool_fn in (builtin.run_command, builtin.make_command_tool(env_passthrough=("GH_TOKEN",))):
        assert tool_fn.__name__ == "run_command"
        assert tool_fn.__tool_spec__["function"]["name"] == "run_command"
        assert "NOT A SECURITY BOUNDARY" in tool_fn.__tool_spec__["function"]["description"]


def test_the_run_command_docstring_refuses_to_claim_containment():
    """The same standard execute_python's docstring is held to. A shell string reaches the
    filesystem, the network, and other processes as the calling user, and the docstring is
    where a caller finds that out.
    """
    doc = builtin.run_command.__doc__
    assert "NOT A SECURITY BOUNDARY" in doc
    assert "container" in doc.lower()


def test_the_run_command_docstring_names_the_credential_and_signalling_limits():
    """Two specifics that are easy to omit and expensive to discover: credentials sitting in
    files are readable, and process signalling is unconfined.
    """
    doc = builtin.run_command.__doc__.lower()
    assert ".env" in doc
    assert "signal" in doc


def test_run_command_is_in_the_compute_group():
    """What Kokua's `compute` toolset and every other consumer of the group depend on. Note
    this is a widening: a host already declaring `compute` gains a shell on upgrade.
    """
    assert builtin.run_command in builtin.compute


def test_run_command_is_not_in_all_tools():
    """Same reasoning that keeps execute_python out: this is isolation, not containment, so
    it is opt-in rather than part of the default set.
    """
    assert builtin.run_command not in builtin.ALL_TOOLS


def test_allow_code_execution_does_not_reach_the_shell_tool():
    """`make_tools(allow_code_execution=True)` appends execute_python and deliberately not
    this tool: the flag says *code* execution, and widening it would grant a shell to every
    caller already passing it. `builtin.compute` is the one route in. Pinned so the
    narrowness is on the record rather than incidental to how make_tools happens to be
    written.
    """
    client = SimpleNamespace(model=SimpleNamespace(supports_vision=False))
    tools = builtin.make_tools(client, allow_code_execution=True)
    assert builtin.execute_python in tools
    assert builtin.run_command not in tools
