"""Tool groups: tools an agent holds without sending their schemas until it loads them.

An agent pays for every tool schema on every model call, whether or not the turn uses the tool. A tool
group lets a host give an agent tools it *holds* without advertising them: the agent lists what it can
load (``available_tools``), loads a group by name (``load_tools``), and the group's tools join its tool
list from the next round on. This is the pattern behind Anthropic's tool search (``defer_loading``) and
Claude Code's deferred tools, written as plain Python a reader can follow.

Two kinds of group, told apart by ``build``:

- **Deferred** (``build`` set): built once per agent instance, when loaded (again only after a failed
  build), so a side effect of building (opening a store, loading a model) is paid only by an agent that
  actually loads the group.
- **Resident** (``build=None``): the host already put this group's tools in ``Agent(tools=...)``. It is
  listed so ``available_tools`` shows the agent its whole picture, and it is never built.

Loading is append-only: a loaded group's tools go after every tool the agent already holds, and nothing is
unloaded, so within one agent's life a request's tool list only grows, which keeps a provider's prompt
cache valid. A loaded tool whose name is already taken is skipped and reported rather than allowed to
shadow the earlier tool, since two tools with one name is a request some providers reject and a dispatch
the loop would resolve silently.

This module holds state and renders text; it never awaits. The sync and async ``Agent`` each build their
own ``available_tools`` / ``load_tools`` over a :class:`ToolGroupState`, because the async one runs
``build`` off the event loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

_logger = logging.getLogger(__name__)

#: The two tools an agent carries when it has at least one deferred group.
CORE_TOOL_NAMES = ("available_tools", "load_tools")


@dataclass(frozen=True)
class ToolGroup:
    """A named set of tools an agent can list and, when deferred, load.

    Args:
        name: How the model names the group to ``load_tools``. Unique within one agent.
        description: One line saying what the group's tools do; the model chooses by it.
        build: Returns the group's tool callables. ``None`` marks a resident group.
        guidance: How to use the tools, returned as ``load_tools``' result when the group loads.
        note: Free text appended to the group's ``available_tools`` line, for example which of its
            tools ask before running.
    """

    name: str
    description: str
    build: Optional[Callable[[], list]] = None
    guidance: str = ""
    note: str = ""

    @property
    def resident(self) -> bool:
        return self.build is None


def check_tool_groups(groups, where: str) -> None:
    """Raise ``ValueError`` for a malformed group list when it is configured, naming ``where``.

    Found mid-run instead, a bad group would surface as a tool failure the model is asked to recover from,
    which is the wrong reader for a programmer error.
    """
    if not isinstance(groups, (list, tuple)):
        raise ValueError(f"{where} must be a list of ToolGroup, got {groups!r}.")
    seen: set[str] = set()
    for group in groups:
        if not isinstance(group, ToolGroup):
            raise ValueError(f"{where} must contain only ToolGroup instances, got {group!r}.")
        if not group.name or group.name != group.name.strip():
            raise ValueError(f"{where} has a group with an empty or padded name: {group.name!r}.")
        if group.build is not None and not callable(group.build):
            raise ValueError(f"{where}: group {group.name!r} has a build that is not callable.")
        if group.name in seen:
            raise ValueError(f"{where} names the group {group.name!r} twice.")
        seen.add(group.name)


def check_core_names(tools: Iterable, groups: Sequence[ToolGroup], where: str) -> None:
    """Refuse a host tool named like a core tool, but only when the core tools will exist.

    An agent with a deferred group carries ``available_tools`` and ``load_tools``; a host tool with either
    name would be a second tool with that name in every request.
    """
    if not any(not g.resident for g in groups):
        return
    for fn in tools:
        name = getattr(fn, "__name__", None)
        if name in CORE_TOOL_NAMES:
            raise ValueError(
                f"{where} contains a tool named {name!r}, which an agent with a deferred tool group already "
                "provides. Rename the tool."
            )


def missing_tool_hint(tool_names: Iterable[str]) -> str:
    """Text appended to the loop's "not found" result when the agent can load tool groups.

    A model that has seen a group's name tends to call a tool it guesses lives there without loading the
    group first; a bare "not found" leaves it guessing, while this names the two steps that work.
    """
    if "load_tools" in set(tool_names):
        return (
            " If it belongs to a tool group you have not loaded, call available_tools to see what you can load, "
            "then load_tools."
        )
    return ""


class ToolGroupState:
    """One agent instance's groups and what it has loaded.

    Never shared between agent instances: a spawned worker is built fresh per call, so its loads are its
    own even when several spawns of one worker type run concurrently from one shared spec.
    """

    def __init__(self, groups: Sequence[ToolGroup]):
        self.groups = list(groups)
        self._by_name = {g.name: g for g in self.groups}
        # Insertion order is load order, which is the order loaded tools are appended in.
        self._loaded: dict[str, list] = {}

    @property
    def has_deferred(self) -> bool:
        return any(not g.resident for g in self.groups)

    def loaded_tools(self) -> list:
        return [fn for tools in self._loaded.values() for fn in tools]

    def listing(self, filter_text: str = "") -> str:
        needle = filter_text.strip().lower()
        lines = []
        for group in self.groups:
            if needle and needle not in group.name.lower() and needle not in group.description.lower():
                continue
            if group.resident:
                state = "resident"
            elif group.name in self._loaded:
                state = "loaded"
            else:
                state = "loadable"
            line = f"{group.name} [{state}]: {group.description}"
            if group.note:
                line += f" {group.note}"
            lines.append(line)
        if not lines:
            return f"No tool group matches {filter_text!r}. Call available_tools with no filter to see them all."
        return "\n".join(lines)

    def precheck(self, name: str) -> Optional[str]:
        """``load_tools``' result when ``name`` cannot be loaded now, else ``None``."""
        group = self._by_name.get(name.strip())
        if group is None:
            loadable = [g.name for g in self.groups if not g.resident and g.name not in self._loaded]
            if loadable:
                return f"No tool group named {name!r}. Loadable: {', '.join(loadable)}."
            return f"No tool group named {name!r}. Everything available is already in your tools."
        if group.resident:
            return f"{group.name!r} is resident: its tools are already in your tool list."
        if group.name in self._loaded:
            return f"{group.name!r} is already loaded."
        return None

    def group(self, name: str) -> ToolGroup:
        return self._by_name[name.strip()]

    def build_failed(self, name: str, exc: BaseException) -> str:
        """``load_tools``' result for a build that raised; the group stays loadable so a retry can work."""
        _logger.warning("Tool group %r failed to build; it stays unloaded.", name, exc_info=exc)
        return f"Could not load {name!r}: {exc}"

    def record(self, name: str, built, existing_names: set) -> str:
        """Record a finished build and return ``load_tools``' result.

        ``existing_names`` is every tool name the agent holds right now. A built tool reusing one is
        skipped, never appended, and the result says which group (if any) holds the name.
        """
        # Checked before any state changes: the loop indexes tools by ``__name__`` every round, so a nameless
        # callable recorded here would break every later round rather than just this load.
        if not isinstance(built, list) or not all(
            callable(fn) and isinstance(getattr(fn, "__name__", None), str) for fn in built
        ):
            return self.build_failed(name, TypeError(f"build returned {built!r}, not a list of named tool callables"))
        group = self._by_name[name]
        kept, skipped = [], []
        taken = set(existing_names)
        for fn in built:
            tool_name = getattr(fn, "__name__", None)
            if tool_name in taken:
                skipped.append(tool_name)
                continue
            kept.append(fn)
            taken.add(tool_name)
        self._loaded[name] = kept
        if not built:
            lines = [f"Loaded {name!r}: it provides no tools right now."]
        elif kept:
            lines = [f"Loaded {name!r}: {', '.join(fn.__name__ for fn in kept)}."]
        else:
            lines = [f"Loaded {name!r}: no new tools."]
        if skipped:
            held = ", ".join(f"{tool_name} (by {self._holder(tool_name, name)})" for tool_name in skipped)
            lines.append(f"Not loaded, because the name is already taken: {held}.")
        if group.guidance:
            lines.append(group.guidance)
        return "\n".join(lines)

    def _holder(self, tool_name: str, loading: str) -> str:
        for group_name, tools in self._loaded.items():
            if group_name != loading and any(getattr(fn, "__name__", None) == tool_name for fn in tools):
                return repr(group_name)
        return "a tool given to this agent directly"
