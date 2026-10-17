"""ToolGroup validation and ToolGroupState: the surface-agnostic half of tool groups (mock-free)."""

from __future__ import annotations

import pytest

from aimu.tools import ToolGroup, tool
from aimu.tools.groups import (
    CORE_TOOL_NAMES,
    ToolGroupState,
    check_core_names,
    check_tool_groups,
    missing_tool_hint,
)


@tool
def search_web(query: str) -> str:
    """Search the web."""
    return query


@tool
def read_file(path: str) -> str:
    """Read a file."""
    return path


WEB = ToolGroup("web", "Search the web and read pages.", build=lambda: [search_web], guidance="Cite sources.")
MEMORY = ToolGroup("memory", "Remember facts across conversations.")
COMPUTE = ToolGroup("compute", "Run Python and shell commands.", build=lambda: [], note="Asks first: run_command.")


def test_a_group_without_build_is_resident():
    assert MEMORY.resident
    assert not WEB.resident


def test_check_rejects_a_non_list():
    with pytest.raises(ValueError, match="tool_groups must be a list"):
        check_tool_groups(WEB, "tool_groups")


def test_check_rejects_a_non_group_entry():
    with pytest.raises(ValueError, match="only ToolGroup instances"):
        check_tool_groups([WEB, "memory"], "tool_groups")


def test_check_rejects_a_duplicate_name():
    with pytest.raises(ValueError, match="names the group 'web' twice"):
        check_tool_groups([WEB, ToolGroup("web", "again", build=list)], "tool_groups")


def test_check_rejects_an_empty_or_padded_name():
    with pytest.raises(ValueError, match="empty or padded name"):
        check_tool_groups([ToolGroup(" web", "x", build=list)], "tool_groups")


def test_check_rejects_a_non_callable_build():
    with pytest.raises(ValueError, match="build that is not callable"):
        check_tool_groups([ToolGroup("web", "x", build="nope")], "tool_groups")


def test_core_names_collide_only_when_a_group_is_deferred():
    @tool
    def load_tools(name: str) -> str:
        """A host tool that happens to share a core name."""
        return name

    check_core_names([load_tools], [MEMORY], "tools")  # resident only: no core tools, no collision
    with pytest.raises(ValueError, match="'load_tools'"):
        check_core_names([load_tools], [WEB], "tools")
    assert CORE_TOOL_NAMES == ("available_tools", "load_tools")


def test_listing_shows_every_group_with_its_state_and_note_in_given_order():
    state = ToolGroupState([MEMORY, WEB, COMPUTE])
    assert state.listing() == (
        "memory [resident]: Remember facts across conversations.\n"
        "web [loadable]: Search the web and read pages.\n"
        "compute [loadable]: Run Python and shell commands. Asks first: run_command."
    )


def test_listing_filter_matches_name_or_description_case_insensitively():
    state = ToolGroupState([MEMORY, WEB])
    assert state.listing("PAGES") == "web [loadable]: Search the web and read pages."
    assert state.listing("mem") == "memory [resident]: Remember facts across conversations."


def test_listing_filter_with_no_match_says_how_to_see_everything():
    assert ToolGroupState([WEB]).listing("zzz") == (
        "No tool group matches 'zzz'. Call available_tools with no filter to see them all."
    )


def test_precheck_unknown_name_lists_what_is_loadable():
    state = ToolGroupState([MEMORY, WEB, COMPUTE])
    assert state.precheck("wbe") == "No tool group named 'wbe'. Loadable: web, compute."


def test_precheck_when_nothing_is_left_to_load():
    state = ToolGroupState([MEMORY])
    assert state.precheck("x") == "No tool group named 'x'. Everything available is already in your tools."


def test_precheck_resident_and_already_loaded():
    state = ToolGroupState([MEMORY, WEB])
    assert state.precheck("memory") == "'memory' is resident: its tools are already in your tool list."
    assert state.precheck(" web ") is None
    state.record("web", [search_web], existing_names=set())
    assert state.precheck("web") == "'web' is already loaded."


def test_record_appends_in_load_order_and_returns_names_then_guidance():
    state = ToolGroupState([WEB, ToolGroup("fs", "Files.", build=lambda: [read_file])])
    assert state.record("fs", [read_file], existing_names=set()) == "Loaded 'fs': read_file."
    assert state.record("web", [search_web], existing_names={"read_file"}) == (
        "Loaded 'web': search_web.\nCite sources."
    )
    assert [fn.__name__ for fn in state.loaded_tools()] == ["read_file", "search_web"]
    assert "web [loaded]" in state.listing()


def test_record_skips_a_taken_name_and_says_who_holds_it():
    fs = ToolGroup("fs", "Files.", build=lambda: [read_file])
    other = ToolGroup("other", "Also reads files.", build=lambda: [read_file, search_web])
    state = ToolGroupState([fs, other])
    state.record("fs", [read_file], existing_names=set())
    text = state.record("other", [read_file, search_web], existing_names={"read_file"})
    assert text == ("Loaded 'other': search_web.\nNot loaded, because the name is already taken: read_file (by 'fs').")
    assert [fn.__name__ for fn in state.loaded_tools()] == ["read_file", "search_web"]


def test_record_names_a_directly_given_tool_as_the_holder():
    state = ToolGroupState([WEB])
    text = state.record("web", [search_web], existing_names={"search_web"})
    assert text == (
        "Loaded 'web': no new tools.\n"
        "Not loaded, because the name is already taken: search_web (by a tool given to this agent directly).\n"
        "Cite sources."
    )


def test_record_a_group_with_no_tools():
    state = ToolGroupState([COMPUTE])
    assert state.record("compute", [], existing_names=set()) == "Loaded 'compute': it provides no tools right now."
    assert state.precheck("compute") == "'compute' is already loaded."


def test_record_rejects_a_build_that_did_not_return_a_list_of_callables():
    state = ToolGroupState([WEB])
    assert state.record("web", "not a list", existing_names=set()).startswith("Could not load 'web': ")
    assert state.precheck("web") is None  # still loadable


def test_build_failed_text_and_state_unchanged():
    state = ToolGroupState([WEB])
    assert state.build_failed("web", RuntimeError("store locked")) == "Could not load 'web': store locked"
    assert state.precheck("web") is None


def test_has_deferred():
    assert ToolGroupState([MEMORY, WEB]).has_deferred
    assert not ToolGroupState([MEMORY]).has_deferred
    assert not ToolGroupState([]).has_deferred


def test_missing_tool_hint_only_when_load_tools_is_present():
    assert missing_tool_hint({"search_web"}) == ""
    assert missing_tool_hint({"load_tools", "available_tools"}) == (
        " If it belongs to a tool group you have not loaded, call available_tools to see what you can load, "
        "then load_tools."
    )


def test_record_rejects_a_tool_without_a_name_and_leaves_the_group_loadable():
    # A nameless callable (a functools.partial) recorded as loaded would crash every later round, since
    # the loop indexes tools by __name__; it must fail the load instead, before any state changes.
    import functools

    state = ToolGroupState([WEB])
    nameless = functools.partial(lambda query: query, "x")
    assert state.record("web", [nameless], existing_names=set()).startswith("Could not load 'web': ")
    assert state.precheck("web") is None
    assert state.loaded_tools() == []
