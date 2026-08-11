"""Mock-only unit tests for the timezone-aware time tools.

Covers ``get_current_date_and_time`` (offset-aware output, IANA lookup) and
``convert_time`` (cross-zone conversion, DST gap/fold notes). No network access.
Zone-name derivation is exercised by monkeypatching ``TZ`` and the
``/etc/localtime`` lookup, so the suite never depends on the host's zone.
"""

from __future__ import annotations

import datetime
import re
from zoneinfo import ZoneInfo

from aimu.tools.builtin import convert_time, get_current_date_and_time

# Leading ISO-8601 timestamp with a mandatory UTC offset.
AWARE_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}\b")


class TestGetCurrentDateAndTime:
    def test_local_output_carries_a_utc_offset(self):
        assert AWARE_ISO.match(get_current_date_and_time())

    def test_local_output_omits_microseconds(self):
        assert "." not in get_current_date_and_time().split()[0]

    def test_named_zone_reports_that_zone(self):
        result = get_current_date_and_time(timezone="Asia/Tokyo")

        assert AWARE_ISO.match(result)
        assert "Asia/Tokyo" in result
        assert "+09:00" in result

    def test_named_zone_reports_the_utc_equivalent(self):
        result = get_current_date_and_time(timezone="Asia/Tokyo")

        utc_now = datetime.datetime.now(datetime.timezone.utc)
        assert utc_now.strftime("%Y-%m-%dT%H:%M")[:15] in result
        assert result.rstrip(")").endswith("Z")

    def test_unknown_zone_returns_a_teaching_string(self):
        result = get_current_date_and_time(timezone="Tokyo")

        assert "Unknown timezone: 'Tokyo'" in result
        assert "IANA" in result
        assert "Asia/Tokyo" in result

    def test_tool_spec_exposes_optional_timezone(self):
        spec = get_current_date_and_time.__tool_spec__["function"]

        assert spec["name"] == "get_current_date_and_time"
        assert "timezone" in spec["parameters"]["properties"]
        assert spec["parameters"].get("required", []) == []


class TestConvertTime:
    def test_converts_naive_input_between_zones(self):
        result = convert_time("2026-11-02T15:00:00", "Europe/Berlin", "America/Denver")

        assert "2026-11-02T15:00:00+01:00" in result
        assert "Europe/Berlin" in result
        assert "2026-11-02T07:00:00-07:00" in result
        assert "America/Denver" in result

    def test_accepts_a_space_separated_timestamp(self):
        result = convert_time("2026-11-02 15:00", "Europe/Berlin", "America/Denver")

        assert "2026-11-02T07:00:00-07:00" in result

    def test_aware_input_keeps_its_offset_and_notes_the_override(self):
        result = convert_time("2026-11-02T15:00:00+00:00", "Europe/Berlin", "America/Denver")

        assert "2026-11-02T07:00:00-07:00" not in result
        assert "2026-11-02T08:00:00-07:00" in result
        assert "from_timezone ignored" in result
        assert "+00:00" in result

    def test_notes_a_nonexistent_spring_forward_time(self):
        # 2026-03-08 is the second Sunday in March: 02:00-03:00 local does not exist.
        result = convert_time("2026-03-08T02:30:00", "America/Los_Angeles", "UTC")

        assert "does not exist" in result
        assert "America/Los_Angeles" in result

    def test_nonexistent_time_reports_the_resolved_instant_as_the_source(self):
        result = convert_time("2026-03-08T02:30:00", "America/Los_Angeles", "UTC")

        # The source line must not present a wall-clock time that never occurred.
        assert result.splitlines()[0].startswith("2026-03-08T03:30:00-07:00")

    def test_notes_an_ambiguous_fall_back_time(self):
        # 2026-11-01 is the first Sunday in November: 01:00-02:00 local happens twice.
        result = convert_time("2026-11-01T01:30:00", "America/Los_Angeles", "UTC")

        assert "ambiguous" in result
        assert "first occurrence" in result

    def test_unambiguous_time_has_no_dst_note(self):
        result = convert_time("2026-06-15T12:00:00", "America/Los_Angeles", "UTC")

        assert "note:" not in result

    def test_round_trips_back_to_the_source_zone(self):
        forward = convert_time("2026-06-15T12:00:00", "America/Los_Angeles", "Asia/Tokyo")
        tokyo = forward.splitlines()[-1].split()[1]

        back = convert_time(tokyo, "Asia/Tokyo", "America/Los_Angeles")

        assert "2026-06-15T12:00:00-07:00" in back

    def test_unknown_source_zone_returns_a_teaching_string(self):
        result = convert_time("2026-06-15T12:00:00", "Berlin", "UTC")

        assert "Unknown timezone: 'Berlin'" in result

    def test_unknown_target_zone_returns_a_teaching_string(self):
        result = convert_time("2026-06-15T12:00:00", "UTC", "Mars/Olympus")

        assert "Unknown timezone: 'Mars/Olympus'" in result

    def test_unparseable_timestamp_returns_a_teaching_string(self):
        result = convert_time("next tuesday", "UTC", "Asia/Tokyo")

        assert "next tuesday" in result
        assert "ISO 8601" in result

    def test_tool_spec_requires_all_three_arguments(self):
        spec = convert_time.__tool_spec__["function"]

        assert spec["name"] == "convert_time"
        assert set(spec["parameters"]["required"]) == {"datetime_str", "from_timezone", "to_timezone"}


class TestLocalZoneDerivation:
    def test_prefers_the_tz_environment_variable(self, monkeypatch):
        monkeypatch.setenv("TZ", "Asia/Tokyo")

        assert "Asia/Tokyo" in get_current_date_and_time()

    def test_falls_back_to_the_localtime_symlink(self, monkeypatch):
        monkeypatch.delenv("TZ", raising=False)
        monkeypatch.setattr(
            "aimu.tools.builtin.os.path.realpath",
            lambda _: "/usr/share/zoneinfo/Europe/Berlin",
        )

        assert "Europe/Berlin" in get_current_date_and_time()

    def test_omits_the_zone_name_when_undeterminable(self, monkeypatch):
        monkeypatch.delenv("TZ", raising=False)
        monkeypatch.setattr("aimu.tools.builtin.os.path.realpath", lambda _: "/etc/localtime")

        result = get_current_date_and_time()

        assert AWARE_ISO.match(result)
        assert "/" not in result


class TestRegistration:
    def test_both_tools_are_in_the_misc_subgroup(self):
        from aimu.tools import builtin

        names = {fn.__name__ for fn in builtin.misc}
        assert {"get_current_date_and_time", "convert_time"} <= names

    def test_convert_time_is_in_all_tools(self):
        from aimu.tools import builtin

        assert convert_time in builtin.ALL_TOOLS

    def test_convert_time_is_re_exported_from_the_async_surface(self):
        from aimu.aio.tools import builtin as aio_builtin

        assert aio_builtin.convert_time is convert_time


class TestSandbox:
    def test_execute_python_can_import_zoneinfo(self):
        from aimu.tools.builtin import execute_python

        code = "from zoneinfo import ZoneInfo\nprint(ZoneInfo('Asia/Tokyo').key)"
        assert "Asia/Tokyo" in execute_python(code)

    def test_zoneinfo_resolves_a_real_offset(self):
        # Guards the tzdata availability the tools depend on.
        assert ZoneInfo("Asia/Tokyo").utcoffset(datetime.datetime(2026, 6, 15)) == datetime.timedelta(hours=9)
