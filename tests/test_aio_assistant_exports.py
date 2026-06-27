"""Smoke test: personal-assistant primitives are importable and exported."""

from __future__ import annotations

import aimu.aio as aio
import aimu.skills as skills


def test_aio_exports_channels_and_scheduler():
    from aimu.aio import CLIChannel, Channel, ChannelMessage, Scheduler  # noqa: F401

    for name in ("Channel", "ChannelMessage", "CLIChannel", "Scheduler"):
        assert name in aio.__all__


def test_skills_exports_authoring():
    from aimu.skills import make_skill_authoring_tool, write_skill  # noqa: F401

    for name in ("write_skill", "make_skill_authoring_tool"):
        assert name in skills.__all__


def test_skill_manager_has_refresh():
    assert callable(skills.SkillManager.refresh)
