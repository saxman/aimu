"""Pulling an Ollama model: what gets logged, and when a client pulls at all.

No server. The SDK's ``pull(stream=True)`` is replaced with a list of ``ProgressResponse`` events in
the shape Ollama streams them: status-only events (manifest, verification, success) carry no
``total``, and each layer reports ``completed`` against ``total``. A layer already on disk arrives
complete; only one that is actually downloading has ``completed < total``.
"""

from __future__ import annotations

import logging

import ollama
import pytest

from aimu.aio.providers.ollama import apull_ollama_model
from aimu.models import ModelNotInstalledError
from aimu.models.providers.ollama import pull_ollama_model

_GIB = 1024**3


def _event(status="pulling", digest=None, total=None, completed=None) -> ollama.ProgressResponse:
    return ollama.ProgressResponse(status=status, digest=digest, total=total, completed=completed)


_UP_TO_DATE = [
    _event("pulling manifest"),
    _event(digest="sha256:a", total=100, completed=100),
    _event("verifying sha256 digest"),
    _event("success"),
]

_DOWNLOADING = [
    _event("pulling manifest"),
    _event(digest="sha256:b", total=10 * 10**9, completed=0),
    _event(digest="sha256:b", total=10 * 10**9, completed=5 * 10**9),
    _event(digest="sha256:b", total=10 * 10**9, completed=10 * 10**9),
    _event("success"),
]


class _FakeSdk:
    def __init__(self, events=None, error=None):
        self.events = events or []
        self.error = error
        self.pulls: list[str] = []

    def pull(self, model, *, stream=False):
        self.pulls.append(model)
        assert stream, "the helper must stream the pull to see progress"
        if self.error is not None:
            raise self.error
        return iter(self.events)


class _AsyncFakeSdk(_FakeSdk):
    async def pull(self, model, *, stream=False):
        events = super().pull(model, stream=stream)

        async def stream_events():
            for event in events:
                yield event

        return stream_events()


def _warnings(caplog):
    return [r for r in caplog.records if r.levelno == logging.WARNING]


def test_up_to_date_pull_logs_no_warning(caplog):
    caplog.set_level(logging.DEBUG)
    pull_ollama_model(_FakeSdk(_UP_TO_DATE), "qwen3:8b")
    assert _warnings(caplog) == []
    assert any(r.levelno == logging.DEBUG and "up to date" in r.getMessage() for r in caplog.records)


def test_partial_layer_logs_one_warning(caplog):
    caplog.set_level(logging.DEBUG)
    pull_ollama_model(_FakeSdk(_DOWNLOADING), "qwen3:8b")
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "qwen3:8b" in warnings[0].getMessage()
    assert "9.3 GiB" in warnings[0].getMessage()
    infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert any("9.3 GiB downloaded" in message for message in infos)


def test_events_without_total_are_ignored(caplog):
    caplog.set_level(logging.DEBUG)
    events = [_event("pulling manifest"), _event("verifying sha256 digest"), _event("success")]
    pull_ollama_model(_FakeSdk(events), "qwen3:8b")
    assert _warnings(caplog) == []


def test_pull_error_propagates_unwrapped():
    error = ollama.ResponseError("pull model manifest: file does not exist", status_code=404)
    with pytest.raises(ollama.ResponseError) as info:
        pull_ollama_model(_FakeSdk(error=error), "nope:1b")
    assert info.value is error
    assert not isinstance(info.value, ModelNotInstalledError)


async def test_async_up_to_date_pull_logs_no_warning(caplog):
    caplog.set_level(logging.DEBUG)
    await apull_ollama_model(_AsyncFakeSdk(_UP_TO_DATE), "qwen3:8b")
    assert _warnings(caplog) == []


async def test_async_partial_layer_logs_one_warning(caplog):
    caplog.set_level(logging.DEBUG)
    await apull_ollama_model(_AsyncFakeSdk(_DOWNLOADING), "qwen3:8b")
    assert len(_warnings(caplog)) == 1
