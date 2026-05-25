"""Async provider clients. Mirrors ``aimu/models/{provider}/``.

Each provider module is imported lazily by ``aimu.aio._model_client`` so a missing
optional dependency doesn't break the rest of the async surface.
"""
