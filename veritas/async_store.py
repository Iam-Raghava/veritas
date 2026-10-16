"""Async wrapper for BeliefStore.

Modern agent frameworks are asyncio-based. AsyncBeliefStore exposes the
same API as BeliefStore but with async methods, running the thread-safe
sync implementation in a thread pool via asyncio.to_thread. This is the
standard pattern for wrapping sync I/O-bound or CPU-bound libraries
(cf. asyncpg, httpx sync/async duality).

Usage:
    store = AsyncBeliefStore()
    await store.assert_belief("Sky is blue", source="sensor")
    beliefs = await store.active_beliefs()
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from .belief import Belief
from .store import BeliefStore


class AsyncBeliefStore:
    """Async interface to a thread-safe BeliefStore."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._store = BeliefStore(*args, **kwargs)
        # Capture the running event loop (if any) so async subscriber
        # callbacks fired from worker threads can be scheduled on the
        # main loop via run_coroutine_threadsafe (instead of creating a
        # second loop, which breaks callbacks bound to the main loop).
        try:
            self._store._async_main_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._store._async_main_loop = None

    @property
    def sync_store(self) -> BeliefStore:
        """Access the underlying sync store (for sync contexts)."""
        return self._store

    async def assert_belief(
        self,
        proposition: str,
        confidence: float = 0.8,
        source: str = "unknown",
        **kwargs: Any,
    ) -> Belief | None:
        return await asyncio.to_thread(
            self._store.assert_belief, proposition, confidence, source, **kwargs
        )

    async def retract_belief(
        self, belief_id: str, reason: str = "manual retraction"
    ) -> list[Belief]:
        return await asyncio.to_thread(
            self._store.retract_belief, belief_id, reason
        )

    async def assert_batch(
        self, items: list[dict],
    ) -> list[Belief | None]:
        """Async wrapper for BeliefStore.assert_batch.

        Runs in a worker thread (the sync version holds the store lock
        for the entire batch, which would block the event loop).
        """
        return await asyncio.to_thread(self._store.assert_batch, items)

    async def get(self, belief_id: str) -> Belief | None:
        return await asyncio.to_thread(self._store.get, belief_id)

    async def active_beliefs(self) -> list[Belief]:
        return await asyncio.to_thread(self._store.active_beliefs)

    async def all_beliefs(self) -> list[Belief]:
        return await asyncio.to_thread(self._store.all_beliefs)

    async def contradictors_of(self, belief: Belief) -> list[Belief]:
        return await asyncio.to_thread(self._store.contradictors_of, belief)

    async def find_contradictions(self) -> list[tuple[Belief, Belief]]:
        return await asyncio.to_thread(self._store.find_contradictions)

    async def is_consistent(self) -> bool:
        return await asyncio.to_thread(self._store.is_consistent)

    async def dependents_of(self, belief_id: str) -> list[Belief]:
        return await asyncio.to_thread(self._store.dependents_of, belief_id)

    async def entrenchment_of(self, belief: Belief) -> float:
        return await asyncio.to_thread(self._store.entrenchment_of, belief)

    async def subscribe(
        self, belief_id: str, callback: Callable[[str, str], None]
    ) -> None:
        # Offload to thread: subscribe acquires the store lock, which
        # may be held by a worker thread running assert_belief.
        await asyncio.to_thread(self._store.subscribe, belief_id, callback)

    async def unsubscribe(
        self, belief_id: str, callback: Callable[[str, str], None]
    ) -> None:
        await asyncio.to_thread(self._store.unsubscribe, belief_id, callback)

    async def add_constraint(
        self, name: str, check: Callable[[list[Belief]], list[str]]
    ) -> None:
        await asyncio.to_thread(self._store.add_constraint, name, check)

    async def remove_constraint(self, name: str) -> None:
        await asyncio.to_thread(self._store.remove_constraint, name)

    async def check_constraints(self) -> dict[str, list[str]]:
        return await asyncio.to_thread(self._store.check_constraints)

    async def stats(self) -> dict:
        return await asyncio.to_thread(self._store.stats)

    def __repr__(self) -> str:
        return f"AsyncBeliefStore({self._store!r})"
