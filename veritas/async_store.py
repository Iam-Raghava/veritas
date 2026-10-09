"""Async wrapper for BeliefStore.

Modern agent frameworks (LangChain, CrewAI, AutoGen) are asyncio-based.
AsyncBeliefStore exposes the same API as BeliefStore but with async methods,
running the thread-safe sync implementation in a thread pool via
asyncio.to_thread. This is the standard pattern for wrapping sync I/O-bound
or CPU-bound libraries (cf. asyncpg, httpx sync/async duality).

Usage:
    store = AsyncBeliefStore()
    await store.assert_belief("Sky is blue", source="sensor")
    beliefs = await store.active_beliefs()
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from .belief import Belief
from .detect import Detector
from .store import BeliefStore


class AsyncBeliefStore:
    """Async interface to a thread-safe BeliefStore."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._store = BeliefStore(*args, **kwargs)

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
        # Subscription registry is sync; no I/O to offload.
        self._store.subscribe(belief_id, callback)

    async def unsubscribe(
        self, belief_id: str, callback: Callable[[str, str], None]
    ) -> None:
        self._store.unsubscribe(belief_id, callback)

    async def add_constraint(
        self, name: str, check: Callable[[list[Belief]], list[str]]
    ) -> None:
        self._store.add_constraint(name, check)

    async def remove_constraint(self, name: str) -> None:
        self._store.remove_constraint(name)

    async def check_constraints(self) -> dict[str, list[str]]:
        return await asyncio.to_thread(self._store.check_constraints)

    def __repr__(self) -> str:
        return f"AsyncBeliefStore({self._store!r})"
