"""LangGraph integration for Veritas.

Provides a Veritas-backed store for LangGraph state graphs that
truth-maintains cross-thread state: when new information contradicts
stored state, Veritas retracts the weaker side instead of overwriting.

Usage:
    from veritas.integrations.langgraph import VeritasMemoryStore

    store = VeritasMemoryStore()
    # Use as a LangGraph Store (if langgraph is installed)
    # or directly:
    store.put(("thread_1",), "ceo", {"proposition": "Acme's CEO is Jane"})
    item = store.get(("thread_1",), "ceo")
"""

from __future__ import annotations

from typing import Any

from ..store import BeliefStore


class VeritasMemoryStore:
    """Veritas-backed key-value store for LangGraph.

    Implements the LangGraph Store interface (put/get/search/delete)
    with truth maintenance: putting a value that contradicts an
    existing value triggers entrenchment-ordered retraction rather
    than silent overwrite.

    Namespace is a tuple (e.g., ("thread_1",)); key is a string.
    Values are dicts with at least a "proposition" key.
    """

    def __init__(
        self,
        store: BeliefStore | None = None,
        source_reliability: float = 0.7,
    ) -> None:
        self.store = store or BeliefStore()
        self.source_reliability = source_reliability
        # Map (namespace, key) -> belief_id for lookup.
        self._index: dict[tuple[tuple, str], str] = {}

    def _ns_key(self, namespace: tuple, key: str) -> str:
        return f"{'/'.join(map(str, namespace))}/{key}"

    def put(
        self,
        namespace: tuple,
        key: str,
        value: dict[str, Any],
        *,
        source: str | None = None,
        ground: bool = True,
    ) -> None:
        """Store a value with truth maintenance.

        If the new proposition contradicts an existing belief for the
        same (namespace, key), entrenchment decides the winner.
        """
        proposition = value.get("proposition", str(value))
        ns_key = self._ns_key(namespace, key)
        # Temporal supersession: if a belief exists at this location,
        # retract it (the new value supersedes, not depends on, the old).
        # Do NOT use justifications=[[old_id]] — that's a category error:
        # a state update doesn't derive warrant from the prior value.
        #
        # TRANSACTIONAL: assert the new belief FIRST. Only retract the old
        # if the new one succeeds. If assert_belief returns None (rejected),
        # the old belief remains intact — no data loss.
        old_id = self._index.get((namespace, key))
        belief = self.store.assert_belief(
            proposition,
            source=source or f"langgraph:{ns_key}",
            source_reliability=value.get(
                "reliability", self.source_reliability),
            ground=ground,
            metadata={
                "namespace": list(namespace),
                "key": key,
                "ns_key": ns_key,
                "supersedes": old_id,
                **{k: v for k, v in value.items()
                   if k not in ("proposition", "reliability")},
            },
        )
        if belief is not None:
            # New belief accepted: now retract the old one.
            if old_id:
                old_belief = self.store.get(old_id)
                if old_belief and old_belief.is_active:
                    self.store.retract_belief(
                        old_id,
                        reason=f"superseded by state update at '{ns_key}'",
                    )
            self._index[(namespace, key)] = belief.id
        # If belief is None (rejected), old_id remains in _index and the
        # old belief stays active. No data loss.

    def get(
        self, namespace: tuple, key: str,
    ) -> dict[str, Any] | None:
        """Retrieve the current (active) value."""
        belief_id = self._index.get((namespace, key))
        if belief_id is None:
            return None
        belief = self.store.get(belief_id)
        if belief is None or not belief.is_active:
            return None
        return {
            "proposition": belief.proposition,
            "belief_id": belief.id,
            "confidence": belief.confidence,
            "timestamp": belief.timestamp,
            **belief.metadata,
        }

    def search(
        self, namespace: tuple, query: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search active beliefs in a namespace."""
        results = []
        for (ns, key), bid in self._index.items():
            if ns != namespace:
                continue
            val = self.get(ns, key)
            if val is None:
                continue
            if query and query.lower() not in val["proposition"].lower():
                continue
            results.append(val)
        return results

    def delete(self, namespace: tuple, key: str) -> None:
        """Retract the belief at (namespace, key)."""
        belief_id = self._index.pop((namespace, key), None)
        if belief_id:
            self.store.retract_belief(
                belief_id, reason="deleted from LangGraph store")
