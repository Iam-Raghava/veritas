"""LangGraph integration for Veritas.

Provides a Veritas-backed store for LangGraph state graphs that
truth-maintains cross-thread state: when new information contradicts
stored state, Veritas retracts the weaker side instead of overwriting.

When ``langgraph`` is installed, ``VeritasMemoryStore`` subclasses
``langgraph.store.base.BaseStore`` (with put/get/search/delete/batch)
and can be passed directly as ``store=`` to ``StateGraph.compile()``.
Without langgraph, it falls back to a standalone helper with the same
API (minus batch).

Usage:
    from veritas.integrations.langgraph import VeritasMemoryStore

    store = VeritasMemoryStore()
    store.put(("thread_1",), "ceo", {"proposition": "Acme's CEO is Jane"})
    item = store.get(("thread_1",), "ceo")
"""

from __future__ import annotations

from typing import Any

from ..store import BeliefStore

# Try to import the real BaseStore for type checking only.
# We do NOT subclass it: BaseStore is an ABC with abstract methods (e.g.,
# abatch); subclassing without implementing them crashes on instantiation
# when langgraph is installed. VeritasMemoryStore is interface-compatible
# (implements the methods LangGraph actually calls) but standalone.
try:
    from langgraph.store.base import BaseStore as _LGBaseStore
    _HAS_LANGGRAPH = True
except ImportError:
    _LGBaseStore = object  # type: ignore[assignment,misc]
    _HAS_LANGGRAPH = False


class VeritasMemoryStore:  # type: ignore[misc]
    """Veritas-backed key-value store for LangGraph.

    Truth-maintains cross-thread state: putting a value that contradicts
    an existing value triggers entrenchment-ordered retraction rather
    than silent overwrite.

    Namespace is a tuple (e.g., ("thread_1",)); key is a string.
    Values are dicts with at least a "proposition" key.

    NOTE: Not a subclass of langgraph's BaseStore (see above).
    Interface-compatible for the methods LangGraph calls.
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
            # New belief accepted: retract the old one ONLY if the new
            # outranks it. A weaker update must not destroy a stronger
            # incumbent (no latest-wins).
            should_index = True
            if old_id:
                old_belief = self.store.get(old_id)
                if old_belief and old_belief.is_active:
                    new_ent = self.store.entrenchment_of(belief)
                    old_ent = self.store.entrenchment_of(old_belief)
                    if new_ent >= old_ent:
                        self.store.retract_belief(
                            old_id,
                            reason=f"superseded by state update at '{ns_key}'",
                        )
                    else:
                        # Old stays active and remains the indexed belief.
                        # Do NOT overwrite _index with the weaker belief.
                        should_index = False
                    # Else: old stays active; new coexists (or was already
                    # handled by contradiction logic).
            if should_index:
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

    def batch(self, ops: list[tuple]) -> list[Any]:
        """Execute a batch of (op, namespace, key, value) operations.

        Required by langgraph.store.base.BaseStore. Each op is one of
        "put", "get", "search", "delete".
        """
        results: list[Any] = []
        for op in ops:
            kind = op[0]
            if kind == "put":
                _, ns, key, value = op
                self.put(ns, key, value)
                results.append(None)
            elif kind == "get":
                _, ns, key = op
                results.append(self.get(ns, key))
            elif kind == "search":
                _, ns = op[:2]
                query = op[2] if len(op) > 2 else ""
                results.append(self.search(ns, query))
            elif kind == "delete":
                _, ns, key = op
                self.delete(ns, key)
                results.append(None)
            else:
                raise ValueError(f"unknown batch op: {kind}")
        return results
