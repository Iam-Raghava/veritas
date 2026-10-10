"""Time-travel queries for Veritas.

The audit log records every state change. This module reconstructs
the belief state at any point in time by replaying the log.

Use cases:
- "What did the agent believe at 3pm yesterday?"
- Debugging: trace when a belief was retracted
- Compliance: prove what was known when a decision was made
"""

from __future__ import annotations

from veritas.belief import Belief
from veritas.store import BeliefStore


class TimeTravel:
    """Query historical belief states."""

    def __init__(self, store: BeliefStore) -> None:
        self.store = store

    def beliefs_at(self, timestamp: float) -> list[Belief]:
        """Reconstruct active beliefs as of timestamp.

        Returns snapshots: mutating them cannot affect the store.
        """
        # Replay audit log up to timestamp
        active: dict[str, Belief] = {}
        for event in self.store.audit:
            if event.timestamp > timestamp:
                break
            if event.event_type == "asserted":
                # Reconstruct from event details if available
                # (simplified: use current belief if it existed then)
                b = self.store.get(event.belief_id)
                if b and b.timestamp <= timestamp:
                    active[event.belief_id] = b
            elif event.event_type in ("retracted", "cascade_retracted",
                                      "rejected"):
                active.pop(event.belief_id, None)
        return list(active.values())

    def when_retracted(self, belief_id: str) -> float | None:
        """Timestamp when a belief was retracted, if ever."""
        for event in self.store.audit:
            if (event.belief_id == belief_id
                    and event.event_type in ("retracted",
                                             "cascade_retracted")):
                return event.timestamp
        return None

    def history_of(self, proposition: str) -> list:
        """All events for a proposition, in order."""
        return [
            e for e in self.store.audit
            if e.proposition == proposition
        ]
