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

        Reconstructs SOLELY from immutable audit event payloads — never
        touches the live store's mutated objects. Each ASSERTED event
        carries a belief_snapshot in details; we replay assertions and
        retractions in order.

        Events are sorted by timestamp before replay (not assumed sorted):
        historical backfills may interleave old assertion timestamps with
        new retraction timestamps.

        Returns snapshots: mutating them cannot affect the store.
        """
        from veritas.belief import Belief
        # Sort by (timestamp, seq) — seq breaks ties for same-timestamp
        # events (e.g., assert then retract in the same millisecond).
        events = sorted(self.store.audit,
                        key=lambda e: (e.timestamp, e.seq))
        active: dict[str, Belief] = {}
        for event in events:
            if event.timestamp > timestamp:
                break
            if event.event_type == "asserted":
                snap = event.details.get("belief_snapshot")
                if snap:
                    # Reconstruct from immutable snapshot.
                    b = Belief.from_dict(snap)
                    active[event.belief_id] = b
                # else: old event without snapshot; skip (cannot reconstruct)
            elif event.event_type in ("retracted", "cascade_retracted",
                                      "rejected"):
                active.pop(event.belief_id, None)
            elif event.event_type == "rederived":
                # Re-derivation: belief came back IN.
                snap = event.details.get("belief_snapshot")
                if snap:
                    b = Belief.from_dict(snap)
                    active[event.belief_id] = b
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
