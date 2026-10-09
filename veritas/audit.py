"""Append-only audit log. Every belief change is recorded with its reason.

The audit trail is what makes Veritas's retractions *accountable*: not just
"belief X is gone" but "belief X was retracted because new evidence Y
(press release, reliability 0.97) outranked it (article, reliability 0.85),
and beliefs Z, W were cascade-retracted for losing justification."
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

ASSERTED = "asserted"
RETRACTED = "retracted"
CASCADE_RETRACTED = "cascade_retracted"
REJECTED = "rejected"


@dataclass
class AuditEvent:
    seq: int
    timestamp: float
    event_type: str
    belief_id: str
    proposition: str
    reason: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "belief_id": self.belief_id,
            "proposition": self.proposition,
            "reason": self.reason,
            "details": dict(self.details),
        }


class AuditLog:
    """Simple append-only log. Kept in memory; serialize via to_dicts()."""

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    def record(
        self,
        event_type: str,
        belief_id: str,
        proposition: str,
        reason: str,
        details: dict | None = None,
        *,
        seq: int | None = None,
        timestamp: float | None = None,
    ) -> AuditEvent:
        """Append an event. seq/timestamp overrides are for persistence
        restore only — normal callers must not pass them."""
        event = AuditEvent(
            seq=len(self._events) if seq is None else seq,
            timestamp=time.time() if timestamp is None else timestamp,
            event_type=event_type,
            belief_id=belief_id,
            proposition=proposition,
            reason=reason,
            details=dict(details or {}),
        )
        self._events.append(event)
        return event

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self):
        return iter(self._events)

    def events_for(self, belief_id: str) -> list[AuditEvent]:
        return [e for e in self._events if e.belief_id == belief_id]

    def of_type(self, event_type: str) -> list[AuditEvent]:
        return [e for e in self._events if e.event_type == event_type]

    def to_dicts(self) -> list[dict]:
        return [e.to_dict() for e in self._events]
