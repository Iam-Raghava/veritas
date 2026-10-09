"""Core belief model for Veritas.

A Belief is the atomic unit of an agent's memory. It carries not just a
proposition but its epistemic pedigree: where it came from, how reliable
that source is, and which other beliefs justify it. This pedigree is what
makes principled retraction possible — without it, the system can only
do "latest wins".
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

ACTIVE = "active"
RETRACTED = "retracted"


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _normalize_justifications(
    justifications: list[str] | list[list[str]] | None,
) -> list[list[str]]:
    """Normalize to Horn clauses (list of AND-sets).

    - None or [] → []
    - ["a", "b"] (flat) → [["a", "b"]] (single AND-set)
    - [["a", "b"], ["c"]] → [["a", "b"], ["c"]] (two derivations)
    Deduplicates within each set, drops empty sets.
    """
    if not justifications:
        return []
    # Detect flat vs nested: flat if all elements are strings.
    if all(isinstance(j, str) for j in justifications):
        flat: list[str] = list(dict.fromkeys(justifications))  # type: ignore[arg-type]
        return [flat] if flat else []
    # Nested: normalize each set.
    result: list[list[str]] = []
    for s in justifications:
        if isinstance(s, str):
            # Mixed: treat lone string as singleton set.
            result.append([s])
        else:
            deduped: list[str] = list(dict.fromkeys(s))
            if deduped:
                result.append(deduped)
    return result


@dataclass
class Belief:
    """A single belief held by an agent.

    Attributes:
        proposition: The claim, as natural language text.
        confidence: The agent's credence in the claim, 0.0-1.0.
        source: Human-readable description of where this came from
            (e.g. "press release", "user statement", "tool observation").
        source_reliability: How trustworthy the source is, 0.0-1.0.
        justifications: Horn clauses — a list of AND-sets, where each
            inner list is a set of belief IDs that jointly warrant this
            belief (all premises needed). Multiple sets represent
            independent derivations (OR across sets). A flat list of
            strings is accepted and treated as a single AND-set.
            Empty means this is a *base* belief standing on its source.
        id: Unique identifier.
        timestamp: Unix time when the belief was formed.
        status: "active" or "retracted".
        metadata: Free-form dict for detectors and integrations
            (e.g. {"contradicts": ["<belief-id>"]} for explicit pairs).
    """

    proposition: str
    confidence: float = 0.8
    source: str = "unknown"
    source_reliability: float = 0.5
    justifications: list[list[str]] = field(default_factory=list)
    # id/timestamp default via __post_init__ (not default_factory) so the
    # generators are looked up dynamically and remain patchable for
    # deterministic testing.
    id: str = field(default="")
    timestamp: float = field(default=0.0)
    status: str = ACTIVE
    metadata: dict = field(default_factory=dict)
    # Ground observations (tool output, user statements, sensor readings)
    # vs derived inferences. An orphaned belief (all justifications lost)
    # survives ONLY if grounded: a derived belief cannot outlive its
    # premises (zombie-belief prevention). See store._retract_single.
    ground: bool = False

    def __post_init__(self) -> None:
        if not self.proposition or not self.proposition.strip():
            raise ValueError("proposition must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not 0.0 <= self.source_reliability <= 1.0:
            raise ValueError("source_reliability must be in [0, 1]")
        if not self.id:
            self.id = _new_id()
        if not self.timestamp:
            self.timestamp = time.time()
        # Normalize justifications to Horn clauses (list of AND-sets).
        # A flat list of strings becomes a single AND-set.
        self.justifications = _normalize_justifications(self.justifications)

    @property
    def is_derived(self) -> bool:
        """True if this belief was inferred from other beliefs."""
        return len(self.justifications) > 0

    @property
    def is_active(self) -> bool:
        return self.status == ACTIVE

    @property
    def all_premise_ids(self) -> list[str]:
        """All belief IDs appearing in any justification set."""
        seen = []
        for s in self.justifications:
            for pid in s:
                if pid not in seen:
                    seen.append(pid)
        return seen

    @property
    def age_days(self) -> float:
        return max(0.0, (time.time() - self.timestamp) / 86400.0)

    def snapshot(self) -> "Belief":
        """Independent copy. Mutating it cannot affect the store.

        The store returns snapshots from all public query methods so
        callers can never corrupt the contradiction index by mutating
        a returned belief.
        """
        return Belief(
            proposition=self.proposition,
            confidence=self.confidence,
            source=self.source,
            source_reliability=self.source_reliability,
            justifications=[list(s) for s in self.justifications],
            id=self.id,
            timestamp=self.timestamp,
            status=self.status,
            metadata=dict(self.metadata),
            ground=self.ground,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "proposition": self.proposition,
            "confidence": self.confidence,
            "source": self.source,
            "source_reliability": self.source_reliability,
            "justifications": [list(s) for s in self.justifications],
            "timestamp": self.timestamp,
            "status": self.status,
            "metadata": dict(self.metadata),
            "ground": self.ground,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Belief":
        return cls(
            proposition=d["proposition"],
            confidence=d.get("confidence", 0.8),
            source=d.get("source", "unknown"),
            source_reliability=d.get("source_reliability", 0.5),
            justifications=d.get("justifications", []),
            id=d.get("id", _new_id()),
            timestamp=d.get("timestamp", time.time()),
            status=d.get("status", ACTIVE),
            metadata=d.get("metadata", {}),
            ground=d.get("ground", False),
        )
