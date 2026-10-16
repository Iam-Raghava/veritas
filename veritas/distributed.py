"""Distributed Veritas: shared truth maintenance across machines.

The problem: multiple agents on different machines need to share one
consistent belief state. Network partitions, concurrent writes, and
cross-node justifications make this genuinely hard.

The solution: event sourcing. Veritas's state is fully determined by
the ordered sequence of assert/retract events (this is what TimeTravel
proves — replay the audit log, get the same state). If all nodes
replay the same ordered event log, they converge to identical state.

Architecture:
    EventLog (abstract): ordered, append-only, shared by all nodes.
    SQLiteEventLog: the log is a SQLite table. Works across processes
        on one machine, and across machines via a network filesystem.
        Zero new dependencies. Single point of failure (the file).
    GossipEventLog (distributed_gossip.py): P2P replication, no SPOF,
        partition-tolerant. Each node has its own SQLite file; events
        replicate via gossip with Lamport timestamps.
    ByzantineTolerantLog (distributed_bft.py): Gossip + Ed25519
        signatures, equivocation detection, author-only retraction.
    DistributedBeliefStore: wraps BeliefStore. Writes append to the
        log; reads come from the local store (instant). sync() pulls
        new events and replays them through the deterministic engine.

Consistency model: eventual. A node is at most one sync() behind the
log. All nodes that have replayed the same log prefix have identical
state — guaranteed by determinism, not by luck.

What this does NOT do:
    - Strong consistency (there is a sync lag; use sync() before
      reads that must be fresh).
    - Real-time total order across partitions (eventual convergence;
      see GossipEventLog for the partition story).
    - Full PBFT (see ByzantineTolerantLog for the honest scope).

Usage:
    log = SQLiteEventLog("/shared/veritas_log.db")
    node_a = DistributedBeliefStore("agent-1", log)
    node_b = DistributedBeliefStore("agent-2", log)  # another machine

    node_a.assert_belief("Acme's CEO is Alice", source_reliability=0.9)
    node_b.sync()  # now node_b sees Alice too
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from abc import ABC, abstractmethod

from .belief import Belief, fixed_now
from .store import BeliefStore


# ---------------------------------------------------------------------------
# Event log abstraction
# ---------------------------------------------------------------------------

class EventLog(ABC):
    """Ordered, append-only, shared event log."""

    @abstractmethod
    def append(self, event: dict) -> int:
        """Append an event. Returns its sequence number."""
        ...

    @abstractmethod
    def read_since(self, seq: int) -> list[tuple[int, dict]]:
        """All (seq, event) with seq > given. Ordered by seq."""
        ...

    @property
    def needs_rebuild_on_sync(self) -> bool:
        """True if positional seq numbers are not stable (gossip logs).

        Gossip logs order by (lamport, node_id); new events can sort
        before old ones, so incremental replay by position is wrong.
        Rebuild from scratch instead (deterministic => same result).
        """
        return False


class SQLiteEventLog(EventLog):
    """Event log backed by SQLite.

    The database is the ordering authority: AUTOINCREMENT gives every
    event a total order, even with concurrent writers. WAL mode +
    busy timeout handle contention.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS veritas_events (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        node_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        payload TEXT NOT NULL,
        created_at REAL NOT NULL
    );
    """

    def __init__(self, path: str) -> None:
        self.path = path
        conn = sqlite3.connect(path, timeout=10.0, isolation_level=None)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=10000;")
            conn.execute(self._SCHEMA)
        finally:
            conn.close()

    def append(self, event: dict) -> int:
        conn = sqlite3.connect(self.path, timeout=10.0, isolation_level=None)
        try:
            conn.execute("PRAGMA busy_timeout=10000;")
            cur = conn.execute(
                "INSERT INTO veritas_events (node_id, event_type, payload, created_at)"
                " VALUES (?, ?, ?, ?)",
                (
                    event.get("node_id", ""),
                    event.get("event_type", ""),
                    json.dumps(event.get("payload", {})),
                    time.time(),
                ),
            )
            assert cur.lastrowid is not None
            return cur.lastrowid
        finally:
            conn.close()

    def read_since(self, seq: int) -> list[tuple[int, dict]]:
        conn = sqlite3.connect(self.path, timeout=10.0)
        try:
            rows = conn.execute(
                "SELECT seq, node_id, event_type, payload FROM veritas_events"
                " WHERE seq > ? ORDER BY seq ASC",
                (seq,),
            ).fetchall()
            return [
                (r[0], {"node_id": r[1], "event_type": r[2],
                        "payload": json.loads(r[3])})
                for r in rows
            ]
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Distributed store
# ---------------------------------------------------------------------------

class DistributedBeliefStore:
    """A BeliefStore shared across machines via an event log.

    Each node keeps a local BeliefStore for instant reads. Writes go
    to the shared log first, then are replayed locally. sync() pulls
    other nodes' events and replays them.

    Belief IDs are assigned by the originating node and carried in the
    event, so cross-node justifications resolve to the same belief
    everywhere.
    """

    def __init__(self, node_id: str | None, log: EventLog) -> None:
        self.node_id = node_id or f"node-{uuid.uuid4().hex[:8]}"
        self.log = log
        self.store = BeliefStore()
        self._seq = 0
        self._lock = threading.RLock()
        # Catch up on existing history on join.
        self.sync()

    # -- writes ----------------------------------------------------------

    def assert_belief(
        self,
        proposition: str,
        confidence: float = 0.8,
        source: str = "unknown",
        source_reliability: float = 0.5,
        justifications: list[str] | list[list[str]] | None = None,
        metadata: dict | None = None,
        ground: bool = False,
    ) -> Belief | None:
        """Assert a belief cluster-wide.

        The event is appended to the shared log, then this node syncs
        (replaying its own event plus any concurrent events from other
        nodes, in log order). Returns the belief, or None if it was
        rejected during replay.
        """
        belief_id = uuid.uuid4().hex
        event = {
            "node_id": self.node_id,
            "event_type": "assert",
            "payload": {
                "belief_id": belief_id,
                "proposition": proposition,
                "confidence": confidence,
                "source": source,
                "source_reliability": source_reliability,
                "justifications": justifications,
                "metadata": metadata,
                "ground": ground,
                "timestamp": time.time(),
            },
        }
        with self._lock:
            self.log.append(event)
            self.sync()
            return self.store.get(belief_id)

    def retract_belief(self, belief_id: str, reason: str = "manual retraction") -> None:
        """Retract a belief cluster-wide."""
        event = {
            "node_id": self.node_id,
            "event_type": "retract",
            "payload": {"belief_id": belief_id, "reason": reason},
        }
        with self._lock:
            self.log.append(event)
            self.sync()

    # -- sync ------------------------------------------------------------

    def sync(self) -> int:
        """Replay all new log events. Returns number of events applied."""
        with self._lock:
            # Gossip logs have non-stable positional ordering, so rebuild
            # from scratch to guarantee convergence. Deterministic replay
            # => same state.
            if self.log.needs_rebuild_on_sync:
                return self._rebuild()
            events = self.log.read_since(self._seq)
            # Deterministic clock: score using max event timestamp, not
            # wall-clock. Identical logs => identical entrenchment =>
            # identical winners, regardless of when the replay runs.
            replay_now = self._replay_now(events)
            applied = 0
            with fixed_now(replay_now):
                for seq, event in events:
                    try:
                        self._apply(event)
                    except Exception:
                        # Malformed event (e.g., invalid confidence):
                        # skip it, don't poison the sync cursor.
                        # (BFT mode validates signatures; this is for
                        # crash-fault gossip mode.)
                        pass
                    self._seq = seq
                    applied += 1
            return applied

    def _replay_now(self, events: list) -> float:
        """Deterministic 'now' for replay: max event timestamp."""
        max_ts = 0.0
        for _, event in events:
            ts = event.get("payload", {}).get("timestamp", 0) or 0
            if ts > max_ts:
                max_ts = ts
        return max_ts if max_ts > 0 else time.time()

    def _rebuild(self) -> int:
        """Clear local state and replay the entire log. Used for gossip
        logs where positional sequence numbers are not stable."""
        self.store = BeliefStore()
        events = self.log.read_since(0)
        replay_now = self._replay_now(events)
        with fixed_now(replay_now):
            for seq, event in events:
                try:
                    self._apply(event)
                except Exception:
                    # Malformed event: skip, don't break rebuild.
                    pass
        self._seq = len(events)
        return len(events)

    def _apply(self, event: dict) -> None:
        etype = event["event_type"]
        payload = event["payload"]
        if etype == "assert":
            # _belief_id ensures every node assigns the same ID.
            self.store.assert_belief(
                proposition=payload["proposition"],
                confidence=payload.get("confidence", 0.8),
                source=payload.get("source", "unknown"),
                source_reliability=payload.get("source_reliability", 0.5),
                justifications=payload.get("justifications"),
                metadata=payload.get("metadata"),
                timestamp=payload.get("timestamp"),
                ground=payload.get("ground", False),
                _belief_id=payload["belief_id"],
            )
        elif etype == "retract":
            # BFT: only the original author can explicitly retract.
            # (Cross-node corrections use contradiction, not fiat.)
            get_author = getattr(self.log, "get_belief_author", None)
            if get_author is not None:
                author = get_author(payload["belief_id"])
                signer = event.get("node_id", "")
                if author is None or author != signer:
                    return  # Unauthorized retract: ignore.
            self.store.retract_belief(
                payload["belief_id"],
                reason=payload.get("reason", "manual retraction"),
            )

    # -- reads (local, instant) ------------------------------------------

    def get(self, belief_id: str) -> Belief | None:
        return self.store.get(belief_id)

    def active_beliefs(self) -> list[Belief]:
        return self.store.active_beliefs()

    def is_consistent(self) -> bool:
        return self.store.is_consistent()

    @property
    def audit(self):
        return self.store.audit

    def state_hash(self) -> str:
        """Hash of active belief IDs + statuses. Two nodes with the
        same hash have converged to identical state."""
        import hashlib
        parts = sorted(
            f"{b.id}:{b.status}:{b.proposition}"
            for b in self.store.all_beliefs()
        )
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]
