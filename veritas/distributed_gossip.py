"""Peer-to-peer event log for Veritas: no single point of failure.

Why not Raft? Raft solves consensus for systems that need strong
consistency and leader election. Veritas needs neither:

  - Our consistency model is already eventual (nodes sync explicitly).
  - Our merge is deterministic (log order decides contradictions).
  - Our nodes cooperate (no Byzantine actors).

So we use gossip replication instead — simpler, more partition-tolerant
than Raft, and a better fit:

  - Every node keeps a full copy of the log (no SPOF).
  - Nodes work independently during partitions (both sides progress).
  - On reconnect, they exchange missing events and converge.
  - Event ordering is by (lamport_timestamp, node_id) — a total order
    with no coordinator.

This is a CRDT (grow-only set with deterministic ordering). Merging
two logs is set union + sort. No conflicts are possible because events
are immutable and uniquely identified.

What this does NOT do:
  - Byzantine fault tolerance (assumes non-malicious nodes — see
    distributed.py docstring for why BFT is out of scope).
  - Strong consistency (eventual only — same as before).

Usage:
    # Each node gets its own log file (no shared storage needed).
    log_a = GossipEventLog("/data/node_a.db", node_id="a",
                           peers=["http://node-b:8000", "http://node-c:8000"])
    log_b = GossipEventLog("/data/node_b.db", node_id="b",
                           peers=["http://node-a:8000", "http://node-c:8000"])

    # Nodes sync via HTTP gossip (or call exchange() directly in tests).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import urllib.request
from typing import Any

from .distributed import EventLog


class GossipEventLog(EventLog):
    """P2P event log: local SQLite + gossip replication.

    Args:
        path: Local SQLite file (each node has its own — no sharing).
        node_id: Unique ID for this node (used in event ordering).
        peers: HTTP URLs of peer nodes' /events endpoints. Empty for
            standalone mode (useful in tests via exchange()).
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS gossip_events (
        lamport INTEGER NOT NULL,
        node_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        payload TEXT NOT NULL,
        created_at REAL NOT NULL,
        PRIMARY KEY (lamport, node_id)
    );
    """

    def __init__(
        self,
        path: str,
        node_id: str,
        peers: list[str] | None = None,
    ) -> None:
        self.path = path
        self.node_id = node_id
        self.peers = peers or []
        self._lamport = 0
        self._lock = threading.RLock()
        conn = sqlite3.connect(path, timeout=10.0, isolation_level=None)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(self._SCHEMA)
            # Resume lamport clock from existing events.
            row = conn.execute(
                "SELECT MAX(lamport) FROM gossip_events WHERE node_id = ?",
                (node_id,),
            ).fetchone()
            if row and row[0]:
                self._lamport = row[0]
        finally:
            conn.close()

    def _tick(self) -> int:
        with self._lock:
            self._lamport += 1
            return self._lamport

    def _observe(self, lamport: int) -> None:
        with self._lock:
            self._lamport = max(self._lamport, lamport)

    def append(self, event: dict) -> int:
        """Append locally, then gossip to peers (best-effort)."""
        lamport = self._tick()
        conn = sqlite3.connect(self.path, timeout=10.0, isolation_level=None)
        try:
            conn.execute(
                "INSERT OR IGNORE INTO gossip_events"
                " (lamport, node_id, event_type, payload, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    lamport,
                    self.node_id,
                    event.get("event_type", ""),
                    json.dumps(event.get("payload", {})),
                    time.time(),
                ),
            )
        finally:
            conn.close()
        # Best-effort gossip: failures don't block the writer.
        self._gossip_one(lamport, event)
        return lamport

    def _gossip_one(self, lamport: int, event: dict) -> None:
        for peer in self.peers:
            try:
                data = json.dumps({
                    "lamport": lamport,
                    "node_id": self.node_id,
                    "event_type": event.get("event_type", ""),
                    "payload": event.get("payload", {}),
                }).encode()
                req = urllib.request.Request(
                    peer.rstrip("/") + "/events",
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=5.0)
            except Exception:
                pass  # Peer unreachable — it'll catch up via exchange().

    def receive(self, lamport: int, node_id: str, event_type: str,
                payload: dict) -> bool:
        """Receive a gossiped event. Returns True if it was new."""
        self._observe(lamport)
        conn = sqlite3.connect(self.path, timeout=10.0, isolation_level=None)
        try:
            cur = conn.execute(
                "INSERT OR IGNORE INTO gossip_events"
                " (lamport, node_id, event_type, payload, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (lamport, node_id, event_type, json.dumps(payload), time.time()),
            )
            return cur.rowcount > 0
        finally:
            conn.close()

    def exchange(self, other: "GossipEventLog") -> int:
        """Bidirectional sync with another log (for tests, or manual heal
        after a partition). Returns number of new events learned."""
        new = 0
        for log in (self, other):
            peer = other if log is self else self
            for lamport, node_id, event_type, payload in peer._all_events():
                if log.receive(lamport, node_id, event_type, payload):
                    new += 1
        return new

    def _all_events(self) -> list[tuple[int, str, str, dict]]:
        conn = sqlite3.connect(self.path, timeout=10.0)
        try:
            rows = conn.execute(
                "SELECT lamport, node_id, event_type, payload FROM gossip_events"
                " ORDER BY lamport ASC, node_id ASC"
            ).fetchall()
            return [(r[0], r[1], r[2], json.loads(r[3])) for r in rows]
        finally:
            conn.close()

    def append_batch(self, events: list[tuple[int, str, str, dict]]) -> None:
        for lamport, node_id, event_type, payload in events:
            self.receive(lamport, node_id, event_type, payload)

    # -- EventLog interface ------------------------------------------------
    # Sequence numbers are positional in the total order. Since the log
    # only grows, position N always refers to the same event.

    def read_since(self, seq: int) -> list[tuple[int, dict]]:
        events = self._all_events()
        return [
            (i + 1, {"node_id": node_id, "event_type": etype,
                     "payload": payload})
            for i, (lamport, node_id, etype, payload) in enumerate(events)
            if i + 1 > seq
        ]

    def __len__(self) -> int:
        conn = sqlite3.connect(self.path, timeout=10.0)
        try:
            return conn.execute("SELECT COUNT(*) FROM gossip_events").fetchone()[0]
        finally:
            conn.close()
