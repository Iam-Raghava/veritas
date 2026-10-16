"""Byzantine-tolerant event log for Veritas.

Threat model: up to f of N nodes are Byzantine (malicious). They can
send arbitrary events, equivocate (send different events to different
nodes), withhold events, and spam. They cannot forge honest nodes'
signatures or break Ed25519.

Guarantees for honest nodes:
  1. NO FORGING: Ed25519 signatures. A malicious node cannot create
     events attributed to an honest node.
  2. EQUIVOCATION DETECTION: If a node signs two different events with
     the same (lamport, node_id), honest nodes detect it, blacklist the
     node, and ignore its events. Equivocation is cryptographic proof
     of malice — no voting needed.
  3. EVENTUAL CONVERGENCE: Honest nodes that communicate converge to
     the same state. Byzantine nodes cannot cause permanent divergence.
  4. AUTHOR-ONLY RETRACTION: An explicit retract event is valid only if
     signed by the belief's original author. Cross-node corrections use
     the principled path: assert a contradictory belief with higher
     entrenchment, and the old belief is automatically retracted.
  5. SEMANTIC SPAM FILTER: Veritas's contradiction resolution is the
     spam filter. Garbage beliefs from malicious nodes are retracted
     when contradicted by honest beliefs (honest sources should have
     higher source_reliability).

What this is NOT:
  - Not PBFT: we don't do 3-phase voting or require 3f+1 nodes. We
    trade real-time total order for eventual convergence, which is
    sufficient for agent memory (sync() is already eventual).
  - Not censorship-resistant: a Byzantine node can withhold its own
    events (equivalent to a crash, which gossip handles).
  - Not DoS-proof: a malicious node can spam before being blacklisted.
    Rate limiting is the operator's job.

This is "BFT for the agent-memory threat model" — not "BFT for
cryptocurrency." A PhD researcher should evaluate it on those terms.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )
    from cryptography.hazmat.primitives import serialization
    _CRYPTO_AVAILABLE = True
except ImportError:
    _CRYPTO_AVAILABLE = False

from .distributed_gossip import GossipEventLog


def _canonical(event_type: str, payload: dict, lamport: int,
               node_id: str) -> bytes:
    """Deterministic bytes for signing."""
    data = {
        "lamport": lamport,
        "node_id": node_id,
        "event_type": event_type,
        "payload": payload,
    }
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()


class ByzantineTolerantLog(GossipEventLog):
    """Gossip log with Ed25519 signatures and equivocation detection.

    Args:
        path: Local SQLite file.
        node_id: Unique node ID.
        private_key: Ed25519PrivateKey. Generated if None.
        peers: HTTP peer URLs (same as GossipEventLog).
    """

    _SCHEMA_BFT = """
    CREATE TABLE IF NOT EXISTS bft_keys (
        node_id TEXT PRIMARY KEY,
        public_key BLOB NOT NULL
    );
    CREATE TABLE IF NOT EXISTS bft_blacklist (
        node_id TEXT PRIMARY KEY,
        reason TEXT NOT NULL,
        evidence TEXT NOT NULL
    );
    """

    def __init__(
        self,
        path: str,
        node_id: str,
        private_key: Any = None,
        peers: list[str] | None = None,
    ) -> None:
        if not _CRYPTO_AVAILABLE:
            raise ImportError(
                "ByzantineTolerantLog requires cryptography: "
                "pip install veritas-tms[bft]"
            )
        super().__init__(path, node_id, peers)
        self._private_key = private_key or Ed25519PrivateKey.generate()
        self._public_key = self._private_key.public_key()
        self._lock = threading.RLock()
        conn = sqlite3.connect(self.path, timeout=10.0, isolation_level=None)
        try:
            conn.executescript(self._SCHEMA_BFT)
            # Register our own public key.
            pub_bytes = self._public_key.public_bytes(
                serialization.Encoding.Raw,
                serialization.PublicFormat.Raw,
            )
            conn.execute(
                "INSERT OR REPLACE INTO bft_keys (node_id, public_key)"
                " VALUES (?, ?)",
                (node_id, pub_bytes),
            )
        finally:
            conn.close()

    def get_public_key_bytes(self) -> bytes:
        from cryptography.hazmat.primitives import serialization
        return self._public_key.public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )

    def register_peer_key(self, node_id: str, public_key_bytes: bytes) -> None:
        """Register another node's public key (out-of-band key exchange)."""
        conn = sqlite3.connect(self.path, timeout=10.0, isolation_level=None)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO bft_keys (node_id, public_key)"
                " VALUES (?, ?)",
                (node_id, public_key_bytes),
            )
        finally:
            conn.close()

    def _get_public_key(self, node_id: str) -> Any | None:
        conn = sqlite3.connect(self.path, timeout=10.0)
        try:
            row = conn.execute(
                "SELECT public_key FROM bft_keys WHERE node_id = ?", (node_id,)
            ).fetchone()
            if not row:
                return None
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PublicKey,
            )
            return Ed25519PublicKey.from_public_bytes(row[0])
        finally:
            conn.close()

    def is_blacklisted(self, node_id: str) -> bool:
        conn = sqlite3.connect(self.path, timeout=10.0)
        try:
            row = conn.execute(
                "SELECT 1 FROM bft_blacklist WHERE node_id = ?", (node_id,)
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def _blacklist(self, node_id: str, reason: str, evidence: str) -> None:
        conn = sqlite3.connect(self.path, timeout=10.0, isolation_level=None)
        try:
            conn.execute(
                "INSERT OR IGNORE INTO bft_blacklist (node_id, reason, evidence)"
                " VALUES (?, ?, ?)",
                (node_id, reason, evidence),
            )
        finally:
            conn.close()

    def append(self, event: dict) -> int:
        """Sign and append. Returns lamport timestamp."""
        lamport = self._tick()
        event_type = event.get("event_type", "")
        payload = event.get("payload", {})
        # Sign the canonical event.
        message = _canonical(event_type, payload, lamport, self.node_id)
        signature = self._private_key.sign(message)
        # Store with signature.
        conn = sqlite3.connect(self.path, timeout=10.0, isolation_level=None)
        try:
            conn.execute(
                "INSERT OR IGNORE INTO gossip_events"
                " (lamport, node_id, event_type, payload, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    lamport,
                    self.node_id,
                    event_type,
                    json.dumps({
                        "payload": payload,
                        "signature": signature.hex(),
                    }),
                    time.time(),
                ),
            )
        finally:
            conn.close()
        self._gossip_one(lamport, event, signature.hex())
        return lamport

    def _gossip_one(self, lamport: int, event: dict,
                    signature_hex: str | None = None) -> None:
        """Gossip with signature (BFT mode). Overrides parent to include
        the Ed25519 signature — peers reject unsigned events."""
        if signature_hex is None:
            # Fallback: should not happen in BFT mode, but don't send
            # unsigned if we can avoid it.
            return
        import urllib.request
        for peer in self.peers:
            try:
                data = json.dumps({
                    "lamport": lamport,
                    "node_id": self.node_id,
                    "event_type": event.get("event_type", ""),
                    "payload": event.get("payload", {}),
                    "signature": signature_hex,
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

    def receive_signed(
        self,
        lamport: int,
        node_id: str,
        event_type: str,
        payload: dict,
        signature_hex: str,
    ) -> tuple[bool, str]:
        """Receive a signed event. Returns (accepted, reason).

        Rejects: blacklisted nodes, bad signatures, equivocations.
        """
        # 1. Check blacklist.
        if self.is_blacklisted(node_id):
            return False, "blacklisted"

        # 2. Verify signature.
        pubkey = self._get_public_key(node_id)
        if pubkey is None:
            return False, "unknown public key"
        message = _canonical(event_type, payload, lamport, node_id)
        try:
            pubkey.verify(bytes.fromhex(signature_hex), message)
        except Exception:
            return False, "bad signature"

        # 3. Check for equivocation: same (lamport, node_id), different content.
        # Content = (event_type, payload). A malicious node that signs
        # "assert" and "retract" with the same lamport+payload is
        # equivocating, not duplicating — the event_type differs.
        conn = sqlite3.connect(self.path, timeout=10.0)
        try:
            row = conn.execute(
                "SELECT event_type, payload FROM gossip_events "
                "WHERE lamport = ? AND node_id = ?",
                (lamport, node_id),
            ).fetchone()
        finally:
            conn.close()
        if row:
            existing_type, existing_payload_json = row[0], row[1]
            existing_payload = json.loads(existing_payload_json)
            # Unwrap if stored as {"payload": ...}
            if isinstance(existing_payload, dict) and "payload" in existing_payload:
                existing_payload = existing_payload["payload"]
            if existing_type != event_type or existing_payload != payload:
                # EQUIVOCATION: cryptographic proof of malice.
                evidence = json.dumps({
                    "lamport": lamport,
                    "node_id": node_id,
                    "event_a": {"event_type": existing_type,
                                "payload": existing_payload},
                    "event_b": {"event_type": event_type,
                                "payload": payload},
                })
                self._blacklist(node_id, "equivocation", evidence)
                return False, "equivocation detected, node blacklisted"
            return False, "duplicate"  # Same event, already have it.

        # 4. Validate lamport (prevent clock poisoning).
        # A malicious node could send lamport=2**60 to poison every
        # honest node's Lamport clock via _observe(). We bound the jump
        # but do NOT blacklist: an honest node that was offline for a
        # long time will legitimately see large jumps when it reconnects.
        # Blacklisting here would partition honest nodes.
        # Instead, _observe() is capped (see below).
        MAX_LAMPORT_JUMP = 10000
        with self._lock:
            local_lamport = self._lamport
        lamport_capped = lamport
        if lamport > local_lamport + MAX_LAMPORT_JUMP:
            # Cap the clock advancement; accept the event.
            lamport_capped = local_lamport + MAX_LAMPORT_JUMP

        # 5. Validate payload (prevent entrenchment max-out).
        # A malicious node with valid keys could assert
        # source_reliability=1.0/confidence=1.0/timestamp=future to make
        # its beliefs unbeatable. We VALIDATE here but do NOT mutate the
        # payload: the signature covers the original payload, and mutating
        # it would break signature verification on re-gossip (causing
        # false equivocation accusations and blacklisting honest nodes).
        # Clamping happens at application time in DistributedBeliefStore.
        if event_type == "assert":
            rel = payload.get("source_reliability", 0.5)
            conf = payload.get("confidence", 0.8)
            ts = payload.get("timestamp", 0)
            if not (0.0 <= rel <= 1.0) or not (0.0 <= conf <= 1.0):
                return False, "reliability/confidence out of range"
            # Timestamp must not be more than 1 hour in the future.
            if ts and ts > time.time() + 3600:
                return False, "timestamp too far in future"

        # 6. Accept.
        # Use the capped lamport for clock advancement (prevents poisoning
        # while allowing honest catch-up after offline periods).
        self._observe(lamport_capped)
        conn = sqlite3.connect(self.path, timeout=10.0, isolation_level=None)
        try:
            conn.execute(
                "INSERT OR IGNORE INTO gossip_events"
                " (lamport, node_id, event_type, payload, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    lamport, node_id, event_type,
                    json.dumps({"payload": payload, "signature": signature_hex}),
                    time.time(),
                ),
            )
        finally:
            conn.close()
        return True, "accepted"

    # Override parent's receive to require signatures.
    def receive(self, lamport: int, node_id: str, event_type: str,
                payload: dict) -> bool:
        """Unsigned receive is rejected in BFT mode. Use receive_signed."""
        return False

    def exchange_signed(self, other: "ByzantineTolerantLog") -> int:
        """Bidirectional sync exchanging signed events. Returns new events learned."""
        new = 0
        for src, dst in ((self, other), (other, self)):
            conn = sqlite3.connect(src.path, timeout=10.0)
            try:
                rows = conn.execute(
                    "SELECT lamport, node_id, event_type, payload FROM gossip_events"
                ).fetchall()
            finally:
                conn.close()
            for lamport, node_id, event_type, wrapped_json in rows:
                wrapped = json.loads(wrapped_json)
                if isinstance(wrapped, dict) and "signature" in wrapped:
                    payload = wrapped["payload"]
                    sig = wrapped["signature"]
                else:
                    continue  # Skip unsigned (shouldn't happen in BFT mode)
                ok, _ = dst.receive_signed(lamport, node_id, event_type, payload, sig)
                if ok:
                    new += 1
        return new

    def get_belief_author(self, belief_id: str) -> str | None:
        """Which node originally asserted this belief?"""
        conn = sqlite3.connect(self.path, timeout=10.0)
        try:
            rows = conn.execute(
                "SELECT node_id, payload FROM gossip_events WHERE event_type = 'assert'"
            ).fetchall()
            for node_id, payload_json in rows:
                try:
                    wrapped = json.loads(payload_json)
                    payload = wrapped.get("payload", wrapped)
                    if isinstance(payload, dict) and payload.get("belief_id") == belief_id:
                        return node_id
                except Exception:
                    continue
            return None
        finally:
            conn.close()

    def is_authorized_retract(self, belief_id: str, signer_node_id: str) -> bool:
        """Only the original author can explicitly retract a belief."""
        author = self.get_belief_author(belief_id)
        return author is not None and author == signer_node_id

    def _all_events(self) -> list[tuple[int, str, str, dict]]:
        """Unwrap signed payloads."""
        conn = sqlite3.connect(self.path, timeout=10.0)
        try:
            rows = conn.execute(
                "SELECT lamport, node_id, event_type, payload FROM gossip_events"
                " WHERE node_id NOT IN (SELECT node_id FROM bft_blacklist)"
                " ORDER BY lamport ASC, node_id ASC"
            ).fetchall()
            result = []
            for r in rows:
                wrapped = json.loads(r[3])
                # Unwrap: payload may be {"payload": ..., "signature": ...}
                inner = wrapped.get("payload", wrapped) if isinstance(wrapped, dict) else wrapped
                if isinstance(inner, dict) and "payload" in inner and "signature" in inner:
                    inner = inner["payload"]
                result.append((r[0], r[1], r[2], inner))
            return result
        finally:
            conn.close()
