"""Persistence backend abstraction for Veritas.

Supports multiple storage engines behind a common interface:
- SQLiteBackend (default): single-file, zero-config. Good for
  single-agent deployments.
- PostgresBackend: horizontally scalable, for multi-agent enterprise
  deployments. Requires `psycopg` (pip install psycopg[binary]) and a
  PostgreSQL server.

Usage:
    from veritas.persist_backend import get_backend

    backend = get_backend("sqlite:///path/to/db.sqlite")
    # or
    backend = get_backend("postgresql://user:pass@host/db")

    backend.save(store)
    store2 = backend.load()
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from veritas.store import BeliefStore


def _safe_dumps(obj) -> str:
    return json.dumps(obj, default=str)


def _safe_loads(s: str):
    return json.loads(s) if s else []


class PersistenceBackend(ABC):
    """Abstract persistence backend."""

    @abstractmethod
    def save(
        self, store: BeliefStore, detector_name: str = "heuristic",
    ) -> None:
        """Persist the store."""
        ...

    @abstractmethod
    def load(self) -> BeliefStore:
        """Load a store."""
        ...


class SQLiteBackend(PersistenceBackend):
    """SQLite backend (wraps veritas.persist)."""

    def __init__(self, path: str) -> None:
        self.path = path

    def save(
        self, store: BeliefStore, detector_name: str = "heuristic",
    ) -> None:
        from veritas import persist
        persist.save(store, self.path, detector_name=detector_name)

    def load(self) -> BeliefStore:
        from veritas import persist
        return persist.load(self.path)


class PostgresBackend(PersistenceBackend):
    """PostgreSQL backend for single-process persistence.

    WARNING: NOT safe for concurrent writers. The save() method uses
    DELETE WHERE id NOT IN (live_ids) which will DELETE beliefs written
    by other processes. This is a single-writer sync pattern, not a
    multi-writer database. Do NOT deploy with multiple pods/workers
    writing to the same database.

    Veritas is in-process epistemic middleware, not a distributed
    database. Use this backend for single-process durability only.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS veritas_beliefs (
        id TEXT PRIMARY KEY,
        proposition TEXT NOT NULL,
        confidence REAL,
        source TEXT,
        source_reliability REAL,
        justifications TEXT,
        timestamp REAL,
        status TEXT,
        metadata TEXT,
        ground INTEGER,
        valid_from REAL,
        valid_until REAL,
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS veritas_audit (
        seq INTEGER PRIMARY KEY,
        timestamp REAL,
        event_type TEXT,
        belief_id TEXT,
        proposition TEXT,
        reason TEXT,
        details TEXT
    );
    CREATE TABLE IF NOT EXISTS veritas_meta (
        key TEXT PRIMARY KEY,
        value TEXT
    );
    """

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._conn = None

    def _connect(self):
        if self._conn is None:
            try:
                import psycopg
            except ImportError:
                raise ImportError(
                    "PostgreSQL backend requires psycopg: "
                    "pip install 'psycopg[binary]'"
                )
            self._conn = psycopg.connect(self.dsn)
            self._conn.autocommit = False
        return self._conn

    def save(
        self, store: BeliefStore, detector_name: str = "heuristic",
    ) -> None:
        from veritas.store import BeliefStore as _BS  # noqa
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(self.SCHEMA)
            # Hold the store lock during save.
            with store._lock:
                current_ids = set()
                for b in store.all_beliefs():
                    current_ids.add(b.id)
                    cur.execute(
                        """
                        INSERT INTO veritas_beliefs
                        (id, proposition, confidence, source,
                         source_reliability, justifications, timestamp,
                         status, metadata, ground, valid_from, valid_until)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (id) DO UPDATE SET
                            proposition=EXCLUDED.proposition,
                            confidence=EXCLUDED.confidence,
                            source=EXCLUDED.source,
                            source_reliability=EXCLUDED.source_reliability,
                            justifications=EXCLUDED.justifications,
                            timestamp=EXCLUDED.timestamp,
                            status=EXCLUDED.status,
                            metadata=EXCLUDED.metadata,
                            ground=EXCLUDED.ground,
                            valid_from=EXCLUDED.valid_from,
                            valid_until=EXCLUDED.valid_until,
                            updated_at=NOW()
                        """,
                        (
                            b.id, b.proposition, b.confidence, b.source,
                            b.source_reliability,
                            _safe_dumps(b.justifications),
                            b.timestamp, b.status,
                            _safe_dumps(b.metadata),
                            1 if b.ground else 0,
                            b.valid_from, b.valid_until,
                        ),
                    )
                # Remove stale beliefs via temp table (avoids var limit).
                cur.execute(
                    "CREATE TEMP TABLE IF NOT EXISTS live_ids "
                    "(id TEXT PRIMARY KEY) ON COMMIT DROP"
                )
                cur.execute("DELETE FROM live_ids")
                # Chunked inserts.
                ids_list = list(current_ids)
                for i in range(0, len(ids_list), 500):
                    chunk = ids_list[i:i+500]
                    cur.executemany(
                        "INSERT INTO live_ids VALUES (%s)",
                        [(bid,) for bid in chunk],
                    )
                cur.execute(
                    "DELETE FROM veritas_beliefs WHERE id NOT IN "
                    "(SELECT id FROM live_ids)"
                )
                # Audit: append-only, ignore existing seq.
                for e in store.audit:
                    cur.execute(
                        """
                        INSERT INTO veritas_audit
                        (seq, timestamp, event_type, belief_id,
                         proposition, reason, details)
                        VALUES (%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (seq) DO NOTHING
                        """,
                        (
                            e.seq, e.timestamp, e.event_type,
                            e.belief_id, e.proposition, e.reason,
                            _safe_dumps(e.details),
                        ),
                    )
                cur.execute(
                    """
                    INSERT INTO veritas_meta (key, value)
                    VALUES ('config', %s)
                    ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value
                    """,
                    (json.dumps({
                        "detector_name": detector_name,
                        "entrenchment_weights": store.entrenchment_weights,
                        "survival_threshold": store.survival_threshold,
                        "policy": store.policy,
                    }),),
                )
            conn.commit()

    def load(self) -> BeliefStore:
        from veritas.belief import Belief
        from veritas.store import BeliefStore
        conn = self._connect()
        store = BeliefStore()
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM veritas_beliefs")
            cols = [d[0] for d in cur.description]
            for row in cur.fetchall():
                d = dict(zip(cols, row))
                b = Belief(
                    proposition=d["proposition"],
                    confidence=d["confidence"],
                    source=d["source"],
                    source_reliability=d["source_reliability"],
                    justifications=_safe_loads(d["justifications"]),
                    timestamp=d["timestamp"],
                    metadata=_safe_loads(d["metadata"]),
                    ground=bool(d["ground"]),
                    valid_from=d["valid_from"],
                    valid_until=d["valid_until"],
                )
                b.id = d["id"]
                b.status = d["status"]
                store._beliefs[b.id] = b
                if b.is_active:
                    store._index(b)
            # Load audit.
            cur.execute(
                "SELECT seq, timestamp, event_type, belief_id, "
                "proposition, reason, details FROM veritas_audit "
                "ORDER BY seq"
            )
            from veritas.audit import AuditEvent
            for row in cur.fetchall():
                seq, ts, et, bid, prop, reason, details = row
                store._audit._events.append(AuditEvent(
                    seq=seq, timestamp=ts, event_type=et,
                    belief_id=bid, proposition=prop, reason=reason,
                    details=_safe_loads(details),
                ))
            # Load config.
            cur.execute(
                "SELECT value FROM veritas_meta WHERE key='config'")
            row = cur.fetchone()
            if row:
                cfg = json.loads(row[0])
                weights = cfg.get("entrenchment_weights")
                if weights and store.entrenchment_weights is not None:
                    store.entrenchment_weights.update(weights)
                if "policy" in cfg:
                    store.policy = cfg["policy"]
        return store

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def get_backend(url: str) -> PersistenceBackend:
    """Create a backend from a URL.

    - sqlite:///path/to/file.db -> SQLiteBackend
    - sqlite:///:memory: -> SQLiteBackend (in-memory)
    - postgresql://... -> PostgresBackend
    """
    if url.startswith("sqlite://"):
        path = url[len("sqlite://"):]
        return SQLiteBackend(path)
    if url.startswith("postgresql://") or url.startswith("postgres://"):
        return PostgresBackend(url)
    raise ValueError(f"Unsupported backend URL: {url}")
