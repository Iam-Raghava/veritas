"""Storage backend protocol for Veritas.

Decouples BeliefStore from specific storage engines. The store uses
in-memory dictionaries as its working set for speed; the backend
handles durability and cross-process visibility.

Protocol:
    class StorageBackend(Protocol):
        def get_belief(self, belief_id: str) -> Belief | None: ...
        def save_belief(self, belief: Belief) -> None: ...
        def append_audit(self, event: AuditEvent) -> None: ...
        def query_active(self) -> list[Belief]: ...

Implementations:
- InMemoryBackend: default, wraps the store's own dictionaries.
- SQLiteBackend: persists to SQLite (WAL mode, multi-process safe).
"""

from __future__ import annotations

from typing import Any, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from veritas.audit import AuditEvent
    from veritas.belief import Belief


class StorageBackend(Protocol):
    """Abstract storage backend for belief persistence."""

    def get_belief(self, belief_id: str) -> Belief | None:
        """Retrieve a belief by ID, or None if not found."""
        ...

    def save_belief(self, belief: Belief) -> None:
        """Persist a belief (insert or update)."""
        ...

    def append_audit(self, event: AuditEvent) -> None:
        """Append an audit event (append-only)."""
        ...

    def query_active(self) -> list[Belief]:
        """Return all active beliefs."""
        ...


class InMemoryBackend:
    """Default backend: delegates to the store's in-memory dictionaries.

    This is a pass-through — the store remains the source of truth.
    Useful for testing and single-process deployments.
    """

    def __init__(self, store: Any = None) -> None:
        self._store = store

    def attach(self, store: Any) -> None:
        self._store = store

    def get_belief(self, belief_id: str) -> Belief | None:
        if self._store is None:
            return None
        b = self._store._beliefs.get(belief_id)
        return b  # type: ignore[no-any-return]

    def save_belief(self, belief: Belief) -> None:
        # In-memory: store already holds it. No-op for durability.
        # (Subclasses override for real persistence.)
        pass

    def append_audit(self, event: AuditEvent) -> None:
        # In-memory: audit log already holds it.
        pass

    def query_active(self) -> list[Belief]:
        if self._store is None:
            return []
        return [b for b in self._store._beliefs.values() if b.is_active]


class SQLiteStorageBackend(InMemoryBackend):
    """SQLite-backed storage with WAL mode for multi-process safety.

    Extends InMemoryBackend: the store remains the working set, but
    every save_belief and append_audit also writes through to SQLite.
    """

    def __init__(self, path: str, store=None) -> None:
        super().__init__(store)
        self.path = path
        self._init_db()

    def _init_db(self) -> None:
        import sqlite3
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS backend_beliefs (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS backend_audit (
                seq INTEGER PRIMARY KEY,
                data TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def save_belief(self, belief: Belief) -> None:
        import sqlite3, json
        conn = sqlite3.connect(self.path, timeout=5.0)
        try:
            conn.execute("PRAGMA busy_timeout=5000;")
            conn.execute(
                "INSERT OR REPLACE INTO backend_beliefs (id, data) "
                "VALUES (?, ?)",
                (belief.id, json.dumps(belief.to_dict(), default=str)),
            )
            conn.commit()
        finally:
            conn.close()

    def append_audit(self, event: AuditEvent) -> None:
        import sqlite3, json
        conn = sqlite3.connect(self.path, timeout=5.0)
        try:
            conn.execute("PRAGMA busy_timeout=5000;")
            conn.execute(
                "INSERT OR IGNORE INTO backend_audit (seq, data) "
                "VALUES (?, ?)",
                (event.seq, json.dumps({
                    "seq": event.seq,
                    "timestamp": event.timestamp,
                    "event_type": event.event_type,
                    "belief_id": event.belief_id,
                    "proposition": event.proposition,
                    "reason": event.reason,
                    "details": event.details,
                }, default=str)),
            )
            conn.commit()
        finally:
            conn.close()
