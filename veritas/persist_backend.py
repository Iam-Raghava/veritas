"""Persistence backend abstraction for Veritas.

Currently backed by a single storage engine behind a common interface:
- SQLiteBackend (default): single-file, zero-config. Good for
  single-agent deployments.

(PostgresBackend was removed in v0.11.0: never tested against a real
server, destructive multi-writer behavior.)

Usage:
    from veritas.persist_backend import get_backend

    backend = get_backend("sqlite:///path/to/db.sqlite")

    backend.save(store)
    store2 = backend.load()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from veritas.store import BeliefStore


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


def get_backend(url: str) -> PersistenceBackend:
    """Create a backend from a URL.

    - sqlite:///path/to/file.db -> SQLiteBackend
    - sqlite:///:memory: -> SQLiteBackend (in-memory)

    PostgreSQL was removed in v0.11.0: the backend was never tested
    against a real server and its save() used a destructive DELETE
    unsafe for any multi-writer deployment. SQLite covers persistence.
    """
    if url.startswith("sqlite://"):
        path = url[len("sqlite://"):]
        return SQLiteBackend(path)
    if url.startswith("postgresql://") or url.startswith("postgres://"):
        raise ValueError(
            "PostgreSQL backend was removed in v0.11.0 (untested, "
            "destructive multi-writer behavior). Use sqlite:// URLs."
        )
    raise ValueError(f"Unsupported backend URL: {url}")
