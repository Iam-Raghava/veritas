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
