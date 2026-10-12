"""SQLite persistence for Veritas belief stores.

A store is fully serializable: beliefs, audit log, and configuration.
`save(store, path)` writes everything; `load(path)` rebuilds an identical
store. The contradiction detector is referenced by name — built-ins
resolve automatically; custom detectors must be re-registered on load
(see `register_detector`).
"""
from __future__ import annotations

import json
import sqlite3
from typing import Callable

from .audit import AuditLog
from .belief import Belief
from .detect import ContradictionFn, heuristic_contradiction
from .store import BeliefStore

# Increment when the on-disk format changes. load() refuses newer
# versions it cannot understand, rather than misreading them.
# v3: justifications are Horn clauses (list of AND-sets). v2 DBs with
# flat justifications are auto-migrated by Belief.__post_init__.
SCHEMA_VERSION = 4


def _safe_dumps(obj) -> str:
    """JSON-encode, falling back to repr() for unserializable values.

    save() must never crash on data the store accepted in memory.
    Unserializable values round-trip as their repr() string.
    """
    return json.dumps(obj, default=repr)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS beliefs (
    id TEXT PRIMARY KEY,
    proposition TEXT NOT NULL,
    confidence REAL NOT NULL,
    source TEXT NOT NULL,
    source_reliability REAL NOT NULL,
    justifications TEXT NOT NULL,
    timestamp REAL NOT NULL,
    status TEXT NOT NULL,
    metadata TEXT NOT NULL,
    ground INTEGER NOT NULL DEFAULT 0,
    valid_from REAL,
    valid_until REAL
);
CREATE TABLE IF NOT EXISTS audit (
    seq INTEGER PRIMARY KEY,
    timestamp REAL NOT NULL,
    event_type TEXT NOT NULL,
    belief_id TEXT NOT NULL,
    proposition TEXT NOT NULL,
    reason TEXT NOT NULL,
    details TEXT NOT NULL
);
"""

# Built-in detectors resolvable by name on load.
_BUILTIN_DETECTORS: dict[str, ContradictionFn] = {
    "heuristic": heuristic_contradiction,
}

_custom_detectors: dict[str, ContradictionFn] = {}


def register_detector(name: str, fn: ContradictionFn) -> None:
    """Register a named detector so persisted stores can resolve it on load."""
    _custom_detectors[name] = fn


def _resolve_detector(name: str) -> ContradictionFn:
    if name in _custom_detectors:
        return _custom_detectors[name]
    if name in _BUILTIN_DETECTORS:
        return _BUILTIN_DETECTORS[name]
    raise ValueError(
        f"unknown detector '{name}'. Register it with register_detector() "
        f"before loading."
    )


def save(store: BeliefStore, path: str, detector_name: str = "heuristic") -> None:
    """Persist a belief store to a SQLite file (created if missing).

    Uses incremental updates (INSERT OR REPLACE) instead of full table
    wipes, and holds the store lock during the entire save to prevent
    concurrent mutations from corrupting the iteration.
    """
    # Hold the store lock for the entire save to prevent the
    # "list changed size during iteration" race.
    with store._lock:
        # Use a dedicated connection with WAL mode for multi-process safety.
        # WAL allows concurrent readers; busy_timeout handles writer contention.
        conn = sqlite3.connect(path, timeout=5.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        try:
            conn.executescript(_SCHEMA)
            # Incremental: upsert beliefs (no full DELETE).
            current_ids = set()
            for b in store.all_beliefs():
                current_ids.add(b.id)
                conn.execute(
                    "INSERT OR REPLACE INTO beliefs VALUES "
                    "(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        b.id,
                        b.proposition,
                        b.confidence,
                        b.source,
                        b.source_reliability,
                        _safe_dumps(b.justifications),
                        b.timestamp,
                        b.status,
                        _safe_dumps(b.metadata),
                        1 if b.ground else 0,
                        b.valid_from,
                        b.valid_until,
                    ),
                )
            # Remove beliefs that are no longer in the store (e.g., if the
            # store was cleared or beliefs were purged).
            # Chunked to avoid SQLITE_MAX_VARIABLE_NUMBER limit (999 on old
            # SQLite, 32766 on 3.32+). 500 per chunk is safe everywhere.
            if current_ids:
                ids_list = list(current_ids)
                # Use a temp table to avoid huge IN clauses.
                conn.execute("CREATE TEMP TABLE IF NOT EXISTS live_ids "
                             "(id TEXT PRIMARY KEY)")
                conn.execute("DELETE FROM live_ids")
                for i in range(0, len(ids_list), 500):
                    chunk = ids_list[i:i+500]
                    conn.executemany(
                        "INSERT OR IGNORE INTO live_ids VALUES (?)",
                        [(bid,) for bid in chunk],
                    )
                conn.execute(
                    "DELETE FROM beliefs WHERE id NOT IN "
                    "(SELECT id FROM live_ids)"
                )
                conn.execute("DELETE FROM live_ids")
            else:
                conn.execute("DELETE FROM beliefs")
            # Audit: append-only; INSERT OR IGNORE skips already-persisted
            # seq numbers (idempotent on retry).
            for e in store.audit:
                conn.execute(
                    "INSERT OR IGNORE INTO audit VALUES (?,?,?,?,?,?,?)",
                    (
                        e.seq,
                        e.timestamp,
                        e.event_type,
                        e.belief_id,
                        e.proposition,
                        e.reason,
                        _safe_dumps(e.details),
                    ),
                )
            conn.execute(
                "INSERT OR REPLACE INTO meta VALUES (?,?)",
                ("config", json.dumps({
                    "detector_name": detector_name,
                    "entrenchment_weights": store.entrenchment_weights,
                    "survival_threshold": store.survival_threshold,
                    "policy": store.policy,
                    "schema_version": SCHEMA_VERSION,
                })),
            )
            conn.commit()
        finally:
            conn.close()


def load(path: str) -> BeliefStore:
    """Rebuild a belief store from a SQLite file."""
    conn = sqlite3.connect(path, timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    try:
        try:
            row = conn.execute(
                "SELECT value FROM meta WHERE key='config'").fetchone()
        except sqlite3.OperationalError:
            # Valid SQLite, but not a Veritas database (no meta table).
            raise ValueError(f"{path} is not a Veritas store (no config)")
        if row is None:
            raise ValueError(f"{path} is not a Veritas store (no config)")
        config = json.loads(row[0])
        ver = config.get("schema_version", 0)
        if ver > SCHEMA_VERSION:
            raise ValueError(
                f"{path} uses schema v{ver}, this Veritas understands "
                f"v{SCHEMA_VERSION}. Upgrade Veritas to read it."
            )
        detector_name = config.get("detector_name", "heuristic")
        store = BeliefStore(
            contradiction_fn=_resolve_detector(detector_name),
            entrenchment_weights=config.get("entrenchment_weights"),
            survival_threshold=config.get(
                "survival_threshold", 0.5),
            policy=config.get("policy", "entrenchment"),
        )
        store.detector_name = detector_name
        # Migrations: beliefs table may lack ground (v1->v2) or temporal
        # columns (v3->v4).
        cols = [r[1] for r in conn.execute("PRAGMA table_info(beliefs)")]
        has_ground = "ground" in cols
        has_temporal = "valid_from" in cols
        # Bulk-load mode: skip per-belief re-derivation (O(N^2)).
        store._bulk_loading = True
        try:
            for r in conn.execute(
                "SELECT id, proposition, confidence, source, source_reliability,"
                " justifications, timestamp, status, metadata"
                + (", ground" if has_ground else "")
                + (", valid_from, valid_until" if has_temporal else "")
                + " FROM beliefs"
            ):
                idx = 9
                ground_val = bool(r[idx]) if has_ground else False
                if has_ground:
                    idx += 1
                vf = r[idx] if has_temporal else None
                vu = r[idx + 1] if has_temporal else None
                b = Belief(
                    proposition=r[1],
                    confidence=r[2],
                    source=r[3],
                    source_reliability=r[4],
                    justifications=json.loads(r[5]),
                    id=r[0],
                    timestamp=r[6],
                    status=r[7],
                    metadata=json.loads(r[8]),
                    ground=ground_val,
                    valid_from=vf,
                    valid_until=vu,
                )
                # Bypass assert_belief: we are restoring, not reasoning.
                store._store(b)
        finally:
            # Bulk load complete: re-enable re-derivation.
            store._bulk_loading = False

        # Restore audit log preserving original sequence/timestamps.
        log = AuditLog()
        for r in conn.execute(
            "SELECT seq, timestamp, event_type, belief_id, proposition,"
            " reason, details FROM audit ORDER BY seq"
        ):
            log.record(r[2], r[3], r[4], r[5], json.loads(r[6]),
                       seq=r[0], timestamp=r[1])
        store._audit = log
        return store
    finally:
        conn.close()
