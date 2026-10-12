"""Baseline systems for the governance benchmark.

Three systems, one workload each. The interface is scenario-driven rather
than a single abstract API, because the honest comparison is per-capability:
a baseline that cannot do temporal queries reports n/a instead of a fake 0.

Systems:
    LatestWins  - dict overwrite on fact key. Last writer wins, no
                  reliability check, no history, no derivations.
                  (This is what a naive `memory[key] = value` agent does.)
    AppendOnly  - keeps every version; query returns the most recent.
                  (This is what "we keep full history" frameworks do
                  without any adjudication logic.)
    VeritasSys  - the real veritas.BeliefStore: entrenchment-ordered
                  contraction, TMS cascades, temporal validity, audit log.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from veritas import BeliefStore  # noqa: E402
from veritas.explain import Explainer  # noqa: E402
from veritas.timetravel import TimeTravel  # noqa: E402


class LatestWins:
    """Naive baseline 1: last write wins. No reliability, no history."""

    name = "latest-wins"

    def __init__(self):
        # key -> (value, reliability, source, timestamp)
        self._facts: dict[str, tuple] = {}
        # derived key -> parent key (recorded but never used for cascades)
        self._derived: dict[str, str] = {}

    def assert_fact(self, key, value, reliability=0.5, source="x",
                    timestamp=0.0, ground=True, parent=None,
                    valid_from=None, valid_until=None):
        self._facts[key] = (value, reliability, source, timestamp)
        if parent is not None:
            self._derived[key] = parent

    def query(self, key):
        rec = self._facts.get(key)
        return rec[0] if rec else None

    def retract_key(self, key):
        # Removes ONLY the key itself. Dependents are left orphaned:
        # the system has no derivation tracking to cascade through.
        self._facts.pop(key, None)

    def forget_source(self, source):
        for k in [k for k, v in self._facts.items() if v[2] == source]:
            del self._facts[k]

    def query_at(self, key, timestamp):
        return None  # n/a: no temporal concept

    def explain_retraction(self, key):
        return None  # n/a: no audit log

    def state_at(self, timestamp):
        return None  # n/a: no history

    def orphan_count(self):
        # Transitive: a derived key is orphaned if ANY ancestor in its
        # parent chain is gone (the whole justification chain is broken).
        orphans = 0
        for k, p in self._derived.items():
            if k not in self._facts:
                continue
            chain = p
            seen = set()
            while chain is not None and chain not in seen:
                seen.add(chain)
                if chain not in self._facts:
                    orphans += 1
                    break
                chain = self._derived.get(chain)
        return orphans

    def count(self):
        return len(self._facts)


class AppendOnly:
    """Naive baseline 2: full history, most-recent-wins on read."""

    name = "append-only"

    def __init__(self):
        # key -> list of (value, reliability, source, timestamp)
        self._facts: dict[str, list[tuple]] = {}
        self._derived: dict[str, str] = {}
        self._deleted: set[str] = set()

    def assert_fact(self, key, value, reliability=0.5, source="x",
                    timestamp=0.0, ground=True, parent=None,
                    valid_from=None, valid_until=None):
        self._facts.setdefault(key, []).append(
            (value, reliability, source, timestamp))
        if parent is not None:
            self._derived[key] = parent

    def query(self, key):
        if key in self._deleted:
            return None
        hist = self._facts.get(key)
        if not hist:
            return None
        return hist[-1][0]  # most recent wins, regardless of reliability

    def retract_key(self, key):
        # "Retraction" only hides the latest version; history (and any
        # dependents derived from older versions) remains.
        self._deleted.add(key)

    def forget_source(self, source):
        for k in [k for k, v in self._facts.items()
                  if any(r[2] == source for r in v)]:
            self._deleted.add(k)

    def query_at(self, key, timestamp):
        return None  # n/a: versions are append-ordered, not time-indexed

    def explain_retraction(self, key):
        return None  # n/a: no causal audit log

    def state_at(self, timestamp):
        return None  # n/a

    def orphan_count(self):
        # Transitive orphan check over the parent chain.
        live = {k for k in self._facts if k not in self._deleted}
        orphans = 0
        for k, p in self._derived.items():
            if k not in live:
                continue
            chain = p
            seen = set()
            while chain is not None and chain not in seen:
                seen.add(chain)
                if chain not in live:
                    orphans += 1
                    break
                chain = self._derived.get(chain)
        return orphans

    def count(self):
        return sum(1 for k in self._facts if k not in self._deleted)


class VeritasSys:
    """The real veritas.BeliefStore, wrapped in the scenario interface."""

    name = "veritas"

    def __init__(self):
        self.store = BeliefStore()
        # fact key -> list of belief ids asserted for it (in order)
        self._ids: dict[str, list[str]] = {}
        # fact key -> list of (belief_id, value) for derived tracking
        self._derived_ids: dict[str, str] = {}

    # -- scenario interface -------------------------------------------

    def assert_fact(self, key, value, reliability=0.5, source="x",
                    timestamp=0.0, ground=True, parent=None,
                    valid_from=None, valid_until=None):
        # Resolve the parent's belief id: the latest asserted id for the
        # parent fact key (roots live in _ids, not _derived_ids).
        just = None
        if parent is not None:
            pids = self._ids.get(parent, [])
            just = [pids[-1]] if pids else None
        b = self.store.assert_belief(
            value,  # proposition text carries the fact
            confidence=0.9,
            source=source,
            source_reliability=reliability,
            timestamp=timestamp,
            ground=ground,
            justifications=just,
            valid_from=valid_from,
            valid_until=valid_until,
            metadata={"fact_key": key},
        )
        if b is not None:
            self._ids.setdefault(key, []).append(b.id)
            if parent is not None:
                self._derived_ids[key] = b.id
        return b

    def query(self, key):
        """Current winning value for a fact key (highest entrenchment)."""
        props = self._active_propositions(key)
        return props[-1] if props else None

    def query_at(self, key, timestamp):
        """Value of a fact key as of timestamp T (temporal validity)."""
        tt = TimeTravel(self.store)
        cands = []
        for b in tt.beliefs_at(timestamp):
            if b.metadata.get("fact_key") != key:
                continue
            vf = b.valid_from if b.valid_from is not None else float("-inf")
            vu = b.valid_until if b.valid_until is not None else float("inf")
            if vf <= timestamp < vu:
                cands.append(b)
        return cands[-1].proposition if cands else None

    def retract_key(self, key):
        for bid in self._ids.get(key, []):
            self.store.retract_belief(bid, reason="benchmark retraction")

    def forget_source(self, source):
        for b in list(self.store.active_beliefs()):
            if b.source == source:
                self.store.retract_belief(
                    b.id, reason=f"benchmark: forget source {source}")

    def explain_retraction(self, key):
        ids = self._ids.get(key, [])
        if not ids:
            return None
        return Explainer(self.store).why_retracted(ids[0])

    def state_at(self, timestamp):
        return TimeTravel(self.store).beliefs_at(timestamp)

    def orphan_count(self):
        # Active beliefs whose justifications are all dead/non-ground.
        n = 0
        for b in self.store.active_beliefs():
            if b.ground or not b.justifications:
                continue
            if not self.store._has_ground_path_locked(b.id):
                n += 1
        return n

    def count(self):
        return len(self.store.active_beliefs())

    # -- helpers -------------------------------------------------------

    def _active_propositions(self, key):
        ids = set(self._ids.get(key, []))
        live = {b.id: b for b in self.store.active_beliefs()}
        return [live[i].proposition for i in self._ids.get(key, [])
                if i in live and i in ids]
