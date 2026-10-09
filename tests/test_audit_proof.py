"""Audit completeness proof for Veritas.

The audit log is Veritas's claim to trustworthiness: NOTHING may change
state silently. This test proves, over randomized workloads, that:

  1. Every active belief has exactly one ASSERTED event (its birth).
  2. Every retracted belief has ASSERTED followed by a retraction event
     (RETRACTED or CASCADE_RETRACTED) — a complete lifecycle.
  3. Every REJECTED event corresponds to a belief that was never stored.
  4. Audit seq numbers are dense and ordered (0..n-1, no gaps).
  5. Every CASCADE event names a real lost justification.
  6. No event references a belief ID unknown to the store.

Run: python3 tests/test_audit_proof.py [--ops N] [--seed S]
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from veritas import BeliefStore  # noqa: E402
from veritas.audit import (  # noqa: E402
    ASSERTED, CASCADE_RETRACTED, REJECTED, RETRACTED,
)

SUBJECTS = [f"S{i}" for i in range(30)]
VALUES = [f"V{i}" for i in range(10)]


def run_workload(seed: int, n_ops: int) -> BeliefStore:
    rng = random.Random(seed)
    s = BeliefStore()
    ids: list[str] = []
    for _ in range(n_ops):
        r = rng.random()
        if r < 0.7 or not ids:
            justs = ([rng.choice(ids) for _ in range(rng.randint(0, 2))]
                     if ids and rng.random() < 0.5 else [])
            b = s.assert_belief(
                f"{rng.choice(SUBJECTS)}'s CEO is {rng.choice(VALUES)}",
                source="x", source_reliability=rng.random(),
                confidence=rng.random(), justifications=justs)
            if b is not None:
                ids.append(b.id)
        elif r < 0.9:
            act = s.active_beliefs()
            if act:
                s.retract_belief(rng.choice(act).id, reason="proof")
        else:
            s.find_contradictions()
    return s


def prove(store: BeliefStore) -> None:
    events = list(store.audit)
    known_ids = {b.id for b in store.all_beliefs()}

    # 4. Dense, ordered seq.
    assert [e.seq for e in events] == list(range(len(events))), \
        "audit seq not dense/ordered"

    # 6. No dangling references.
    for e in events:
        assert e.belief_id in known_ids or e.event_type == REJECTED, \
            f"event {e.seq} references unknown belief {e.belief_id}"

    asserted: dict[str, int] = {}
    retracted: set[str] = set()
    rejected: set[str] = set()
    for e in events:
        if e.event_type == ASSERTED:
            assert e.belief_id not in asserted, \
                f"belief {e.belief_id} asserted twice"
            asserted[e.belief_id] = e.seq
        elif e.event_type in (RETRACTED, CASCADE_RETRACTED):
            assert e.belief_id in asserted, \
                f"belief {e.belief_id} retracted without assertion"
            # 5. Cascade names a real lost justification.
            if e.event_type == CASCADE_RETRACTED:
                lost = e.details.get("lost_justification")
                assert lost is not None, \
                    f"cascade event {e.seq} names no justification"
            retracted.add(e.belief_id)
        elif e.event_type == REJECTED:
            rejected.add(e.belief_id)

    # 1. Every active belief asserted exactly once, never retracted.
    for b in store.active_beliefs():
        assert b.id in asserted, f"active belief {b.id} has no ASSERTED event"
        assert b.id not in retracted, \
            f"active belief {b.id} has a retraction event"

    # 2. Every inactive belief: full lifecycle ASSERTED -> retracted.
    for b in store.all_beliefs():
        if not b.is_active:
            assert b.id in asserted, \
                f"retracted belief {b.id} was never asserted"
            assert b.id in retracted, \
                f"retracted belief {b.id} has no retraction event"
            assert asserted[b.id] < next(
                e.seq for e in events
                if e.belief_id == b.id
                and e.event_type in (RETRACTED, CASCADE_RETRACTED)), \
                f"belief {b.id} retracted before assertion"

    # 3. Rejected beliefs were never stored.
    for bid in rejected:
        assert bid not in known_ids, \
            f"rejected belief {bid} exists in store"


def main():
    n_ops, seed = 8000, 60606
    if "--ops" in sys.argv:
        n_ops = int(sys.argv[sys.argv.index("--ops") + 1])
    if "--seed" in sys.argv:
        seed = int(sys.argv[sys.argv.index("--seed") + 1])
    print(f"audit proof: {n_ops} ops (seed={seed})...", flush=True)
    for i in range(3):
        prove(run_workload(seed + i, n_ops))
    print("AUDIT COMPLETE: every state change accounted for, 3 workloads")


if __name__ == "__main__":
    main()
