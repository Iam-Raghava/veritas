"""Determinism test for Veritas.

Same seed + frozen time => byte-identical stores and audit logs,
ACROSS Python hash seeds. This matters for debugging: any reported
bug must be reproducible.

Run: python3 tests/test_determinism.py
"""
import itertools
import os
import random
import subprocess
import sys
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from veritas import BeliefStore  # noqa: E402


def build(seed: int) -> BeliefStore:
    rng = random.Random(seed)
    s = BeliefStore()
    ids: list[str] = []
    for _ in range(500):
        r = rng.random()
        if r < 0.7:
            justs = ([rng.choice(ids) for _ in range(rng.randint(0, 2))]
                     if ids else [])
            # Quantized reliability/confidence: forces EXACT entrenchment
            # ties, so the deterministic tiebreak is actually exercised.
            q = rng.choice([0.3, 0.5, 0.7, 0.9])
            b = s.assert_belief(
                f"S{rng.randint(0, 8)}'s CEO is V{rng.randint(0, 3)}",
                source="x", source_reliability=q,
                confidence=q, justifications=justs)
            if b is not None:
                ids.append(b.id)
        else:
            act = s.active_beliefs()
            if act:
                s.retract_belief(rng.choice(act).id, reason="det")
    return s


def fingerprint(s: BeliefStore) -> tuple:
    beliefs = tuple(sorted(
        (b.id, b.proposition, b.confidence, b.source, b.source_reliability,
         tuple(sorted(b.justifications)), b.status,
         json_dumps(b.metadata))
        for b in s.all_beliefs()))
    audit = tuple(
        (e.seq, e.event_type, e.belief_id, e.proposition, e.reason,
         json_dumps(e.details))
        for e in s.audit)
    return beliefs, audit


def json_dumps(d) -> str:
    import json
    return json.dumps(d, sort_keys=True)


def run(seed: int):
    """Build with frozen IDs and a frozen incrementing clock."""
    t = itertools.count(1_700_000_000, 1)
    i = itertools.count()
    # _new_id / time.time are looked up dynamically in __post_init__,
    # so plain mock.patch works. One incrementing clock for everything:
    # deterministic because the operation sequence is seeded.
    with mock.patch("veritas.belief._new_id",
                     lambda: f"id-{next(i):06d}"), \
         mock.patch("time.time", lambda: float(next(t))):
        return fingerprint(build(seed))


_FINGERPRINT_HELPER = r"""
import itertools, json, sys
from unittest import mock
sys.path.insert(0, '.')
sys.path.insert(0, 'tests')
from test_determinism import build, fingerprint
t = itertools.count(1700000000, 1)
i = itertools.count()
with mock.patch("veritas.belief._new_id", lambda: f"id-{next(i):06d}"), \
     mock.patch("time.time", lambda: float(next(t))):
    fp = fingerprint(build(42))
print(json.dumps([[list(b) for b in fp[0]],
                  [list(e) for e in fp[1]]], default=str))
"""


_TIE_HELPER = r"""
import itertools, json, sys
from unittest import mock
sys.path.insert(0, '.')
from veritas import BeliefStore
W = {"source": 0.5, "corroboration": 0.0, "confidence": 0.5, "recency": 0.0}
s = BeliefStore(entrenchment_weights=W)
t = itertools.count(1700000000, 1)
i = itertools.count()
with mock.patch("veritas.belief._new_id", lambda: f"id-{next(i):06d}"), \
     mock.patch("time.time", lambda: float(next(t))):
    ids = []
    for subj in ["Acme", "Globex", "Initech"]:
        b = s.assert_belief(f"{subj}'s CEO is Alice", source="x",
                            source_reliability=0.7, confidence=0.7)
        ids.append(b.id)
    new = s.assert_belief("All CEOs are Bob", source="x",
                          source_reliability=0.8, confidence=0.8,
                          metadata={"contradicts": ids})
    order = [e.belief_id for e in s.audit if e.event_type == "retracted"]
    print(json.dumps(order))
"""


def main():
    # Part 1: in-process determinism.
    f1, f2 = run(42), run(42)
    assert f1 == f2, "same seed diverged!"
    f3 = run(43)
    assert f1 != f3, "different seeds identical (suspicious)"
    print(f"DETERMINISTIC: {len(f1[0])} beliefs, {len(f1[1])} audit events "
          f"identical across runs")

    # Part 2: cross-process determinism across hash seeds. This is the
    # real guarantee: Python's hash randomization must not affect output.
    root = os.path.join(os.path.dirname(__file__), "..")
    fps = []
    for hs in ("0", "1", "42"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        p = subprocess.run(
            [sys.executable, "-c", _FINGERPRINT_HELPER],
            capture_output=True, text=True, timeout=120,
            cwd=root, env=env)
        assert p.returncode == 0, \
            f"hash seed {hs} failed: {p.stderr[-300:]}"
        fps.append(p.stdout.strip())
    assert fps[0] == fps[1] == fps[2], \
        "PYTHONHASHSEED changed output: determinism broken"
    print(f"CROSS-SEED DETERMINISTIC: byte-identical across "
          f"PYTHONHASHSEED 0/1/42 ({len(fps[0])} chars each)")

    # Part 3: tie-torture. Three incumbents with EXACTLY tied entrenchment,
    # one newcomer retracting all three. Without the ID tiebreak +
    # contradictor sort, the retraction order varies by hash seed
    # (verified: 5 distinct orders across 6 seeds). With them, it must
    # be ID order on every seed.
    orders = []
    for hs in ("0", "1", "42", "99"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        p = subprocess.run(
            [sys.executable, "-c", _TIE_HELPER],
            capture_output=True, text=True, timeout=60,
            cwd=root, env=env)
        assert p.returncode == 0, \
            f"tie helper failed on seed {hs}: {p.stderr[-300:]}"
        orders.append(p.stdout.strip())
    assert len(set(orders)) == 1, \
        f"tie retraction order varied by hash seed: {orders}"
    assert orders[0] == '["id-000000", "id-000001", "id-000002"]', \
        f"tie order not ID-sorted: {orders[0]}"
    print(f"TIE-TORTURE DETERMINISTIC: 3-way exact tie retracts in ID "
          f"order on all seeds")


if __name__ == "__main__":
    main()
