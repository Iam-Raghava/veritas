"""Stress tests: concurrency safety and 100k-belief scale.

  1. Threads hammering one store: the store must never corrupt
     (no lost beliefs, no phantom actives, audit length == events).
  2. 100k beliefs: assertion throughput and full-scan latency.

Run: python3 tests/test_stress.py [--quick]
"""
import os
import random
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from veritas import BeliefStore  # noqa: E402

FAIL = []


def check(name, fn):
    try:
        fn()
        print(f"  ok: {name}")
    except Exception as e:  # noqa: BLE001
        FAIL.append((name, e))
        print(f"  FAIL: {name}: {type(e).__name__}: {e}")


def test_concurrent_assert():
    """8 threads x 500 asserts: no corruption, audit complete."""
    s = BeliefStore()
    errors: list = []

    def worker(wid: int):
        rng = random.Random(wid)
        try:
            for i in range(500):
                s.assert_belief(
                    f"Worker{wid} claim {i} is value{rng.randint(0, 50)}",
                    source="x", source_reliability=rng.random(),
                    confidence=rng.random(),
                )
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"thread errors: {errors[:2]}"
    # Every belief object is accounted for exactly once.
    ids = [b.id for b in s.all_beliefs()]
    assert len(ids) == len(set(ids)), "duplicate belief ids!"
    # Audit length matches recorded events; store is consistent.
    assert s.is_consistent(), "concurrent asserts left contradictions"
    n_asserted = sum(1 for e in s.audit if e.event_type == "asserted")
    n_rejected = sum(1 for e in s.audit if e.event_type == "rejected")
    assert n_asserted + n_rejected == 4000, (
        f"audit gap: {n_asserted} asserted + {n_rejected} rejected != 4000"
    )
    print(f"    ({len(s.active_beliefs())} active, "
          f"{len(s.all_beliefs()) - len(s.active_beliefs())} retracted)")


def test_concurrent_mixed():
    """4 assert-threads + 4 retract-threads: invariants hold at the end."""
    s = BeliefStore()
    seed_ids: list[str] = []
    for i in range(200):
        b = s.assert_belief(f"Seed claim {i} is v{i % 10}", source="x",
                            source_reliability=0.9)
        seed_ids.append(b.id)
    errors: list = []
    stop = threading.Event()

    def asserter(wid: int):
        rng = random.Random(100 + wid)
        try:
            while not stop.is_set():
                s.assert_belief(
                    f"Mix{wid} claim {rng.randint(0, 10**6)}",
                    source="x", source_reliability=rng.random())
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def retracter(wid: int):
        rng = random.Random(200 + wid)
        try:
            while not stop.is_set():
                act = s.active_beliefs()
                if act:
                    s.retract_belief(rng.choice(act).id, reason="stress")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = ([threading.Thread(target=asserter, args=(w,)) for w in range(4)]
               + [threading.Thread(target=retracter, args=(w,)) for w in range(4)])
    for t in threads:
        t.start()
    time.sleep(3)
    stop.set()
    for t in threads:
        t.join()

    assert not errors, f"thread errors: {errors[:2]}"
    assert s.is_consistent(), "mixed workload left contradictions"
    # No dangling justifications anywhere.
    for b in s.active_beliefs():
        for j in b.justifications:
            t = s.get(j)
            assert t is not None and t.is_active, "dangling justification"


def test_100k_scale(quick: bool):
    n = 20000 if quick else 100000
    s = BeliefStore()
    rng = random.Random(42)
    t0 = time.perf_counter()
    for i in range(n):
        s.assert_belief(
            f"Company{i % 2000}'s CEO is "
            f"{['Alice', 'Bob', 'Carol', 'Dave'][rng.randint(0, 3)]}",
            source="x", source_reliability=rng.random(), confidence=rng.random())
    dt = time.perf_counter() - t0
    print(f"    {n} asserts with contraction: {dt:.1f}s ({n/dt:,.0f}/sec)")
    t0 = time.perf_counter()
    pairs = s.find_contradictions()
    dt2 = time.perf_counter() - t0
    print(f"    full scan over {len(s.active_beliefs())} active: {dt2:.2f}s "
          f"({len(pairs)} pairs)")
    assert not pairs, "scale test left contradictions"
    assert n / dt > 1000, f"assertion throughput collapsed: {n/dt:.0f}/sec"


def main():
    quick = "--quick" in sys.argv
    print("stress tests:")
    check("concurrent assert (8x500)", test_concurrent_assert)
    check("concurrent mixed assert/retract", test_concurrent_mixed)
    check("100k scale" if not quick else "20k scale (quick)",
          lambda: test_100k_scale(quick))
    print(f"\n{3 - len(FAIL)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
