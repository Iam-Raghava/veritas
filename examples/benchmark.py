"""Veritas benchmarks: assertion and contradiction-search at scale.

Measures the hot paths with synthetic belief stores:
  - bulk assertion throughput (beliefs/second)
  - contradictors_of() latency for a single new belief (the hot path)
  - find_contradictions() over the full store, indexed vs brute force

Run: python3 examples/benchmark.py [--n 10000]
"""
import os
import random
import resource
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from veritas import Belief, BeliefStore, heuristic_contradiction  # noqa: E402

NAMES = ["Alice", "Bob", "Carol", "Dave", "Erin", "Frank", "Grace", "Heidi"]
SUBJECTS = [f"Company{i}" for i in range(500)]
RELS = ["CEO", "HQ", "founder"]


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def make_beliefs(n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.append({
            "proposition": f"{rng.choice(SUBJECTS)}'s {rng.choice(RELS)} "
                           f"is {rng.choice(NAMES)}",
            "source_reliability": rng.random(),
            "confidence": rng.random(),
        })
    return out


def bench_assertion(n: int) -> tuple[BeliefStore, float]:
    store = BeliefStore()
    specs = make_beliefs(n)
    t0 = time.perf_counter()
    for s in specs:
        # Bypass contradiction handling for pure insert throughput;
        # use _store directly (index maintenance included).
        b = Belief(s["proposition"], confidence=s["confidence"],
                   source_reliability=s["source_reliability"])
        store._store(b)
    dt = time.perf_counter() - t0
    return store, dt


def bench_contradictors(store: BeliefStore, trials: int = 200) -> float:
    rng = random.Random(99)
    probes = [
        Belief(f"{rng.choice(SUBJECTS)}'s {rng.choice(RELS)} is {rng.choice(NAMES)}")
        for _ in range(trials)
    ]
    t0 = time.perf_counter()
    total_hits = 0
    for p in probes:
        total_hits += len(store.contradictors_of(p))
    dt = time.perf_counter() - t0
    return dt / trials, total_hits


def bench_full_scan(store: BeliefStore, do_brute: bool) -> tuple[float, float, int]:
    # Indexed
    t0 = time.perf_counter()
    indexed_pairs = store.find_contradictions()
    t_indexed = time.perf_counter() - t0
    t_brute = 0.0
    if do_brute:
        # Brute force reference (only feasible at small n).
        active = store.active_beliefs()
        t0 = time.perf_counter()
        brute = set()
        for i, a in enumerate(active):
            for b in active[i + 1:]:
                if heuristic_contradiction(a, b):
                    brute.add(tuple(sorted((a.id, b.id))))
        t_brute = time.perf_counter() - t0
        assert {tuple(sorted((a.id, b.id))) for a, b in indexed_pairs} == brute, \
            "indexed search diverged from brute force!"
    return t_indexed, t_brute, len(indexed_pairs)


def main() -> None:
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 10000
    print(f"=== Veritas benchmark: n={n} ===")
    print(f"memory before: {mem_mb():.1f} MB")

    store, dt = bench_assertion(n)
    print(f"bulk insert: {n} beliefs in {dt:.2f}s "
          f"({n/dt:,.0f} beliefs/sec, {dt/n*1e6:.1f} us/belief)")
    print(f"memory after insert: {mem_mb():.1f} MB")

    per_probe, hits = bench_contradictors(store)
    print(f"contradictors_of(): {per_probe*1e6:.1f} us/probe "
          f"(avg {hits/200:.1f} candidates)")

    t_idx, t_brute, npairs = bench_full_scan(store, do_brute=(n <= 2000))
    if t_brute:
        print(f"find_contradictions(): indexed {t_idx:.3f}s vs brute {t_brute:.3f}s "
              f"({t_brute/max(t_idx,1e-9):.0f}x speedup, {npairs} pairs)")
    else:
        print(f"find_contradictions(): indexed {t_idx:.3f}s over {n} beliefs "
              f"({npairs} pairs; brute force skipped at this scale)")

    # Assertion WITH contradiction handling (the real hot path).
    s2 = BeliefStore()
    specs = make_beliefs(2000)
    t0 = time.perf_counter()
    for sp in specs:
        s2.assert_belief(sp["proposition"], confidence=sp["confidence"],
                         source_reliability=sp["source_reliability"],
                         source="bench")
    dt2 = time.perf_counter() - t0
    st = s2.stats()
    print(f"assert_belief() with contraction: 2000 in {dt2:.2f}s "
          f"({2000/dt2:,.0f}/sec); active={st['active']} "
          f"retracted={st['retracted']}")
    print("benchmark complete")


if __name__ == "__main__":
    main()
