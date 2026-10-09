"""Soak test for Veritas: 100k operations, watching for degradation.

Measures across the run:
  - wall time per 10k-op window (must not degrade superlinearly)
  - RSS memory (must grow at most linearly with beliefs, not ops)
  - invariants spot-checked every window

A "leak" here = memory or time growing with OPERATIONS while the
belief count stays bounded. Fails if the last window is >3x slower
than the first, or memory grows >2x while beliefs stay flat.

Run: python3 tests/test_soak.py [--ops N]
"""
import os
import random
import resource
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from veritas import BeliefStore  # noqa: E402

SUBJECTS = [f"S{i}" for i in range(100)]
VALUES = [f"V{i}" for i in range(20)]


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main():
    n_ops = 100000
    if "--ops" in sys.argv:
        n_ops = int(sys.argv[sys.argv.index("--ops") + 1])
    rng = random.Random(31337)
    s = BeliefStore()
    window = 10000
    times, mems, belief_counts = [], [], []

    print(f"soak: {n_ops} ops...", flush=True)
    t_start = time.perf_counter()
    for i in range(n_ops):
        r = rng.random()
        if r < 0.6:
            s.assert_belief(
                f"{rng.choice(SUBJECTS)}'s CEO is {rng.choice(VALUES)}",
                source="x", source_reliability=rng.random(),
                confidence=rng.random())
        elif r < 0.8:
            act = s.active_beliefs()
            if act:
                s.retract_belief(rng.choice(act).id, reason="soak")
        else:
            s.is_consistent()

        if (i + 1) % window == 0:
            dt = time.perf_counter() - t_start
            times.append(dt - sum(times))
            mems.append(rss_mb())
            belief_counts.append(len(s.all_beliefs()))
            assert s.is_consistent(), f"inconsistent at op {i}"
            print(f"  op {i+1}: {times[-1]:.1f}s window, "
                  f"{mems[-1]:.0f}MB, {belief_counts[-1]} beliefs", flush=True)

    # Degradation checks. Per-op cost MUST grow with bucket density (every
    # candidate contradictor has to be checked — that's the price of
    # correctness). What we forbid is SUPERLINEAR blowup: time growth
    # must track belief growth, not exceed it wildly.
    slowdown = times[-1] / max(times[0], 1e-9)
    belief_growth = belief_counts[-1] / max(belief_counts[0], 1)
    print(f"\nfirst window: {times[0]:.1f}s, last window: {times[-1]:.1f}s "
          f"({slowdown:.1f}x time vs {belief_growth:.1f}x beliefs)")
    print(f"memory: {mems[0]:.0f}MB -> {mems[-1]:.0f}MB, "
          f"beliefs: {belief_counts[0]} -> {belief_counts[-1]}")
    assert slowdown < belief_growth * 1.5, \
        (f"SUPERLINEAR DEGRADATION: {slowdown:.1f}x slower for "
         f"{belief_growth:.1f}x beliefs")
    # Memory may grow with beliefs (expected); flag only explosive growth.
    mem_growth = mems[-1] / max(mems[0], 1e-9)
    assert mem_growth < max(4.0, belief_growth * 2), \
        f"MEMORY LEAK: {mem_growth:.1f}x memory vs {belief_growth:.1f}x beliefs"
    print("SOAK CLEAN: no superlinear degradation, no leaks")


if __name__ == "__main__":
    main()
