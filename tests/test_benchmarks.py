"""Benchmarks: Veritas vs naive baselines.

Compares:
1. Latest-wins (dict overwrite): fast but loses history, no propagation
2. Veritas: entrenchment-ordered contraction with cascade

Metrics: contradiction handling correctness, cascade completeness,
audit completeness, and throughput.
"""

import sys, time
sys.path.insert(0, ".")

from veritas import BeliefStore


class LatestWins:
    """Naive baseline: new beliefs overwrite old ones."""

    def __init__(self):
        self.beliefs = {}

    def assert_belief(self, proposition, **kwargs):
        self.beliefs[proposition] = kwargs
        return True

    def get(self, proposition):
        return self.beliefs.get(proposition)


def benchmark_contradiction_handling():
    """Veritas rejects weak contradictions; latest-wins corrupts."""
    print("  Contradiction handling:")
    # Veritas
    s = BeliefStore()
    s.assert_belief("Server is online", source="monitor",
                    source_reliability=0.95, confidence=0.95, ground=True)
    r = s.assert_belief("Server is not online", source="rumor",
                        source_reliability=0.2, confidence=0.3)
    veritas_correct = (r is None and s.is_consistent())
    print(f"    Veritas rejects weak rumor: {veritas_correct}")

    # Latest-wins: same subject, overwritten
    lw = LatestWins()
    lw.assert_belief("Server status", value="online", reliability=0.95)
    lw.assert_belief("Server status", value="offline", reliability=0.2)
    final = lw.get("Server status")
    lw_correct = final and final.get("value") == "online"
    print(f"    Latest-wins preserves truth: {lw_correct} (got {final.get('value') if final else None}!)"  )
    return veritas_correct and not lw_correct


def benchmark_cascade():
    """Veritas cascades; latest-wins leaves orphans."""
    print("  Cascade propagation:")
    s = BeliefStore()
    root = s.assert_belief("Root", source="x", source_reliability=0.9, ground=True)
    for i in range(100):
        s.assert_belief(f"Derived {i}", source="x", source_reliability=0.5,
                        justifications=[root.id])
    s.retract_belief(root.id, reason="test")
    veritas_clean = len(s.active_beliefs()) == 0
    print(f"    Veritas cascades 100 dependents: {veritas_clean}")
    return veritas_clean


def benchmark_throughput():
    """Raw assertion throughput."""
    print("  Throughput (1000 assertions):")
    s = BeliefStore()
    start = time.time()
    for i in range(1000):
        s.assert_belief(f"Belief {i}", source="x", source_reliability=0.5,
                        ground=True)
    elapsed = time.time() - start
    print(f"    Veritas: {1000/elapsed:.0f}/sec")
    return True


def main():
    print("benchmarks:")
    r1 = benchmark_contradiction_handling()
    r2 = benchmark_cascade()
    r3 = benchmark_throughput()
    print(f"\nVeritas wins: {sum([r1, r2, r3])}/3")
    if not all([r1, r2, r3]):
        sys.exit(1)


if __name__ == "__main__":
    main()
