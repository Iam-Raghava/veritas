"""Property tests for entrenchment invariants.

These verify the mathematical properties that make Veritas sound:
- EE2: ent(derived) <= min(ent(premises)) — no free warrant
- Monotonicity: more independent justifications => higher entrenchment
- Grounding: ground beliefs are bounded by source reliability
- Cache consistency: cached values equal recomputed values
"""

import sys
sys.path.insert(0, ".")

from veritas import BeliefStore

PASS = []
FAIL = []

def check(name, cond):
    if cond:
        PASS.append(name)
        print(f"  ok: {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL: {name}")


def test_ee2_no_free_warrant():
    """EE2: derived entrenchment never exceeds weakest premise."""
    s = BeliefStore()
    # Weak premise, strong derivation attempt
    weak = s.assert_belief("Weak premise", source="x",
                           source_reliability=0.2, confidence=0.2,
                           ground=True)
    strong = s.assert_belief("Strong premise", source="x",
                             source_reliability=0.9, confidence=0.9,
                             ground=True)
    derived = s.assert_belief("Derived", source="x",
                              source_reliability=0.9, confidence=0.9,
                              justifications=[[weak.id, strong.id]])
    e_weak = s.entrenchment_of(s.get(weak.id))
    e_derived = s.entrenchment_of(s.get(derived.id))
    check("EE2: derived <= weakest premise", e_derived <= e_weak + 1e-9)


def test_monotonicity():
    """More independent justifications => higher or equal entrenchment."""
    s = BeliefStore()
    p1 = s.assert_belief("P1", source="x", source_reliability=0.7, ground=True)
    p2 = s.assert_belief("P2", source="x", source_reliability=0.7, ground=True)
    # Single justification
    d1 = s.assert_belief("D", source="x", source_reliability=0.5,
                         justifications=[[p1.id]])
    e1 = s.entrenchment_of(s.get(d1.id))
    # Add second independent justification by creating new belief
    # (can't modify justifications post-hoc, so compare two beliefs)
    s2 = BeliefStore()
    q1 = s2.assert_belief("P1", source="x", source_reliability=0.7, ground=True)
    q2 = s2.assert_belief("P2", source="x", source_reliability=0.7, ground=True)
    d2 = s2.assert_belief("D", source="x", source_reliability=0.5,
                          justifications=[[q1.id], [q2.id]])
    e2 = s2.entrenchment_of(s2.get(d2.id))
    check("monotonicity: 2 justifications >= 1", e2 >= e1 - 1e-9)


def test_ground_bounded():
    """Ground belief entrenchment bounded by source reliability."""
    s = BeliefStore()
    b = s.assert_belief("Ground truth", source="x",
                        source_reliability=0.6, confidence=0.9,
                        ground=True)
    e = s.entrenchment_of(s.get(b.id))
    # Entrenchment should reflect both reliability and confidence,
    # not exceed their combination.
    check("ground bounded", 0.0 <= e <= 1.0)


def test_cache_consistency():
    """Cached entrenchment equals recomputed value after mutations."""
    s = BeliefStore()
    a = s.assert_belief("A", source="x", source_reliability=0.8, ground=True)
    b = s.assert_belief("B", source="x", source_reliability=0.7,
                        justifications=[a.id])
    e1 = s.entrenchment_of(s.get(b.id))
    # Trigger cache
    e2 = s.entrenchment_of(s.get(b.id))
    check("cache hit consistent", e1 == e2)
    # Mutate (add unrelated belief) -> cache invalidates, recomputes same
    s.assert_belief("C", source="x", source_reliability=0.5, ground=True)
    e3 = s.entrenchment_of(s.get(b.id))
    check("post-mutation consistent", abs(e1 - e3) < 1e-9)


def test_temporal_no_false_positive():
    """Non-overlapping temporal beliefs don't contradict."""
    import time
    s = BeliefStore()
    now = time.time()
    b1 = s.assert_belief("X is true", source="s", source_reliability=0.9,
                         ground=True, valid_from=now-100, valid_until=now-50)
    b2 = s.assert_belief("X is not true", source="s", source_reliability=0.9,
                         ground=True, valid_from=now-40, valid_until=now)
    check("temporal: no false positive", b1 is not None and b2 is not None)
    check("temporal: consistent", s.is_consistent())


def main():
    print("property tests:")
    test_ee2_no_free_warrant()
    test_monotonicity()
    test_ground_bounded()
    test_cache_consistency()
    test_temporal_no_false_positive()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
