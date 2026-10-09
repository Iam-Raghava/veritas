"""Entrenchment-ordering proof for Veritas.

The store's core claim is AGM-style contraction: when a newcomer
contradicts existing beliefs, the LEAST entrenched die first, and a
newcomer that can't beat every contradictor is rejected. These tests
verify the decision rule itself — not just that the result is
consistent, but that the RIGHT beliefs died in the RIGHT order.

Run: python3 tests/test_ordering.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from veritas import BeliefStore  # noqa: E402
from veritas.audit import RETRACTED  # noqa: E402

# Recency zeroed: entrenchment is then fully determined by our inputs,
# making the expected outcome exactly predictable.
W = {"source": 0.5, "corroboration": 0.0, "confidence": 0.5, "recency": 0.0}

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  ok: {name}")
    except Exception as e:  # noqa: BLE001
        FAIL.append((name, e))
        print(f"  FAIL: {name}: {type(e).__name__}: {e}")


def ent(s, b):
    return s.entrenchment_of(b)


def test_weaker_newcomer_rejected():
    s = BeliefStore(entrenchment_weights=W)
    old = s.assert_belief("Acme's CEO is Alice", source="x",
                          source_reliability=0.9, confidence=0.9)
    assert ent(s, old) == 0.9
    new = s.assert_belief("Acme's CEO is Bob", source="x",
                          source_reliability=0.5, confidence=0.5)
    assert new is None, "weaker newcomer must be rejected"
    assert s.get(old.id).is_active
    assert len(s.active_beliefs()) == 1


def test_stronger_newcomer_contracts():
    s = BeliefStore(entrenchment_weights=W)
    old = s.assert_belief("Acme's CEO is Alice", source="x",
                          source_reliability=0.9, confidence=0.9)
    new = s.assert_belief("Acme's CEO is Bob", source="x",
                          source_reliability=0.95, confidence=0.95)
    assert new is not None and new.is_active
    assert not s.get(old.id).is_active, "weaker incumbent must be retracted"
    assert s.is_consistent()


def test_one_strong_blocker_rejects():
    """Three contradictors; the newcomer beats two but not the third."""
    s = BeliefStore(entrenchment_weights=W)
    # Use distinct subjects so they don't contradict each other.
    b1 = s.assert_belief("Acme's CEO is Alice", source="x",
                         source_reliability=0.3, confidence=0.3)   # ent 0.3
    b2 = s.assert_belief("Globex's CEO is Alice", source="x",
                         source_reliability=0.6, confidence=0.6)  # ent 0.6
    b3 = s.assert_belief("Initech's CEO is Alice", source="x",
                         source_reliability=0.8, confidence=0.8)  # ent 0.8
    # Newcomer contradicts all three via explicit metadata.
    new = s.assert_belief(
        "All CEOs are Bob", source="x",
        source_reliability=0.7, confidence=0.7,  # ent 0.7
        metadata={"contradicts": [b1.id, b2.id, b3.id]})
    assert new is None, "b3 (0.8) must block newcomer (0.7)"
    assert all(s.get(b.id).is_active for b in (b1, b2, b3)), \
        "rejection must change nothing"


def test_all_weaker_contract_weakest_first():
    """Newcomer beats all three: all die, weakest first in the audit."""
    s = BeliefStore(entrenchment_weights=W)
    b1 = s.assert_belief("Acme's CEO is Alice", source="x",
                         source_reliability=0.3, confidence=0.3)
    b2 = s.assert_belief("Globex's CEO is Alice", source="x",
                         source_reliability=0.6, confidence=0.6)
    b3 = s.assert_belief("Initech's CEO is Alice", source="x",
                         source_reliability=0.8, confidence=0.8)
    new = s.assert_belief(
        "All CEOs are Bob", source="x",
        source_reliability=0.85, confidence=0.85,
        metadata={"contradicts": [b1.id, b2.id, b3.id]})
    assert new is not None and new.is_active
    assert not any(s.get(b.id).is_active for b in (b1, b2, b3))
    # Audit must show retraction in ascending entrenchment: b1, b2, b3.
    retraction_order = [e.belief_id for e in s.audit
                        if e.event_type == RETRACTED]
    assert retraction_order == [b1.id, b2.id, b3.id], \
        f"expected weakest-first {[b1.id, b2.id, b3.id]}, got {retraction_order}"
    assert s.is_consistent()


def test_corroboration_counts():
    """A well-justified belief outranks a lonely one with better source."""
    s = BeliefStore(entrenchment_weights={
        "source": 0.3, "corroboration": 0.5,
        "confidence": 0.2, "recency": 0.0})
    lonely = s.assert_belief("Acme's CEO is Alice", source="x",
                             source_reliability=1.0, confidence=1.0)
    # Supported: 3 justifications -> corroboration ~0.67.
    j1 = s.assert_belief("Board met today", source="x", source_reliability=0.5)
    j2 = s.assert_belief("Press release issued", source="x", source_reliability=0.5)
    j3 = s.assert_belief("CEO tweeted", source="x", source_reliability=0.5)
    supported = s.assert_belief(
        "Acme's CEO is Bob", source="y", source_reliability=0.2,
        confidence=0.2, justifications=[j1.id, j2.id, j3.id])
    # lonely ent = 0.3*1 + 0.2*1 = 0.50
    # newcomer ent = 0.3*0.2 + 0.5*0.667 + 0.2*0.2 = 0.433 -> rejected.
    assert supported is None, \
        f"lonely ent={ent(s, lonely):.3f} should beat newcomer"
    assert s.get(lonely.id).is_active


def test_ordering_randomized():
    """Randomized: brute-force check the decision rule on 200 scenarios."""
    import random
    rng = random.Random(20261008)
    for trial in range(200):
        s = BeliefStore(entrenchment_weights=W)
        incumbents = []
        for i in range(rng.randint(1, 4)):
            r = rng.random()
            b = s.assert_belief(f"Subj{i}'s CEO is Alice", source="x",
                                source_reliability=r, confidence=r)
            incumbents.append((b, r))  # ent == r with these weights
        nr = rng.random()
        new = s.assert_belief(
            "Every CEO is Bob", source="x",
            source_reliability=nr, confidence=nr,
            metadata={"contradicts": [b.id for b, _ in incumbents]})
        max_inc = max(r for _, r in incumbents)
        if nr > max_inc:
            assert new is not None, \
                f"trial {trial}: newcomer {nr:.3f} beats all, must be accepted"
            assert all(not s.get(b.id).is_active for b, _ in incumbents)
        else:
            assert new is None, \
                f"trial {trial}: newcomer {nr:.3f} blocked by {max_inc:.3f}"
            assert all(s.get(b.id).is_active for b, _ in incumbents)
        assert s.is_consistent()


def main():
    print("entrenchment ordering proof:")
    check("weaker newcomer rejected", test_weaker_newcomer_rejected)
    check("stronger newcomer contracts", test_stronger_newcomer_contracts)
    check("one strong blocker rejects", test_one_strong_blocker_rejects)
    check("all weaker contract weakest-first", test_all_weaker_contract_weakest_first)
    check("corroboration counts", test_corroboration_counts)
    check("randomized rule check (200 trials)", test_ordering_randomized)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
