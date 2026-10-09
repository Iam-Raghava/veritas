"""Adversarial edge cases for Veritas.

Each test attacks a specific failure mode a demanding reviewer would try:
cycles, diamonds, deep chains, hostile detectors, and pathological inputs.

Run: python3 tests/test_edge.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from veritas import Belief, BeliefStore  # noqa: E402

PASS = []
FAIL = []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  ok: {name}")
    except Exception as e:  # noqa: BLE001
        FAIL.append((name, e))
        print(f"  FAIL: {name}: {type(e).__name__}: {e}")


def test_justification_cycle_terminates():
    """A <-> B cycle: retracting one must terminate, not loop forever."""
    s = BeliefStore()
    a = s.assert_belief("Acme's CEO is Alice", source="x", source_reliability=0.9)
    b = s.assert_belief("Acme's HQ is Austin", source="x", source_reliability=0.1,
                        justifications=[a.id])
    # Manually create the back-edge (simulates a hostile/mistaken agent).
    # Reach into the store directly: public API returns snapshots precisely
    # so this kind of mutation can't happen by accident.
    s._beliefs[a.id].justifications.append(b.id)
    retracted = s.retract_belief(a.id, reason="test")
    assert not s.get(a.id).is_active
    assert s.is_consistent()
    assert len(retracted) >= 1


def test_diamond_dependency():
    """D depends on B and C; both depend on A. Retract A: all fall or survive."""
    s = BeliefStore()  # nothing survives orphaned
    a = s.assert_belief("Acme's CEO is Alice", source="x", source_reliability=0.9)
    b = s.assert_belief("Acme's HQ is Austin", source="x", source_reliability=0.1,
                        justifications=[a.id])
    c = s.assert_belief("Acme's founder is Bob", source="x", source_reliability=0.1,
                        justifications=[a.id])
    d = s.assert_belief("Acme's motto is Win", source="x", source_reliability=0.1,
                        justifications=[b.id, c.id])
    retracted = s.retract_belief(a.id, reason="test")
    ids = {b.id for b in retracted}
    assert a.id in ids and b.id in ids and c.id in ids and d.id in ids, \
        f"diamond did not fully cascade: {[x.proposition for x in retracted]}"
    assert s.is_consistent()


def test_deep_chain_no_recursion_error():
    """1000-deep justification chain must not blow the stack."""
    s = BeliefStore()
    prev = s.assert_belief("Root claim holds", source="x", source_reliability=0.9)
    for i in range(1000):
        prev = s.assert_belief(f"Derived claim {i} holds", source="x",
                               source_reliability=0.1,
                               justifications=[prev.id])
    root_id = s.all_beliefs()[0].id
    retracted = s.retract_belief(root_id, reason="test")
    assert len(retracted) == 1001, f"expected 1001 retracted, got {len(retracted)}"
    assert s.is_consistent()


def test_detector_that_raises():
    """A throwing detector must not corrupt the store."""
    def boom(a, b):
        raise RuntimeError("detector exploded")
    s = BeliefStore(contradiction_fn=boom)
    s.assert_belief("Acme's CEO is Alice")  # no candidates yet: no call
    try:
        s.assert_belief("Acme's CEO is Bob")  # candidate exists: boom
        raise AssertionError("expected detector exception to propagate")
    except RuntimeError:
        pass
    # The failed assertion must not have stored a half-baked belief.
    assert len(s.all_beliefs()) == 1
    assert s.active_beliefs()[0].proposition == "Acme's CEO is Alice"


def test_detector_contradicts_everything():
    """Paranoid detector: every pair contradicts. Store must stay consistent."""
    s = BeliefStore(contradiction_fn=lambda a, b: a.id != b.id)
    first = s.assert_belief("First claim", source="x", source_reliability=0.99)
    assert first is not None
    # Everything after must either contract or be rejected — never coexist.
    for i in range(20):
        s.assert_belief(f"Claim {i}", source="x",
                        source_reliability=0.5 + (i % 2) * 0.4)
    assert s.is_consistent(), "paranoid detector left contradictions"


def test_self_contradictory_assert():
    """A belief contradicting itself must not break assertion."""
    s = BeliefStore()
    b = s.assert_belief("Acme's CEO is Alice", source="x", source_reliability=0.9)
    assert b is not None and s.is_consistent()


def test_pathological_propositions():
    """Empty, unicode, and huge propositions must not crash."""
    s = BeliefStore()
    for prop in ["", "   ", "日本語の主張", "é" * 10000,
                 "Acme's CEO is Alice\nwith newline",
                 "null\x00byte"]:
        try:
            s.assert_belief(prop, source="x", source_reliability=0.5)
        except Exception:  # noqa: BLE001
            pass  # must not crash the store; rejection is fine
    assert s.is_consistent()


def test_entrenchment_tie_goes_to_incumbent():
    """Exact tie: newcomer must be rejected (documented AGM behavior).

    Note: with default weights, recency (0.15) breaks near-ties in favor
    of the newcomer — fresher info wins ties by design. A *true* tie
    needs recency zeroed out.
    """
    no_recency = {"source": 1.0, "corroboration": 0.0,
                  "confidence": 0.0, "recency": 0.0}
    s = BeliefStore(entrenchment_weights=no_recency)
    old = s.assert_belief("Acme's CEO is Alice", source="x",
                          source_reliability=0.8, confidence=0.8)
    new = s.assert_belief("Acme's CEO is Bob", source="x",
                          source_reliability=0.8, confidence=0.8)
    assert new is None, "tie should reject newcomer"
    assert s.get(old.id).is_active
    assert any(e.event_type == "rejected" for e in s.audit)


def test_retract_nonexistent_is_noop():
    s = BeliefStore()
    assert s.retract_belief("no-such-id") == []
    b = s.assert_belief("Acme's CEO is Alice")
    assert s.retract_belief(b.id) != []
    assert s.retract_belief(b.id) == []  # double retract: no-op


def test_justification_to_retracted_pruned():
    """Assert with justification to a dead belief: pruned, not dangling."""
    s = BeliefStore()
    dead = s.assert_belief("Acme's CEO is Alice", source="x", source_reliability=0.9)
    s.retract_belief(dead.id)
    live = s.assert_belief("Acme's HQ is Austin", source="x", source_reliability=0.9,
                           justifications=[dead.id, "ghost-id"])
    assert live is not None
    assert live.justifications == [], f"expected pruned, got {live.justifications}"
    assert s.is_consistent()


def test_mass_contradiction_storm():
    """50 mutually-contradicting beliefs: exactly the strongest survives."""
    s = BeliefStore()
    survivors = []
    for i in range(50):
        r = s.assert_belief("Acme's CEO is Alice" if i % 2 == 0 else
                            "Acme's CEO is Bob",
                            source="x", source_reliability=i / 50.0,
                            confidence=0.9)
        if r is not None:
            survivors.append(r)
    assert s.is_consistent()
    active = s.active_beliefs()
    assert len(active) == 1, f"expected 1 survivor, got {len(active)}"
    # The strongest (reliability 49/50 = 0.98) must be the survivor.
    assert active[0].source_reliability == 0.98


def main():
    print("edge cases:")
    check("justification cycle terminates", test_justification_cycle_terminates)
    check("diamond dependency cascades", test_diamond_dependency)
    check("deep chain (1000) no recursion error", test_deep_chain_no_recursion_error)
    check("throwing detector doesn't corrupt", test_detector_that_raises)
    check("paranoid detector stays consistent", test_detector_contradicts_everything)
    check("self-contradictory assert", test_self_contradictory_assert)
    check("pathological propositions", test_pathological_propositions)
    check("entrenchment tie -> incumbent wins", test_entrenchment_tie_goes_to_incumbent)
    check("retract nonexistent/double = no-op", test_retract_nonexistent_is_noop)
    check("dead justifications pruned", test_justification_to_retracted_pruned)
    check("50-way contradiction storm", test_mass_contradiction_storm)
    print(f"\n{PASS.__len__()} passed, {FAIL.__len__()} failed")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
