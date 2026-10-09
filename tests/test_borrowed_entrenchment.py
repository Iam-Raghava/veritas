"""No borrowed entrenchment: a belief cannot win on corroboration from
beliefs its own assertion will kill.

Attack: assert P justified by its own contradictor Q. Before the fix,
the all-or-nothing comparison used P's entrenchment *including* the
corroboration boost from Q; Q was then retracted and the justification
pruned, leaving P holding a win its own entrenchment could not earn.

Run: python3 tests/test_borrowed_entrenchment.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from veritas import BeliefStore  # noqa: E402
from veritas.audit import REJECTED  # noqa: E402
from veritas.entrenchment import entrenchment  # noqa: E402

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


def make_q(store):
    """Strong incumbent: entrenchment ~0.63."""
    return store.assert_belief(
        "Acme's CEO is Jane",
        confidence=0.8,
        source="press release",
        source_reliability=0.8,
    )


def test_newcomer_cannot_borrow_from_contradictor():
    """P justified by Q; P's own entrenchment (0.57) < Q's (0.63).

    With Q's borrowed corroboration P would reach 0.653 and win.
    Correct: P is rejected and Q survives.
    """
    s = BeliefStore()
    q = make_q(s)
    assert abs(entrenchment(q) - 0.63) < 0.01, entrenchment(q)

    p = s.assert_belief(
        "Acme's CEO is John",
        confidence=0.7,
        source="press release",
        source_reliability=0.7,
        justifications=[q.id],
    )
    assert p is None, "P won on borrowed entrenchment"
    assert s.get(q.id).is_active, "Q was wrongly retracted"
    assert s.is_consistent()


def test_strong_newcomer_still_wins():
    """P justified by Q, but P's OWN entrenchment (0.72) beats Q (0.63).

    The fix must not over-correct: a genuinely stronger newcomer still
    displaces the incumbent, and the doomed justification is pruned.
    """
    s = BeliefStore()
    q = make_q(s)

    p = s.assert_belief(
        "Acme's CEO is John",
        confidence=0.95,
        source="press release",
        source_reliability=0.95,
        justifications=[q.id],
    )
    assert p is not None, "strong P was wrongly rejected"
    assert p.justifications == [], p.justifications
    assert not s.get(q.id).is_active, "Q should be retracted"
    assert s.is_consistent()


def test_transitive_doomed_justification():
    """P justified by D; D justified by contradictor Q; D cannot stand
    alone (independent entrenchment 0.37 < 0.5 survival threshold).

    Asserting P kills Q, which cascade-kills D. P must not borrow D's
    corroboration either: P's own 0.60 < Q's 0.63 -> rejected.
    """
    s = BeliefStore()
    q = make_q(s)
    d = s.assert_belief(
        "Acme's HQ is Reno",
        confidence=0.5,
        source="rumor",
        source_reliability=0.3,
        justifications=[q.id],
    )
    assert d is not None

    p = s.assert_belief(
        "Acme's CEO is John",
        confidence=0.75,
        source="press release",
        source_reliability=0.75,
        justifications=[d.id],
    )
    assert p is None, "P won on transitively borrowed entrenchment"
    assert s.get(q.id).is_active
    assert s.get(d.id).is_active
    assert s.is_consistent()


def test_live_justification_still_counts():
    """P justified by R (unrelated, survives): R's corroboration is
    legitimate and must still count toward P's entrenchment."""
    s = BeliefStore()
    q = make_q(s)
    r = s.assert_belief(
        "Beta's CEO is Ann",
        confidence=0.7,
        source="press release",
        source_reliability=0.7,
    )

    p = s.assert_belief(
        "Acme's CEO is John",
        confidence=0.7,
        source="press release",
        source_reliability=0.7,
        justifications=[r.id],
    )
    # 0.57 own + 0.083 corroboration = 0.653 > 0.63: wins fairly.
    assert p is not None, "P with live justification was wrongly rejected"
    assert p.justifications == [r.id], p.justifications
    assert s.get(r.id).is_active
    assert not s.get(q.id).is_active
    assert s.is_consistent()


def test_rejected_audit_names_doomed_prune():
    """The REJECTED audit event records the doomed justification prune,
    so the decision is explainable after the fact."""
    s = BeliefStore()
    q = make_q(s)
    p_id = None
    before = len(s.audit)
    p = s.assert_belief(
        "Acme's CEO is John",
        confidence=0.7,
        source="press release",
        source_reliability=0.7,
        justifications=[q.id],
    )
    assert p is None
    events = [e for e in s.audit if e.event_type == REJECTED]
    assert len(events) == 1, [e.event_type for e in s.audit[before:]]
    details = events[0].details
    assert "pruned_justifications" in details, details.keys()
    assert q.id in details["pruned_justifications"]


def main():
    print("borrowed entrenchment:")
    check("newcomer cannot borrow from contradictor",
          test_newcomer_cannot_borrow_from_contradictor)
    check("strong newcomer still wins", test_strong_newcomer_still_wins)
    check("transitive doomed justification", test_transitive_doomed_justification)
    check("live justification still counts", test_live_justification_still_counts)
    check("rejected audit names doomed prune",
          test_rejected_audit_names_doomed_prune)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
