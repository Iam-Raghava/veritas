"""v0.4.0 tests: well-foundedness, Horn clauses, commutativity.

Run: python3 tests/test_v04.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from veritas import BeliefStore  # noqa: E402
from veritas.belief import Belief  # noqa: E402
from veritas.detect import as_function, heuristic_contradiction  # noqa: E402

PASS, FAIL = [], []


def check(name, cond):
    if cond:
        PASS.append(name)
        print(f"  ok: {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL: {name}")


def test_circular_bypass_prevented():
    """Reviewer's Attack 1: A<->B with no ground must not survive."""
    s = BeliefStore()
    p = s.assert_belief("Ground premise", source="x",
                        source_reliability=0.9, ground=True)
    a = s.assert_belief("Belief A", source="x", source_reliability=0.5,
                        justifications=[p.id])
    b = s.assert_belief("Belief B", source="x", source_reliability=0.5,
                        justifications=[a.id])
    # Create cycle: A now depends on B as well.
    s._beliefs[a.id].justifications.append([b.id])
    # Retract P: both A and B should fall (no ground path).
    s.retract_belief(p.id, reason="test")
    check("circular A retracts", not s.get(a.id).is_active)
    check("circular B retracts", not s.get(b.id).is_active)


def test_well_founded_survives():
    """A belief with a ground path through a cycle survives."""
    s = BeliefStore()
    p = s.assert_belief("Ground", source="x", source_reliability=0.9,
                        ground=True)
    a = s.assert_belief("A", source="x", source_reliability=0.5,
                        justifications=[p.id])
    b = s.assert_belief("B", source="x", source_reliability=0.5,
                        justifications=[a.id])
    # A also depends on B (cycle), but A has direct ground path via P.
    s._beliefs[a.id].justifications.append([b.id])
    # Retract something unrelated; A and B should survive (ground path).
    c = s.assert_belief("Unrelated", source="x", source_reliability=0.9,
                        ground=True)
    s.retract_belief(c.id, reason="test")
    check("well-founded A survives", s.get(a.id).is_active)
    check("well-founded B survives", s.get(b.id).is_active)


def test_horn_and_semantics():
    """AND-set: all premises needed; losing one kills the set."""
    s = BeliefStore()
    p1 = s.assert_belief("Premise 1", source="x", source_reliability=0.9,
                         ground=True)
    p2 = s.assert_belief("Premise 2", source="x", source_reliability=0.9,
                         ground=True)
    # C requires BOTH p1 and p2 (single AND-set).
    c = s.assert_belief("Conclusion", source="x", source_reliability=0.5,
                        justifications=[[p1.id, p2.id]])
    check("AND-set stored", c.justifications == [[p1.id, p2.id]])
    # Retract p1: the set dies, C has no other sets -> orphan -> retract.
    s.retract_belief(p1.id, reason="test")
    check("AND: losing one premise kills derived",
          not s.get(c.id).is_active)


def test_horn_or_semantics():
    """OR across sets: independent derivations; one surviving set suffices."""
    s = BeliefStore()
    p1 = s.assert_belief("Premise 1", source="x", source_reliability=0.9,
                         ground=True)
    p2 = s.assert_belief("Premise 2", source="x", source_reliability=0.9,
                         ground=True)
    # C has two independent derivations.
    c = s.assert_belief("Conclusion", source="x", source_reliability=0.5,
                        justifications=[[p1.id], [p2.id]])
    check("OR-sets stored", len(c.justifications) == 2)
    # Retract p1: set1 dies, but set2 ([p2]) survives -> C survives.
    s.retract_belief(p1.id, reason="test")
    check("OR: alternative derivation sustains",
          s.get(c.id).is_active)
    check("OR: dead set pruned",
          s._beliefs[c.id].justifications == [[p2.id]])


def test_ee2_horn_best_proof():
    """EE2 with Horn: capped by BEST set, not weakest alternative."""
    s = BeliefStore()
    strong = s.assert_belief("Strong", source="x",
                             source_reliability=0.9, confidence=0.9,
                             ground=True)
    weak = s.assert_belief("Weak", source="x",
                           source_reliability=0.2, confidence=0.2,
                           ground=True)
    # C has strong proof [strong] and weak proof [weak].
    # Should be capped by strong (0.9-ish), not weak (0.2).
    c = s.assert_belief("C", source="x",
                        source_reliability=0.9, confidence=0.9,
                        justifications=[[strong.id], [weak.id]])
    ent_c = s.entrenchment_of(s._beliefs[c.id])
    ent_strong = s.entrenchment_of(s._beliefs[strong.id])
    # C should be close to strong, not dragged to weak.
    check("Horn EE2: best proof warrants",
          ent_c > 0.5)  # well above weak's 0.2


def test_ee2_horn_and_weakest():
    """EE2 with Horn: within a set, weakest premise bounds."""
    s = BeliefStore()
    p1 = s.assert_belief("P1", source="x",
                         source_reliability=0.9, confidence=0.9,
                         ground=True)
    p2 = s.assert_belief("P2 fragile", source="x",
                         source_reliability=0.2, confidence=0.2,
                         ground=True)
    # C requires both: capped by weakest (p2).
    c = s.assert_belief("C", source="x",
                        source_reliability=0.9, confidence=0.9,
                        justifications=[[p1.id, p2.id]])
    ent_c = s.entrenchment_of(s._beliefs[c.id])
    ent_p2 = s.entrenchment_of(s._beliefs[p2.id])
    check("Horn EE2: AND weakest bounds",
          ent_c <= ent_p2 + 1e-9)


def test_detector_commutativity():
    """Commutativity harness: contradicts(A,B) == contradicts(B,A)."""
    sym = as_function(heuristic_contradiction)
    b1 = Belief(proposition="Acme's CEO is Alice")
    b2 = Belief(proposition="Acme's CEO is Bob")
    check("commutative true/true",
          sym(b1, b2) and sym(b2, b1))
    b3 = Belief(proposition="The sky is blue")
    check("commutative false/false",
          not sym(b1, b3) and not sym(b3, b1))


def test_asymmetric_detector_symmetrized():
    """A detector that fires in only one direction is symmetrized."""
    def one_way(a, b):
        # Only fires when a.proposition < b.proposition (artificial).
        return a.proposition < b.proposition and "X" in a.proposition

    sym = as_function(one_way)
    b1 = Belief(proposition="X is true")
    b2 = Belief(proposition="Y is true")
    # one_way(b1,b2)=True, one_way(b2,b1)=False. Symmetric should be True both.
    check("asymmetric detector symmetrized",
          sym(b1, b2) and sym(b2, b1))


def test_constraint_joint_inconsistency():
    """N-ary constraint: budget + costs jointly inconsistent."""
    s = BeliefStore()

    def budget_check(beliefs):
        budget = None
        costs = []
        for b in beliefs:
            if "budget" in b.metadata:
                budget = b.metadata["budget"]
            if "cost" in b.metadata:
                costs.append(b)
        if budget is None or not costs:
            return []
        total = sum(b.metadata["cost"] for b in costs)
        if total > budget:
            return [b.id for b in costs]
        return []

    s.add_constraint("budget", budget_check)
    b1 = s.assert_belief("Budget is $10,000", source="x",
                         source_reliability=0.9, confidence=0.9,
                         metadata={"budget": 10000}, ground=True)
    b2 = s.assert_belief("Hosting costs $7,000", source="x",
                         source_reliability=0.7, confidence=0.7,
                         metadata={"cost": 7000}, ground=True)
    # $7k + $5k > $10k: weakest (b3) should be rejected.
    b3 = s.assert_belief("Marketing costs $5,000", source="x",
                         source_reliability=0.5, confidence=0.5,
                         metadata={"cost": 5000}, ground=True)
    check("constraint: joint violator rejected", b3 is None)
    check("constraint: existing beliefs survive",
          s.get(b1.id).is_active and s.get(b2.id).is_active)
    check("constraint: satisfied after",
          s.check_constraints() == {})


def test_constraint_weakest_retracted():
    """Constraint retracts weakest involved, not newcomer if stronger."""
    s = BeliefStore()

    def budget_check(beliefs):
        budget = None
        costs = []
        for b in beliefs:
            if "budget" in b.metadata:
                budget = b.metadata["budget"]
            if "cost" in b.metadata:
                costs.append(b)
        if budget is None or not costs:
            return []
        total = sum(b.metadata["cost"] for b in costs)
        if total > budget:
            return [b.id for b in costs]
        return []

    s.add_constraint("budget", budget_check)
    b1 = s.assert_belief("Budget is $10,000", source="x",
                         source_reliability=0.9, confidence=0.9,
                         metadata={"budget": 10000}, ground=True)
    # Weak existing cost.
    b2 = s.assert_belief("Old hosting costs $7,000", source="x",
                         source_reliability=0.3, confidence=0.3,
                         metadata={"cost": 7000}, ground=True)
    # Strong newcomer: $5k, but $7k + $5k > $10k.
    # Weakest is b2 ($7k at 0.3), so b2 should go, b3 stays.
    b3 = s.assert_belief("Marketing costs $5,000", source="x",
                         source_reliability=0.9, confidence=0.9,
                         metadata={"cost": 5000}, ground=True)
    check("constraint: newcomer accepted", b3 is not None)
    if b3:
        check("constraint: weakest existing retracted",
              not s.get(b2.id).is_active)
        check("constraint: newcomer survives",
              s.get(b3.id).is_active)


def test_constraint_broken_isolated():
    """A throwing constraint must not break assertion."""
    s = BeliefStore()

    def bad(beliefs):
        raise RuntimeError("constraint bug")

    s.add_constraint("bad", bad)
    b = s.assert_belief("Test", source="x", source_reliability=0.9,
                        ground=True)
    check("broken constraint isolated", b is not None)


def test_constraint_remove():
    """remove_constraint works."""
    s = BeliefStore()
    s.add_constraint("c1", lambda beliefs: [])
    s.remove_constraint("c1")
    check("constraint removed", "c1" not in s._constraints)

def main():
    print("v0.4.0:")
    test_circular_bypass_prevented()
    test_well_founded_survives()
    test_horn_and_semantics()
    test_horn_or_semantics()
    test_ee2_horn_best_proof()
    test_ee2_horn_and_weakest()
    test_detector_commutativity()
    test_asymmetric_detector_symmetrized()
    test_constraint_joint_inconsistency()
    test_constraint_weakest_retracted()
    test_constraint_broken_isolated()
    test_constraint_remove()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)




if __name__ == "__main__":
    main()
