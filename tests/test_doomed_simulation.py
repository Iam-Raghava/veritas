"""Differential test: _doomed_targets (pure simulation) must exactly match
the real cascade in _retract_single, over randomized justification graphs.

This guards the borrowed-entrenchment fix: the simulation decides which
justifications are "dead on arrival", so any divergence from the real
cascade would be a soundness hole in the all-or-nothing decision.

Run: python3 tests/test_doomed_simulation.py [--trials N] [--seed S]
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from veritas import BeliefStore  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} {extra}")


def random_store(rng, n=40):
    """Build a store with random justification DAG (unique subjects, so no
    accidental contradictions disturb the graph)."""
    s = BeliefStore()
    ids = []
    for i in range(n):
        justs = [j for j in ids if rng.random() < 0.25]
        b = s.assert_belief(
            f"Subject{i}'s status is Value{rng.randrange(3)}",
            confidence=round(rng.uniform(0.1, 1.0), 2),
            source="test",
            source_reliability=round(rng.uniform(0.1, 1.0), 2),
            justifications=justs[:3],
        )
        if b is not None:
            ids.append(b.id)
    return s, ids


def trial(rng, trial_no):
    s, ids = random_store(rng)
    active = [i for i in ids if s.get(i).is_active]
    if len(active) < 3:
        return True  # degenerate; skip
    targets = set(rng.sample(active, k=rng.randint(1, min(5, len(active)))))

    # Membership oracle: for a sample of beliefs, the simulation's
    # doomed-membership must match the real cascade's retraction set.
    probe = rng.sample(active, k=min(12, len(active)))
    sim_doomed = s._doomed_justifications(targets, probe)
    for pid in targets:
        if pid in probe and pid not in sim_doomed:
            return f"target {pid} not reported doomed (trial {trial_no})"

    # Now really retract and compare per-belief.
    actually_retracted = set()
    for t in targets:
        for b in s.retract_belief(t, reason="differential test"):
            actually_retracted.add(b.id)

    for pid in probe:
        sim, real = (pid in sim_doomed), (pid in actually_retracted)
        if sim != real:
            return (
                f"simulation != cascade for {pid} (trial {trial_no}): "
                f"sim_doomed={sim}, actually_retracted={real}"
            )
    if not s.is_consistent():
        return f"store inconsistent after retractions (trial {trial_no})"
    return True


def main():
    trials = int(sys.argv[sys.argv.index("--trials") + 1]) if "--trials" in sys.argv else 300
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 20261008
    rng = random.Random(seed)
    print(f"doomed-simulation differential: {trials} trials, seed {seed}")
    bad = 0
    for t in range(trials):
        r = trial(rng, t)
        if r is not True:
            bad += 1
            print(f"  FAIL: {r}")
            if bad >= 3:
                break
    check(f"{trials} randomized simulation==cascade trials", bad == 0, f"{bad} diverged")
    print(f"\n{PASS} passed, {FAIL} failed")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
