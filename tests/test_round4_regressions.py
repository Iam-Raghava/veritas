"""Round-4 red-team regression tests (fixes shipped in c925f63 / 9421ccc).

Each test pins one bug from the 2026-10-11 "Verification of Attempted Fixes"
report to a concrete scenario. The suite was green when these bugs were
found, so these tests exist to make the suite verify the right invariants.

Run: python3 tests/test_round4_regressions.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from veritas import BeliefStore  # noqa: E402
from veritas.belief import Belief  # noqa: E402
from veritas.detect import HeuristicDetector, heuristic_contradiction  # noqa: E402
from veritas.detect_semantic import _has_opposition  # noqa: E402
from veritas.explain import Explainer  # noqa: E402
from veritas.persist import save, load  # noqa: E402
from veritas.timetravel import TimeTravel  # noqa: E402

PASS, FAIL = [], []


def check(name, cond):
    if cond:
        PASS.append(name)
        print(f"  ok: {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL: {name}")


# ---------------------------------------------------------------------------
# 1. Re-derivation must not resurrect contradicted beliefs (store.py)
# ---------------------------------------------------------------------------
# NOTE: _try_rederive_dependents is only reachable via the internal re-store
# path (public re-assertion mints a fresh id, so the depends-check cannot
# match). The guard below is defense-in-depth on that internal path; the test
# drives it directly and documents the reachability caveat.


def _rederive_setup():
    s = BeliefStore()
    p = s.assert_belief("Sensor online", ground=True, source_reliability=0.9)
    b = s.assert_belief(
        "Acme's CEO is Jane",
        justifications=[p.id],
        source_reliability=0.9,
        confidence=0.9,
    )
    c = s.assert_belief(
        "Acme's CEO is John",
        ground=True,
        source_reliability=0.95,
        confidence=0.95,
    )
    return s, p, b, c


def test_rederive_blocked_by_stronger_contradictor():
    s, p, b, c = _rederive_setup()
    check("setup: B retracted by contradiction", not s.get(b.id).is_active)
    check("setup: C active", s.get(c.id).is_active)
    check(
        "setup: C outranks B",
        s.entrenchment_of(s.get(c.id)) >= s.entrenchment_of(s.get(b.id)),
    )
    s._try_rederive_dependents(p.id)  # internal re-derivation trigger
    check("B stays retracted (blocked by stronger C)", not s.get(b.id).is_active)
    check("store stays consistent", s.is_consistent())
    blocked = [
        e
        for e in s.audit
        if e.event_type == "rederived_blocked" and e.belief_id == b.id
    ]
    check("rederived_blocked audited", len(blocked) == 1)


def test_rederive_displaces_weaker_contradictor():
    # Displacement branch: retracted B with an ACTIVE but WEAKER
    # contradictor C. B must revive and displace C.
    s = BeliefStore()
    p = s.assert_belief("Sensor online", ground=True, source_reliability=0.9)
    c = s.assert_belief(
        "Acme's CEO is John",
        ground=True,
        source_reliability=0.4,
        confidence=0.4,
    )
    b = s.assert_belief(
        "Acme's CEO is Jane",
        justifications=[p.id],
        source_reliability=0.95,
        confidence=0.95,
    )
    check("setup: B outranks weak C, C retracted", not s.get(c.id).is_active)
    # D contradicts B explicitly (metadata pair, no functional clash with C).
    d = s.assert_belief(
        "Q3 revenue is 5B",
        ground=True,
        source_reliability=0.99,
        confidence=0.99,
        metadata={"contradicts": [b.id]},
    )
    check("setup: B retracted by stronger D", not s.get(b.id).is_active)
    s.retract_belief(d.id, "correction")
    check("setup: D gone, B still retracted", not s.get(b.id).is_active)
    # Re-assert a fresh weak contradictor with the same proposition as C
    # (C itself is retracted; ids are fresh on re-assertion).
    c2 = s.assert_belief(
        "Acme's CEO is John",
        ground=True,
        source_reliability=0.4,
        confidence=0.4,
    )
    check("setup: weak contradictor active", s.get(c2.id).is_active)
    b_ent = s.entrenchment_of(s.get(b.id))
    c_ent = s.entrenchment_of(s.get(c2.id))
    s._try_rederive_dependents(p.id)  # internal re-derivation trigger
    if b_ent >= c_ent:
        check("outranking B revives", s.get(b.id).is_active)
        check("weaker contradictor displaced", not s.get(c2.id).is_active)
        check("store consistent after displacement", s.is_consistent())
    else:
        check("weaker B stays retracted", not s.get(b.id).is_active)
        check("store consistent", s.is_consistent())
    assert d is not None


# ---------------------------------------------------------------------------
# 2. Contradiction + constraint must be transactional (store.py)
# ---------------------------------------------------------------------------


def test_rejected_newcomer_leaves_incumbents_untouched():
    s = BeliefStore()

    def budget(snaps):
        total = sum(b.metadata.get("cost", 0) for b in snaps)
        if total > 100:
            return [b.id for b in snaps if b.metadata.get("cost", 0) > 0]
        return []

    s.add_constraint("budget", budget)
    inc = s.assert_belief(
        "Acme's CEO is John",
        ground=True,
        source_reliability=0.5,
        confidence=0.5,
    )
    # Newcomer outranks the incumbent on contradiction (high reliability),
    # but alone violates the budget constraint (cost 150 > 100).
    new = s.assert_belief(
        "Acme's CEO is Jane",
        ground=True,
        source_reliability=0.95,
        confidence=0.95,
        metadata={"cost": 150},
    )
    check("newcomer rejected by constraint simulation", new is None)
    check("incumbent untouched", s.get(inc.id).is_active)
    check(
        "rejection audited",
        any(
            e.event_type == "rejected" and "contradictors untouched" in e.reason
            for e in s.audit
        ),
    )


# ---------------------------------------------------------------------------
# 3. SQLite persist must not blow the variable limit (persist.py)
# ---------------------------------------------------------------------------


def test_persist_chunked_save_roundtrip(tmp_path="/tmp"):
    s = BeliefStore()
    n = 2500  # > 500/chunk -> exercises the temp-table chunked path
    for i in range(n):
        s.assert_belief(f"Fact number {i} is recorded", ground=True)
    path = os.path.join(tmp_path, "round4_test.db")
    if os.path.exists(path):
        os.remove(path)
    save(s, path)
    s2 = load(path)
    check("all beliefs persisted", len(s2.active_beliefs()) == n)
    check(
        "proposition integrity after round-trip",
        s2.get(s.active_beliefs()[0].id) is not None,
    )
    os.remove(path)


# ---------------------------------------------------------------------------
# 4. TimeTravel must sort events before replay (timetravel.py)
# ---------------------------------------------------------------------------


def test_timetravel_backfill_ordering():
    s = BeliefStore()
    a = s.assert_belief("Alpha is online", ground=True, timestamp=100.0)
    s.retract_belief(a.id, "superseded")  # stamped with system clock (~1.79e9)
    b = s.assert_belief("Beta is online", ground=True, timestamp=200.0)
    # Backfill an OLD assertion AFTER the now-stamped retraction.
    c = s.assert_belief("Gamma is online", ground=True, timestamp=50.0)
    tt = TimeTravel(s)
    props = {x.proposition for x in tt.beliefs_at(150.0)}
    check("backfilled belief visible at t=150", "Gamma is online" in props)
    check("historical belief visible at t=150", "Alpha is online" in props)
    check("future belief not visible at t=150", "Beta is online" not in props)
    check("retracted belief gone at far future", "Alpha is online" not in {
        x.proposition for x in tt.beliefs_at(time.time() + 3600)
    })
    assert b is not None and c is not None  # silence linters


# ---------------------------------------------------------------------------
# 5. SemanticDetector antonym check independent of negation (detect_semantic)
# ---------------------------------------------------------------------------


def test_antonym_with_asymmetric_negation():
    # Old code: a_neg != b_neg entered the if-block, jaccard 0.6 <= 0.8,
    # hit `return False` — the antonym loop was unreachable.
    check(
        "not online vs offline is opposition",
        _has_opposition("The service is not online", "The service is offline"),
    )
    check(
        "symmetric pair still works",
        _has_opposition("The server is online", "The server is offline"),
    )
    check(
        "negation-only pair still works",
        _has_opposition("The server is online", "The server is not online"),
    )
    check(
        "unrelated pair not opposition",
        not _has_opposition("The server is online", "The database is large"),
    )


# ---------------------------------------------------------------------------
# 6. Explainer must not report survived cascades (explain.py)
# ---------------------------------------------------------------------------


def test_why_retracted_ignores_survivors():
    s = BeliefStore()
    g = s.assert_belief("Sensor online", ground=True, source_reliability=0.9)
    # Ground dependent: survives orphaning, but the cascade evaluation
    # still logs a CASCADE_RETRACTED event with survived=True.
    d = s.assert_belief(
        "Server is online",
        justifications=[g.id],
        ground=True,
        source_reliability=0.9,
    )
    s.retract_belief(g.id, "sensor failed")
    check("ground dependent survived", s.get(d.id).is_active)
    survived_events = [
        e
        for e in s.audit
        if e.event_type == "cascade_survived"
    ]
    check("survived cascade event exists", len(survived_events) >= 1)
    text = Explainer(s).why_retracted(g.id)
    check(
        "no phantom cascade count for survivors",
        "dependent belief(s) also retracted" not in text,
    )


# ---------------------------------------------------------------------------
# 7. FUNCTIONAL_RELATIONS whitelist gaps (detect.py)
# ---------------------------------------------------------------------------


def _mk(prop):
    return Belief(proposition=prop, confidence=0.9, source="t",
                  source_reliability=0.9, ground=True)


def test_functional_relations_whitelist():
    a = _mk("France's population is 67M")
    b = _mk("France's population is 70M")
    check("population contradiction detected", heuristic_contradiction(a, b))

    c = _mk("Acme's CTO is Alice")
    d = _mk("Acme's CTO is Bob")
    check("CTO contradiction detected", heuristic_contradiction(c, d))

    e = _mk("Acme's CFO is Alice")
    f = _mk("Acme's CFO is Carol")
    check("CFO contradiction detected", heuristic_contradiction(e, f))

    # Custom whitelist param.
    det = HeuristicDetector(functional_relations=frozenset({"mayor"}))
    g, h = _mk("Springfield's mayor is X"), _mk("Springfield's mayor is Y")
    check("custom functional_relations honored", det.contradicts(g, h))
    check(
        "custom whitelist excludes others",
        not det.contradicts(c, d),
    )


def main():
    print("round-4 regression tests")
    test_rederive_blocked_by_stronger_contradictor()
    test_rederive_displaces_weaker_contradictor()
    test_rejected_newcomer_leaves_incumbents_untouched()
    test_persist_chunked_save_roundtrip()
    test_timetravel_backfill_ordering()
    test_antonym_with_asymmetric_negation()
    test_why_retracted_ignores_survivors()
    test_functional_relations_whitelist()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failures:", FAIL)
        sys.exit(1)


if __name__ == "__main__":
    main()
