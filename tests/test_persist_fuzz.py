"""Persistence fuzz for Veritas.

Builds random stores (contradictions, cascades, retractions, unicode,
weird metadata), saves to SQLite, loads back, and verifies the loaded
store is IDENTICAL: every belief field, every justification, every
audit event, and the same query results.

Run: python3 tests/test_persist_fuzz.py [--trials N] [--seed S]
"""
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from veritas import BeliefStore, load, save  # noqa: E402

PROPS = [
    "Acme's CEO is Alice", "Acme's CEO is Bob", "Globex's HQ is Austin",
    "Initech is funded", "日本語の信念", "émojis 🎉 here",
    "null\x00byte?", "x" * 5000, "  padded  ", "UPPER vs lower",
    "O'Brien's \"quoted\" claim", "tab\there", "newline\nthere",
]


def random_store(rng: random.Random) -> BeliefStore:
    s = BeliefStore()
    ids: list[str] = []
    for _ in range(rng.randint(5, 60)):
        justs = [rng.choice(ids) for _ in range(rng.randint(0, 2))] if ids else []
        meta = {}
        if rng.random() < 0.3:
            meta = {"k": rng.choice(["v", 123, 4.5, True, None, ["a", 1]]),
                    "uni": "日本語"}
        b = s.assert_belief(
            rng.choice(PROPS) + f" #{rng.randint(0, 99999)}",
            confidence=rng.random(),
            source=rng.choice(["blog", "wiki", "", "x" * 200]),
            source_reliability=rng.random(),
            justifications=justs,
            metadata=meta,
        )
        if b is not None:
            ids.append(b.id)
    # Random retractions to exercise inactive beliefs + cascades.
    for _ in range(rng.randint(0, 10)):
        act = s.active_beliefs()
        if act:
            s.retract_belief(rng.choice(act).id, reason="fuzz")
    return s


def beliefs_equal(a, b) -> str | None:
    for f in ("id", "proposition", "confidence", "source",
              "source_reliability", "status"):
        if getattr(a, f) != getattr(b, f):
            return f"field {f}: {getattr(a, f)!r} != {getattr(b, f)!r}"
    if sorted(a.justifications) != sorted(b.justifications):
        return f"justifications: {a.justifications} != {b.justifications}"
    if a.metadata != b.metadata:
        return f"metadata: {a.metadata} != {b.metadata}"
    # Timestamps: allow small float serialization tolerance.
    if abs(a.timestamp - b.timestamp) > 0.01:
        return f"created_at drift: {a.timestamp} vs {b.timestamp}"
    return None


def check_roundtrip(seed: int, trial: int) -> None:
    rng = random.Random(seed)
    s1 = random_store(rng)
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        save(s1, path)
        # Save twice: idempotent, no duplication.
        save(s1, path)
        s2 = load(path)

        b1 = {b.id: b for b in s1.all_beliefs()}
        b2 = {b.id: b for b in s2.all_beliefs()}
        assert set(b1) == set(b2), \
            f"belief id sets differ: {set(b1) ^ set(b2)}"
        for bid in b1:
            diff = beliefs_equal(b1[bid], b2[bid])
            assert diff is None, f"belief {bid}: {diff}"

        # Audit events identical.
        a1 = [(e.event_type, e.belief_id, e.proposition, e.reason)
              for e in s1.audit]
        a2 = [(e.event_type, e.belief_id, e.proposition, e.reason)
              for e in s2.audit]
        assert a1 == a2, f"audit diverged ({len(a1)} vs {len(a2)} events)"

        # Query results identical.
        assert s1.is_consistent() == s2.is_consistent()
        assert len(s1.active_beliefs()) == len(s2.active_beliefs())
        assert s1.stats()["consistent"] == s2.stats()["consistent"]

        # The loaded store is fully functional: assert + contract.
        probe = s2.assert_belief(
            f"Zzzqy unique probe {trial} does not contradict anything",
            source="x", source_reliability=0.99)
        assert probe is not None
        assert s2.is_consistent()
    finally:
        os.unlink(path)


def main():
    trials = 30
    seed0 = 777
    if "--trials" in sys.argv:
        trials = int(sys.argv[sys.argv.index("--trials") + 1])
    if "--seed" in sys.argv:
        seed0 = int(sys.argv[sys.argv.index("--seed") + 1])
    print(f"persistence fuzz: {trials} trials...", flush=True)
    for i in range(trials):
        check_roundtrip(seed0 + i, i)
    print(f"ALL {trials} ROUND-TRIPS IDENTICAL")


if __name__ == "__main__":
    main()
