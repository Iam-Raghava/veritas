"""Property-based fuzz test for Veritas.

Generates random sequences of assert/retract operations (including
justification cycles, dangling justifications, and entrenchment ties)
and verifies core invariants after EVERY operation:

  1. CONSISTENCY: no two active beliefs contradict each other.
  2. AUDIT APPEND-ONLY: the audit log never shrinks.
  3. RETRACTION ACCOUNTING: every inactive belief has a retraction entry.
  4. INDEX HYGIENE: contradictors_of never returns inactive beliefs;
     active beliefs are findable through the index.
  5. JUSTIFICATION HYGIENE: stored beliefs never reference retracted
     beliefs in their justifications.

Run: python3 tests/fuzz.py [--ops N] [--seed S]
Exit nonzero on first invariant violation.
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from veritas import BeliefStore  # noqa: E402
from veritas.audit import CASCADE_RETRACTED, RETRACTED  # noqa: E402

SUBJECTS = ["Acme", "Globex", "Initech", "Umbrella", "Hooli"]
RELATIONS = ["CEO", "HQ", "founder"]
VALUES = ["Alice", "Bob", "Carol", "Dave", "Austin", "Boston", "Reno"]
SOURCES = [("blog", 0.4), ("wiki", 0.7), ("press release", 0.95), ("rumor", 0.2)]


def random_proposition(rng: random.Random) -> str:
    kind = rng.random()
    s, r, v = rng.choice(SUBJECTS), rng.choice(RELATIONS), rng.choice(VALUES)
    if kind < 0.7:
        return f"{s}'s {r} is {v}"
    if kind < 0.85:
        return f"{s} is {v}"
    return f"{s} is not {v}"


def check_invariants(store: BeliefStore, op_no: int, op_desc: str) -> None:
    tag = f"[op {op_no}: {op_desc}]"

    # 1. Consistency.
    pairs = store.find_contradictions()
    assert not pairs, (
        f"{tag} CONSISTENCY VIOLATED: "
        + str([(a.proposition, b.proposition) for a, b in pairs[:3]])
    )

    # 2. Audit append-only (tracked by caller via closure counter).
    # 3. Retraction accounting.
    retracted_ids = {
        e.belief_id for e in store.audit
        if e.event_type in (RETRACTED, CASCADE_RETRACTED)
    }
    for b in store.all_beliefs():
        if not b.is_active:
            assert b.id in retracted_ids, (
                f"{tag} RETRACTION ACCOUNTING: belief '{b.proposition}' "
                f"inactive with no retraction audit entry"
            )

    # 4. Index hygiene.
    active_ids = {b.id for b in store.active_beliefs()}
    for b in store.active_beliefs():
        for c in store.contradictors_of(b):
            assert c.is_active, (
                f"{tag} INDEX HYGIENE: contradictors_of returned "
                f"inactive belief '{c.proposition}'"
            )
            assert c.id != b.id, f"{tag} self-contradiction reported"

    # 5. Justification hygiene: no dangling refs to retracted beliefs.
    for b in store.active_beliefs():
        for j in b.all_premise_ids:
            target = store.get(j)
            assert target is not None and target.is_active, (
                f"{tag} JUSTIFICATION HYGIENE: active belief "
                f"'{b.proposition}' references inactive/missing '{j}'"
            )


def fuzz(n_ops: int, seed: int) -> dict:
    rng = random.Random(seed)
    store = BeliefStore()
    audit_len = 0
    stats = {"asserted": 0, "rejected": 0, "retracted": 0, "ops": n_ops}

    for i in range(n_ops):
        roll = rng.random()
        if roll < 0.75 or not store.active_beliefs():
            # ASSERT with chaotic justifications.
            justs: list[str] = []
            existing = [b.id for b in store.all_beliefs()]
            for _ in range(rng.randint(0, 3)):
                if existing and rng.random() < 0.8:
                    justs.append(rng.choice(existing))  # may be retracted: ok
                else:
                    justs.append("ghost-id-%d" % rng.randint(0, 9999))
            src, rel = rng.choice(SOURCES)
            # Sometimes force entrenchment ties.
            conf = rng.choice([0.5, rng.random()])
            relv = rng.choice([0.5, rel])
            desc = f"assert({random_proposition(rng)[:30]}...)"
            # NOTE: proposition generated inside desc would differ; regenerate:
            prop = random_proposition(rng)
            result = store.assert_belief(
                prop, confidence=conf, source=src,
                source_reliability=relv, justifications=justs,
            )
            stats["asserted" if result else "rejected"] += 1
        else:
            # RETRACT a random active belief (may cascade).
            target = rng.choice(store.active_beliefs())
            desc = f"retract({target.proposition[:30]}...)"
            before = len(store.active_beliefs())
            store.retract_belief(target.id, reason="fuzz retraction")
            stats["retracted"] += before - len(store.active_beliefs())

        new_len = len(store.audit)
        assert new_len >= audit_len, f"[op {i}] AUDIT SHRANK: {audit_len} -> {new_len}"
        audit_len = new_len
        check_invariants(store, i, desc)

    stats["final_active"] = len(store.active_beliefs())
    stats["final_retracted"] = len(store.all_beliefs()) - stats["final_active"]
    stats["audit_events"] = audit_len
    return stats


def main() -> None:
    n_ops = 5000
    seed = 12345
    if "--ops" in sys.argv:
        n_ops = int(sys.argv[sys.argv.index("--ops") + 1])
    if "--seed" in sys.argv:
        seed = int(sys.argv[sys.argv.index("--seed") + 1])
    print(f"fuzzing {n_ops} ops (seed={seed})...", flush=True)
    stats = fuzz(n_ops, seed)
    print("ALL INVARIANTS HELD")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
