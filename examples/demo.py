"""Veritas end-to-end demonstration.

Scenario: an agent has been researching Acme Corp for weeks. Its memory
holds the old CEO, facts derived from the old CEO, and unrelated facts.
Then a press release announces a new CEO.

Watch what happens:
  1. The contradiction is detected automatically.
  2. Entrenchment decides: press release outranks the old article.
  3. The old CEO belief is retracted; facts derived from it CASCADE.
  4. Unrelated beliefs survive untouched.
  5. A weak rumor contradicting the press release is REJECTED.
  6. The audit log tells the whole story.

Run: python3 examples/demo.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from veritas import BeliefStore

DAY = 86400


def show(store, title):
    print(f"\n--- {title} ---")
    for b in store.active_beliefs():
        e = store.entrenchment_of(b)
        print(f"  [{b.id}] (entrenchment {e:.3f}) {b.proposition}")
        print(f"           src={b.source} rel={b.source_reliability} "
              f"just={len(b.justifications)}")


def main():
    store = BeliefStore()

    print("=" * 70)
    print("PHASE 1: the agent accumulates beliefs over weeks of research")
    print("=" * 70)

    ceo = store.assert_belief(
        "Acme's CEO is Jane Smith",
        confidence=0.8, source="tech blog", source_reliability=0.65,
    )
    ceo.timestamp -= 45 * DAY  # learned 45 days ago

    founder = store.assert_belief(
        "Jane Smith founded Acme in 2020",
        confidence=0.7, source="derived", source_reliability=0.4,
        justifications=[ceo.id],
    )
    strategy = store.assert_belief(
        "Acme's strategy follows Jane Smith's vision",
        confidence=0.6, source="derived", source_reliability=0.4,
        justifications=[founder.id],
    )
    hq = store.assert_belief(
        "Acme's HQ is Austin",
        confidence=0.95, source="company website", source_reliability=0.9,
    )
    hq.timestamp -= 10 * DAY

    show(store, "memory before the press release")
    print(f"\nconsistent: {store.is_consistent()}")

    print("\n" + "=" * 70)
    print("PHASE 2: press release — Acme's CEO is John Doe")
    print("=" * 70)

    new_ceo = store.assert_belief(
        "Acme's CEO is John Doe",
        confidence=0.98, source="official press release",
        source_reliability=0.97,
    )
    assert new_ceo is not None

    show(store, "memory after the press release")
    print(f"\nconsistent: {store.is_consistent()}")
    s = store.stats()
    print(f"stats: {s['active']} active, {s['retracted']} retracted, "
          f"{s['audit_events']} audit events")

    print("\n" + "=" * 70)
    print("PHASE 3: a weak rumor tries to contradict the press release")
    print("=" * 70)

    rumor = store.assert_belief(
        "Acme's CEO is Jane Smith",
        confidence=0.4, source="social media rumor",
        source_reliability=0.2,
    )
    print(f"rumor accepted? {rumor is not None}  (expected: False — rejected)")
    print(f"consistent: {store.is_consistent()}")

    print("\n" + "=" * 70)
    print("AUDIT TRAIL — every decision, with its reason")
    print("=" * 70)
    for e in store.audit:
        print(f"  [{e.seq:02d}] {e.event_type:16s} | {e.proposition[:52]}")
        print(f"       reason: {e.reason[:100]}")

    # Self-checks: the demo asserts its own correctness.
    assert store.is_consistent()
    active = {b.proposition for b in store.active_beliefs()}
    assert "Acme's CEO is John Doe" in active
    assert "Acme's HQ is Austin" in active
    assert "Acme's CEO is Jane Smith" not in active
    assert "Jane Smith founded Acme in 2020" not in active
    assert "Acme's strategy follows Jane Smith's vision" not in active
    print("\nAll demo assertions passed: contraction, cascade, "
          "survival, and rejection behave as designed.")


if __name__ == "__main__":
    main()
