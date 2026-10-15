"""The skeptic demo: why an agent's memory rots without Veritas.

Run:  python examples/skeptic_demo.py

Simulates 6 weeks of an AI assistant tracking a startup. The same
sequence of facts is fed to two memories:

  1. A plain dict (what most agents do today: append everything).
  2. A Veritas BeliefStore.

Watch what happens to contradictions and to conclusions derived
from retracted facts ("zombie beliefs").
"""

from veritas import BeliefStore

# ---------------------------------------------------------------------------
# The scenario: 6 weeks of facts an assistant learns about "Acme Corp".
# Each fact: (week, proposition, source, reliability, kind)
# ---------------------------------------------------------------------------
FACTS = [
    # Week 1: solid ground truth from the board meeting.
    (1, "Acme's CEO is Alice", "board_meeting", 0.7, "ground"),
    # Week 2: the assistant derives a conclusion from the CEO fact.
    (2, "Alice approved the Q4 budget", "finance_doc", 0.8, "derived:Acme's CEO is Alice"),
    # Week 3: a stronger source names a new CEO. Alice is retracted;
    # the budget conclusion loses its premise and is cascade-retracted.
    (3, "Acme's CEO is Bob", "press_release", 0.85, "ground"),
    # Week 4: the assistant builds on the new CEO fact.
    (4, "Bob hired a new CFO", "blog_post", 0.5, "derived:Acme's CEO is Bob"),
    # Week 5: the board corrects the record again. Bob is retracted;
    # the CFO conclusion is cascade-retracted with it.
    (5, "Acme's CEO is Carol", "board_meeting", 0.95, "ground"),
    # Week 6: stale re-assertion of an old truth. Rejected: Carol stands.
    (6, "Acme's CEO is Alice", "old_newsletter", 0.4, "ground"),
]


def run_naive():
    """What most agents do: a dict. Last write wins nothing; everything stays."""
    memory = {}  # proposition -> (week, source)
    derived = {}  # conclusion -> premise proposition
    for week, prop, source, _rel, kind in FACTS:
        if kind.startswith("derived:"):
            premise = kind.split(":", 1)[1]
            derived[prop] = premise
        memory[prop] = (week, source)
    return memory, derived


def run_veritas():
    store = BeliefStore()
    premise_ids = {}  # proposition -> belief id (for justifications)
    for week, prop, source, rel, kind in FACTS:
        if kind.startswith("derived:"):
            premise_prop = kind.split(":", 1)[1]
            premise_id = premise_ids.get(premise_prop)
            just = [premise_id] if premise_id else None
            b = store.assert_belief(
                prop, source=source, source_reliability=rel,
                justifications=just,
            )
        else:
            b = store.assert_belief(
                prop, source=source, source_reliability=rel, ground=True,
            )
        if b is not None:
            premise_ids[prop] = b.id
    return store


def main():
    print("=" * 70)
    print("THE SKEPTIC DEMO: 6 weeks, one startup, two memories")
    print("=" * 70)

    # --- Naive memory ------------------------------------------------------
    memory, derived = run_naive()
    print("\n--- WITHOUT Veritas (plain dict: append everything) ---\n")
    ceos = [p for p in memory if "CEO is" in p]
    print(f"CEO facts stored: {len(ceos)}")
    for p in ceos:
        week, source = memory[p]
        print(f"  week {week}: {p!r}  (from {source})")
    zombies = [c for c, prem in derived.items() if prem in ceos and c in memory]
    print(f"\nDerived conclusions still stored: {len(derived)}")
    for c in derived:
        prem = derived[c]
        print(f"  {c!r}  (derived from {prem!r})")
    print("\nProblem: the agent simultaneously believes Alice, Bob, AND Carol")
    print("are CEO. 'Bob hired a new CFO' survives even though Bob was a rumor.")
    print("Nothing was ever questioned. Nothing was ever retracted.")

    # --- Veritas memory ----------------------------------------------------
    store = run_veritas()
    print("\n--- WITH Veritas (BeliefStore) ---\n")
    active = store.active_beliefs()
    ceo_props = sorted(set(
        b.proposition for b in active if "CEO is" in b.proposition
    ))
    print(f"CEO claims active: {len(ceo_props)}")
    for p in ceo_props:
        srcs = sorted(set(
            b.source for b in active if b.proposition == p
        ))
        print(f"  {p!r}  (from {', '.join(srcs)})")
    derived_v = [b for b in active if b.proposition in derived]
    print(f"\nDerived conclusions still active: {len(derived_v)}")
    for b in derived_v:
        print(f"  {b.proposition!r}")
    print("\nWhat happened, from the audit trail:")
    for e in store.audit:
        if e.event_type in ("rejected", "retracted", "cascade_retracted"):
            print(f"  [{e.event_type}] {e.proposition!r} — {e.reason}")

    # --- Scoreboard ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("SCOREBOARD")
    print("=" * 70)
    naive_ok = len(ceos) == 1 and not zombies
    veritas_ok = len(ceo_props) == 1 and store.is_consistent()
    print(f"  Naive memory consistent:  {naive_ok}  ({len(ceos)} CEOs, {len(zombies)} zombie conclusions)")
    print(f"  Veritas memory consistent: {veritas_ok}  ({len(ceo_props)} CEO, {len(derived_v)} derived)")
    print()
    if veritas_ok and not naive_ok:
        print("The naive agent is confidently wrong about three CEOs at once.")
        print("Veritas kept exactly one — the best-evidenced one — and can")
        print("prove why, from the audit log.")
    print("=" * 70)


if __name__ == "__main__":
    main()
