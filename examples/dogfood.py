"""Dogfood: Veritas audits Muse's own long-term memory.

Reads ~/MEMORY.md, loads every recorded fact/preference/commitment as a
belief (source="MEMORY.md"), and checks the whole set for contradictions.
Then demonstrates the correction workflow: when the user corrects a fact,
the old belief is retracted through the principled path, not silently
overwritten.

Run: python3 examples/dogfood.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from veritas import BeliefStore

MEMORY_PATH = os.path.expanduser("~/MEMORY.md")
SECTIONS = ("## Facts", "## Preferences", "## Boundaries", "## Commitments")


def extract_propositions(path: str) -> list[tuple[str, str]]:
    """Pull (section, proposition) pairs from MEMORY.md bullet lines."""
    out: list[tuple[str, str]] = []
    section = ""
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("## "):
                section = line.strip()
            elif line.strip().startswith("- ") and section in SECTIONS:
                prop = line.strip()[2:].strip()
                # Strip parenthetical citations for cleaner propositions.
                prop = re.sub(r"\s*\([^()]*\d{4}[^()]*\)\.?\s*$", "", prop).strip()
                if len(prop) > 20:
                    out.append((section, prop))
    return out


def main():
    store = BeliefStore()
    props = extract_propositions(MEMORY_PATH)
    print(f"extracted {len(props)} propositions from MEMORY.md")

    for section, prop in props:
        store.assert_belief(
            prop,
            confidence=0.85,
            source=f"MEMORY.md {section}",
            source_reliability=0.9,
        )

    pairs = store.find_contradictions()
    print(f"active beliefs: {len(store.active_beliefs())}")
    print(f"contradictions found: {len(pairs)}")
    for a, b in pairs:
        print(f"  CONFLICT:\n    [{a.id}] {a.proposition[:80]}\n"
              f"    [{b.id}] {b.proposition[:80]}")

    # Correction workflow (synthetic example, clearly labeled):
    # the user corrects a recorded preference.
    print("\n--- synthetic correction demo ---")
    old = store.assert_belief(
        "The user's timezone is America/Chicago",
        confidence=0.7, source="inferred", source_reliability=0.5,
    )
    new = store.assert_belief(
        "The user's timezone is not America/Chicago",
        confidence=0.95, source="user correction", source_reliability=0.99,
    )
    print(f"correction accepted: {new is not None}")
    print(f"old belief active: {store.get(old.id).is_active} (expected False)")
    print(f"store consistent: {store.is_consistent()}")
    assert (new is not None and not store.get(old.id).is_active
            and store.is_consistent())
    print("dogfood complete: memory audited, correction workflow verified.")


if __name__ == "__main__":
    main()
