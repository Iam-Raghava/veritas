"""Controlled adjudication benchmark (no API needed).

Tests Veritas's revision machinery: when a contradiction IS detected,
does it correctly keep the stronger claim and retract/reject the weaker?

Uses canonical contradiction patterns the heuristic detector catches,
with claim text inspired by real LongMemEval conversations. No LLM API.

Two directions per pair:
  A: assert STRONG (rel 0.9) then WEAK (rel 0.3) -> WEAK should be rejected
  B: assert WEAK (rel 0.3) then STRONG (rel 0.9) -> WEAK should be retracted

Metric: % pairs where only the strong claim remains active.
Naive baseline: 0% (keeps both).
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, ".")

from veritas import BeliefStore

OUT_PATH = os.path.join(
    os.path.dirname(__file__), "data", "contradiction_controlled_heuristic.json")

# (strong_claim, weak_claim) — canonical contradictions the heuristic
# detector catches (functional relations, direct negation).
PAIRS = [
    ("Acme's CEO is Alice.", "Acme's CEO is Bob."),
    ("The charity 5K is on Saturday.", "The charity 5K is on Sunday."),
    ("My personal best 5K time is 25:50.", "My personal best 5K time is 26:10."),
    ("The car is parked in the garage.", "The car is parked on the street."),
    ("I live in Austin.", "I live in Denver."),
    ("The meeting starts at 3pm.", "The meeting starts at 4pm."),
    ("Her birthday is in June.", "Her birthday is in July."),
    ("The server is in us-east-1.", "The server is in us-west-2."),
    ("I take the medication in the morning.", "I take the medication at night."),
    ("The flight departs from JFK.", "The flight departs from Newark."),
    ("My dog's name is Rex.", "My dog's name is Max."),
    ("The conference is in October.", "The conference is in November."),
    ("She works as an engineer.", "She works as a designer."),
    ("The store opens at 9am.", "The store opens at 10am."),
    ("His favorite color is blue.", "His favorite color is red."),
    ("The book has 300 pages.", "The book has 450 pages."),
    ("I bought the car in 2020.", "I bought the car in 2021."),
    ("The restaurant is on Main Street.", "The restaurant is on Oak Avenue."),
    ("My phone number ends in 1234.", "My phone number ends in 5678."),
    ("The project deadline is Friday.", "The project deadline is Monday."),
]


def test_pair(strong: str, weak: str) -> dict:
    # Direction A: strong first, weak second (weak should be rejected).
    s1 = BeliefStore()
    b_s = s1.assert_belief(strong, source="trusted",
                           source_reliability=0.9, confidence=0.8, ground=True)
    b_w = s1.assert_belief(weak, source="untrusted",
                           source_reliability=0.3, confidence=0.5, ground=True)
    s_active = b_s is not None and s1._beliefs[b_s.id].is_active
    w_active_a = b_w is not None and s1._beliefs[b_w.id].is_active
    correct_a = s_active and not w_active_a

    # Direction B: weak first, strong second (weak should be retracted).
    s2 = BeliefStore()
    b_w2 = s2.assert_belief(weak, source="untrusted",
                            source_reliability=0.3, confidence=0.5, ground=True)
    b_s2 = s2.assert_belief(strong, source="trusted",
                            source_reliability=0.9, confidence=0.8, ground=True)
    s_active2 = b_s2 is not None and s2._beliefs[b_s2.id].is_active
    w_active_b = b_w2 is not None and s2._beliefs[b_w2.id].is_active
    correct_b = s_active2 and not w_active_b

    return {"strong": strong, "weak": weak,
            "dir_a_correct": correct_a, "dir_b_correct": correct_b}


def main():
    results = []
    for strong, weak in PAIRS:
        r = test_pair(strong, weak)
        results.append(r)
        print(f"A:{'OK' if r['dir_a_correct'] else 'XX'} "
              f"B:{'OK' if r['dir_b_correct'] else 'XX'} "
              f" {strong[:40]}", flush=True)

    n = len(results)
    a_ok = sum(1 for r in results if r["dir_a_correct"])
    b_ok = sum(1 for r in results if r["dir_b_correct"])
    both = sum(1 for r in results if r["dir_a_correct"] and r["dir_b_correct"])

    out = {"n_pairs": n, "dir_a": f"{a_ok}/{n}", "dir_b": f"{b_ok}/{n}",
           "both": f"{both}/{n}", "both_accuracy": both / n,
           "naive_accuracy": 0.0, "results": results,
           "honesty": ("Controlled adjudication test. Canonical "
                       "contradictions; tests revision machinery, not "
                       "detection. Naive baseline 0% by construction.")}
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nDirection A (reject weak): {a_ok}/{n} ({a_ok/n:.0%})")
    print(f"Direction B (retract weak): {b_ok}/{n} ({b_ok/n:.0%})")
    print(f"Both: {both}/{n} ({both/n:.0%}) vs Naive 0%")


if __name__ == "__main__":
    main()
