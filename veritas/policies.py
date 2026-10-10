"""Pluggable contraction policies for Veritas.

Different belief revision strategies for different use cases:

- **entrenchment** (default): Hansson's non-prioritized base revision.
  Retract weakest-first by entrenchment. Minimal change, principled.
- **conservative**: Reject the newcomer if *any* contradiction exists.
  Never retracts existing beliefs; safest for critical systems.

Usage:
    store = BeliefStore(policy="conservative")

(Future work: a maxichoice variant — retract the minimal-cardinality set
restoring consistency — is not yet implemented and therefore not offered.)
"""

from __future__ import annotations

from typing import Literal

Policy = Literal["entrenchment", "conservative"]

POLICIES = ("entrenchment", "conservative")


def describe_policy(policy: Policy) -> str:
    descriptions = {
        "entrenchment": (
            "Hansson's non-prioritized base revision: order contradictors "
            "by entrenchment (weakest first), retract until consistent. "
            "Rejects newcomers weaker than any contradictor."
        ),
        "conservative": (
            "Reject any newcomer that contradicts existing beliefs. "
            "Existing beliefs are never retracted automatically."
        ),
    }
    return descriptions[policy]
