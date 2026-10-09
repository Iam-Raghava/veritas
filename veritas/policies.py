"""Pluggable contraction policies for Veritas.

Different belief revision strategies for different use cases:

- **entrenchment** (default): Hansson's non-prioritized base revision.
  Retract weakest-first by entrenchment. Minimal change, principled.
- **maxichoice**: Retract the minimal set that restores consistency.
  Keeps maximal beliefs; good when every belief is valuable.
- **conservative**: Reject the newcomer if *any* contradiction exists.
  Never retracts existing beliefs; safest for critical systems.

Usage:
    store = BeliefStore(policy="maxichoice")
"""

from __future__ import annotations

from typing import Literal

Policy = Literal["entrenchment", "maxichoice", "conservative"]

POLICIES = ("entrenchment", "maxichoice", "conservative")


def describe_policy(policy: Policy) -> str:
    descriptions = {
        "entrenchment": (
            "Hansson's non-prioritized base revision: order contradictors "
            "by entrenchment (weakest first), retract until consistent. "
            "Rejects newcomers weaker than any contradictor."
        ),
        "maxichoice": (
            "Retract the minimal cardinality set restoring consistency. "
            "Prefers keeping existing beliefs over the newcomer when tied."
        ),
        "conservative": (
            "Reject any newcomer that contradicts existing beliefs. "
            "Existing beliefs are never retracted automatically."
        ),
    }
    return descriptions[policy]
