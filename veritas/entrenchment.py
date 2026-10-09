"""Epistemic entrenchment: how resistant a belief is to retraction.

Entrenchment is the core of Veritas's retraction policy. When new evidence
contradicts old beliefs, the system retracts the *least entrenched* first.
This is Hansson's non-prioritized belief base revision made computable:
give up as little as possible, give up the weakest-held beliefs first,
and reject incoming beliefs that cannot outrank their contradictors.

The score is a transparent weighted combination — no black box. Every
component is in [0, 1] and the weights sum to 1, so entrenchment is in
[0, 1] and comparable across beliefs.
"""
from __future__ import annotations

import math

from .belief import Belief

DEFAULT_WEIGHTS = {
    "source": 0.40,         # how reliable the origin is
    "corroboration": 0.25,  # how many independent beliefs support it
    "confidence": 0.20,     # the agent's own credence
    "recency": 0.15,        # fresher beliefs are harder to dislodge
}

# Beliefs older than this lose half their recency component.
RECENCY_HALFLIFE_DAYS = 30.0

# Corroboration saturates: 0 -> 0.0, 1 -> ~0.33, 3 -> ~0.67, 7+ -> 1.0
_CORROBORATION_SCALE = 3.0


def corroboration_score(n_justifications: int) -> float:
    """Log-scaled corroboration in [0, 1]."""
    if n_justifications <= 0:
        return 0.0
    return min(1.0, math.log2(1 + n_justifications) / _CORROBORATION_SCALE)


def recency_score(belief: Belief, halflife_days: float = RECENCY_HALFLIFE_DAYS) -> float:
    """Exponential decay in (0, 1]; brand-new beliefs score ~1.0."""
    return 0.5 ** (belief.age_days / halflife_days)


def entrenchment(
    belief: Belief,
    weights: dict[str, float] | None = None,
    n_justifications: int | None = None,
) -> float:
    """Compute a belief's entrenchment in [0, 1].

    Higher means harder to retract. `n_justifications` lets callers score
    a belief *as if* it had a different justification count — used by the
    store to compute "independent entrenchment" (would this belief survive
    on its source alone?).
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    n = len(belief.justifications) if n_justifications is None else n_justifications
    score = (
        w["source"] * belief.source_reliability
        + w["corroboration"] * corroboration_score(n)
        + w["confidence"] * belief.confidence
        + w["recency"] * recency_score(belief)
    )
    return max(0.0, min(1.0, score))


def independent_entrenchment(belief: Belief, weights: dict[str, float] | None = None) -> float:
    """Entrenchment with zero justifications: can this belief stand alone?"""
    return entrenchment(belief, weights=weights, n_justifications=0)


def order_by_entrenchment(
    beliefs: list[Belief],
    weights: dict[str, float] | None = None,
    ascending: bool = True,
) -> list[Belief]:
    """Sort beliefs by entrenchment. Ascending = weakest first (retraction order).

    Ties break by belief ID: the order is fully deterministic, independent
    of hash randomization. (Tied beliefs are equivalent for the retraction
    decision; determinism is for reproducibility.)
    """
    return sorted(
        beliefs,
        key=lambda b: (entrenchment(b, weights), b.id),
        reverse=not ascending,
    )
