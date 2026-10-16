"""Contradiction detection. Pluggable; ships with a transparent heuristic.

Veritas separates *detection* (do these two beliefs conflict?) from
*retraction* (which one goes?). Detection is pluggable because the right
detector depends on the domain: a cheap heuristic for tests, an NLI model
for production, explicit pairs for curated knowledge.

The default heuristic handles the most common real case in agent memory:
two beliefs assigning *different values* to the same *functional property*
of the same subject ("Acme's CEO is Jane" vs "Acme's CEO is John").
"""
from __future__ import annotations

import functools
import re
from typing import Callable, Protocol, runtime_checkable

from .belief import Belief

ContradictionFn = Callable[[Belief, Belief], bool]


@runtime_checkable
class Detector(Protocol):
    """Plugin interface for contradiction detectors."""

    def contradicts(self, a: Belief, b: Belief) -> bool:
        """True if beliefs a and b cannot both hold."""
        ...


def as_function(detector: Detector | ContradictionFn) -> ContradictionFn:
    """Adapt a Detector (or pass through a plain function).

    Wraps the result in a commutativity harness: contradicts(A, B) is
    defined as Detector(A, B) OR Detector(B, A). This forces symmetry
    across all callers and prevents directional assertion order (or NLI
    model asymmetry) from altering the contradiction graph — which would
    otherwise cause epistemic thrashing in multi-agent deployments.
    """
    base = detector.contradicts if isinstance(detector, Detector) else detector

    def symmetric(a, b) -> bool:
        # Short-circuit: if base(a, b) is True, skip the reverse call.
        # If False, try the reverse — a detector that only fires in one
        # direction still yields a symmetric contradiction.
        return bool(base(a, b)) or bool(base(b, a))

    # Preserve the detector name for index selection.
    symmetric.__name__ = getattr(base, "__name__", "contradiction_fn")
    return symmetric


# ---------------------------------------------------------------------------
# Heuristic detector (default): transparent, no model required.
# ---------------------------------------------------------------------------

_SPOSSESSIVE = re.compile(
    r"^\s*(?P<subj>.+?)'s\s+(?P<rel>.+?)\s+is\s+(?P<obj>.+?)\s*$", re.IGNORECASE
)
_SPLAIN = re.compile(
    r"^\s*(?P<subj>.+?)\s+(?P<rel>CEO|ceo|headquarters|founder|capital|population|subscription|status|plan|tier|owner|manager|lead)\s+is\s+(?P<obj>.+?)\s*$",
    re.IGNORECASE,
)
_SIS = re.compile(r"^\s*(?P<subj>.+?)\s+is\s+(?P<obj>.+?)\s*$", re.IGNORECASE)
_SIS_NOT = re.compile(r"^\s*(?P<subj>.+?)\s+is\s+not\s+(?P<obj>.+?)\s*$", re.IGNORECASE)
_SWILL = re.compile(r"^\s*(?P<subj>.+?)\s+will\s+(?P<obj>.+?)\s*$", re.IGNORECASE)
_SWILL_NOT = re.compile(
    r"^\s*(?P<subj>.+?)\s+will\s+not\s+(?P<obj>.+?)\s*$", re.IGNORECASE
)

# Precompiled: _norm runs on every comparison in dense stores.
_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s.strip().lower().rstrip("."))


@functools.lru_cache(maxsize=65536)
def functional_parts(proposition: str) -> tuple[str, str, str] | None:
    """Extract (subject, relation, object) for functional-property claims.

    Public so the store can build its contradiction index.
    Parsed once per unique proposition; cached for dense stores.

    Returns None for negated objects ("X's CEO is not Bob") — those are
    handled by the negation detector, not as positive functional
    assignments. Otherwise "CEO is Jane" vs "CEO is not Bob" would be
    falsely flagged as contradictory.
    """
    m = _SPOSSESSIVE.match(proposition) or _SPLAIN.match(proposition)
    if m:
        obj = _norm(m.group("obj"))
        # Negated object: not a positive assignment.
        if obj.lower().startswith("not "):
            return None
        return _norm(m.group("subj")), _norm(m.group("rel")), obj
    return None


@functools.lru_cache(maxsize=65536)
def _negation_parts(proposition: str) -> tuple[str, str, str, bool] | None:
    """Extract (subject, object, verb, negated) for 'S is[/not] O' claims."""
    for verb, pos, neg in (("is", _SIS, _SIS_NOT), ("will", _SWILL, _SWILL_NOT)):
        m = neg.match(proposition)
        if m:
            return _norm(m.group("subj")), _norm(m.group("obj")), verb, True
        m = pos.match(proposition)
        if m:
            return _norm(m.group("subj")), _norm(m.group("obj")), verb, False
    return None


def subject_object_parts(proposition: str) -> tuple[str, str] | None:
    """Extract (subject, object) for 'S is O' / 'S will O' claims.

    Used for the negation index: "S is O" vs "S is not O" share a key.
    """
    parsed = _negation_parts(proposition)
    if parsed:
        return parsed[0], parsed[1]
    return None


# Whitelist of relations that are functionally single-valued (a subject
# can have only one). "X's CEO is A" vs "X's CEO is B" is a contradiction.
# But "X's investor is A" vs "X's investor is B" is NOT — investors are
# multi-valued. Without this whitelist, the heuristic treats every
# possessive as functional, causing false positives.
FUNCTIONAL_RELATIONS = frozenset({
    "ceo", "chief executive officer",
    "cto", "chief technology officer",
    "cfo", "chief financial officer",
    "coo", "chief operating officer",
    "chairman", "chairwoman", "chair",
    "capital",  # of a country/state
    "population",  # of a place (at a given time)
    "headquarters", "hq",
    "date_of_birth", "birthday", "born",
    "founder",  # typically single, though debatable
    "spouse", "husband", "wife",  # legally single in most jurisdictions
    "president",  # of a country (at a given time)
    "os", "operating system",  # a device/user has one at a time
    "location", "address", "residence",  # primary location
    "phone", "phone number", "email", "email address",  # contact (primary)
    "age",  # single value
    "gender", "sex",
    "nationality", "citizenship",  # primary
    "occupation", "job", "title", "role",  # primary role
    "company", "employer",  # primary employer
    "school", "university", "college",  # current
    "major",  # primary field of study
    "subscription", "plan", "tier", "status",  # service level (one at a time)
    "owner",  # primary owner
    "manager", "lead",  # primary manager/lead (matches _SPLAIN regex)
    "target", "destination", "deployment target",  # single active target
})

def heuristic_contradiction(
    a: Belief,
    b: Belief,
    functional_relations: frozenset[str] | None = None,
) -> bool:
    """Default detector.

    Fires when:
      1. Either belief explicitly names the other in metadata["contradicts"].
      2. Both assign different objects to the same (subject, relation)
         functional property ("X's CEO is A" vs "X's CEO is B"), where the
         relation is in FUNCTIONAL_RELATIONS.
      3. Direct negation: "S is O" vs "S is not O", "S will O" vs
         "S will not O".
    """
    if a.id == b.id:
        return False

    # 1. Explicit pairs (tests, curated knowledge).
    if b.id in (a.metadata.get("contradicts") or []) or a.id in (
        b.metadata.get("contradicts") or []
    ):
        return True

    pa, pb = functional_parts(a.proposition), functional_parts(b.proposition)

    # 2. Same functional property, different value — but ONLY if the
    # relation is in the functional whitelist. Generic possessives
    # (investor, office, employee) are multi-valued.
    _fr = (FUNCTIONAL_RELATIONS if functional_relations is None
           else functional_relations)
    if (pa and pb and pa[0] == pb[0] and pa[1] == pb[1] and pa[2] != pb[2]
            and pa[1].lower() in _fr):
        return True

    # 3. Direct negation (cached parses; no regex per comparison).
    # Verb must match: "is not" contradicts "is", "will not" vs "will".
    na, nb = _negation_parts(a.proposition), _negation_parts(b.proposition)
    if (na and nb and na[0] == nb[0] and na[1] == nb[1]
            and na[2] == nb[2] and na[3] != nb[3]):
        return True

    return False


class HeuristicDetector:
    """Detector-class wrapper around the default heuristic."""

    name = "heuristic"

    def __init__(
        self,
        functional_relations: frozenset[str] | set[str] | None = None,
    ) -> None:
        """Optionally override the functional-relation whitelist.

        Pass a custom set to treat domain-specific relations as
        single-valued (or empty set to disable functional detection).
        """
        self._functional_relations = (
            FUNCTIONAL_RELATIONS if functional_relations is None
            else frozenset(r.lower() for r in functional_relations)
        )

    def contradicts(self, a: Belief, b: Belief) -> bool:
        return heuristic_contradiction(
            a, b, functional_relations=self._functional_relations,
        )
