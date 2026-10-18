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

    def contradicts_batch(
        self, belief: Belief, candidates: list[Belief]
    ) -> list[Belief]:
        """Batch contradiction check: return the subset of candidates
        that contradict `belief`.

        Default implementation loops over `contradicts`. Detectors backed
        by neural models (HuggingFace NLI, LLM judges) SHOULD override this
        to run a single batched forward pass / single prompt instead of N
        sequential calls — 10-20x faster and cheaper.

        The store calls this OUTSIDE the global lock when available, so
        slow detectors don't block concurrent readers.
        """
        return [c for c in candidates if self.contradicts(belief, c)]


def as_function(detector: Detector | ContradictionFn) -> ContradictionFn:
    """Adapt a Detector (or pass through a plain function).

    Wraps the result in a commutativity harness: contradicts(A, B) is
    defined as Detector(A, B) OR Detector(B, A). This forces symmetry
    across all callers and prevents directional assertion order (or NLI
    model asymmetry) from altering the contradiction graph — which would
    otherwise cause epistemic thrashing in multi-agent deployments.
    """
    # Use hasattr, not isinstance(Detector): the Protocol now includes
    # contradicts_batch, and detectors that don't override it (e.g.,
    # HeuristicDetector) would fail the runtime_checkable isinstance.
    base = detector.contradicts if hasattr(detector, "contradicts") else detector

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
# Possessive adjectives: "My X is Y", "His X is Y", etc.
_SPOSSESSIVE_ADJ = re.compile(
    r"^\s*(?P<subj>my|his|her|their|our|your)\s+(?P<rel>.+?)\s+is\s+(?P<obj>.+?)\s*$",
    re.IGNORECASE,
)
# _SPLAIN is built dynamically from FUNCTIONAL_RELATIONS below (after
# the frozenset is defined). See _build_splain().
_SPLAIN = None  # type: ignore[assignment]

# Location patterns: "X is in Y", "X is located in Y", "X moved to Y",
# "X lives in Y", "X is on Y" (streets), "X departs from Y".
# These are functional (a subject has one location at a time).
_LOCATION_IN = re.compile(
    r"^\s*(?P<subj>.+?)\s+is\s+(?:located\s+)?in\s+(?P<obj>.+?)\s*$",
    re.IGNORECASE,
)
_LIVE_IN = re.compile(
    r"^\s*(?P<subj>.+?)\s+lives?\s+in\s+(?P<obj>.+?)\s*$",
    re.IGNORECASE,
)
_IS_ON = re.compile(
    r"^\s*(?P<subj>.+?)\s+is\s+on\s+(?P<obj>.+?)\s*$",
    re.IGNORECASE,
)
_MOVED_TO = re.compile(
    r"^\s*(?P<subj>.+?)\s+moved\s+to\s+(?P<obj>.+?)\s*$",
    re.IGNORECASE,
)
_DEPARTS_FROM = re.compile(
    r"^\s*(?P<subj>.+?)\s+departs?\s+from\s+(?P<obj>.+?)\s*$",
    re.IGNORECASE,
)
# Occupation: "X works as Y" (functional: primary occupation).
_WORKS_AS = re.compile(
    r"^\s*(?P<subj>.+?)\s+works?\s+as\s+(?:an?\s+)?(?P<obj>.+?)\s*$",
    re.IGNORECASE,
)
# Temporal functional: "X starts at Y", "X begins at Y", "X opens at Y",
# "X closes at Y". An event has one start/open time.
_STARTS_AT = re.compile(
    r"^\s*(?P<subj>.+?)\s+(?:starts?|begins?)\s+at\s+(?P<obj>.+?)\s*$",
    re.IGNORECASE,
)
_OPENS_AT = re.compile(
    r"^\s*(?P<subj>.+?)\s+(?:opens?|closes?)\s+at\s+(?P<obj>.+?)\s*$",
    re.IGNORECASE,
)
# Location via parking: "X is parked in/on Y".
_PARKED_IN = re.compile(
    r"^\s*(?P<subj>.+?)\s+(?:is\s+)?parked\s+(?:in|on)\s+(?P<obj>.+?)\s*$",
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

    Subject and relation are canonicalized (via governor.canon_entity
    and governor.canon_relation) so "Acme Corp's chief executive is Jane"
    matches "Acme's CEO is John" in the O(1) heuristic index.
    """
    # Local import to avoid circular dependency (governor imports detect).
    from .governor import canon_entity, canon_relation
    m = _SPOSSESSIVE.match(proposition) or _SPLAIN.match(proposition)
    if m:
        obj = _norm(m.group("obj"))
        # Negated object: not a positive assignment.
        if obj.lower().startswith("not "):
            return None
        subj = canon_entity(_norm(m.group("subj")))
        rel = canon_relation(_norm(m.group("rel")))
        return subj, rel, obj
    # Location patterns: "X is in Y", "X lives in Y", "X is on Y",
    # "X moved to Y", "X departs from Y".
    # These are functional (location is single-valued at a time).
    m_loc = (
        _LOCATION_IN.match(proposition)
        or _LIVE_IN.match(proposition)
        or _IS_ON.match(proposition)
        or _MOVED_TO.match(proposition)
        or _DEPARTS_FROM.match(proposition)
    )
    if m_loc:
        obj = _norm(m_loc.group("obj"))
        if obj.lower().startswith("not "):
            return None
        subj = canon_entity(_norm(m_loc.group("subj")))
        # Canonical relation for location.
        return subj, "location", obj
    # Occupation: "X works as Y" -> (X, occupation, Y).
    m_works = _WORKS_AS.match(proposition)
    if m_works:
        obj = _norm(m_works.group("obj"))
        if obj.lower().startswith("not "):
            return None
        subj = canon_entity(_norm(m_works.group("subj")))
        return subj, "occupation", obj
    # Temporal: "X starts/begins at Y" -> (X, start_time, Y).
    m_starts = _STARTS_AT.match(proposition)
    if m_starts:
        obj = _norm(m_starts.group("obj"))
        if obj.lower().startswith("not "):
            return None
        subj = canon_entity(_norm(m_starts.group("subj")))
        return subj, "start_time", obj
    # Temporal: "X opens/closes at Y" -> (X, open_time/close_time, Y).
    m_opens = _OPENS_AT.match(proposition)
    if m_opens:
        obj = _norm(m_opens.group("obj"))
        if obj.lower().startswith("not "):
            return None
        subj = canon_entity(_norm(m_opens.group("subj")))
        # Distinguish open vs close by the verb.
        verb = "open_time" if "open" in proposition.lower() else "close_time"
        return subj, verb, obj
    # Location: "X is parked in/on Y" -> (X, location, Y).
    m_parked = _PARKED_IN.match(proposition)
    if m_parked:
        obj = _norm(m_parked.group("obj"))
        if obj.lower().startswith("not "):
            return None
        subj = canon_entity(_norm(m_parked.group("subj")))
        return subj, "location", obj
    # Possessive adjectives: "My X is Y", "His X is Y", "Her X is Y",
    # "Their X is Y", "Our X is Y". Treat like "X's" possessive.
    m_poss_adj = _SPOSSESSIVE_ADJ.match(proposition)
    if m_poss_adj:
        obj = _norm(m_poss_adj.group("obj"))
        if obj.lower().startswith("not "):
            return None
        # "my" -> "speaker", "his"/"her" -> keep as is for matching.
        subj_raw = _norm(m_poss_adj.group("subj"))
        subj = canon_entity(subj_raw)
        rel = canon_relation(_norm(m_poss_adj.group("rel")))
        return subj, rel, obj
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
    "name",  # a subject has one name at a time
    "color", "favorite color",  # single favorite
    "personal best", "best time", "record",  # single best at a time
    "deadline", "due date",  # single deadline
    "start_time", "open_time", "close_time",  # event times (functional)
})


def _build_splain() -> "re.Pattern":
    """Build _SPLAIN regex dynamically from FUNCTIONAL_RELATIONS.

    Sorts longest-first so multi-word relations ("chief executive officer")
    match before their prefixes. Uses re.escape for safety.
    """
    import re
    # Sort by length descending, then alphabetically for determinism.
    sorted_rels = sorted(FUNCTIONAL_RELATIONS, key=lambda r: (-len(r), r))
    alternation = "|".join(re.escape(r) for r in sorted_rels)
    return re.compile(
        rf"^\s*(?P<subj>.+?)\s+(?P<rel>{alternation})\s+is\s+(?P<obj>.+?)\s*$",
        re.IGNORECASE,
    )


_SPLAIN = _build_splain()

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
    # relation is in the functional whitelist (or contains a whitelisted
    # phrase). Generic possessives (investor, office, employee) are
    # multi-valued.
    _fr = (FUNCTIONAL_RELATIONS if functional_relations is None
           else functional_relations)
    def _is_functional(rel: str) -> bool:
        r = rel.lower()
        if r in _fr:
            return True
        # Substring match: "personal best 5k time" contains "personal best".
        # Use word boundaries to avoid partial matches.
        for f in _fr:
            # Skip very short terms to avoid false positives.
            if len(f) < 4:
                continue
            if re.search(r"\b" + re.escape(f) + r"\b", r):
                return True
        return False
    if (pa and pb and pa[0] == pb[0] and pa[1] == pb[1] and pa[2] != pb[2]
            and _is_functional(pa[1])):
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
