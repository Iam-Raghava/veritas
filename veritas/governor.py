"""MemoryGovernor: the ingestion pipeline that makes Veritas drop-in.

Raw agent/tool output (unstructured text) flows through four stages:

    Raw Text ──► [1. Claim & Provenance Extractor]
              ──► [2. Automatic Entity & Premise Linker]
              ──► [3. Hybrid Candidate Retriever]
              ──► [4. Batched NLI / LLM Adjudicator + Veritas TMS Core]

Before the governor, developers had to manually spoon-feed atomic strings,
manually look up parent UUIDs, and manually pass justifications=[[parent.id]].
The governor automates all of it:

- Extracts atomic claims from unstructured text (via ClaimExtractor).
- Resolves each claim's ``premises_referred`` substrings to live belief IDs
  (fuzzy proposition match + entity overlap) and passes them as JTMS
  justifications — so the cascade engine actually fires in real workloads.
- Detects when a new observation corroborates an existing active belief
  (handled by the store's duplicate-linking) or re-activates a retracted
  belief's lineage (so ``_try_rederive_dependents`` fires).
- Canonicalizes entities and relations ("Acme Corp" → "acme",
  "chief executive" → "ceo") so surface-form variants link correctly.

Usage:
    store = BeliefStore()
    gov = MemoryGovernor(store)
    results = gov.ingest_text(
        "Acme's CEO is Jane. She was appointed in January 2024. "
        "Revenue grew because sales increased.",
        source="agent",
        source_reliability=0.8,
    )
    for claim, belief in results:
        print(claim.proposition, "->", belief.id if belief else "REJECTED")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .belief import Belief
from .detect import _norm
from .extractor import ClaimExtractor, ExtractedClaim, HeuristicExtractor
from .store import BeliefStore

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer 2: Canonical entity & relation normalization
# ---------------------------------------------------------------------------

# Common entity aliases: "Acme Corp", "Acme Inc." -> "acme".
# Stripped of corporate suffixes, punctuation, and case.
_ENTITY_SUFFIXES = re.compile(
    r"\b(inc|llc|ltd|corp|corporation|company|co|group|holdings|"
    r"technologies|tech|systems|solutions|services|partners)\b\.?",
    re.IGNORECASE,
)

# Relation synonyms mapped to canonical form.
_RELATION_SYNONYMS: dict[str, str] = {
    "chief executive": "ceo",
    "chief executive officer": "ceo",
    "head": "ceo",
    "boss": "ceo",
    "chief financial officer": "cfo",
    "chief operating officer": "coo",
    "chief technology officer": "cto",
    "wife": "spouse",
    "husband": "spouse",
    "married to": "spouse",
    "located in": "location",
    "based in": "location",
    "headquartered in": "location",
    "works at": "employer",
    "employed by": "employer",
    "founded": "founder",
    "founded by": "founder",
    "created by": "founder",
}


def canon_entity(text: str) -> str:
    """Canonicalize an entity mention: 'Acme Corp' -> 'acme'."""
    t = text.strip().lower()
    t = _ENTITY_SUFFIXES.sub("", t)
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def canon_relation(text: str) -> str:
    """Canonicalize a relation: 'chief executive' -> 'ceo'."""
    t = text.strip().lower()
    t = re.sub(r"\s+", " ", t)
    return _RELATION_SYNONYMS.get(t, t)


def entities_of(proposition: str) -> set[str]:
    """Extract canonical entity mentions from a proposition.

    Uses the functional/subject-object parsers plus a fallback that
    treats capitalized word sequences as candidate entities.
    """
    from .detect import functional_parts, subject_object_parts

    entities: set[str] = set()
    fp = functional_parts(proposition)
    if fp:
        entities.add(canon_entity(fp[0]))
        # The object may also be an entity ("CEO is Jane Smith").
        obj = canon_entity(fp[2])
        if obj and len(obj) > 1:
            entities.add(obj)
    else:
        sp = subject_object_parts(proposition)
        if sp:
            entities.add(canon_entity(sp[0]))
            obj = canon_entity(sp[1])
            if obj and len(obj) > 1:
                entities.add(obj)
    # Fallback: capitalized sequences ("Jane Smith", "New York").
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", proposition):
        ent = canon_entity(m.group(1))
        if ent and len(ent) > 1:
            entities.add(ent)
    entities.discard("")
    return entities


# ---------------------------------------------------------------------------
# Ingestion result
# ---------------------------------------------------------------------------

@dataclass
class IngestedClaim:
    """One claim's journey through the governor."""
    claim: ExtractedClaim
    belief: Belief | None  # None if rejected by entrenchment
    linked_premise_ids: list[str] = field(default_factory=list)
    reactivated: bool = False  # True if this re-activated a retracted lineage


# ---------------------------------------------------------------------------
# MemoryGovernor
# ---------------------------------------------------------------------------

class MemoryGovernor:
    """Drop-in epistemic memory engine for agents.

    Wraps a BeliefStore with automatic claim extraction, premise linking,
    and corroboration handling. The agent just calls ``ingest_text`` with
    raw output; the governor handles the JTMS bookkeeping.
    """

    def __init__(
        self,
        store: BeliefStore,
        extractor: ClaimExtractor | None = None,
    ) -> None:
        self.store = store
        self.extractor = extractor or HeuristicExtractor()

    # -- public API ------------------------------------------------------

    def ingest_text(
        self,
        text: str,
        source: str = "agent",
        source_reliability: float = 0.5,
        ground: bool = True,
        **assert_kwargs,
    ) -> list[IngestedClaim]:
        """Ingest raw agent/tool text into the store.

        Extracts atomic claims, links each claim's premises to live belief
        IDs, and asserts with full JTMS justifications. Returns one
        IngestedClaim per extracted claim.
        """
        claims = self.extractor.extract_claims(text, source=source)
        return self.ingest_claims(
            claims,
            source_reliability=source_reliability,
            ground=ground,
            **assert_kwargs,
        )

    def ingest_claims(
        self,
        claims: list[ExtractedClaim],
        source_reliability: float = 0.5,
        ground: bool = True,
        **assert_kwargs,
    ) -> list[IngestedClaim]:
        """Ingest pre-extracted claims (skips the extractor).

        Claims are ordered: ground claims (no premises_referred) are
        asserted BEFORE derived claims (with premises_referred), so
        intra-paragraph premises resolve to the newly created belief IDs.
        E.g., "Revenue grew because sales increased": "Sales increased"
        (ground) is asserted first, then "Revenue grew" (derived) can
        link to it.
        """
        # Sort: ground first (no premises), then derived.
        # Stable sort preserves original order within each group.
        # Track original indices to return results in original order.
        indexed = list(enumerate(claims))
        indexed.sort(key=lambda pair: 1 if pair[1].premises_referred else 0)
        ingested_by_idx: dict[int, IngestedClaim] = {}
        for orig_idx, claim in indexed:
            ingested = self._ingest_one(
                claim,
                source_reliability=source_reliability,
                ground=ground,
                **assert_kwargs,
            )
            ingested_by_idx[orig_idx] = ingested
        # Return in original claim order.
        return [ingested_by_idx[i] for i in range(len(claims))]

    # -- internals -------------------------------------------------------

    def _ingest_one(
        self,
        claim: ExtractedClaim,
        source_reliability: float,
        ground: bool,
        **assert_kwargs,
    ) -> IngestedClaim:
        # Layer 2: link premises_referred substrings to live belief IDs.
        linked = self._link_premises(claim)

        # Check for re-activation: does a RETRACTED belief hold this
        # proposition? If so, re-asserting re-activates the lineage and
        # lets _try_rederive_dependents fire.
        reactivated = self._check_reactivation(claim)

        belief = self.store.assert_belief(
            claim.proposition,
            confidence=claim.confidence,
            source=claim.source,
            source_reliability=source_reliability,
            justifications=linked if linked else None,
            ground=ground if not linked else False,
            # Derived (has premises) unless it's a pure observation.
            valid_from=claim.valid_from,
            valid_until=claim.valid_until,
            metadata={"extracted": True, **assert_kwargs.pop("metadata", {})},
            # Wire reactivation: if a retracted belief holds this proposition,
            # reuse its ID so dependents revive via _try_rederive_dependents.
            reuse_retracted_id=reactivated,
            **assert_kwargs,
        )
        return IngestedClaim(
            claim=claim,
            belief=belief,
            linked_premise_ids=linked,
            reactivated=reactivated,
        )

    def _link_premises(self, claim: ExtractedClaim) -> list[str]:
        """Resolve premises_referred substrings to active belief IDs.

        Strategy (in order):
        1. If a premise ref looks like a belief ID, use it directly.
        2. Exact normalized proposition match.
        3. Substring match (ref appears in proposition or vice versa).
        4. Entity overlap (shared canonical entities).
        Returns a list of belief IDs (may be empty).

        IMPORTANT: Only links when claim.premises_referred is non-empty.
        Sharing an entity ("Acme") is a candidate retrieval signal for
        contradiction detection, NEVER a logical premise (justification).
        Auto-linking by entity turns independent ground facts into fragile
        derived chains that falsely cascade-retract.
        """
        if not claim.premises_referred:
            # No explicit premises: this is a ground fact. Do NOT guess
            # justifications from shared entities.
            return []

        linked: list[str] = []
        active = self.store.active_beliefs()
        for ref in claim.premises_referred:
            ref_norm = _norm(ref)
            if not ref_norm:
                continue
            # 1. Direct belief ID.
            direct = self.store.get(ref)
            if direct is not None and direct.is_active:
                if direct.id not in linked:
                    linked.append(direct.id)
                continue
            best: Belief | None = None
            best_score = 0
            for b in active:
                b_norm = _norm(b.proposition)
                score = 0
                # 2. Exact match.
                if b_norm == ref_norm:
                    score = 100
                # 3. Substring match.
                elif ref_norm in b_norm or b_norm in ref_norm:
                    # Longer overlap = stronger.
                    overlap = len(ref_norm) if ref_norm in b_norm else len(b_norm)
                    score = 50 + min(overlap, 40)
                # 4. Entity overlap.
                else:
                    ref_ents = {canon_entity(w) for w in ref_norm.split()}
                    b_ents = entities_of(b.proposition)
                    shared = ref_ents & b_ents
                    if shared:
                        score = 10 + 5 * len(shared)
                if score > best_score:
                    best_score = score
                    best = b
            if best is not None and best_score >= 10:
                if best.id not in linked:
                    linked.append(best.id)
        return linked

    def _check_reactivation(self, claim: ExtractedClaim) -> bool:
        """True if a retracted belief holds this proposition.

        The subsequent assert_belief will create a new belief; the store's
        re-derivation logic then re-evaluates dependents of the retracted
        lineage.
        """
        claim_norm = _norm(claim.proposition)
        for b in self.store.all_beliefs():
            if (not b.is_active
                    and _norm(b.proposition) == claim_norm):
                return True
        return False
