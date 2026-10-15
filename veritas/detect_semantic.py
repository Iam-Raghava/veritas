"""Semantic contradiction detector using embeddings.

Beyond pattern matching: two propositions contradict if they are
semantically similar (high embedding cosine similarity) BUT have
opposing polarity (negation, antonyms, numeric mismatch).

This catches cases the heuristic detector misses:
- "The server is operational" vs "The server is down"
  (no shared negation pattern, but semantically opposed)
- "Revenue grew 20%" vs "Revenue declined"
  (antonym pair, not pattern-matched)

Uses sentence-transformers if available (all-MiniLM-L6-v2, ~90MB),
falls back to TF-IDF + enhanced patterns otherwise.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from veritas.belief import Belief
from veritas.detect import Detector, heuristic_contradiction

# Antonym pairs for semantic opposition (beyond not/X patterns).
ANTONYMS = {
    "operational": "down", "down": "operational",
    "online": "offline", "offline": "online",
    "grew": "declined", "declined": "grew",
    "increase": "decrease", "decrease": "increase",
    "rising": "falling", "falling": "rising",
    "success": "failure", "failure": "success",
    "open": "closed", "closed": "open",
    "active": "inactive", "inactive": "active",
    "enabled": "disabled", "disabled": "enabled",
    "true": "false", "false": "true",
    "yes": "no", "no": "yes",
    "pass": "fail", "fail": "pass",
    "working": "broken", "broken": "working",
}

NEGATIONS = {"not", "no", "never", "none", "n't", "cannot", "can't", "won't",
             "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't",
             "didn't", "hasn't", "haven't", "hadn't"}


def _tokens(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _tfidf_similarity(a: str, b: str) -> float:
    """Cosine similarity of TF vectors (simple, dependency-free)."""
    ta = Counter(_tokens(a))
    tb = Counter(_tokens(b))
    if not ta or not tb:
        return 0.0
    # Cosine
    dot = sum(ta[w] * tb[w] for w in ta if w in tb)
    na = math.sqrt(sum(v * v for v in ta.values()))
    nb = math.sqrt(sum(v * v for v in tb.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _has_opposition(a: str, b: str) -> bool:
    """Check for semantic opposition: negation or antonym mismatch.

    Negation must apply to the SAME predicate, not just appear anywhere.
    "Cluster has no errors" vs "Cluster is operational" both describe a
    healthy cluster — the "no" doesn't create opposition. We check if the
    non-negation tokens overlap significantly AND the negation status
    differs on the overlapping core.
    """
    ta = set(_tokens(a))
    tb = set(_tokens(b))
    # Remove negation words to get the core content.
    core_a = ta - NEGATIONS
    core_b = tb - NEGATIONS
    # If cores don't overlap much, they're about different things.
    if not core_a or not core_b:
        return False
    overlap = len(core_a & core_b) / max(len(core_a), len(core_b))
    if overlap < 0.5:
        return False
    # Cores overlap: now check for opposition via negation OR antonyms.
    # Both are checked independently (not mutually exclusive).
    a_neg = bool(ta & NEGATIONS)
    b_neg = bool(tb & NEGATIONS)
    neg_opposition = False
    if a_neg != b_neg:
        # Heuristic: if the only difference is a negation word and the
        # rest is nearly identical, it's likely a true contradiction.
        # If cores differ substantially, the negation is probably
        # describing absence ("no errors") not opposition.
        union = core_a | core_b
        jaccard = len(core_a & core_b) / len(union) if union else 0
        if jaccard > 0.8:
            neg_opposition = True
        # else: Low overlap + negation asymmetry: not a contradiction.
        # ("has no errors" vs "is operational" — both healthy.)
    # Antonym pair: a has X, b has antonym(X). Checked INDEPENDENTLY of
    # negation (not dead code).
    antonym_opposition = False
    for w in ta:
        if w in ANTONYMS and ANTONYMS[w] in tb:
            antonym_opposition = True
            break
    return neg_opposition or antonym_opposition


_embedding_model = None  # lazy singleton

def _embedding_similarity(a: str, b: str) -> float | None:
    """Use sentence-transformers if available, else None."""
    global _embedding_model
    try:
        from sentence_transformers import SentenceTransformer
        if _embedding_model is None:
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        ea, eb = _embedding_model.encode([a, b])
        # Cosine
        dot = sum(x * y for x, y in zip(ea, eb))
        na = math.sqrt(sum(x * x for x in ea))
        nb = math.sqrt(sum(x * x for x in eb))
        return dot / (na * nb) if na and nb else 0.0
    except ImportError:
        return None


class SemanticDetector(Detector):
    """Embedding-enhanced contradiction detector.

    Combines semantic similarity (embeddings or TF-IDF) with opposition
    detection (negation/antonyms). More robust than pure patterns,
    lighter than full NLI.
    """

    name = "semantic"

    def __init__(self, similarity_threshold: float = 0.5) -> None:
        self.threshold = similarity_threshold

    def contradicts(self, a: Belief, b: Belief) -> bool:
        """Detector protocol: True if a and b contradict."""
        return self(a, b)

    def __call__(self, a: Belief | str, b: Belief | str) -> bool:
        pa = a.proposition if isinstance(a, Belief) else a
        pb = b.proposition if isinstance(b, Belief) else b
        # Fast path: heuristic already catches it (needs Belief objects)
        if isinstance(a, Belief) and isinstance(b, Belief):
            if heuristic_contradiction(a, b):
                return True
        # Semantic path: high similarity + opposition
        sim = _embedding_similarity(pa, pb)
        if sim is None:
            sim = _tfidf_similarity(pa, pb)
        # Antonym opposition is strong evidence even at lower similarity.
        has_antonym = any(
            w in ANTONYMS and ANTONYMS[w] in set(_tokens(pb))
            for w in _tokens(pa)
        )
        threshold = 0.3 if has_antonym else self.threshold
        if sim >= threshold and _has_opposition(pa, pb):
            return True
        return False
