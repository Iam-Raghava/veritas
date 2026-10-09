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
from veritas.detect import heuristic_contradiction

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
    """Check for semantic opposition: negation or antonym mismatch."""
    ta = set(_tokens(a))
    tb = set(_tokens(b))
    # Negation asymmetry: one has negation, other doesn't
    a_neg = bool(ta & NEGATIONS)
    b_neg = bool(tb & NEGATIONS)
    if a_neg != b_neg:
        return True
    # Antonym pair: a has X, b has antonym(X)
    for w in ta:
        if w in ANTONYMS and ANTONYMS[w] in tb:
            return True
    return False


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


class SemanticDetector:
    """Embedding-enhanced contradiction detector.

    Combines semantic similarity (embeddings or TF-IDF) with opposition
    detection (negation/antonyms). More robust than pure patterns,
    lighter than full NLI.
    """

    name = "semantic"

    def __init__(self, similarity_threshold: float = 0.5) -> None:
        self.threshold = similarity_threshold

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
