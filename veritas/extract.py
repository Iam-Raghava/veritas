"""Atomic claim extraction for Veritas.

Enterprise inputs are paragraphs, emails, and tool traces — not atomic
propositions. This module splits raw text into atomic claims that the
contradiction detectors can actually evaluate.

Two implementations:
- HeuristicClaimExtractor (default): rule-based, no model required.
  Splits sentences, extracts SPO triplets via patterns.
- ClaimExtractor Protocol: plug in an SLM or LLM for better extraction.

Usage:
    from veritas.extract import HeuristicClaimExtractor
    ex = HeuristicClaimExtractor()
    claims = ex.extract_claims("Acme's CEO is Jane. Revenue grew 20%.")
    # ["Acme's CEO is Jane", "Revenue grew 20%"]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Claim:
    """An atomic claim extracted from raw text."""
    text: str
    # Character offsets in the source text (for provenance).
    start: int = 0
    end: int = 0
    # Optional SPO triplet if extraction succeeded.
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    # Confidence in the extraction (1.0 = rule-based certain).
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)


class ClaimExtractor(Protocol):
    """Protocol for claim extractors. Implement for SLM/LLM backends."""

    def extract_claims(self, text: str) -> list[Claim]:
        """Split raw text into atomic claims."""
        ...


# Sentence splitting: handle abbreviations, decimals, etc.
_SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9\"'])"
)

# SPO patterns for common claim structures.
# These mirror the heuristic detector's patterns so extracted claims
# are directly evaluable.
_SPO_PATTERNS = [
    # "X's Y is Z" (possessive functional)
    re.compile(
        r"^\s*(?P<subj>.+?)'s\s+(?P<pred>.+?)\s+is\s+(?P<obj>.+?)\s*$",
        re.IGNORECASE,
    ),
    # "X is Y" (copular)
    re.compile(
        r"^\s*(?P<subj>.+?)\s+is\s+(?P<obj>.+?)\s*$",
        re.IGNORECASE,
    ),
    # "X has Y" (possessive)
    re.compile(
        r"^\s*(?P<subj>.+?)\s+has\s+(?P<obj>.+?)\s*$",
        re.IGNORECASE,
    ),
    # "X will Y" (future)
    re.compile(
        r"^\s*(?P<subj>.+?)\s+will\s+(?P<obj>.+?)\s*$",
        re.IGNORECASE,
    ),
]


class HeuristicClaimExtractor:
    """Rule-based claim extractor. No model required.

    Splits text into sentences, then attempts SPO extraction on each.
    Sentences that don't match SPO patterns are returned as-is (atomic
    by sentence boundary).
    """

    def __init__(self, min_length: int = 3) -> None:
        self.min_length = min_length

    def extract_claims(self, text: str) -> list[Claim]:
        """Extract atomic claims from raw text."""
        if not text or not text.strip():
            return []
        claims: list[Claim] = []
        # Split into sentences, track offsets.
        offset = 0
        sentences = _SENTENCE_SPLIT.split(text)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < self.min_length:
                offset += len(sent) + 1
                continue
            # Find start offset (approximate).
            start = text.find(sent, offset)
            if start == -1:
                start = offset
            end = start + len(sent)
            offset = end
            # Try SPO extraction.
            claim = self._extract_spo(sent, start, end)
            claims.append(claim)
        return claims

    def _extract_spo(self, sentence: str, start: int, end: int) -> Claim:
        """Attempt SPO triplet extraction; fallback to raw sentence."""
        for pat in _SPO_PATTERNS:
            m = pat.match(sentence)
            if m:
                gd = m.groupdict()
                return Claim(
                    text=sentence,
                    start=start,
                    end=end,
                    subject=gd.get("subj", "").strip() or None,
                    predicate=gd.get("pred", "").strip() or None,
                    object=gd.get("obj", "").strip() or None,
                    confidence=0.9,
                    metadata={"pattern": "spo"},
                )
        # No SPO match: return as atomic sentence claim.
        return Claim(
            text=sentence,
            start=start,
            end=end,
            confidence=0.7,
            metadata={"pattern": "sentence"},
        )

    def extract_propositions(self, text: str) -> list[str]:
        """Convenience: return just the claim texts."""
        return [c.text for c in self.extract_claims(text)]
