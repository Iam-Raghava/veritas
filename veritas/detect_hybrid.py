"""Hybrid contradiction detector: heuristic speed + NLI robustness.

The #1 weakness of pure heuristic detection is brittleness on paraphrase
and complex semantics. The #1 weakness of pure NLI is cost (model inference
on every pair). This detector combines them:

1. Heuristic first: if both propositions parse as functional properties
   ("X's Y is Z"), use the fast transparent check. No model needed.
2. NLI fallback: if heuristic can't parse (returns None/uncertain), use
   the NLI model for semantic contradiction judgment.
3. Fail-open: if NLI model is unavailable, fall back to heuristic-only.

This is the recommended production detector. It catches:
- Functional conflicts: "Acme's CEO is Jane" vs "Acme's CEO is John" (heuristic)
- Paraphrases: "Acme's chief executive is Jane" vs "Acme's CEO is John" (NLI)
- Negations: "Server is online" vs "Server is not online" (heuristic + NLI)
- Complex semantics: "All deployments failed" vs "Deployment X succeeded" (NLI)

It does NOT catch:
- Temporal contradictions requiring world knowledge ("CEO in 2020" vs "CEO in 2024"
  are both true at different times — use valid_from/valid_until, not contradiction)
- Contradictions requiring multi-hop reasoning across 3+ beliefs
"""
from __future__ import annotations

from typing import Any

from .belief import Belief
from .detect import Detector, functional_parts, heuristic_contradiction


class HybridDetector(Detector):
    """Heuristic + NLI hybrid. Best of both: speed and robustness."""

    name = "hybrid"

    def __init__(
        self,
        nli_model: str = "roberta-large-mnli",
        nli_threshold: float = 0.7,
        _nli_pipe: Any | None = None,
    ) -> None:
        self._nli_model = nli_model
        self._nli_threshold = nli_threshold
        self._nli_pipe = _nli_pipe
        self._nli_failed = False

    def contradicts(self, a: Belief, b: Belief) -> bool:
        # Layer 1: heuristic (fast, transparent)
        # If both parse as functional, we have a definitive answer.
        pa = functional_parts(a.proposition)
        pb = functional_parts(b.proposition)
        if pa is not None and pb is not None:
            return heuristic_contradiction(a, b)

        # Layer 2: NLI fallback for unparsable pairs
        return self._nli_contradicts(a.proposition, b.proposition)

    def _nli_contradicts(self, p1: str, p2: str) -> bool:
        if self._nli_failed:
            return False
        try:
            pipe = self._ensure_nli()
        except ImportError:
            self._nli_failed = True
            return False

        # Test both directions; contradiction in either direction counts.
        for prem, hyp in [(p1, p2), (p2, p1)]:
            try:
                result = pipe(f"{prem} </s> {hyp}")
                # result is list of {label, score} dicts
                for r in result:
                    label = r.get("label", "").upper()
                    score = r.get("score", 0.0)
                    if "CONTRADICTION" in label and score >= self._nli_threshold:
                        return True
            except Exception:
                continue
        return False

    def _ensure_nli(self):
        if self._nli_pipe is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise ImportError(
                    "HybridDetector NLI fallback requires "
                    "'pip install veritas-tms[nli]'"
                ) from exc
            self._nli_pipe = pipeline(
                "text-classification",
                model=self._nli_model,
                top_k=None,
            )
        return self._nli_pipe
