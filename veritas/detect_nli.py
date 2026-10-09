"""NLI-based contradiction detector (optional).

Uses a natural-language-inference model to judge contradiction, for
production use where the heuristic detector is too narrow.

Requires: pip install veritas-tms[nli]

The model is loaded lazily on first use. Both directions are tested
(premise->hypothesis and hypothesis->premise); a pair contradicts if
*either* direction predicts CONTRADICTION above threshold.
"""
from __future__ import annotations

from typing import Any

from .belief import Belief
from .detect import Detector


class HuggingFaceNLIDetector(Detector):
    """NLI contradiction detector backed by HuggingFace transformers."""

    name = "nli"

    def __init__(
        self,
        model_name: str = "roberta-large-mnli",
        threshold: float = 0.7,
        _pipeline: Any | None = None,  # injection point for tests
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self._pipe = _pipeline

    def _ensure_pipeline(self):
        if self._pipe is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise ImportError(
                    "HuggingFaceNLIDetector requires "
                    "'pip install veritas-tms[nli]'"
                ) from exc
            self._pipe = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,
            )
        return self._pipe

    def _contradiction_score(self, premise: str, hypothesis: str) -> float:
        pipe = self._ensure_pipeline()
        results = pipe(f"{premise} </s></s> {hypothesis}")
        # Normalize: pipelines may return [{...}] or [[{...}]].
        if results and isinstance(results[0], list):
            results = results[0]
        for r in results or []:
            label = str(r.get("label", "")).upper()
            if "CONTRADICT" in label:
                return float(r.get("score", 0.0))
        return 0.0

    def contradicts(self, a: Belief, b: Belief) -> bool:
        if a.id == b.id:
            return False
        return (
            self._contradiction_score(a.proposition, b.proposition)
            >= self.threshold
            or self._contradiction_score(b.proposition, a.proposition)
            >= self.threshold
        )
